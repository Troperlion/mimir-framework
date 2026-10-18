#!/usr/bin/env python3
"""Persistent Mímir operation compiler. Production always requires a TPU.

Only --reference-test permits CPU, for differential tests against CpuRuntime.
All tensor arithmetic, including VJPs, is JIT compiled on the selected device.
The binary protocol is private: little endian u32 JSON length, JSON, F32 arrays.
"""
import argparse
from collections import OrderedDict
import json
import math
import os
import struct
import sys

# Never let diagnostics corrupt the binary stdout protocol.
WIRE = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_default_matmul_precision", "highest")

OPS = frozenset("""Conv2d ConvTranspose2d Conv1d DepthwiseConv2d Linear Bilinear
Embedding EmbeddingBag BatchNorm2d BatchNorm1d LayerNorm GroupNorm InstanceNorm2d
RMSNorm ReLU LeakyReLU GELU GEGLU SiLU Tanh Sigmoid Softmax LogSoftmax Softplus Mish
HardSigmoid HardSwish MaxPool2d AvgPool2d AdaptiveAvgPool2d GlobalAvgPool2d MaxPool1d
AvgPool1d TokenMeanPool Dropout Dropout2d AlphaDropout Flatten Reshape Transpose
Permute Squeeze Unsqueeze View Add Subtract Multiply Divide Reparameterize Concat
Split Chunk Stack MatMul BatchMatMul NMS SelfAttention MultiHeadAttention
CrossAttention UpsampleNearest UpsampleBilinear UpsampleBicubic PixelShuffle
ZeroPad2d ReflectionPad2d ReplicationPad2d LSTM GRU RNN Identity Constant PatchEmbed""".split())
DROP = frozenset(("Dropout", "Dropout2d", "AlphaDropout"))


def target_profile(device_kind,device_count,requested="auto"):
    # PJRT names follow JAX's chip_version_from_device_kind mapping:
    # https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py
    kinds={"TPU v5e":"TPU-v5e1","TPU v5 lite":"TPU-v5e1",
           "TPU v6e":"TPU-v6e1","TPU v6 lite":"TPU-v6e1"}
    profile=kinds.get(device_kind)
    if profile is None:raise ValueError(f"unsupported TPU kind {device_kind!r}; expected TPU-v5e1 or TPU-v6e1")
    if device_count!=1:raise ValueError(f"{profile} requires one visible TPU device, got {device_count}")
    key=requested.lower().replace("tpu","").replace("-","").replace("_","").strip()
    aliases={"auto":None,"v5e1":"TPU-v5e1","v6e1":"TPU-v6e1"}
    if key not in aliases:raise ValueError("MIMIR_TPU_TARGET must be auto, TPU-v5e1 or TPU-v6e1")
    if aliases[key] is not None and aliases[key]!=profile:
        raise ValueError(f"requested {aliases[key]}, but PJRT detected {profile} ({device_kind})")
    return profile


def rounded(x):
    """std::llround: half away from zero (not numpy/JAX round-to-even)."""
    return (jnp.sign(x) * jnp.floor(jnp.abs(x) + .5)).astype(jnp.int32)


def gelu(x):
    return .5*x*(1+jnp.tanh(math.sqrt(2/math.pi)*(x+.044715*x*x*x)))


@jax.custom_jvp
def protected_divisor(x):
    return jnp.where(jnp.abs(x)<1e-8, jnp.where(x>=0, 1e-8, -1e-8), x)


@protected_divisor.defjvp
def divisor_jvp(primals, tangents):
    # CPU differentiates using the protected divisor, even inside the clamp.
    return protected_divisor(primals[0]), tangents[0]


def activation(x, kind, param=0.01):
    if kind == 0: return x
    if kind == 1: return jnp.where(x>0, x, 0.)
    if kind == 2: return jnp.where(x<=0, 0., jnp.where(x>=6, 6., x))
    if kind == 3: return jnp.where(x>0, x, param*x)
    if kind == 4: return jnp.where(x>0, x, param*jnp.expm1(x))
    if kind == 5: return 1.0507009873554805*jnp.where(x>0, x, 1.6732632423543773*jnp.expm1(x))
    if kind == 6: return gelu(x)
    if kind == 7: return x*jax.nn.sigmoid(x)
    if kind == 8: return x*jnp.tanh(jax.nn.softplus(x))
    if kind == 9: return jnp.tanh(x)
    if kind == 10: return jax.nn.sigmoid(x)
    if kind == 11: return jax.nn.softmax(x)
    if kind == 12: return jax.nn.softplus(x)
    if kind == 13: return x/(1+jnp.abs(x))
    raise ValueError("unknown activation")


def axis_shape(l,size):
    shape=l["shape"]
    if not shape or any(d<=0 for d in shape) or math.prod(shape)!=size:
        shape=[size]
        for key in ("in_features","embed_dim","seq_len"):
            d=l[key]
            if d>0 and size%d==0:
                shape=[d,size//d] if key=="seq_len" else [size//d,d];break
    axis=l["axis"]
    if axis<0:axis+=len(shape)
    if not 0<=axis<len(shape):axis=len(shape)-1
    return shape,axis


def spatial(l, n, channels, infer=False, default=32):
    h, w = l["input_height"] or default, l["input_width"] or default
    if infer and n % channels == 0 and h*w != n//channels:
        area = n//channels
        if l["input_height"]>0 and area%l["input_height"]==0: h,w=l["input_height"],area//l["input_height"]
        elif l["input_width"]>0 and area%l["input_width"]==0: h,w=area//l["input_width"],l["input_width"]
        elif math.isqrt(area)**2 == area: h=w=math.isqrt(area)
        else: h,w=1,area
    return h,w


def norm_affine(y, w, l, dim, channel_axis=-1, beta=True, partial=False):
    if not l["affine"]: return y
    gamma=jnp.ones((dim,),jnp.float32)
    n=min(dim,w.size)
    if n: gamma=gamma.at[:n].set(w[:n])
    if not partial and w.size < dim: raise ValueError("affine weights missing")
    shape=[1]*y.ndim;shape[channel_axis]=dim
    y=y*gamma.reshape(shape)
    if beta and l["use_bias"] and w.size>=dim*2: y=y+w[dim:dim*2].reshape(shape)
    return y


def normalize(x, w, l, training):
    t=l["type"];eps=l["eps"]
    if t in ("LayerNorm","RMSNorm"):
        d=l["in_features"]
        if t=="RMSNorm" and d<=0:
            d=l["embed_dim"] or (l["shape"][-1] if l["shape"] and l["shape"][-1]>0 else 0) or (l["target_shape"][-1] if l["target_shape"] and l["target_shape"][-1]>0 else 0)
        d=d or x.size
        a=x.reshape((-1,d))
        z=a if t=="RMSNorm" else a-jnp.mean(a,axis=-1,keepdims=True)
        y=z*lax.rsqrt(jnp.mean(z*z,axis=-1,keepdims=True)+eps)
        return norm_affine(y,w,l,d,beta=t!="RMSNorm",partial=t=="RMSNorm").ravel()
    if t.startswith("BatchNorm"):
        c=l["in_channels"] or l["out_channels"] or len(l["running_mean"])
        if not c and l["affine"]: c=w.size//2 if l["use_bias"] and w.size%2==0 else w.size
        s=l["input_height"]*l["input_width"] if t=="BatchNorm2d" else 1
        if s<=0 or x.size%(c*s):s=1
        a=x.reshape((-1,c,s))
        running=l["track_running_stats"] and len(l["running_mean"])==c and len(l["running_var"])==c
        if training or not running:
            mean=jnp.mean(a,axis=(0,2),keepdims=True)
            var=jnp.mean((a-mean)**2,axis=(0,2),keepdims=True)
        else:
            mean=jnp.asarray(l["running_mean"],jnp.float32).reshape((1,c,1))
            var=jnp.asarray(l["running_var"],jnp.float32).reshape((1,c,1))
        return norm_affine((a-mean)*lax.rsqrt(var+eps),w,l,c,1,partial=True).ravel()
    c=l["in_channels"] or (3 if t=="InstanceNorm2d" else 64)
    h,width=spatial(l,x.size,c)
    groups=c if t=="InstanceNorm2d" else l["num_groups"]
    a=x.reshape((groups,-1));z=a-jnp.mean(a,axis=1,keepdims=True)
    y=(z*lax.rsqrt(jnp.mean(z*z,axis=1,keepdims=True)+eps)).reshape((c,h,width))
    # InstanceNorm's CPU forward always reads beta when affine is true.
    cfg=dict(l)
    if t=="InstanceNorm2d":cfg["use_bias"]=True
    return norm_affine(y,w,cfg,c,0).ravel()


def convolution(x,w,l):
    t=l["type"]
    if t=="Conv1d":
        ci=l["in_channels"] or 1;co=l["out_channels"] or 1;k=l["kernel_h"] or 3
        s=l["stride_h"] if l["stride_h"]>0 else 1;p=max(0,l["pad_h"])
        count=co*ci*k
        y=lax.conv_general_dilated(x.reshape((1,ci,-1)),w[:count].reshape((co,ci,k)),(s,),((p,p),),dimension_numbers=("NCH","OIH","NCH"),precision=lax.Precision.HIGHEST)
    else:
        depth=t=="DepthwiseConv2d"
        ci=l["in_channels"] or (3 if depth else 1);co=ci if depth else l["out_channels"] or 1
        h,width=spatial(l,x.size,ci,infer=not depth,default=0 if not depth else 32)
        k=l["kernel_h"] or (3 if depth else l["kh"])
        s=(l["stride_h"] or 1) if depth else l["sh"]
        p=max(0,l["pad_h"]) if depth else l["ph"]
        count=co*(1 if depth else ci)*k*k
        a=x.reshape((1,ci,h,width));kernel=w[:count].reshape((co,1 if depth else ci,k,k))
        if t=="ConvTranspose2d":
            # CPU stores [out,in,kh,kw], scatters without a spatial flip.
            pad=k-1-p
            y=lax.conv_general_dilated(a,kernel[:,:,::-1,::-1],(1,1),((pad,pad),(pad,pad)),lhs_dilation=(s,s),dimension_numbers=("NCHW","OIHW","NCHW"),precision=lax.Precision.HIGHEST)
        else:
            dilation=1 if depth else (l["dilation_h"] if l["dilation_h"]!=1 else l["dilation"])
            y=lax.conv_general_dilated(a,kernel,(s,s),((p,p),(p,p)),rhs_dilation=(dilation,dilation),feature_group_count=ci if depth else 1,dimension_numbers=("NCHW","OIHW","NCHW"),precision=lax.Precision.HIGHEST)
    if l["use_bias"]: y=y+w[count:count+co].reshape((1,co)+((1,)*(y.ndim-2)))
    return y.ravel()


def pooling(x,l):
    t=l["type"]
    if t in ("GlobalAvgPool2d","AdaptiveAvgPool2d"):
        c=l["in_channels"] or 64;h,w=spatial(l,x.size,c)
        return jnp.mean(x.reshape((c,h*w)),axis=1)
    if t=="TokenMeanPool":return jnp.mean(x.reshape((l["seq_len"],l["embed_dim"])),axis=0)
    one=t.endswith("1d");avg=t.startswith("Avg")
    c=l["in_channels"] or (1 if one or not avg else 64)
    if one:
        h=1;width=x.size//c;kh=1;kw=l["kernel_h"] or 2
        sh=1;sw=l["stride_h"] if l["stride_h"]>0 else kw;ph=0;pw=max(0,l["pad_h"])
    else:
        h,width=spatial(l,x.size,c,default=32 if avg else 1)
        kh=l["kh"];kw=l["kw"] if avg else kh
        sh=l["sh"];sw=l["sw"] if avg else sh;ph=l["ph"];pw=l["pw"] if avg else ph
    oh=(h+2*ph-kh)//sh+1;ow=(width+2*pw-kw)//sw+1
    if min(oh,ow,sh,sw)<=0:raise ValueError("invalid pooling shape")
    # Explicit gather ensures max ties choose the first valid element, as CPU.
    yy=jnp.arange(oh)[:,None,None,None]*sh+jnp.arange(kh)[None,None,:,None]-ph
    xx=jnp.arange(ow)[None,:,None,None]*sw+jnp.arange(kw)[None,None,None,:]-pw
    valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<width)
    patches=x.reshape((c,h,width))[:,jnp.clip(yy,0,h-1),jnp.clip(xx,0,width-1)]
    patches=jnp.where(valid,patches,0. if avg else -jnp.inf).reshape((c,oh,ow,kh*kw))
    if avg:return (jnp.sum(patches,axis=-1)/jnp.maximum(1,jnp.sum(valid,axis=(-1,-2)))).ravel()
    ix=jnp.argmax(patches,axis=-1,keepdims=True)
    return jnp.take_along_axis(patches,ix,axis=-1).ravel()


def resize(x,l):
    t=l["type"];h=l["input_height"] or l["out_h"];w=l["input_width"] or l["out_w"]
    c=l["in_channels"] or (3 if t=="UpsampleBicubic" else 0)
    if min(h,w,c)<=0:raise ValueError("invalid resize shape")
    a=x.reshape((c,h,w))
    oh=l["output_height"] or l["out_h"];ow=l["output_width"] or l["out_w"]
    if t=="UpsampleNearest":
        sh=int(math.floor(l["scale_h"]+.5)) if l["scale_h"]>0 else max(1,oh//h) if oh else 2
        sw=int(math.floor(l["scale_w"]+.5)) if l["scale_w"]>0 else max(1,ow//w) if ow else 2
        return jnp.repeat(jnp.repeat(a,sh,axis=1),sw,axis=2).ravel()
    oh=oh or max(1,int(math.floor(h*max(0,l["scale_h"])+.5)))
    ow=ow or max(1,int(math.floor(w*max(0,l["scale_w"])+.5)))
    yy=jnp.arange(oh,dtype=jnp.float32)*h/oh;xx=jnp.arange(ow,dtype=jnp.float32)*w/ow
    if t=="UpsampleBilinear":
        y0=jnp.floor(yy).astype(jnp.int32);x0=jnp.floor(xx).astype(jnp.int32)
        y1=jnp.minimum(y0+1,h-1);x1=jnp.minimum(x0+1,w-1)
        dy=(yy-y0)[:,None];dx=(xx-x0)[None,:]
        return (a[:,y0[:,None],x0[None,:]]*(1-dy)*(1-dx)+a[:,y0[:,None],x1[None,:]]*(1-dy)*dx+a[:,y1[:,None],x0[None,:]]*dy*(1-dx)+a[:,y1[:,None],x1[None,:]]*dy*dx).ravel()
    yy=(jnp.arange(oh,dtype=jnp.float32)+.5)*h/oh-.5;xx=(jnp.arange(ow,dtype=jnp.float32)+.5)*w/ow-.5
    y0=jnp.floor(yy).astype(jnp.int32);x0=jnp.floor(xx).astype(jnp.int32)
    def cubic(v):
        v=jnp.abs(v);a=-.75
        return jnp.where(v<=1,((a+2)*v-(a+3))*v*v+1,jnp.where(v<2,((a*v-5*a)*v+8*a)*v-4*a,0.))
    y=jnp.zeros((c,oh,ow),jnp.float32);total=jnp.zeros((oh,ow),jnp.float32)
    for dy in (-1,0,1,2):
        for dx in (-1,0,1,2):
            weight=cubic(yy-y0-dy)[:,None]*cubic(xx-x0-dx)[None,:]
            y=y+weight*a[:,jnp.clip(y0+dy,0,h-1)[:,None],jnp.clip(x0+dx,0,w-1)[None,:]];total=total+weight
    return (y/total).ravel()


def attention(xs,w,l):
    x=xs[0];t=l["type"];heads=l["num_heads"] or 1
    d=l["embed_dim"] or (l["head_dim"]*heads if t=="CrossAttention" else x.size)
    hd=d//heads
    if d%heads:raise ValueError("attention heads must divide embedding")
    if t=="CrossAttention":
        kd=l["in_features"] or d;q=x.reshape((-1,d));kv=xs[1].reshape((-1,kd))
        a=d*d;b=a+kd*2*d;c=b+d*d
        q=q@w[:a].reshape((d,d));kv=kv@w[a:b].reshape((kd,2*d));k,v=jnp.split(kv,2,axis=1)
        ow=w[b:c].reshape((d,d));bias=w[c:c+d] if w.size>=c+d else jnp.zeros((d,),jnp.float32)
    else:
        n=l["seq_len"] or 1;a=3*d*d;b=a+d*d
        qkv=x.reshape((n,d))@w[:a].reshape((3*d,d)).T
        if w.size>=b+4*d:qkv=qkv+w[b:b+3*d];bias=w[b+3*d:b+4*d]
        elif w.size>=b+d:bias=w[b:b+d]
        else:bias=jnp.zeros((d,),jnp.float32)
        q,k,v=jnp.split(qkv,3,axis=1);ow=w[a:b].reshape((d,d))
    q=q.reshape((-1,heads,hd)).transpose((1,0,2));k=k.reshape((-1,heads,hd)).transpose((1,0,2));v=v.reshape((-1,heads,hd)).transpose((1,0,2))
    score=(q@k.transpose((0,2,1)))/math.sqrt(hd)
    if l["causal"]:score=jnp.where(jnp.arange(k.shape[1])[None,:]>jnp.arange(q.shape[1])[:,None],-1e9,score)
    a=(jax.nn.softmax(score,axis=-1)@v).transpose((1,0,2)).reshape((-1,d))
    # CPU output projection uses row-major [in,out].
    return (a@ow+bias).ravel()


def recurrent(x,w,l):
    t=l["type"];n=l["seq_len"];i=l["in_features"];h=l["out_features"]
    gates={"RNN":1,"GRU":3,"LSTM":4}[t];a=gates*h*i;b=a+gates*h*h
    wi=w[:a].reshape((gates*h,i));wh=w[a:b].reshape((gates*h,h))
    bi=w[b:b+gates*h] if l["use_bias"] else jnp.zeros((gates*h,),jnp.float32)
    bh=w[b+gates*h:b+2*gates*h] if l["use_bias"] else jnp.zeros((gates*h,),jnp.float32)
    z=jnp.zeros((h,),jnp.float32)
    def step(state,xt):
        hp,cp,rprev=state
        xi=wi@xt+bi;hh=wh@hp+bh
        if t=="RNN": hn=jnp.tanh(xi+hh);return (hn,cp,rprev),hn
        if t=="LSTM":
            ig,fg,gg,og=jnp.split(xi+hh,4)
            cn=jax.nn.sigmoid(fg)*cp+jax.nn.sigmoid(ig)*jnp.tanh(gg)
            hn=jax.nn.sigmoid(og)*jnp.tanh(cn);return (hn,cn,rprev),hn
        r=jax.nn.sigmoid(xi[:h]+hh[:h]);zg=jax.nn.sigmoid(xi[h:2*h]+hh[h:2*h])
        # Mirror the CPU forward's ordered reset updates (including previous
        # time step values above the current hidden index).
        def hidden(k,carry):
            reset,ns=carry;reset=reset.at[k].set(r[k])
            sn=xi[2*h+k]+wh[2*h+k]@(reset*hp)+bh[2*h+k]
            return reset,ns.at[k].set(jnp.tanh(sn))
        reset,ns=lax.fori_loop(0,h,hidden,(rprev,z))
        hn=(1-zg)*ns+zg*hp;return (hn,cp,reset),hn
    return lax.scan(step,(z,z,z),x.reshape((n,i)))[1].ravel()


def nms(xs,l):
    boxes=xs[0].reshape((-1,4));scores=xs[1];n=scores.size
    if n==0:return jnp.empty((0,),jnp.float32),jnp.int32(0)
    classes=rounded(xs[2]) if len(xs)>2 else jnp.zeros((n,),jnp.int32)
    valid=jnp.all(jnp.isfinite(boxes),axis=1)&jnp.isfinite(scores)&(scores>=l["nms_score_threshold"])
    if len(xs)>2:valid=valid&jnp.isfinite(xs[2])
    order=jnp.argsort(-jnp.where(valid,scores,-jnp.inf),stable=True)
    out=jnp.full((n,),-1,jnp.int32);suppressed=~valid;limit=l["nms_max_detections"] or n
    def step(k,state):
        result,blocked,count=state;ix=order[k]
        accept=(~blocked[ix])&(count<limit)
        result=lax.cond(accept,lambda a:a.at[count].set(ix),lambda a:a,result)
        b=boxes[ix];area=jnp.maximum(0,boxes[:,2]-boxes[:,0])*jnp.maximum(0,boxes[:,3]-boxes[:,1])
        ar=jnp.maximum(0,b[2]-b[0])*jnp.maximum(0,b[3]-b[1])
        inter=jnp.maximum(0,jnp.minimum(boxes[:,2],b[2])-jnp.maximum(boxes[:,0],b[0]))*jnp.maximum(0,jnp.minimum(boxes[:,3],b[3])-jnp.maximum(boxes[:,1],b[1]))
        union=area+ar-inter;iou=jnp.where(union>0,inter/jnp.maximum(union,1e-30),0.)
        same=True if l["nms_class_agnostic"] or len(xs)<3 else classes==classes[ix]
        blocked=blocked|(accept&same&(iou>l["nms_iou_threshold"]))
        return result,blocked,count+accept.astype(jnp.int32)
    out,_,count=lax.fori_loop(0,n,step,(out,suppressed,jnp.int32(0)))
    return out.astype(jnp.float32),count


def forward(xs,w,l,training,noise,fused=False):
    t=l["type"];x=xs[0] if xs else jnp.empty((0,),jnp.float32)
    if t in ("Identity","Flatten","Reshape","View","Squeeze","Unsqueeze"): y=x
    elif t=="Linear":
        i=l["in_features"] or x.size;o=l["out_features"]
        y=x.reshape((-1,i))@w[:i*o].reshape((o,i)).T
        if l["use_bias"]:y=y+w[i*o:i*o+o]
    elif t=="Bilinear":
        i=l["in_features"];j=l["out_features"];o=l["embed_dim"] or l["target_shape"][0];n=i*j*o
        y=jnp.einsum("bi,oij,bj->bo",x.reshape((-1,i)),w[:n].reshape((o,i,j)),xs[1].reshape((-1,j)),precision=lax.Precision.HIGHEST)
        if l["use_bias"]:y=y+w[n:n+o]
    elif t in ("Conv1d","Conv2d","ConvTranspose2d","DepthwiseConv2d"):y=convolution(x,w,l)
    elif t in ("BatchNorm1d","BatchNorm2d","LayerNorm","GroupNorm","InstanceNorm2d","RMSNorm"):y=normalize(x,w,l,training)
    elif t=="ReLU":y=jnp.where(x>0,x,0.)
    elif t=="LeakyReLU":y=jnp.where(x>0,x,(l["leaky_relu_alpha"] if l["leaky_relu_alpha"]>0 else .01)*x)
    elif t=="GELU":y=gelu(x)
    elif t=="GEGLU":
        a,b=jnp.split(x.reshape((l["seq_len"],2*l["out_features"])),2,axis=1);y=a*gelu(b)
    elif t=="SiLU":y=x*jax.nn.sigmoid(x)
    elif t=="Tanh":y=jnp.tanh(x)
    elif t=="Sigmoid":y=jax.nn.sigmoid(x)
    elif t in ("Softplus","Mish"):
        sp=jnp.where(x>20,x,jax.nn.softplus(x));y=sp if t=="Softplus" else x*jnp.tanh(sp)
    elif t in ("HardSigmoid","HardSwish"):
        hs=jnp.where(x<=-3,0.,jnp.where(x>=3,1.,(x+3)/6));y=hs if t=="HardSigmoid" else x*hs
    elif t in ("Softmax","LogSoftmax"):
        shape,axis=axis_shape(l,x.size)
        y=(jax.nn.log_softmax if t=="LogSoftmax" else jax.nn.softmax)(x.reshape(shape),axis=axis)
    elif "Pool" in t:y=pooling(x,l)
    elif t in DROP:
        if not training:y=x
        else:
            p=min(1.,max(0.,l["dropout_p"]))
            if p>=1:y=jnp.zeros_like(x)
            elif t=="AlphaDropout":
                ap=-1.6732632423543773*1.0507009873554805;a=1/math.sqrt((1-p)*(1+p*ap*ap));b=-a*ap*p
                y=a*jnp.where(noise>0,x,ap)+b
            else:y=jnp.where(noise>0,x/(1-p),0.)
    elif t=="Transpose":y=x.reshape((l["in_features"],l["out_features"])).T
    elif t=="Permute":y=x.reshape(l["shape"] or [1,x.size]).transpose(l["permute_dims"])
    elif t=="Add":
        n=max(x.size,xs[1].size)
        if not min(x.size,xs[1].size) or n%min(x.size,xs[1].size):raise ValueError("Add size mismatch")
        y=jnp.tile(x,n//x.size)+jnp.tile(xs[1],n//xs[1].size)
    elif t in ("Subtract","Multiply","Divide"):
        if x.size!=xs[1].size:raise ValueError("elementwise size mismatch")
        y=x-xs[1] if t=="Subtract" else x*xs[1] if t=="Multiply" else x/protected_divisor(xs[1])
    elif t=="Reparameterize":y=x+jnp.exp(.5*jnp.clip(xs[1],-20,20))*noise if training else x
    elif t in ("Concat","Stack"):
        if len(xs)<2 or (t=="Stack" and any(a.size!=x.size for a in xs)):raise ValueError("invalid concatenation")
        y=jnp.concatenate(xs)
    elif t in ("Split","Chunk"):
        sizes=l["split_sizes"] if t=="Split" else []
        if not sizes:
            n=l["num_splits"] if t=="Split" else l["num_chunks"]
            if n<=0 or (t=="Split" and x.size%n):raise ValueError("invalid split")
            sizes=[x.size//n+(1 if t=="Chunk" and i<x.size%n else 0) for i in range(n)]
        if sum(sizes)!=x.size or any(s<0 for s in sizes):raise ValueError("invalid split sizes")
        return tuple(jnp.split(x,np.cumsum(sizes)[:-1]))
    elif t in ("MatMul","BatchMatMul"):
        m=l["in_features"];k=l["out_features"];n=l["embed_dim"]
        prefix=(l["seq_len"],) if t=="BatchMatMul" else ()
        y=x.reshape(prefix+(m,k))@xs[1].reshape(prefix+(k,n))
    elif t in ("Embedding","EmbeddingBag"):
        d=l["embed_dim"] or (l["out_features"] if t=="Embedding" else 1);v=max(1,l["vocab_size"])
        ids=rounded(lax.stop_gradient(x));valid=(ids>=0)&(ids<v)&(ids!=l["padding_idx"])
        table=w[:v*d].reshape((v,d));rows=jnp.where(valid[:,None],table[jnp.clip(ids,0,v-1)],0.)
        if t=="Embedding":y=rows
        else:
            offsets=rounded(lax.stop_gradient(xs[1])) if len(xs)>1 and xs[1].size else jnp.array([0,x.size],jnp.int32)
            if offsets.size<2:raise ValueError("EmbeddingBag needs >=2 offsets")
            pos=jnp.arange(x.size);start=jnp.clip(offsets[:-1],0,x.size);end=jnp.maximum(start,jnp.minimum(offsets[1:],x.size))
            y=((pos[None,:]>=start[:,None])&(pos[None,:]<end[:,None])).astype(jnp.float32)@rows
    elif t in ("SelfAttention","MultiHeadAttention","CrossAttention"):y=attention(xs,w,l)
    elif t.startswith("Upsample"):y=resize(x,l)
    elif t=="PixelShuffle":
        r=int(l["scale_h"]);c=l["in_channels"];h=l["input_height"];width=l["input_width"]
        if r<1 or r!=l["scale_h"] or c%(r*r):raise ValueError("invalid PixelShuffle")
        y=x.reshape((c//(r*r),r,r,h,width)).transpose((0,3,1,4,2)).reshape((c//(r*r),h*r,width*r))
    elif t in ("ZeroPad2d","ReflectionPad2d","ReplicationPad2d"):
        c=l["in_channels"] or 3;h,width=spatial(l,x.size,c);p=max(0,l["pad_h"]);q=max(0,l["pad_w"]) if t=="ZeroPad2d" else p
        a=x.reshape((c,h,width))
        if t=="ZeroPad2d":y=jnp.pad(a,((0,0),(p,p),(q,q)))
        else:
            yy=jnp.arange(h+2*p)-p;xx=jnp.arange(width+2*q)-q
            if t=="ReflectionPad2d":
                yy=jnp.where(yy<0,-yy,yy);yy=jnp.where(yy>=h,2*h-yy-2,yy)
                xx=jnp.where(xx<0,-xx,xx);xx=jnp.where(xx>=width,2*width-xx-2,xx)
            y=a[:,jnp.clip(yy,0,h-1)[:,None],jnp.clip(xx,0,width-1)[None,:]]
    elif t in ("LSTM","GRU","RNN"):y=recurrent(x,w,l)
    elif t=="Constant":
        d=l["embed_dim"];n=l["seq_len"]
        if d>0 and n>0 and l["shape"]==[-1,d] and w.size and w.size%d==0:
            rows=w.size//d;pos=jnp.arange(n,dtype=jnp.float32) if n<=rows else jnp.arange(n,dtype=jnp.float32)*(rows-1)/max(1,n-1)
            a=jnp.floor(pos).astype(jnp.int32);b=jnp.minimum(a+1,rows-1);f=pos-a;table=w.reshape((rows,d))
            y=(1-f[:,None])*table[a]+f[:,None]*table[b]
        else:y=w
    elif t=="PatchEmbed":
        d=l["embed_dim"] or l["out_features"];st=max(1,l["seq_text"]);np_=max(1,l["num_patches"]);pd=max(1,l["patch_dim"]);prefix=(st+1)*d
        patch=x[prefix:].reshape((np_,pd))/math.sqrt(pd)
        y=jnp.concatenate((x[:prefix],(patch@w[:pd*d].reshape((pd,d))+w[pd*d:pd*d+d]).ravel()))
    else:raise ValueError("unsupported layer "+t)
    y=y.ravel()
    if fused and t in ("Conv2d","ConvTranspose2d"):y=activation(y,l["activation"],l["activation_param"])
    return (y,)


def bias_gradient(gw,l):
    """Mirror CPU's separate grad_bias view; packed biases stay in gw too."""
    t=l["type"]
    if not l["use_bias"]:return jnp.empty((0,),jnp.float32)
    if t=="Linear":d=l["out_features"]
    elif t=="Bilinear":d=l["embed_dim"] or l["target_shape"][0]
    elif t in ("Conv1d","Conv2d","ConvTranspose2d"):d=l["out_channels"] or 1
    elif t=="DepthwiseConv2d":d=l["in_channels"] or 3
    elif t in ("LayerNorm","RMSNorm"):d=l["in_features"]
    elif t in ("GroupNorm","InstanceNorm2d"):d=l["in_channels"]
    else:return jnp.empty((0,),jnp.float32)
    return gw[-d:] if d and gw.size>=d else jnp.empty((0,),jnp.float32)


def gru_backward(x,w,go,l):
    # CPU backward reconstructs GRU state with a scalar reset for each row,
    # then uses the vector-reset adjoint. Preserve both conventions verbatim.
    h=l["out_features"];i=l["in_features"];t=l["seq_len"];a=3*h*i;b=a+3*h*h
    wi=w[:a].reshape((3*h,i));wh=w[a:b].reshape((3*h,h))
    bi=w[b:b+3*h] if l["use_bias"] else jnp.zeros((3*h,),jnp.float32)
    bh=w[b+3*h:b+6*h] if l["use_bias"] else jnp.zeros((3*h,),jnp.float32)
    zero=jnp.zeros((h,),jnp.float32)
    def step(hp,xt):
        xi=wi@xt+bi;hh=wh@hp+bh
        r=jax.nn.sigmoid(xi[:h]+hh[:h]);z=jax.nn.sigmoid(xi[h:2*h]+hh[h:2*h])
        n=jnp.tanh(xi[2*h:]+r*(wh[2*h:]@hp)+bh[2*h:])
        hn=(1-z)*n+z*hp
        return hn,(hp,r,z,n,xt)
    _,states=lax.scan(step,zero,x.reshape((t,i)))
    def reverse(carry,item):
        dhnext,gwi,gwh,gbi=carry
        hp,r,z,n,xt,g=item
        dh=g+dhnext;dn=dh*(1-z)*(1-n*n);dz=dh*(hp-n)*z*(1-z)
        drh=wh[2*h:].T@dn;dr=drh*hp*r*(1-r)
        da=jnp.concatenate((dr,dz,dn))
        dhp=dh*z+drh*r+wh[:h].T@dr+wh[h:2*h].T@dz
        dw=jnp.concatenate((jnp.outer(dr,hp),jnp.outer(dz,hp),jnp.outer(dn,r*hp)),axis=0)
        return (dhp,gwi+jnp.outer(da,xt),gwh+dw,gbi+da),da@wi
    init=(zero,jnp.zeros_like(wi),jnp.zeros_like(wh),jnp.zeros((3*h,),jnp.float32))
    (_,gwi,gwh,gb),gx=lax.scan(reverse,init,(*states,go.reshape((t,h))),reverse=True)
    gw=jnp.zeros_like(w).at[:a].set(gwi.ravel()).at[a:b].set(gwh.ravel())
    if l["use_bias"]:gw=gw.at[b:b+3*h].set(gb).at[b+3*h:b+6*h].set(gb)
    return (gx.ravel(),),gw


def lstm_backward(x,w,go,l):
    # The existing CPU backward reconstructs hidden components in place within
    # each time step. Its subsequent BPTT uses the previous complete state.
    h=l["out_features"];i=l["in_features"];t=l["seq_len"];a=4*h*i;b=a+4*h*h
    wi=w[:a].reshape((4*h,i));wh=w[a:b].reshape((4*h,h))
    bias=w[b:b+4*h]+w[b+4*h:b+8*h] if l["use_bias"] else jnp.zeros((4*h,),jnp.float32)
    zero=jnp.zeros((h,),jnp.float32)
    def step(state,xt):
        hp,cp=state;oldh=hp;oldc=cp;xi=wi@xt+bias
        def hidden(k,carry):
            hp,cp,gates=carry;ix=jnp.array([k,h+k,2*h+k,3*h+k])
            v=xi[ix]+wh[ix]@hp
            ig=jax.nn.sigmoid(v[0]);fg=jax.nn.sigmoid(v[1]);gg=jnp.tanh(v[2]);og=jax.nn.sigmoid(v[3])
            cn=fg*cp[k]+ig*gg;hn=og*jnp.tanh(cn)
            return hp.at[k].set(hn),cp.at[k].set(cn),gates.at[:,k].set(jnp.array([ig,fg,gg,og]))
        hn,cn,gates=lax.fori_loop(0,h,hidden,(hp,cp,jnp.zeros((4,h),jnp.float32)))
        return (hn,cn),(oldh,oldc,cn,gates,xt)
    _,states=lax.scan(step,(zero,zero),x.reshape((t,i)))
    def reverse(carry,item):
        dhnext,dcnext,gwi,gwh,gb=carry;hp,cp,cn,gates,xt,g=item
        ig,fg,gg,og=gates;dh=g+dhnext;tc=jnp.tanh(cn);dc=dcnext+dh*og*(1-tc*tc)
        da=jnp.concatenate((dc*gg*ig*(1-ig),dc*cp*fg*(1-fg),dc*ig*(1-gg*gg),dh*tc*og*(1-og)))
        return (da@wh,dc*fg,gwi+jnp.outer(da,xt),gwh+jnp.outer(da,hp),gb+da),da@wi
    init=(zero,zero,jnp.zeros_like(wi),jnp.zeros_like(wh),jnp.zeros((4*h,),jnp.float32))
    (_,_,gwi,gwh,gb),gx=lax.scan(reverse,init,(*states,go.reshape((t,h))),reverse=True)
    gw=jnp.zeros_like(w).at[:a].set(gwi.ravel()).at[a:b].set(gwh.ravel())
    if l["use_bias"]:gw=gw.at[b:b+4*h].set(gb).at[b+4*h:b+8*h].set(gb)
    return (gx.ravel(),),gw


def backward(xs,w,gos,saved,mask,l,training):
    t=l["type"];x=xs[0] if xs else jnp.empty((0,),jnp.float32);go=gos[0]
    if t=="GRU":return gru_backward(x,w,go,l)
    if t=="LSTM":return lstm_backward(x,w,go,l)
    if t=="LogSoftmax":
        shape,axis=axis_shape(l,x.size)
        logy=jax.nn.log_softmax(x.reshape(shape),axis=axis)
        # CPU uses log probabilities directly in this backward formula.
        g=go.reshape(shape)
        return ((g-logy*jnp.sum(g,axis=axis,keepdims=True)).ravel(),),jnp.zeros_like(w)
    if t=="NMS":return tuple(jnp.zeros_like(a) for a in xs),jnp.zeros_like(w)
    if t=="Reparameterize":
        z=saved if saved.size else xs[2] if len(xs)>2 else jnp.empty((0,),jnp.float32)
        gv=.5*go*(z-x)*((xs[1]>=-20)&(xs[1]<=20)) if z.size else jnp.zeros_like(x)
        return (go,gv),jnp.zeros_like(w)
    if t in DROP:
        if not training:return (go,),jnp.zeros_like(w)
        p=min(1.,max(0.,l["dropout_p"]))
        if p>=1:return (jnp.zeros_like(x),),jnp.zeros_like(w)
        ap=-1.6732632423543773*1.0507009873554805
        a=1/math.sqrt((1-p)*(1+p*ap*ap)) if t=="AlphaDropout" else 1/(1-p)
        if mask.size:keep=mask>0
        elif saved.size:keep=jnp.abs(saved-(a*ap*(1-p) if t=="AlphaDropout" else 0.))>1e-6
        elif len(xs)>1:keep=jnp.abs(xs[-1]-(a*ap*(1-p) if t=="AlphaDropout" else 0.))>1e-6
        else:keep=jnp.ones_like(x,dtype=jnp.bool_)
        return (jnp.where(keep,go*a,0.),),jnp.zeros_like(w)
    if t=="Constant" and not l["trainable_parameter"]:return (),jnp.zeros_like(w)
    # CpuRuntime's normalization backward always uses batch statistics, including
    # inference. Reproduce that contract rather than differentiating running stats.
    back_training=True if t in ("BatchNorm1d","BatchNorm2d") else training
    if t in ("SelfAttention","MultiHeadAttention","CrossAttention"):
        d=l["embed_dim"] or l["head_dim"]*(l["num_heads"] or 1) or x.size
        kd=l["in_features"] or d
        limit=2*d*d+2*kd*d if t=="CrossAttention" else 4*d*d
        # CPU backward reconstructs attention without projection/output biases
        # and leaves their parameter gradients zero.
        fn=lambda args,weights:(attention(args,weights[:limit],l),)
    else:fn=lambda args,weights:forward(args,weights,l,back_training,jnp.zeros_like(x),False)
    _,pullback=jax.vjp(fn,tuple(xs),w)
    gx,gw=pullback(tuple(gos))
    if t in ("Conv2d","ConvTranspose2d") and l["activation"]:
        raw=forward(tuple(xs),w,l,training,jnp.zeros_like(x),False)[0]
        _,gate=jax.vjp(lambda z:activation(z,l["activation"],l["activation_param"]),raw)
        gated=gate(go)[0];gx,gw=pullback((gated,))
    if t=="InstanceNorm2d" and l["affine"] and not l["use_bias"]:
        dim=l["in_channels"] or 3
        gw=gw.at[dim:].set(0.)
    if t=="RMSNorm" and l["affine"] and l["use_bias"]:
        dim=l["in_features"] or x.size
        if w.size>=2*dim:gw=gw.at[-dim:].add(go.reshape((-1,dim)).sum(axis=0))
    return gx,gw


def optimizer_update(kind,w,g,m,v,p):
    """All update/reduction arithmetic is compiled onto the selected PJRT device."""
    lr,b1,b2,eps,wd,bc1,bc2,decay,abeta,clip,eps2,relative,scale,rect,rho,alpha=p
    u=g
    if kind in (1,2,5,6,8):
        m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g
        mh=m/bc1;vh=v/bc2;u=mh/(jnp.sqrt(vh)+eps)
        if kind==5:u=jnp.where(rho>5,rect*mh/(jnp.sqrt(vh)+eps),mh)
        if kind==6:u=(b1*mh+(1-b1)*g/bc1)/(jnp.sqrt(vh)+eps)
        if kind==8:u=u+wd*w
    elif kind==3:
        update=b1*m+(1-b1)*g
        u=(update>0).astype(jnp.float32)-(update<0).astype(jnp.float32)+wd*w
        m=b2*m+(1-b2)*g
    elif kind==4:
        v=decay*v+(1-decay)*g*g
        u=g/jnp.sqrt(v+jnp.maximum(jnp.float32(1e-30),eps))
    elif kind==7:
        v=alpha*v+(1-alpha)*g*g;u=g/(jnp.sqrt(v)+eps)
    def stable_rms(x):
        maximum=jnp.max(jnp.abs(x));scaled=x/jnp.where(maximum>0,maximum,1)
        return maximum*jnp.sqrt(jnp.mean(scaled*scaled))
    if kind==4:
        clip_scale=1/jnp.maximum(1,stable_rms(u)/clip)
        param_scale=jnp.where(scale>0,jnp.maximum(eps2,stable_rms(w)),1)
        u=clip_scale*g/jnp.sqrt(v+jnp.maximum(jnp.float32(1e-30),eps))
        momentum=abeta*m+(1-abeta)*u
        m=jnp.where(abeta>0,momentum,m);u=jnp.where(abeta>0,momentum,u)
        w=w-relative*param_scale*u
    elif kind==8:
        wn=stable_rms(w)*jnp.sqrt(jnp.float32(w.size));un=stable_rms(u)*jnp.sqrt(jnp.float32(u.size))
        ratio=jnp.where((wn>0)&(un>0),wn/jnp.maximum(un,jnp.float32(1e-30)),1)
        w=w-lr*ratio*u
    elif kind==2:w=w-lr*u-lr*(wd*w)
    else:w=w-lr*u
    return w,m,v


def _loss_double_product(a,b):
    # Two float limbs: TPU has no native FP64. Barriers preserve the error-free
    # transform against XLA contraction and algebraic simplification.
    stop=lax.optimization_barrier
    def split(x):
        bits=lax.bitcast_convert_type(x,jnp.uint32)
        high=lax.bitcast_convert_type(bits & jnp.uint32(0xfffff000),jnp.float32)
        return high,stop(x-high)
    high=stop(a*b);ah,al=split(a);bh,bl=split(b)
    error=stop(stop(ah*bh)-high)
    error=stop(error+stop(ah*bl));error=stop(error+stop(al*bh));error=stop(error+stop(al*bl))
    return high,error


def _loss_two_sum(a,b):
    stop=lax.optimization_barrier
    total=stop(a+b);part=stop(total-a)
    error=stop(stop(a-stop(total-part))+stop(b-part))
    return total,error


def _loss_accumulate(carry,a,b):
    # Three float limbs, rounded to 53 bits after each addition, reproduce the
    # existing CPU double accumulator even at a zero DCT coefficient.
    stop=lax.optimization_barrier
    high,middle,low=carry;product,error=_loss_double_product(a,b)
    total,remainder=_loss_two_sum(high,product)
    mid,mid_error=_loss_two_sum(middle,error)
    mid2,error2=_loss_two_sum(remainder,mid)
    tail=stop(stop(low+mid_error)+error2)
    head,mid=_loss_two_sum(total,mid2)
    middle,low=_loss_two_sum(mid,tail)
    high,mid=_loss_two_sum(head,middle)
    middle,low=_loss_two_sum(mid,low)
    bits=lax.bitcast_convert_type(high,jnp.uint32)
    exponent=((bits>>23)&255).astype(jnp.int32)-127
    crossing=((bits&8388607)==0)&(((high>0)&(middle<0))|((high<0)&(middle>0)))
    exponent=exponent-crossing.astype(jnp.int32)
    biased=exponent+75
    step_bits=jnp.where(biased>0,biased.astype(jnp.uint32)<<23,
        jnp.where(biased>=-22,jnp.uint32(1)<<((biased+22)&31).astype(jnp.uint32),jnp.uint32(0)))
    step=lax.bitcast_convert_type(step_bits,jnp.float32)
    active=(high!=0)&(step>0);step=jnp.where(active,step,1)
    base=jnp.trunc(middle/step);res=stop(middle-stop(base*step))
    z=stop(res/step+low/step);fl=jnp.floor(z);fraction=stop(z-fl)
    odd=(base.astype(jnp.int32)&1)!=(fl.astype(jnp.int32)&1)
    rounded=fl+((fraction>.5)|((fraction==.5)&odd)).astype(jnp.float32)
    return high,jnp.where(active,stop(base*step),middle),jnp.where(active,stop(rounded*step),low)


def _loss_round_scaled(pair,scale):
    high,middle,low=pair;product,error=_loss_double_product(high,scale)
    stop=lax.optimization_barrier
    return stop(product+stop(error+stop(stop(middle*scale)+stop(low*scale))))


def reconstruction_loss(op,kind,relation_only,w,h,c,x,y,weight,p,coefficients):
    """Loss and analytical VJP, JIT compiled on the selected PJRT device."""
    n=x.size
    zero=jnp.float32(0)
    if op==0:
        d=x-y;a=jnp.abs(d);delta,eps,sigma=p[:3]
        if kind==1:values=a;grad=jnp.sign(d)
        elif kind==2:
            values=jnp.where(a<=delta,.5*d*d,delta*(a-.5*delta))
            grad=jnp.where(a<=delta,d,delta*jnp.sign(d))
        elif kind==3:
            values=jnp.sqrt(d*d+eps*eps);grad=d/values
        elif kind==4:
            values=.5*(d*d/(sigma*sigma)+jnp.log(sigma*sigma));grad=d/(sigma*sigma)
        elif kind==5:
            q=jnp.clip(x,1e-7,1-1e-7)
            lp=jnp.where(x>jnp.float32(.99999988),-1e-7,jnp.log(q))
            lq=jnp.where(x>jnp.float32(.99999988),jnp.log(jnp.float32(1e-7)),jnp.log1p(-q))
            values=-y*lp-(1-y)*lq
            grad=jnp.where((x>1e-7)&(x<=jnp.float32(.99999988)),(x-y)/(x*(1-x)),0)
        else:values=d*d;grad=2*d
        if relation_only:return zero,jnp.zeros_like(x)
        return jnp.sum(weight*values)/n,weight*grad/n
    if op==5:
        logits=x.reshape(h,w);labels=y[:h].astype(jnp.int32);valid=(labels>=0)&(labels<w)
        safe=jnp.clip(labels,0,w-1);maximum=jnp.max(logits,axis=1,keepdims=True)
        exponent=jnp.exp(logits-maximum);total=jnp.sum(exponent,axis=1,keepdims=True)
        count=jnp.maximum(1,jnp.sum(valid));normalizer=valid.astype(jnp.float32)/count
        losses=maximum[:,0]-logits[jnp.arange(h),safe]+jnp.log(total[:,0])
        grad=(exponent/total-jax.nn.one_hot(safe,w))*normalizer[:,None]
        return jnp.sum(jnp.where(valid,losses,0))/count,grad.reshape(-1)
    ow,oh=max(1,w//2),max(1,h//2)
    if op==2:
        image=x.reshape(h,w,c)
        # Preserve the reference floor dimensions and singleton edges.
        rows=[]
        for dy in range(min(2,h)):
            for dx in range(min(2,w)):rows.append(image[dy:oh*2:2,dx:ow*2:2,:])
        return zero,(sum(rows)/len(rows)).reshape(-1)
    if op==3:
        image=x[:ow*oh*c].reshape(oh,ow,c)/(min(2,w)*min(2,h))
        expanded=jnp.repeat(jnp.repeat(image,2,axis=0),2,axis=1)
        result=jnp.zeros((h,w,c),jnp.float32)
        result=result.at[:min(h,oh*2),:min(w,ow*2)].set(expanded[:min(h,oh*2),:min(w,ow*2)])
        return zero,result.reshape(-1)
    image=x.reshape(h,w,c);expected=y.reshape(h,w,c)
    if op==1:
        mx=jnp.mean(image,axis=(0,1));my=jnp.mean(expected,axis=(0,1))
        vx=jnp.mean((image-mx)**2,axis=(0,1));vy=jnp.mean((expected-my)**2,axis=(0,1))
        cov=jnp.mean((image-mx)*(expected-my),axis=(0,1))
        c1=(p[3]*p[5])**2;c2=(p[4]*p[5])**2
        A=2*mx*my+c1;B=2*cov+c2;C=mx*mx+my*my+c1;D=vx+vy+c2
        ssim=jnp.where(C*D!=0,A*B/(C*D+1e-12),0)
        derivative=2*my/(A+1e-12)+2*(expected-my)/(B+1e-12)-2*mx/(C+1e-12)-2*(image-mx)/(D+1e-12)
        return jnp.mean(1-ssim),(-ssim*derivative/n).reshape(-1)
    if op==4:
        cw=coefficients[:w*w].reshape(w,w);aw=coefficients[w*w:w*w+w]
        offset=w*w+w;ch=coefficients[offset:offset+h*h].reshape(h,h);ah=coefficients[offset+h*h:offset+h*h+h]
        # Separate transforms retain the reference rounding before subtraction.
        def dct(a):
            initial=tuple(jnp.zeros((h,w,c),jnp.float32) for _ in range(3))
            rows_pair=lax.fori_loop(0,w,lambda j,carry:_loss_accumulate(carry,a[:,j,None,:],cw[j,None,:,None]),initial)
            rows=_loss_round_scaled(rows_pair,aw[None,:,None])
            columns=lax.fori_loop(0,h,lambda j,carry:_loss_accumulate(carry,rows[j,None,:,:],ch[j,:,None,None]),initial)
            return _loss_round_scaled(columns,ah[:,None,None])
        diff=dct(image)-dct(expected);grad_coeff=jnp.sign(diff)/n
        cols=jnp.einsum('lkc,l,yl->ykc',grad_coeff,ah,ch)
        grad=jnp.einsum('ykc,k,xk->yxc',cols,aw,cw)
        return jnp.sum(jnp.abs(diff))/n,grad.reshape(-1)
    raise ValueError('unknown reconstruction operation')

def reconstruction_with_relations(op,kind,relation_only,w,h,c,x,y,weight,p,coefficients,mask):
    value,grad=reconstruction_loss(op,kind,relation_only,w,h,c,x,y,weight,p,coefficients)
    relation_loss=jnp.float32(0)
    if op==0 and coefficients.size:
        entries=coefficients.size//26
        meta=coefficients[:16*entries].reshape(entries,16)
        idx=jnp.arange(x.size)[None,:]
        rw=meta[:,0:1].astype(jnp.int32);rh=meta[:,1:2].astype(jnp.int32);rc=meta[:,2:3].astype(jnp.int32)
        chw=meta[:,3:4]>0;channel=meta[:,15:16].astype(jnp.int32)
        pos=jnp.where(chw,idx%(rw*rh),idx//rc)
        ich=jnp.where(chw,idx//(rw*rh),idx%rc)
        xx=pos%rw;yy=pos//rw
        valid=(idx<rw*rh*rc)&(ich==channel)&(mask[None,:]>0)
        features=[]
        for box in (7,11):
            x0=meta[:,box:box+1]*rw;y0=meta[:,box+1:box+2]*rh
            x1=meta[:,box+2:box+3]*rw;y1=meta[:,box+3:box+4]*rh
            xa=jnp.maximum(x0,xx);xb=jnp.minimum(x1,xx+1)
            ya=jnp.maximum(y0,yy);yb=jnp.minimum(y1,yy+1)
            area=jnp.where(valid,jnp.maximum(0,xb-xa)*jnp.maximum(0,yb-ya),0)
            total=jnp.sum(area,axis=1,keepdims=True)
            average=area/jnp.maximum(total,1e-30)
            fx=((xa+xb)-(x0+x1))/(x1-x0);fy=((ya+yb)-(y0+y1))/(y1-y0)
            features.append((total,average,fx,fy))
        sa,s,fx,fy=features[0];oa,o,gx,gy=features[1];d=(x-y)[None,:]
        active=(sa[:,0]>0)&(oa[:,0]>0)
        d0=jnp.where(active,jnp.sum((s-o)*d,axis=1),0)
        d1=jnp.where(active,jnp.sum((s*fx-o*gx)*d,axis=1)*meta[:,5],0)
        d2=jnp.where(active,jnp.sum((s*fy-o*gy)*d,axis=1)*meta[:,6],0)
        scale=meta[:,4]
        relation_loss=jnp.sum(scale*(d0*d0+d1*d1+d2*d2))
        relation_grad=2*scale[:,None]*(s*(d0[:,None]+d1[:,None]*fx+d2[:,None]*fy)-o*(d0[:,None]+d1[:,None]*gx+d2[:,None]*gy))
        grad=grad+jnp.sum(jnp.where(active[:,None],relation_grad,0),axis=0)
    return value+relation_loss,grad,relation_loss

class Engine:
    def __init__(self,device,target="reference-cpu"):
        self.device=device
        self.target=target
        self.cache=OrderedDict()
        self.key=jax.device_put(jax.random.PRNGKey(1337),device)

    def compiled(self,key,fn):
        if key not in self.cache:
            self.cache[key]=jax.jit(fn)
            if len(self.cache)>256:self.cache.popitem(last=False)
        else:self.cache.move_to_end(key)
        return self.cache[key]

    def execute(self,req,arrays):
        cmd=req["command"]
        if cmd=="hello":return {"platform":self.device.platform,"device":str(self.device),"device_kind":self.device.device_kind,"target":self.target,"ops":sorted(OPS)},[]
        xs=tuple(jax.device_put(a,self.device) for a in arrays)
        if cmd=="reconstruction_loss":
            op,kind=int(req['op']),int(req['kind'])
            w,h,c=int(req['width']),int(req['height']),int(req['channels'])
            relation_only=bool(req.get('relation_only',False))
            if op not in range(6) or kind not in range(6) or len(xs)!=6 or xs[3].size!=6:raise ValueError('invalid reconstruction request')
            if not xs[0].size or xs[1].size!=xs[0].size or xs[2].size!=xs[0].size:raise ValueError('invalid reconstruction buffers')
            if op and (min(w,h,c)<=0 or w*h*c!=xs[0].size):raise ValueError('invalid reconstruction shape')
            if op==4 and xs[4].size!=w*w+w+h*h+h:raise ValueError('invalid DCT coefficients')
            key=('reconstruction',op,kind,relation_only,w,h,c)
            fn=lambda x,y,weight,p,co,mask:reconstruction_with_relations(op,kind,relation_only,w,h,c,x,y,weight,p,co,mask)
            value,grad,relation=self.compiled(key,fn)(*xs)
            return {},[value.reshape(1),grad,relation.reshape(1)]
        if cmd=="optimizer":
            kind=int(req["kind"])
            if kind not in range(9) or len(xs)!=5:raise ValueError("invalid optimizer request")
            if not xs[0].size or any(a.size!=xs[0].size for a in xs[1:4]) or xs[4].size!=16:raise ValueError("invalid optimizer buffers")
            return {},list(self.compiled(("optimizer",kind),lambda w,g,m,v,p:optimizer_update(kind,w,g,m,v,p))(*xs))
        if cmd=="merge":
            op=req["merge"]
            def merge(a,b):
                if op==2:return jnp.concatenate((a,b))
                if a.size!=b.size:raise ValueError("branch merge size mismatch")
                if op==0:return a+b
                if op==1:return a*b
                if op==3:return jnp.maximum(a,b)
                if op==4:return (a+b)*.5
                if op in (5,6):return a+b
                raise ValueError("unknown branch merge")
            return {},[self.compiled(("merge",op),merge)(*xs)]
        l=req["layer"];t=l["type"];n=req["inputs"];training=req["training"]
        if t not in OPS:raise ValueError("unsupported layer "+t)
        inputs=xs[:n];w=xs[n];x=inputs[0] if inputs else jax.device_put(np.empty((0,),np.float32),self.device)
        config=json.dumps(l,sort_keys=True)
        if t=="NMS":
            if n not in (2,3) or inputs[0].size!=4*inputs[1].size or (n==3 and inputs[2].size!=inputs[1].size):raise ValueError("NMS shape mismatch")
            if not 0<=l["nms_iou_threshold"]<=1 or not math.isfinite(l["nms_score_threshold"]) or l["nms_max_detections"]<0:raise ValueError("invalid NMS thresholds")
            selected,count=self.compiled(("nms",config),lambda a:nms(a,l))(inputs)
            count=int(count)
            if cmd=="forward":return {"outputs":1},[selected[:count],np.empty((0,),np.float32)]
            if req["grad_outputs"]!=1 or xs[n+1].size!=count:raise ValueError("NMS gradient shape mismatch")
        if cmd=="forward":
            noise=jnp.zeros_like(x)
            if training and (t in DROP or t=="Reparameterize"):
                self.key,key=jax.random.split(self.key)
                noise=jax.random.normal(key,x.shape) if t=="Reparameterize" else (jax.random.uniform(key,x.shape)>min(1,max(0,l["dropout_p"]))).astype(jnp.float32)
            fused=req.get("fused",False)
            out=self.compiled((cmd,config,training,fused),lambda a,b,c:forward(a,b,l,training,c,fused))(inputs,w,noise)
            mask=noise if training and t in DROP else jnp.ones_like(x) if t in DROP else jnp.empty((0,),jnp.float32)
            return {"outputs":len(out)},list(out)+[mask]
        if cmd=="backward":
            ng=req["grad_outputs"];gos=xs[n+1:n+1+ng];saved,mask=xs[-2:]
            gx,gw=self.compiled((cmd,config,training),lambda a,b,c,d,e:backward(a,b,c,d,e,l,training))(inputs,w,gos,saved,mask)
            gb=bias_gradient(gw,l)
            return {"grad_inputs":len(gx)},list(gx)+[gw,gb]
        raise ValueError("unknown command")


def read_exact(size):
    chunks=[]
    while size:
        part=sys.stdin.buffer.read(size)
        if not part:raise EOFError()
        chunks.append(part);size-=len(part)
    return b"".join(chunks)


def receive():
    size=struct.unpack("<I",read_exact(4))[0]
    if size>1024*1024:raise ValueError("protocol header too large")
    req=json.loads(read_exact(size))
    arrays=[]
    for count in req["sizes"]:
        if not isinstance(count,int) or count<0 or count>2**30:raise ValueError("invalid tensor size")
        arrays.append(np.frombuffer(read_exact(count*4),dtype="<f4").copy())
    return req,arrays


def reply(meta,arrays):
    values=[np.asarray(a,dtype="<f4").reshape(-1) for a in arrays]
    meta=dict(meta,ok="error" not in meta,sizes=[v.size for v in values])
    header=json.dumps(meta,separators=(",",":"),allow_nan=False).encode()
    WIRE.write(struct.pack("<I",len(header)));WIRE.write(header)
    for a in values:WIRE.write(a.tobytes())
    WIRE.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",type=int,default=0)
    parser.add_argument("--target",default=os.environ.get("MIMIR_TPU_TARGET","auto"),help="auto, TPU-v5e1 or TPU-v6e1")
    parser.add_argument("--reference-test",action="store_true",help="test-only CPU compiler; never used by TpuRuntime")
    args=parser.parse_args()
    try:
        devices=jax.devices("cpu" if args.reference_test else "tpu")
        if args.device<0 or args.device>=len(devices):raise ValueError("TPU device index out of range")
        device=devices[args.device]
        if not args.reference_test and device.platform!="tpu":raise RuntimeError("a real TPU is required")
        profile="reference-cpu" if args.reference_test else target_profile(device.device_kind,len(devices),args.target)
        engine=Engine(device,profile)
    except Exception as exc:
        # Wait for the hello request so initialization failures use the same framing.
        receive();reply({"error":str(exc)},[]);return
    while True:
        try:req,arrays=receive()
        except EOFError:return
        try:
            with jax.default_device(device):meta,values=engine.execute(req,arrays)
            reply(meta,values)
        except Exception as exc:reply({"error":f"{type(exc).__name__}: {exc}"},[])


if __name__=="__main__":main()
