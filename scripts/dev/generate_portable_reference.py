#!/usr/bin/env python3
"""Regenerate native OpenCL/Vulkan reference kernels from shared GPU algorithms."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2];
import os
os.chdir(ROOT); OUT=ROOT/'src/runtimes/portable'; OUT.mkdir(exist_ok=True)

def balanced(s,pos,left='{',right='}'):
 d=1;i=pos+1
 while d:
  if s[i]==left:d+=1
  elif s[i]==right:d-=1
  i+=1
 return i

def funcs(s):
 result=[]
 for m in re.finditer(r'__(global|device)__\s+(?:static\s+|inline\s+)*(float|void|bool)\s+(\w+)\s*\(',s):
  a=m.end()-1;e=balanced(s,a,'(',')');b=s.index('{',e);end=balanced(s,b)
  result.append((m.start(),end,m.group(1),m.group(2),m.group(3),s[a+1:e-1],s[b:end]))
 return result

def split(s):
 return [x.strip() for x in s.split(',')]

sources={}
for name in ['Native','Training','Attention','Recurrent','Detection']:
 sources[name]=Path(f'src/runtimes/cuda/Cuda{name}Ops.cuh').read_text()
sources['Runtime']=Path('src/runtimes/cuda/CudaRuntime.cpp').read_text()
sources['Extra']='''__global__ void portable_copy(const float* source,float* dest,int n){for(int i=0;i<n;++i)dest[i]=source[i];}
__global__ void portable_gemm(const float* a,const float* b,float* c,int m,int n,int k,int lda,int ldb,int ldc,bool ta,bool tb,float alpha,float beta){
 for(int j=0;j<n;++j)for(int i=0;i<m;++i){float sum=0;for(int f=0;f<k;++f)sum+=a[ta?f+i*lda:i+f*lda]*b[tb?j+f*ldb:f+j*ldb];c[i+j*ldc]=alpha*sum+beta*c[i+j*ldc];}}
__global__ void portable_ger(const float* x,const float* y,float* a,int m,int n,int sx,int sy,int lda,float alpha){for(int j=0;j<n;++j)for(int i=0;i<m;++i)a[i+j*lda]+=alpha*x[i*sx]*y[j*sy];}'''

# Scalar GPU geometries have no dynamic containers.
gs=Path('src/runtimes/cuda/CudaOperationGeometry.hpp').read_text(); geometries={}
for name in ['Resize','Pool','Projection','ExtendedConv','BatchNorm','Attention','Recurrent']:
 s=gs.split('struct '+name+' {')[1].split('bool valid')[0]
 fields=[]
 for t,vals in re.findall(r'\b(int|bool|size_t)\s+([^;{}]+);',s):
  for val in vals.split(','):
   n=val.split('=')[0].strip()
   if re.fullmatch(r'\w+',n):fields.append(('int' if t=='size_t' else t,n))
 geometries[name]=fields

allfunc=[]
for family,s in sources.items():
 for f in funcs(s):
  if family=='Recurrent' and f[4] not in ['sigmoid_scalar','recurrent_forward_reference','recurrent_backward_reference','recurrent_kernel']:continue
  allfunc.append((family,f))

# GPU source uses a single float arena; pointer arguments become validated offsets.
# Reference launches are serial on device, making gradient scatters deterministic.
seqvars=set(); seqscalar=set()
for f in [x[1] for x in allfunc if x[0]=='Recurrent']:
 for n in re.findall(r'SequenceVector(?:\s*&)?\s+(\w+)',f[6]):seqvars.add(n)
seqvars|={'x','go','outputs','grad_inputs'}

common='''
struct SequenceVector { int values; int count; };
struct SequenceArena { int data; int used; int capacity; };
struct SequenceLayer { int type_enum; int seq_len; int in_features; int out_features; bool use_bias; int weights; int weights_size; SequenceVector grad_weights; };
SequenceVector sequence_new(inout SequenceArena arena,int n,float value) {
 SequenceVector result=SequenceVector(arena.data+arena.used,n);arena.used+=n;
 for(int i=0;i<n;++i)memory[result.values+i]=value;return result;
}
void sequence_assign(inout SequenceVector v,int n,float value){for(int i=0;i<n;++i)memory[v.values+i]=value;v.count=n;}
void sequence_swap(inout SequenceVector a,inout SequenceVector b){SequenceVector c=a;a=b;b=c;}
void sequence_copy(int first,int last,int output){while(first!=last){memory[output++]=memory[first++];}}
float sequence_dot(int w,int x,int n){float s=0;for(int i=0;i<n;++i)s+=memory[w+i]*memory[x+i];return s;}
float reduce_sum(float value,int scratch){return value;}
float reduce_max(float value,int scratch){return value;}
void atomicAdd(int at,float value){memory[at]+=value;}
bool truth(bool v){return v;}
bool truth(int v){return v!=0;}
bool truth(float v){return v!=0.0;}
float round_away(float v){return v>=0.0?floor(v+0.5):ceil(v-0.5);}
float rsqrtf(float v){return inversesqrt(v);}
float log1pf(float v){return log(1.0+v);}
int threadIdx_x=0; int blockIdx_x=0; int blockDim_x=1; int gridDim_x=1;
'''
for name,fields in geometries.items():common='struct G'+name+' { '+' '.join(t+' '+n+';' for t,n in fields)+' };\n'+common
# Enum constants need only integral identities.
enum=Path('src/LayerTypes.hpp').read_text().split('enum class LayerType')[1].split('{')[1].split('};')[0]
names=[]
for item in enum.split(','):
 item=re.sub(r'//[^\n]*','',item).strip()
 if item:names.append(item.split('=')[0].strip())
constants={name:i for i,name in enumerate(names)}
# Correct explicit UNKNOWN=0 numbering by asking C++ source's enum order.
print('enum first',names[:3])

signatures={}
for family,f in allfunc:
 args=f[5]
 if f[4]=='recurrent_forward_reference':args='SequenceVector x, inout SequenceVector outputs, inout SequenceLayer layer, inout SequenceArena arena'
 elif f[4]=='recurrent_backward_reference':args='SequenceVector inputs, SequenceVector grad_outputs, inout SequenceVector grad_inputs, inout SequenceLayer layer, inout SequenceArena arena'
 params=[]
 for a in split(args):
  n=re.findall(r'\w+',a)[-1]
  if '*' in a:typ='int'
  else:typ=re.sub(r'\bconst\b|&','',a).strip().rsplit(' ',1)[0].strip().replace('CudaOperationGeometry::','G').replace('LayerType','int').replace('size_t','int')
  params.append((typ,n,a))
 signatures[f[4]]=params

def transform(body,args,name):
 # Remove local shared storage: width=1 reductions return their argument.
 body=re.sub(r'__shared__\s+float\s+(\w+)\[threads\];',r'int \1=0;',body)
 body=body.replace('__syncthreads();','').replace('threads','1').replace('constexpr ','const ')
 body=re.sub(r'\(void\)\w+;','',body)
 body=re.sub(r'auto sigmoid_scalar = \[\].*?;\s*};','',body)
 body=re.sub(r'(\d+)(?:ULL|UL|LL)\b',r'\1',body)
 body=re.sub(r'\b(?:threadIdx|blockIdx|blockDim|gridDim)\.x',lambda m:m[0].replace('.','_'),body)
 body=re.sub(r'static_cast<([^>]+)>\(',lambda m:('int' if m[1]=='size_t' else 'float' if m[1]=='double' else m[1])+'(',body)
 body=body.replace('CudaOperationGeometry::','G').replace('std::','')
 body=re.sub(r'auto dot_(?:in|h) =.*?};','',body,flags=re.S)
 for fn,extra in [('dot_in',',xt,I'),('dot_h',',h_prev.values,H')]:
  pos=0
  while (m:=re.search(r'\b'+fn+r'\(',body[pos:])):
   start=pos+m.start();a=body.index('(',start);end=balanced(body,a,'(',')');rep='sequence_dot('+body[a+1:end-1]+extra+')';body=body[:start]+rep+body[end:];pos=start+len(rep)
 body=re.sub(r'LayerType::(\w+)',lambda m:str(constants[m[1]]),body)
 for en in ['CudaUnaryOp','CudaBinaryOp']:
  values=dict(re.findall(r'(\w+)\s*=\s*(\d+)',sources['Runtime'].split('enum class '+en)[1].split('};')[0]))
  body=re.sub(en+r'::(\w+)',lambda m:values[m[1]],body)
 body=re.sub(r'\bsize_t\b|\bdouble\b',lambda m:'int' if m[0]=='size_t' else 'float',body)
 for math in ['exp','log','sqrt','tanh','floor','pow','fabs','fmin','fmax']:
  body=re.sub(r'\b'+math+r'f\b',{'fabs':'abs','fmin':'min','fmax':'max'}.get(math,math),body)
 body=body.replace('llround','round_away').replace('lroundf','round_away').replace('lround','round_away').replace('isfinite(', 'finite_value(')
 body=body.replace('INFINITY','uintBitsToFloat(0x7f800000u)').replace('nullptr','0')
 body=re.sub(r'(\d(?:[\deE.+-]*\d)?\.?)[fF]\b',r'\1',body)
 # C++ recurrence views are offsets too. No heap allocation occurs on device.
 if name.startswith('recurrent'):
  body=body.replace('const SequenceVector& x = *inputs[0];','SequenceVector x = inputs;').replace('const SequenceVector& go = *grad_outputs[0];','SequenceVector go = grad_outputs;')
  body=re.sub(r'\boutputs\[0\]', 'outputs',body);body=re.sub(r'\bgrad_inputs\[0\]','grad_inputs',body)
  body=body.replace('outputs[0]','outputs').replace('grad_inputs[0]','grad_inputs')
  body=body.replace('layer.getWeights()','layer.weights').replace('layer.getWeightsSize()','layer.weights_size')
  body=re.sub(r'(\w+(?:\.\w+)?)\.(?:data|begin)\(\)',r'\1.values',body).replace('.size()', '.count')
  body=re.sub(r'(\w+(?:\.\w+)?)\.assign\(([^;]+)\)',r'sequence_assign(\1,\2)',body)
  body=re.sub(r'(\w+)\.swap\((\w+)\)',r'sequence_swap(\1,\2)',body)
  body=re.sub(r'SequenceVector\s+(\w+)\(arena,\s*(.*?)\);',r'SequenceVector \1=sequence_new(arena,\2);',body)
  body=body.replace('SequenceArena arena{workspace,0,g.workspace};','SequenceArena arena=SequenceArena(workspace,0,g.workspace);')
  body=re.sub(r'SequenceVector xv\(x,(.*?)\),yv\(y,(.*?)\),gv\(dy,(.*?)\);',r'SequenceVector xv=SequenceVector(x,\1);SequenceVector yv=SequenceVector(y,\2);SequenceVector gv=SequenceVector(dy,\3);',body)
  body=body.replace('SequenceLayer layer{type,g.steps,g.input,g.hidden,bias,w,weights_size,SequenceVector(dw,weights_size)};','SequenceLayer layer=SequenceLayer(type,g.steps,g.input,g.hidden,bias,w,weights_size,SequenceVector(dw,weights_size));')
  body=re.sub(r'const SequenceVector\* (?:in|grads)\[\]=\{&(?:xv|gv)\};','',body)
  body=body.replace('recurrent_backward_reference(in,grads,&yv,layer,arena)','recurrent_backward_reference(xv,gv,yv,layer,arena)').replace('recurrent_forward_reference(xv,&yv,layer,arena)','recurrent_forward_reference(xv,yv,layer,arena)')
  # SequenceLayer's parameter gradient has a view too.
  body=re.sub(r'layer\.grad_weights\[([^\]]+)\]',r'memory[layer.grad_weights.values+(\1)]',body)
  for v in sorted(seqvars|{'xv','yv','gv'},key=len,reverse=True):
   body=re.sub(r'(?<![\w.])'+v+r'\[([^\]]+)\]',lambda m:'memory['+v+'.values+('+m[1]+')]',body)
 # Pointer offset aliases; nested indices are parsed rather than regex matched.
 floats={n for t,n,a in args if '*' in a and 'float' in a}
 ints={n for t,n,a in args if '*' in a and 'int' in a}
 for m in re.finditer(r'(?:const\s+)?(float|int)\s*\*\s*(\w+)',body):
  (floats if m[1]=='float' else ints).add(m[2])
 body=re.sub(r'(?:const\s+)?(?:float|int)\s*\*\s*(\w+)',r'int \1',body)
 body=re.sub(r'const SequenceVector\s*&\s*', 'SequenceVector ',body)
 for v in floats|ints|{n for t,n,a in args if t in ['int','float'] and '*' not in a}:
  pat=re.compile(r'\b'+v+r'\[');pos=0
  while (m:=pat.search(body,pos)):
   end=balanced(body,m.end()-1,'[',']');idx=body[m.end():end-1];read='memory['+v+'+('+idx+')]'
   if v in ints:read='floatBitsToInt('+read+')'
   body=body[:m.start()]+read+body[end:];pos=m.start()+len(read)
  if v in ints:
   # Only status/count pointers are mutable int scalars in these kernels.
   body=body.replace('*'+v,'floatBitsToInt(memory['+v+'])')
  # Convert offset conditions without touching arithmetic addresses.
  body=re.sub(r'(?<![.\w*/+%<>=!-])\b'+v+r'\s*\?',v+'!=0 ?',body)
  body=re.sub(r'(?<![.\w*/+%<>=!-])\b'+v+r'\s*(&&|\|\|)',lambda m:v+'!=0 '+m[1],body)
  body=re.sub(r'&&\s*'+v+r'\b(?=\s*(?:&&|\|\||\)))','&& '+v+'!=0',body)
 body=re.sub(r'&memory\[([^\]]+)\]',r'(\1)',body)
 # Indexed numeric conditions retain C++ truthiness.
 pos=0
 while (m:=re.search(r'!?memory\[',body[pos:])):
  start=pos+m.start();at=body.index('[',start);end=balanced(body,at,'[',']')
  if body[start]=='!' or re.match(r'\s*\?',body[end:]):
   raw=body[start:end];converted=('!' if raw.startswith('!') else '')+'truth('+raw.lstrip('!')+')'
   body=body[:start]+converted+body[end:];pos=start+len(converted)
  else:pos=end
 # Store integer scalars through the aliased arena.
 body=re.sub(r'floatBitsToInt\(memory\[(\w+)\]\)\s*=([^;]+);',r'memory[\1]=intBitsToFloat(int(\2));',body)
 body=body.replace('output[(floatBitsToInt(memory[count]))++]','output[floatBitsToInt(memory[count])]')
 # Handle NMS count increment explicitly.
 body=body.replace('output[(floatBitsToInt(memory[count]))++]','output[floatBitsToInt(memory[count])]')
 body=body.replace('memory[output+((floatBitsToInt(memory[count]))++)]', 'memory[output+(floatBitsToInt(memory[count]))]')
 body=body.replace('if(maximum>0 && floatBitsToInt(memory[count])>=maximum)', 'memory[count]=intBitsToFloat(floatBitsToInt(memory[count])+1);if(maximum>0 && floatBitsToInt(memory[count])>=maximum)')
 body=re.sub(r'(?<![<>=!])=\s*(\w+)\s*(\?|&&|\|\|)',r'=truth(\1) \2',body)
 body=re.sub(r'([(:,])\s*([A-Za-z_]\w*(?:\.\w+)?)\s*\?',r'\1 truth(\2) ?',body)
 # C++ permits numeric conditions; GLSL does not.
 body=re.sub(r'!(\w+(?:\.\w+)?)\b(?!\[|\s*\()',r'!truth(\1)',body)
 body=re.sub(r'if\s*\(\s*([A-Za-z_]\w*(?:\.\w+)?)\s*\)',r'if(truth(\1))',body)
 body=re.sub(r'\b(?:wsum|valid)\s*\?',lambda m:'truth('+m[0].rstrip(' ?')+') ?',body)
 body=body.replace('if(blockIdx_x || threadIdx_x)', 'if(blockIdx_x!=0 || threadIdx_x!=0)')
 body=re.sub(r'(\|\||&&)\s*(\w+%[\w.]+)(?=\s*(?:\|\||&&|\)))',r'\1 \2!=0',body)
 body=body.replace('grad_inputs.count', '1').replace('outputs.count','1')
 body=re.sub(r'\bconst\s+(?=int|float|bool|SequenceVector)', '',body)
 body=re.sub(r'(\|\||&&)\s*(\w+(?:\.\w+)?)(?=\s*(?:\|\||&&|\)))',r'\1 truth(\2)',body)
 # Pointer dereferences in copy loops are covered by dedicated helpers.
 return body

shader=common+'\nbool finite_value(float x){return !isnan(x) && !isinf(x);}\n'
kernels=[]
for family,f in allfunc:
 if f[4] in ['reduce_sum','reduce_max']:continue
 args=signatures[f[4]]
 sig=f[3]+' '+f[4]+'('+','.join(t+' '+n for t,n,a in args)+')'
 shader+=sig+transform(f[6],args,f[4])+'\n'
 if f[2]=='global':kernels.append((f[4],args))

# A single command entry dispatches one reference kernel over its logical grid.
entry='void reference_main(){\n blockDim_x=arguments[1];gridDim_x=arguments[2];\n for(int gid=0;gid<blockDim_x*gridDim_x;++gid){blockIdx_x=gid/blockDim_x;threadIdx_x=gid%blockDim_x;switch(arguments[0]){\n'
pack=''
for i,(name,args) in enumerate(kernels):
 decode=[];call=[];idx=3
 for t,n,a in args:
  if t.startswith('G'):
   typ=t[1:];vals=[]
   for ft,fn in geometries[typ]:
    val=f'arguments[{idx}]';idx+=1
    vals.append(val+'!=0' if ft=='bool' else val)
   decode.append(f'{t} {n}={t}('+','.join(vals)+');');call.append(n)
  else:
   val=f'arguments[{idx}]';idx+=1
   if t=='float':val=f'intBitsToFloat({val})'
   if t=='bool':val+='!=0'
   call.append(val)
 entry+=f'case {i}: {{'+''.join(decode)+name+'('+','.join(call)+');break;}\n'
entry+='default:break;}}}\n'
shader+=entry
for reserved in ['in','out','input','output','common','filter','half','sample','partition','active','packed']:
 shader=re.sub(r'\b'+reserved+r'\b',reserved+'_value',shader)
(OUT/'reference.comp').write_text('#version 450\nlayout(local_size_x=1)in;\nlayout(set=0,binding=0,std430)buffer Data{float memory[];};\nlayout(set=0,binding=1,std430)readonly buffer Params{int arguments[];};\n'+shader+'\nvoid main(){reference_main();}\n')
# Host wrappers are generated from the existing GPU implementation, not the CPU.
for family,s in sources.items():
 if family=='Extra':continue
 if family=='Recurrent':
  start=s.index('struct SequenceArena');end=s.index('inline bool recurrent(');s=s[:start]+s[end:]
 else:
  for f in reversed(funcs(s)):s=s[:f[0]]+s[f[1]:]
 s=s.replace('#include <cuda_runtime.h>','#include "PortableStaging.hpp"').replace('#include <cublas_v2.h>','')
 s=s.replace('#include "CudaOperationGeometry.hpp"','#include "runtimes/cuda/CudaOperationGeometry.hpp"')
 s=s.replace('CudaNativeOps','PortableNativeOps')
 s=s.replace('CudaRuntime','PortableRuntime').replace('ENABLE_CUDA','ENABLE_PORTABLE_REFERENCE')
 s=s.replace('constexpr int threads = 256;','constexpr int threads = 1;')
 s=s.replace('RuntimeCapabilityLevel::NativeOptimized','RuntimeCapabilityLevel::Native')
 for sub in ['Native','Training','Attention','Recurrent','Detection']:
  s=s.replace(f'runtimes/cuda/Cuda{sub}Ops.cuh',f'runtimes/portable/Portable{sub}Ops.hpp').replace(f'"Cuda{sub}Ops.cuh"',f'"Portable{sub}Ops.hpp"')
 s=s.replace('runtimes/cuda/PortableNativeOps.cuh','runtimes/portable/PortableNativeOps.hpp').replace('"PortableNativeOps.cuh"','"PortableNativeOps.hpp"')
 s=s.replace('runtimes/cuda/PortableRuntime.hpp','runtimes/portable/PortableRuntime.hpp')
 s=re.sub(r'(\w+)<<<(.*?)>>>\((.*?)\);',lambda m:'launch("'+m[1]+'", '+m[2]+', '+m[3]+');',s,flags=re.S)
 if family=='Runtime':
  # Bind the current backend executor for each device operation.
  for method in ['initialize','linearForward','forwardLayer','forwardLayerWithContext','backwardLayer']:
   pos=s.index('PortableRuntime::'+method+'(');start=s.index('{',pos)
   s=s[:start+1]+'\n    DeviceScope device_scope(backend_,config_.device_index);'+s[start+1:]
  s=s.replace('#include "runtimes/portable/PortableRuntime.hpp"','#include "runtimes/portable/PortableRuntime.hpp"\n#include "PortableStaging.hpp"\n#define ENABLE_PORTABLE_REFERENCE 1')
  s='\n'.join(x for x in s.splitlines() if not x.startswith('#include'))+'\n' if False else s
  s=s.replace('DeviceScope device_scope(backend_,config_.device_index);\n    config_ = cfg;', 'DeviceScope device_scope(backend_,cfg.device_index);\n    config_ = cfg;')
  # No CPU arithmetic branch is compiled or retained in the portable runtime.
  clean=[];stack=[];keep=True
  for line in s.splitlines():
   if line.startswith('#ifdef ENABLE_PORTABLE_REFERENCE') or line.startswith('#ifndef ENABLE_PORTABLE_REFERENCE'):
    stack.append((keep,not line.startswith('#ifndef')));keep=keep and stack[-1][1];continue
   if line.startswith('#else') and stack:
    parent,cond=stack[-1];keep=parent and not cond;continue
   if line.startswith('#endif') and stack:
    keep=stack.pop()[0];continue
   if keep:clean.append(line)
  s='\n'.join(clean)+'\n'
  # Match the CPU Lambda backward identity contract. CPU Lambda forward itself returns false.
  start=s.index('PortableRuntime::supportsBackwardLayerType(');end=s.index('RuntimeCapabilityLevel PortableRuntime::queryForwardCapability',start)
  section=s[start:end].replace('        case LayerType::Lambda:\n','');s=s[:start]+section+s[end:]
  pos=s.index('PortableRuntime::queryBackwardOperationCapability(');at=s.index('{',pos)
  s=s[:at+1]+'''\n    if(layer.type_enum==LayerType::Lambda) {
        return !config_.disabled && inputs.size()>=1 && inputs[0] && grad_outputs.size()==1 && grad_outputs[0] && inputs[0]->size()==grad_outputs[0]->size()?RuntimeCapabilityLevel::Native:RuntimeCapabilityLevel::Unsupported;
    }
'''+s[at+1:]
  pos=s.index('PortableRuntime::backwardLayer(');at=s.index('    if(PortableNativeOps::recurrentType',pos) if '    if(PortableNativeOps::recurrentType' in s[pos:] else s.index('    if(CudaOperationGeometry::recurrentType',pos)
  s=s[:at]+'''    if(layer.type_enum==LayerType::Lambda) {
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        Layer identity=layer;identity.type_enum=LayerType::Identity;
        CudaOperationGeometry::Movement shape;if(!shape.valid(identity,grad_outputs))return false;
        return PortableNativeOps::movement(grad_outputs,grad_inputs,identity,shape);
    }
'''+s[at:]
  s=s.replace('if (!supportsBackwardLayerType(type)) return RuntimeCapabilityLevel::Unsupported;', 'if (!supportsBackwardLayerType(type)) return RuntimeCapabilityLevel::Unsupported;\n    if(type==LayerType::Lambda)return RuntimeCapabilityLevel::Native;')
  s=s.replace('default: return AbstractRuntime::queryBackwardOperationCapability(layer,inputs,grad_outputs,training);', 'default: {\n            if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)) || grad_outputs.size()!=1)return RuntimeCapabilityLevel::Unsupported;\n            size_t expected=inputs[0]->size();\n            if(layer.type_enum==LayerType::Linear)expected=inputs[0]->size()/static_cast<size_t>(layer.in_features)*static_cast<size_t>(layer.out_features);\n            if(layer.type_enum==LayerType::MatMul || layer.type_enum==LayerType::BatchMatMul)expected=static_cast<size_t>(layer.in_features)*layer.embed_dim*(layer.type_enum==LayerType::BatchMatMul?layer.seq_len:1);\n            return grad_outputs[0]->size()==expected?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;\n        }')
  s=s.replace('                for (int b = 0; b < batch; ++b) {\n                    const float* row = go.data() + static_cast<size_t>(b) * static_cast<size_t>(out_f);\n                    for (int o = 0; o < out_f; ++o) layer.grad_bias[static_cast<size_t>(o)] += row[o];\n                }','                Impl::DeviceBuf d_ones,d_bias;\n                std::vector<float> ones(batch,1.f), bias_gradient(out_f);\n                if(!d_ones.alloc(batch*sizeof(float)) || !d_bias.alloc(out_f*sizeof(float)) || !d_ones.copyFromHost(ones.data(),batch*sizeof(float)))return false;\n                if(cublasSgemm(impl_->handle,CUBLAS_OP_N,CUBLAS_OP_N,out_f,1,batch,&alpha,reinterpret_cast<const float*>(d_go.ptr),out_f,reinterpret_cast<const float*>(d_ones.ptr),batch,&beta,reinterpret_cast<float*>(d_bias.ptr),out_f)!=CUBLAS_STATUS_SUCCESS)return false;\n                if(!d_bias.copyToHost(bias_gradient.data(),out_f*sizeof(float)))return false;\n                for(int o=0;o<out_f;++o){layer.grad_bias[o]+=bias_gradient[o];layer.grad_weights[w_elems+o]+=bias_gradient[o];}')
  # staging functions live in a namespace to coexist with real CUDA builds.
  lines=s.splitlines();end=max(j for j,x in enumerate(lines) if x.startswith('#include'))+1
  lines.insert(end,'using namespace PortableStaging;');s='\n'.join(lines)+'\n'
  # Keep geometry validation reusable by parallel fast paths without the serial budget.
  pos=s.index('RuntimeCapabilityLevel PortableRuntime::queryForwardOperationCapability(')
  end=s.index('    if(layer.type_enum==LayerType::NMS){',pos)
  s=s[:pos]+"RuntimeCapabilityLevel PortableRuntime::queryForwardOperationCapability(\n    const Layer& layer, const std::vector<const std::vector<float>*>& inputs, bool training) const {\n    if (!referenceInputsFit(layer.getWeightsSize(), inputs)) return RuntimeCapabilityLevel::Unsupported;\n    return queryForwardGeometryCapability(layer, inputs, training);\n}\n\nRuntimeCapabilityLevel PortableRuntime::queryForwardGeometryCapability(\n    const Layer& layer, const std::vector<const std::vector<float>*>& inputs, bool training) const {\n"+s[end:]
  s=s.replace('const std::vector<const std::vector<float>*>& grad_outputs, bool training) const {', 'const std::vector<const std::vector<float>*>& grad_outputs, bool training) const {\n    if (!referenceInputsFit(layer.getWeightsSize(), inputs) || !referenceInputsFit(0, grad_outputs)) return RuntimeCapabilityLevel::Unsupported;')
  for method,guard in [('forwardLayer','!referenceInputsFit(layer.getWeightsSize(), inputs)'),('backwardLayer','!referenceInputsFit(layer.getWeightsSize(), inputs) || !referenceInputsFit(0, grad_outputs)')]:
   pos=s.index('PortableRuntime::'+method+'(');start=s.index('{',pos)
   s=s[:start+1]+'\n    if ('+guard+') return false;'+s[start+1:]
  # Allocate bias workspaces before any gradient update: budget rejection must be atomic.
  s=s.replace('            Impl::DeviceBuf d_x, d_go, d_w, d_gx, d_gw;', '            Impl::DeviceBuf d_x, d_go, d_w, d_gx, d_gw;\n            Impl::DeviceBuf d_ones, d_bias;\n            if (layer.use_bias && (!d_ones.alloc(batch*sizeof(float)) || !d_bias.alloc(out_f*sizeof(float)))) return false;')
  s=s.replace('                Impl::DeviceBuf d_ones,d_bias;\n', '')
  s=s.replace('if(!d_ones.alloc(batch*sizeof(float)) || !d_bias.alloc(out_f*sizeof(float)) || !d_ones.copyFromHost(ones.data(),batch*sizeof(float)))return false;', 'if(!d_ones.copyFromHost(ones.data(),batch*sizeof(float)))return false;')
  (OUT/'PortableRuntime.cpp').write_text(s)
 else:
  s=s.replace('namespace PortableNativeOps {','namespace PortableNativeOps {\nusing namespace PortableStaging;')
  if family=='Native':
   s=s.replace('    out.clear();off=0;for(auto n:g.sizes)', '    launch("portable_copy",1,1,b.ptr,b.ptr,total);if(!finished())return false;\n    out.clear();off=0;for(auto n:g.sizes)')
  (OUT/f'Portable{family}Ops.hpp').write_text(s)
for name,fields in geometries.items():
 pack+='inline void pack(std::vector<int>& p, const CudaOperationGeometry::'+name+'& g){'+''.join('pack(p,g.'+n+');' for t,n in fields)+'}\n'
pack+='inline int kernelId(const std::string& name){\n'+''.join(f'if(name=="{n}")return {i};\n' for i,(n,a) in enumerate(kernels))+'return -1;}\n'
(OUT/'PortableKernelArguments.hpp').write_text(pack)
print('generated',len(kernels),'kernels')

from pathlib import Path
import re

s=shader
s=re.sub(r'bool truth\(.*?\}\n','',s)
s=re.sub(r'int threadIdx_x=0; int blockIdx_x=0; int blockDim_x=1; int gridDim_x=1;','',s)
s=re.sub(r'struct (\w+)\s*\{(.*?)\};',r'typedef struct {\2} \1;',s,flags=re.S)
types=list(geometries.keys());types=['G'+x for x in types]+['SequenceVector','SequenceArena','SequenceLayer']
# Construct scalar views and geometries using C compound literals.
for typ in types:
 pos=0
 while (m:=re.search(r'\b'+typ+r'\(',s[pos:])):
  a=pos+m.start()+len(typ);end=balanced(s,a,'(',')');replacement='('+typ+'){'+s[a+1:end-1]+'}';s=s[:pos+m.start()]+replacement+s[end:];pos=pos+m.start()+len(replacement)

def split_args(s):
 out=[];last=0;depth=0
 for i,c in enumerate(s):
  if c in '([{':depth+=1
  elif c in ')]}':depth-=1
  elif c==',' and depth==0:out.append(s[last:i].strip());last=i+1
 out.append(s[last:].strip());return out if s else []

ff=[]
for m in re.finditer(r'\b(float|int|bool|void|SequenceVector)\s+(\w+)\(',s):
 a=m.end()-1;e=balanced(s,a,'(',')');b=s.find('{',e)
 if b<0 or s[e:b].strip():continue
 end=balanced(s,b);ff.append((m.start(),end,m[1],m[2],split_args(s[a+1:e-1]),s[b:end]))
references={f[3]:[i for i,a in enumerate(f[4]) if a.startswith('inout ')] for f in ff}
names={f[3] for f in ff};result=''
for start,end,rt,name,args,body in ff:
 for a in args:
  if a.startswith('inout '):
   var=a.split()[-1];body=re.sub(r'\b'+var+r'\b','(*'+var+')',body)
 for called in names:
  pos=0
  while (m:=re.search(r'\b'+called+r'\(',body[pos:])):
   at=pos+m.start()+len(called);en=balanced(body,at,'(',')');values=split_args(body[at+1:en-1])
   for idx in references[called]:values[idx]='&('+values[idx]+')'
   values+=['memory','threadIdx_x','blockIdx_x','blockDim_x','gridDim_x']
   rep=called+'('+','.join(values)+')';body=body[:pos+m.start()]+rep+body[en:];pos=pos+m.start()+len(rep)
 body=re.sub(r'\b(float|int|bool)\(',r'(\1)(',body)
 body=body.replace('floatBitsToInt','as_int').replace('uintBitsToFloat','as_float').replace('intBitsToFloat','as_float').replace('abs(', 'fabs(').replace('inversesqrt','rsqrt')
 body=re.sub(r'(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])',r'\1f',body)
 args=[a.replace('inout ','').rsplit(' ',1)[0]+'* '+a.split()[-1] if a.startswith('inout ') else a for a in args]
 if name=='reference_main':
  sig='__kernel void reference(__global float* memory,__global const int* arguments)'
  body=body[:1]+'int threadIdx_x=0,blockIdx_x=0,blockDim_x=1,gridDim_x=1;'+body[1:]
 else:
  args+=['__global float* memory','int threadIdx_x','int blockIdx_x','int blockDim_x','int gridDim_x'];sig=rt+' '+name+'('+','.join(args)+')'
 result+=sig+body+'\n'
preamble=s[:s.index('SequenceVector sequence_new')]
result='#define truth(v) ((v)!=0)\n'+preamble+result
Path('src/runtimes/portable/reference.cl').write_text(result)

import subprocess, tempfile
with tempfile.TemporaryDirectory(prefix="mimir-portable-") as temporary:
    spirv_path=Path(temporary)/"reference.spv"
    subprocess.run(["glslangValidator","-V",str(OUT/"reference.comp"),"-o",str(spirv_path)],check=True)
    subprocess.run(["spirv-val",str(spirv_path)],check=True)
    from pathlib import Path
    import struct
    b=spirv_path.read_bytes();words=struct.unpack('<'+str(len(b)//4)+'I',b)
    s='#pragma once\n#include <cstdint>\nnamespace PortableStaging {\n#ifdef ENABLE_OPENCL\ninline constexpr const char* referenceOpenCL=R"MIMIRCL('+Path('src/runtimes/portable/reference.cl').read_text()+')MIMIRCL";\n#endif\n#ifdef ENABLE_VULKAN\ninline constexpr std::uint32_t referenceSpirv[]={\n'+''.join(','.join(str(x)+'u' for x in words[i:i+16])+',\n' for i in range(0,len(words),16))+'};\n#endif\n}\n';Path('src/runtimes/portable/PortableShaderData.hpp').write_text(s)
