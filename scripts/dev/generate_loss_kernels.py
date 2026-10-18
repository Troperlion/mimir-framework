#!/usr/bin/env python3
"""Generate the same native reconstruction arithmetic for four GPU APIs."""
from pathlib import Path
import re,struct,subprocess,tempfile
root=Path(__file__).resolve().parents[2]
out=root/'src/runtimes/losses'
body=r'''
int n=args[3],w=args[4],h=args[5],c=args[6],op=args[2],pass=args[0];
int ch=i%c,hw=w*h,ow=w/2>0?w/2:1,oh=h/2>0?h/2:1;
if(pass==90 || pass==91) {
 int count=op==1?c:(op==5?h:n),groups=(count+255)/256;
 if(group>=(pass==90?groups:1))return;
 float value=0.0;
 if(pass==90){if(i<count)value=memory[4*n+i];}
 else {for(int j=local;j<groups;j+=256)value+=memory[10*n+j];}
 sums[local]=value;BARRIER;
 for(int stride=128;stride>0;stride/=2){if(local<stride)sums[local]+=sums[local+stride];BARRIER;}
 if(local==0){if(pass==90)memory[10*n+group]=sums[0];else memory[4*n]=sums[0]+(op==0?memory[11*n]:0.0);}
 return;
}
if(op==0 && (pass==3 || pass==4)) {
 int entries=args[15],stats=12*n+16*entries;
 if(pass==3) {
  if(i>=entries)return;
  int meta=12*n+16*i,rw=int(memory[meta]),rh=int(memory[meta+1]),rc=int(memory[meta+2]),channel=int(memory[meta+15]);
  float momentS0=0.0,momentS1=0.0,momentS2=0.0,momentO0=0.0,momentO1=0.0,momentO2=0.0,sa=0.0,oa=0.0;
  for(int box=0;box<2;++box) {
   int at=meta+7+4*box;float x0=memory[at]*float(rw),y0=memory[at+1]*float(rh),x1=memory[at+2]*float(rw),y1=memory[at+3]*float(rh);
   float a=0.0,m0=0.0,m1=0.0,m2=0.0;
   for(int yy=int(floor(y0));yy<int(ceil(y1));++yy)for(int xx=int(floor(x0));xx<int(ceil(x1));++xx) {
    int idx=memory[meta+3]>0.0?channel*rw*rh+yy*rw+xx:(yy*rw+xx)*rc+channel;
    if(memory[5*n+idx]==0.0)continue;
    float xa=max(x0,float(xx)),xb=min(x1,float(xx+1)),ya=max(y0,float(yy)),yb=min(y1,float(yy+1));
    float area=(xb-xa)*(yb-ya),d=memory[idx]-memory[n+idx];a+=area;
    m0+=area*d;m1+=area*((xa+xb)-(x0+x1))/(x1-x0)*d;m2+=area*((ya+yb)-(y0+y1))/(y1-y0)*d;
   }
   if(a>0.0){m0/=a;m1/=a;m2/=a;}
   if(box==0){sa=a;momentS0=m0;momentS1=m1;momentS2=m2;}else{oa=a;momentO0=m0;momentO1=m1;momentO2=m2;}
  }
  float d0=sa>0.0 && oa>0.0?momentS0-momentO0:0.0,d1=(momentS1-momentO1)*memory[meta+5],d2=(momentS2-momentO2)*memory[meta+6];
  if(sa<=0.0 || oa<=0.0){d1=0.0;d2=0.0;}
  int at=stats+10*i;memory[at]=d0;memory[at+1]=d1;memory[at+2]=d2;memory[at+3]=sa;memory[at+4]=oa;memory[at+5]=memory[meta+4]*(d0*d0+d1*d1+d2*d2);
 } else {
  if(i>=n)return;
  float gradient=memory[3*n+i],rloss=0.0;
  for(int entry=0;entry<entries;++entry) {
   int meta=12*n+16*entry,at=stats+10*entry;
   if(i==0)rloss+=memory[at+5];
   int rw=int(memory[meta]),rh=int(memory[meta+1]),rc=int(memory[meta+2]),channel=int(memory[meta+15]);
   if(i>=rw*rh*rc || memory[5*n+i]==0.0 || memory[at+3]<=0.0 || memory[at+4]<=0.0)continue;
   int pidx=memory[meta+3]>0.0?i%(rw*rh):i/rc,ich=memory[meta+3]>0.0?i/(rw*rh):i%rc;
   if(ich!=channel)continue;
   int xx=pidx%rw,yy=pidx/rw;
   for(int box=0;box<2;++box) {
    int region=meta+7+4*box;float x0=memory[region]*float(rw),y0=memory[region+1]*float(rh),x1=memory[region+2]*float(rw),y1=memory[region+3]*float(rh);
    float xa=max(x0,float(xx)),xb=min(x1,float(xx+1)),ya=max(y0,float(yy)),yb=min(y1,float(yy+1));
    if(xb<=xa || yb<=ya)continue;
    float feature=memory[at]+memory[at+1]*((xa+xb)-(x0+x1))/(x1-x0)+memory[at+2]*((ya+yb)-(y0+y1))/(y1-y0);
    gradient+=(box==0?1.0:-1.0)*2.0*memory[meta+4]*(xb-xa)*(yb-ya)/memory[at+3+box]*feature;
   }
  }
  memory[3*n+i]=gradient;if(i==0)memory[11*n]=rloss;
 }
 return;
}

if(op==0) {
 if(i>=n)return;
 float x=memory[i],y=memory[n+i],d=x-y,a=abs(d),weight=memory[2*n+i],loss=0.0,g=0.0;
 float delta=F(0),eps=F(1),sigma=F(2),inv=1.0/float(n);
 int kind=args[7];
 if(args[8]==0) {
  if(kind==1){loss=a;g=d>0.0?1.0:(d<0.0?-1.0:0.0);}
  else if(kind==2){if(a<=delta){loss=0.5*d*d;g=d;}else{loss=delta*(a-0.5*delta);g=delta*(d>0.0?1.0:-1.0);}}
  else if(kind==3){loss=sqrt(d*d+eps*eps);g=d/loss;}
  else if(kind==4){loss=0.5*(d*d/(sigma*sigma)+log(sigma*sigma));g=d/(sigma*sigma);}
  else if(kind==5){
   float p=max(1e-7,min(0.9999999,x));
   float lp=x>0.99999988?-1e-7:log(p),lq=x>0.99999988?log(1e-7):log(1.0-p);
   loss=-y*lp-(1.0-y)*lq;
   if(x>1e-7 && x<=0.99999988)g=(x-y)/(x*(1.0-x));
  } else {loss=d*d;g=2.0*d;}
 }
 memory[3*n+i]=weight*inv*g;memory[4*n+i]=weight*inv*loss;return;
}
if(op==2 || op==3) {
 if(i>=n)return;
 if(op==2) {
  if(i>=ow*oh*c)return;
  int channel=i%c,x=(i/c)%ow,y=(i/c)/ow;
  float value=0.0;int count=0;
  for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int xx=x*2+dx,yy=y*2+dy;if(xx<w && yy<h){value+=memory[(yy*w+xx)*c+channel];++count;}}
  memory[3*n+i]=value/float(count);
 } else {
  int x=(i/c)%w,y=(i/c)/w;
  float value=0.0;
  if(x/2<ow && y/2<oh){int nx=w-x/2*2>1?2:1,ny=h-y/2*2>1?2:1;value=memory[((y/2)*ow+x/2)*c+ch]/float(nx*ny);}
  memory[3*n+i]=value;
 }
 return;
}
if(op==1) {
 if(pass==0 || pass==1) {
  int channel=group;if(channel>=c)return;
  float sx=0.0,sy=0.0,sxy=0.0,mx=memory[9*n+8*channel],my=memory[9*n+8*channel+1];
  for(int j=local;j<hw;j+=256){float x=memory[j*c+channel],y=memory[n+j*c+channel];if(pass==1){x-=mx;y-=my;}sx+=pass==0?x:x*x;sy+=pass==0?y:y*y;sxy+=x*y;}
  sums[local]=sx;sums[256+local]=sy;sums[512+local]=sxy;BARRIER;
  for(int stride=128;stride>0;stride/=2){if(local<stride){sums[local]+=sums[local+stride];sums[256+local]+=sums[256+local+stride];sums[512+local]+=sums[512+local+stride];}BARRIER;}
  if(local==0) {
   int at=9*n+8*channel;
   if(pass==0){memory[at]=sums[0]/float(hw);memory[at+1]=sums[256]/float(hw);}
   else {
    float c1=F(3)*F(5),c2=F(4)*F(5);c1*=c1;c2*=c2;
    float A=2.0*mx*my+c1,B=2.0*sums[512]/float(hw)+c2,C=mx*mx+my*my+c1,D=(sums[0]+sums[256])/float(hw)+c2;
    float denom=C*D,ssim=denom!=0.0?A*B/(denom+1e-12):0.0;
    memory[at+2]=A;memory[at+3]=B;memory[at+4]=C;memory[at+5]=D;memory[at+6]=ssim;
    memory[4*n+channel]=(1.0-ssim)/float(c);
   }
  }
 } else if(pass==2 && i<n) {
  int at=9*n+8*ch;float mx=memory[at],my=memory[at+1],x=memory[i],y=memory[n+i];
  float deriv=2.0*my/(memory[at+2]+1e-12)+2.0*(y-my)/(memory[at+3]+1e-12)-2.0*mx/(memory[at+4]+1e-12)-2.0*(x-mx)/(memory[at+5]+1e-12);
  memory[3*n+i]=-memory[at+6]*deriv/float(hw*c);
 }
 return;
}
if(op==5) {
 if(pass==0) {
  int token=group;if(token>=h)return;
  int label=int(memory[n+token]);
  float maximum=-3.402823466e38;
  for(int k=local;k<w;k+=256)maximum=max(maximum,memory[token*w+k]);
  sums[local]=maximum;BARRIER;
  for(int stride=128;stride>0;stride/=2){if(local<stride)sums[local]=max(sums[local],sums[local+stride]);BARRIER;}
  maximum=sums[0];BARRIER;
  float total=0.0;for(int k=local;k<w;k+=256)total+=exp(memory[token*w+k]-maximum);
  sums[local]=total;BARRIER;
  for(int stride=128;stride>0;stride/=2){if(local<stride)sums[local]+=sums[local+stride];BARRIER;}
  total=sums[0];
  bool valid=label>=0 && label<w;
  for(int k=local;k<w;k+=256)memory[3*n+token*w+k]=valid?exp(memory[token*w+k]-maximum)/max(1e-30,total)-(k==label?1.0:0.0):0.0;
  if(local==0){memory[4*n+token]=valid?(maximum-memory[token*w+label])+log(max(1e-30,total)):0.0;memory[5*n+token]=valid?1.0:0.0;}
 } else if(pass==1) {
  if(group>0)return;
  float total=0.0;for(int k=local;k<h;k+=256)total+=memory[5*n+k];
  sums[local]=total;BARRIER;
  for(int stride=128;stride>0;stride/=2){if(local<stride)sums[local]+=sums[local+stride];BARRIER;}
  if(local==0)memory[11*n+1]=sums[0]>0.0?1.0/sums[0]:0.0;
 } else if(pass==2 && i<n){memory[3*n+i]*=memory[11*n+1];if(i<h)memory[4*n+i]*=memory[11*n+1];}
 return;
}
if(op==4 && i<n) {
 int x=(i/c)%w,y=(i/c)/w;
 int cw=12*n,aw=cw+w*w,ct=aw+w,ah=ct+h*h;
 float a=0.0,b=0.0,al=0.0,bl=0.0,at=0.0,bt=0.0;
 if(pass==0){for(int j=0;j<w;++j){float co=memory[cw+j*w+x];LOSS_ACC(memory[(y*w+j)*c+ch],co,a,al,at);LOSS_ACC(memory[n+(y*w+j)*c+ch],co,b,bl,bt);}float pa,pb;LOSS_STORE(a,al,at,memory[aw+x],pa);LOSS_STORE(b,bl,bt,memory[aw+x],pb);memory[5*n+i]=pa;memory[6*n+i]=pb;}
 else if(pass==1){for(int j=0;j<h;++j){float co=memory[ct+j*h+y];LOSS_ACC(memory[5*n+(j*w+x)*c+ch],co,a,al,at);LOSS_ACC(memory[6*n+(j*w+x)*c+ch],co,b,bl,bt);}float pa,pb;LOSS_STORE(a,al,at,memory[ah+y],pa);LOSS_STORE(b,bl,bt,memory[ah+y],pb);memory[7*n+i]=pa;memory[8*n+i]=pb;}
 else if(pass==2){float d=memory[7*n+i]-memory[8*n+i];memory[9*n+i]=(d>0.0?1.0:(d<0.0?-1.0:0.0))/float(n);memory[4*n+i]=abs(d)/float(n);}
 else if(pass==3){for(int j=0;j<h;++j)a+=memory[ah+j]*memory[9*n+(j*w+x)*c+ch]*memory[ct+y*h+j];memory[5*n+i]=a;}
 else if(pass==4){for(int j=0;j<w;++j)a+=memory[aw+j]*memory[5*n+(y*w+j)*c+ch]*memory[cw+x*w+j];memory[3*n+i]=a;}
}
'''
# Error-free float products and compensated sums preserve the CPU DCT sign
# around zero coefficients; a plain float dot changes its L1 subgradient.
macros=r"""
#define LOSS_PRODUCT(a,b,p,e) {float loss_pr_a=(a),loss_pr_b=(b),loss_pr_ah=LOSS_FLOAT(LOSS_BITS(loss_pr_a)&-4096),loss_pr_bh=LOSS_FLOAT(LOSS_BITS(loss_pr_b)&-4096);float loss_pr_al=LOSS_SUB(loss_pr_a,loss_pr_ah),loss_pr_bl=LOSS_SUB(loss_pr_b,loss_pr_bh),loss_pr_p=LOSS_MUL(loss_pr_a,loss_pr_b);float loss_pr_e=LOSS_ADD(LOSS_ADD(LOSS_ADD(LOSS_SUB(LOSS_MUL(loss_pr_ah,loss_pr_bh),loss_pr_p),LOSS_MUL(loss_pr_ah,loss_pr_bl)),LOSS_MUL(loss_pr_al,loss_pr_bh)),LOSS_MUL(loss_pr_al,loss_pr_bl));p=loss_pr_p;e=loss_pr_e;}
#define LOSS_TWO(a,b,s,e) {float loss_ts_a=(a),loss_ts_b=(b),loss_ts_s=LOSS_ADD(loss_ts_a,loss_ts_b),loss_ts_v=LOSS_SUB(loss_ts_s,loss_ts_a);float loss_ts_e=LOSS_ADD(LOSS_SUB(loss_ts_a,LOSS_SUB(loss_ts_s,loss_ts_v)),LOSS_SUB(loss_ts_b,loss_ts_v));s=loss_ts_s;e=loss_ts_e;}
#define LOSS_ACC(a,b,hi,mid,lo) {float va=(a),vb=(b),pr,pe;LOSS_PRODUCT(va,vb,pr,pe);float ss,ee,mm,me,m2,e2,hh,mi,tail;LOSS_TWO(hi,pr,ss,ee);LOSS_TWO(mid,pe,mm,me);LOSS_TWO(ee,mm,m2,e2);float ll=LOSS_ADD(LOSS_ADD(lo,me),e2);LOSS_TWO(ss,m2,hh,mi);LOSS_TWO(mi,ll,mm,tail);LOSS_TWO(hh,mm,hi,mi);LOSS_TWO(mi,tail,mid,lo);if(hi!=0.0){int exponent=((LOSS_BITS(hi)>>23)&255)-127;if((LOSS_BITS(hi)&8388607)==0 && ((hi>0.0 && mid<0.0)||(hi<0.0 && mid>0.0)))--exponent;int biased=exponent+75;float step=LOSS_FLOAT(biased>0?(biased<<23):(biased>=-22?(1<<((biased+22)&31)):0));if(step>0.0){float base=trunc(mid/step),res=LOSS_SUB(mid,LOSS_MUL(base,step)),z=LOSS_ADD(res/step,lo/step),fl=floor(z),fraction=LOSS_SUB(z,fl);bool odd=(int(base)&1)!=(int(fl)&1);float rounded=fl+(fraction>0.5 || (fraction==0.5 && odd)?1.0:0.0);mid=LOSS_MUL(base,step);lo=LOSS_MUL(rounded,step);}}}
#define LOSS_STORE(hi,mid,lo,a,result) {float loss_scale_p,loss_scale_e;LOSS_PRODUCT(hi,a,loss_scale_p,loss_scale_e);result=LOSS_ADD(loss_scale_p,LOSS_ADD(loss_scale_e,LOSS_ADD(LOSS_MUL(mid,a),LOSS_MUL(lo,a))));}
"""
plain="#define LOSS_ADD(a,b) ((a)+(b))\n#define LOSS_SUB(a,b) ((a)-(b))\n#define LOSS_MUL(a,b) ((a)*(b))\n"
body=macros+body
clbody=body
for expr in ['n','count','hw','c','hw*c','nx*ny']:
 clbody=clbody.replace('float('+expr+')','(float)('+expr+')')
clbody=re.sub(r'\bfloat\(([^()]*)\)',r'(float)(\1)',clbody)
clbody=re.sub(r'\bint\(([^()]*)\)',r'(int)(\1)',clbody)
clbody=clbody.replace('int(', '(int)(').replace('abs(','fabs(')
clbody=re.sub(r'\blocal\b','lid',clbody)
clbody=re.sub(r'(?<![\w.])(\d+\.\d*(?:e[-+]?\d+)?|\d+e[-+]?\d+)(?![\w.])',r'\1f',clbody)
cl='#pragma OPENCL FP_CONTRACT OFF\n#define LOSS_BITS(x) as_int(x)\n#define LOSS_FLOAT(x) as_float(x)\n'+plain+'''#define F(k) as_float(args[9+(k)])
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE)
__kernel void reconstruction(__global float* memory,__global const int* args) {
int i=get_global_id(0),lid=get_local_id(0),group=get_group_id(0);
__local float sums[768];
'''+clbody+'\n}\n'
glsl='''#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0,std430) buffer Data {float memory[];};
layout(set=0,binding=1,std430) readonly buffer Params {int args[];};
shared float sums[768];
#define F(k) intBitsToFloat(args[9+(k)])
#define BARRIER barrier()
void main(){int i=int(gl_GlobalInvocationID.x),local=int(gl_LocalInvocationID.x),group=int(gl_WorkGroupID.x);
'''+plain+'#define LOSS_BITS(x) floatBitsToInt(x)\n#define LOSS_FLOAT(x) intBitsToFloat(x)\n'+body.replace('float ','precise float ')+'\n}\n'
cubody=body.replace('fma(','fmaf(').replace('abs(','fabsf(').replace('exp(','expf(').replace('sqrt(','sqrtf(').replace('log(','logf(').replace('max(','fmaxf(').replace('min(','fminf(')
cubody=re.sub(r'(?<![\w.])(\d+\.\d*(?:e[-+]?\d+)?|\d+e[-+]?\d+)(?![\w.])',r'\1f',cubody)
cuda='#define LOSS_BITS(x) __float_as_int(x)\n#define LOSS_FLOAT(x) __int_as_float(x)\n#define LOSS_ADD(a,b) __fadd_rn(a,b)\n#define LOSS_SUB(a,b) __fsub_rn(a,b)\n#define LOSS_MUL(a,b) __fmul_rn(a,b)\n'+'''#define F(k) __int_as_float(args[9+(k)])
#define BARRIER __syncthreads()
extern "C" __global__ void mimir_reconstruction(float* memory,const int* args) {
int i=blockIdx.x*blockDim.x+threadIdx.x,local=threadIdx.x,group=blockIdx.x;
__shared__ float sums[768];
'''+cubody+'\n}\n#undef F\n#undef BARRIER\n#undef LOSS_BITS\n#undef LOSS_FLOAT\n#undef LOSS_ADD\n#undef LOSS_SUB\n#undef LOSS_MUL\n#undef LOSS_TWO\n#undef LOSS_ACC\n#undef LOSS_STORE\n#undef LOSS_PRODUCT\n'
out.mkdir(exist_ok=True)
(out/'reconstruction.cl').write_text(cl);(out/'reconstruction.comp').write_text(glsl);(out/'LossCudaKernel.cuh').write_text('#pragma once\n'+cuda)
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'loss.spv';subprocess.run(['glslangValidator','-V',str(out/'reconstruction.comp'),'-o',str(p)],check=True);subprocess.run(['spirv-val',str(p)],check=True)
 words=struct.unpack('<'+'I'*(p.stat().st_size//4),p.read_bytes())
 header='#pragma once\n#include <cstdint>\nnamespace MimirLossKernels {\n'
 header+='inline constexpr const char* opencl=R"MIMIRCL('+cl+')MIMIRCL";\n'
 header+='inline constexpr const char* hip=R"MIMIRHIP(#include <hip/hip_runtime.h>\n'+cuda+')MIMIRHIP";\n'
 header+='inline constexpr std::uint32_t spirv[]={\n'+''.join(','.join(str(x)+'u' for x in words[i:i+16])+',\n' for i in range(0,len(words),16))+'};\n}\n'
 (out/'LossShaderData.hpp').write_text(header)
