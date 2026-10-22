#!/usr/bin/env python3
"""Generate parallel CUDA/HIP/OpenCL/Vulkan optimizer kernels from one body."""
from pathlib import Path
import struct,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'src/runtimes/optimizers'
body=r'''
int n=args[1],kind=args[2],pass=args[0],groups=(n+255)/256;
int mo=args[20],vo=args[21],go=args[22],uo=args[23],ro=args[24],flags=args[25];
// Compact layout and active moments are supplied by the host; scratch stays on device.
if(kind==0) {
 if(pass==0 && i<n)memory[i]-=F(0)*memory[go+i];
 return;
}
if(pass==0 && i<n) {
 float w=memory[i],g=memory[go+i],m=(flags&1)!=0?memory[mo+i]:0.0,v=(flags&2)!=0?memory[vo+i]:0.0,u=g;
 float b1=F(1),b2=F(2),eps=F(3),wd=F(4),bc1=F(5),bc2=F(6);
 if(kind==1 || kind==2 || kind==5 || kind==6 || kind==8) {
  m=b1*m+(1.0-b1)*g;v=b2*v+(1.0-b2)*g*g;
  float mh=m/bc1,vh=v/bc2;
  u=mh/(sqrt(vh)+eps);
  if(kind==5)u=F(14)>5.0?F(13)*mh/(sqrt(vh)+eps):mh;
  if(kind==6)u=(b1*mh+(1.0-b1)*g/bc1)/(sqrt(vh)+eps);
  if(kind==8)u+=wd*w;
 } else if(kind==3) {
  float update=b1*m+(1.0-b1)*g;
  u=float(update>0.0)-float(update<0.0)+wd*w;
  m=b2*m+(1.0-b2)*g;
 } else if(kind==4) {
  v=F(7)*v+(1.0-F(7))*g*g;u=g/sqrt(v+max(1e-30,eps));
 } else if(kind==7) {
  v=F(15)*v+(1.0-F(15))*g*g;u=g/(sqrt(v)+eps);
 }
 if((flags&1)!=0)memory[mo+i]=m;
 if((flags&2)!=0)memory[vo+i]=v;
 if(kind==4 || kind==8)memory[uo+i]=u;
 if(kind!=4 && kind!=8)memory[i]=kind==2?w-F(0)*u-F(0)*(F(4)*w):w-F(0)*u;
}
if(pass==1 || pass==3) {
 float a=0.0,b=0.0;
 if(pass==1){if(i<n){float u=memory[uo+i],w=memory[i];a=abs(u)/sqrt(float(n));b=abs(w)/sqrt(float(n));}}
 else {for(int j=local;j<groups;j+=256){a=norm_join(a,memory[ro+2*j]);b=norm_join(b,memory[ro+2*j+1]);}}
 sumsA[local]=a;sumsB[local]=b;BARRIER;
 for(int stride=128;stride>0;stride/=2){if(local<stride){sumsA[local]=norm_join(sumsA[local],sumsA[local+stride]);sumsB[local]=norm_join(sumsB[local],sumsB[local+stride]);}BARRIER;}
 if(local==0){int at=ro+2*(pass==1?group:groups);memory[at]=sumsA[0];memory[at+1]=sumsB[0];}
}
if(pass==2 && i<n) {
 float w=memory[i],u=memory[uo+i],lr=F(0);
 if(kind==4) {
  float rms=memory[ro+2*groups];float clip=1.0/max(1.0,rms/F(9));
  float scale=F(12)>0.0?max(F(10),memory[ro+2*groups+1]):1.0;
  u=clip*memory[go+i]/sqrt(memory[vo+i]+max(1e-30,F(3)));
  if(F(8)>0.0){float m=F(8)*memory[mo+i]+(1.0-F(8))*u;memory[mo+i]=m;u=m;}
  w-=F(11)*scale*u;
 } else if(kind==8) {
  float wn=memory[ro+2*groups+1]*sqrt(float(n)),un=memory[ro+2*groups]*sqrt(float(n));float ratio=wn>0.0 && un>0.0?wn/un:1.0;
  w-=lr*ratio*u;
 } else if(kind==2)w=w-lr*u-lr*(F(4)*w);
 else w-=lr*u;
 memory[i]=w;
}
'''
norm="float norm_join(float a,float b){float hi=max(abs(a),abs(b)),lo=min(abs(a),abs(b));if(hi==0.0)return 0.0;float q=lo/hi;return hi*sqrt(1.0+q*q);}\n"
# The same arithmetic body is used by all four GPU APIs. Only workgroup syntax differs.
clbody=body.replace('float(update>0.0)','(float)(update>0.0)').replace('float(update<0.0)','(float)(update<0.0)').replace('float(n)','(float)(n)')
import re
clbody=re.sub(r'(?<![\w.])(\d+\.\d*(?:e[-+]?\d+)?|\d+e[-+]?\d+)(?![\w.])',r'\1f',clbody)
cl=norm+'''#define F(k) as_float(args[3+(k)])
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE)
__kernel void optimizer(__global float* memory,__global const int* args) {
 int i=get_global_id(0),local=get_local_id(0),group=get_group_id(0);
 __local float sumsA[256],sumsB[256];
'''+clbody+'\n}\n'
# 'local' is an OpenCL address-space keyword.
cl=re.sub(r'\blocal\b','lid',cl)
cl=cl.replace('abs(', 'fabs(')
cl=re.sub(r'(?<![\w.])(\d+\.\d*(?:e[-+]?\d+)?|\d+e[-+]?\d+)(?![\w.])',r'\1f',cl)
glsl='#version 450\n'+norm+'''layout(local_size_x=256) in;
layout(set=0,binding=0,std430) buffer Data {float memory[];};
layout(set=0,binding=1,std430) readonly buffer Params {int args[];};
shared float sumsA[256],sumsB[256];
#define F(k) intBitsToFloat(args[3+(k)])
#define BARRIER barrier()
void main(){int i=int(gl_GlobalInvocationID.x),local=int(gl_LocalInvocationID.x),group=int(gl_WorkGroupID.x);
'''+body+'\n}\n'
cuda_body=(norm+body).replace('float(update>0.0)','float(update>0.0)').replace('max(','fmaxf(').replace('sqrt(','sqrtf(')
# CUDA/HIP literals stay float, matching the existing float32 updates.
cuda_body=re.sub(r'(?<![\w.])(\d+\.\d*(?:e[-+]?\d+)?|\d+e[-+]?\d+)(?![\w.])',r'\1f',cuda_body)
cuda_helper,cuda_body=cuda_body.split('\n',1)
cuda='__device__ inline '+cuda_helper.replace('abs(', 'fabsf(').replace('min(', 'fminf(')+'\n'+'''#define F(k) __int_as_float(args[3+(k)])
#define BARRIER __syncthreads()
extern "C" __global__ void mimir_optimizer(float* memory,const int* args,int pass) {
 int i=blockIdx.x*blockDim.x+threadIdx.x,local=threadIdx.x,group=blockIdx.x;
 __shared__ float sumsA[256],sumsB[256];
'''+cuda_body.replace(',pass=args[0]', '')+'\n}\n#undef F\n#undef BARRIER\n'
OUT.mkdir(exist_ok=True)
(OUT/'optimizer.cl').write_text(cl);(OUT/'optimizer.comp').write_text(glsl)
(OUT/'OptimizerCudaKernel.cuh').write_text('#pragma once\n'+cuda)
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'optimizer.spv'
 subprocess.run(['glslangValidator','-V',str(OUT/'optimizer.comp'),'-o',str(p)],check=True)
 subprocess.run(['spirv-val',str(p)],check=True)
 words=struct.unpack('<'+'I'*(p.stat().st_size//4),p.read_bytes())
 header='#pragma once\n#include <cstdint>\nnamespace MimirOptimizerKernels {\n'
 header+='inline constexpr const char* opencl=R"MIMIRCL('+cl+')MIMIRCL";\n'
 hip_cuda=cuda
 header+='inline constexpr const char* hip=R"MIMIRHIP(#include <hip/hip_runtime.h>\n'+hip_cuda+')MIMIRHIP";\n'
 header+='inline constexpr std::uint32_t spirv[]={\n'+''.join(','.join(str(x)+'u' for x in words[i:i+16])+',\n' for i in range(0,len(words),16))+'};\n}\n'
 (OUT/'OptimizerShaderData.hpp').write_text(header)
