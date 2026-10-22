#!/usr/bin/env python3
"""Regenerate native OpenCL/Vulkan reference kernels from shared GPU algorithms."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2];
import os
os.chdir(ROOT); OUT=ROOT/'src/runtimes/portable'; OUT.mkdir(exist_ok=True)

def write_if_changed(path,content):
 if not path.exists() or path.read_text()!=content:path.write_text(content)

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
sources['Extra']='''__global__ void portable_copy(const float* source,float* dest,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dest[i]=source[i];}
__global__ void portable_gemm(const float* a,const float* b,float* c,int m,int n,int k,int lda,int ldb,int ldc,bool ta,bool tb,float alpha,float beta){
 int index=blockIdx.x*blockDim.x+threadIdx.x;int i=index%m,j=index/m;if(j<n){float sum=0;for(int f=0;f<k;++f)sum+=a[ta?f+i*lda:i+f*lda]*b[tb?j+f*ldb:f+j*ldb];c[i+j*ldc]=alpha*sum+beta*c[i+j*ldc];}}
__global__ void portable_ger(const float* x,const float* y,float* a,int m,int n,int sx,int sy,int lda,float alpha){int index=blockIdx.x*blockDim.x+threadIdx.x;int i=index%m,j=index/m;if(j<n)a[i+j*lda]+=alpha*x[i*sx]*y[j*sy];}'''

# Positional tables keep their checkpoint shape. Coordinates come from the
# shared host geometry; tensor interpolation and the adjoint run on device.
sources['Extra'] += r'''
__global__ void portable_sequence_constant_forward(const float* source,const float* rows,float* result,int width,int base_rows,int count) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int row=i/width,c=i%width,a=int(rows[2*row]),b=a+1<base_rows?a+1:base_rows-1;
 float t=rows[2*row+1];result[i]=(1.0f-t)*source[a*width+c]+t*source[b*width+c];
}
__global__ void portable_sequence_constant_backward(const float* source,const float* rows,float* result,int width,int base_rows,int count) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int row=i/width,c=i%width,a=int(rows[2*row]),b=a+1<base_rows?a+1:base_rows-1;
 float t=rows[2*row+1];atomicAdd(result+a*width+c,(1.0f-t)*source[i]);atomicAdd(result+b*width+c,t*source[i]);
}
'''

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
# Kernels with disjoint outputs use a parallel grid; gradient scatters remain
# serialized so they do not require unsupported floating-point atomics.
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
float parallel_sum(float value){
 normScratch[int(gl_LocalInvocationID.x)]=value;barrier();
 for(int stride=128;stride>0;stride/=2){if(int(gl_LocalInvocationID.x)<stride)normScratch[int(gl_LocalInvocationID.x)]+=normScratch[int(gl_LocalInvocationID.x)+stride];barrier();}
 float result=normScratch[0];barrier();return result;
}
float reduce_max(float value,int scratch){return value;}
float parallel_max(float value){
 normScratch[int(gl_LocalInvocationID.x)]=value;barrier();
 for(int stride=128;stride>0;stride/=2){if(int(gl_LocalInvocationID.x)<stride)normScratch[int(gl_LocalInvocationID.x)]=max(normScratch[int(gl_LocalInvocationID.x)],normScratch[int(gl_LocalInvocationID.x)+stride]);barrier();}
 float result=normScratch[0];barrier();return result;
}
void atomicAdd(int at,float value){
 uint old=atomicCompSwap(integer_memory[at],0u,0u);
 for(;;){uint expected=old;old=atomicCompSwap(integer_memory[at],expected,floatBitsToUint(uintBitsToFloat(expected)+value));if(old==expected)return;}
}
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

# Pooling windows cooperate within a workgroup; argmax ties retain CPU order.
start=shader.index('void pool_kernel(');end=balanced(shader,shader.index('{',start))
shader=shader[:start]+r'''
void pool_kernel(int x,int dy,int y,int dx,GPool g,int count){
 int o=blockIdx_x;if(o>=count)return;
 int c=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
 float value=g.maximum?-uintBitsToFloat(0x7f800000u):0.;int valid=0,best=-1;
 for(int k=threadIdx_x;k<g.kh*g.kw;k+=blockDim_x){
  int iy=oy*g.sh+k/g.kw-g.ph,ix=ox*g.sw+k%g.kw-g.pw;
  if(iy<0 || iy>=g.h || ix<0 || ix>=g.w)continue;
  int i=g.token?iy*g.c+c:(c*g.h+iy)*g.w+ix;
  ++valid;if(g.maximum){if(memory[x+i]>value){value=memory[x+i];best=i;}}else value+=memory[x+i];
 }
 float reduced=g.maximum?parallel_max(value):parallel_sum(value);
 float valid_count=parallel_sum(float(valid));
 float winner=g.maximum?-parallel_max(-(best>=0 && value==reduced?float(best):1e30)):0.;
 if(dy==0){if(threadIdx_x==0)memory[y+o]=g.maximum?reduced:(valid_count>0.?reduced/valid_count:0.);return;}
 if(g.maximum){if(threadIdx_x==0 && winner<1e30)atomicAdd(dx+int(winner),memory[dy+o]);return;}
 if(valid_count==0.)return;
 float gradient=memory[dy+o]/valid_count;
 for(int k=threadIdx_x;k<g.kh*g.kw;k+=blockDim_x){
  int iy=oy*g.sh+k/g.kw-g.ph,ix=ox*g.sw+k%g.kw-g.pw;
  if(iy>=0 && iy<g.h && ix>=0 && ix<g.w)atomicAdd(dx+(g.token?iy*g.c+c:(c*g.h+iy)*g.w+ix),gradient);
 }
}
'''+shader[end:]

# Reductions have one workgroup per independent row/channel. Scatter writes
# use portable integer CAS atomics instead of racing non-atomic float stores.
team_kernels={'norm_forward','norm_backward','channel_gradient','softmax_kernel',
              'row_norm_kernel','pool_kernel','batch_norm_kernel','attention_probabilities','attention_score_grad'}
for name in team_kernels:
 row_start=shader.index('void '+name+'(')
 row_end=balanced(shader,shader.index('{',row_start))
 row=shader[row_start:row_end]
 row=re.sub(r'(\b\w+\s*)\+=\s*1\b',r'\1+=blockDim_x',row)
 row=re.sub(r'reduce_sum\((\w+),\s*scratch\)',r'parallel_sum(\1)',row)
 row=re.sub(r'reduce_max\((\w+),\s*scratch\)',r'parallel_max(\1)',row)
 shader=shader[:row_start]+row+shader[row_end:]

# A 16 x 16 tile reuses matrix operands in workgroup memory. Each thread
# owns one result and all threads participate in both tile barriers.
shader+=r'''
void tiled_gemm(int a,int b,int c,int m,int n,int k,int lda,int ldb,int ldc,bool ta,bool tb,float alpha,float beta){
 int local=int(gl_LocalInvocationID.x),group=int(gl_WorkGroupID.x+gl_WorkGroupID.y*gl_NumWorkGroups.x);
 int tiles=(m+15)/16,r=local%16,d=local/16,row=(group%tiles)*16+r,col=(group/tiles)*16+d;
 float sum=0;
 for(int base=0;base<k;base+=16){
  int fa=base+d,fb=base+r;
  normScratch[local]=row<m && fa<k?memory[a+(ta?fa+row*lda:row+fa*lda)]:0.;
  normScratch[256+local]=col<n && fb<k?memory[b+(tb?col+fb*ldb:fb+col*ldb)]:0.;
  barrier();
  for(int f=0;f<16;++f)sum+=normScratch[r+f*16]*normScratch[256+f+d*16];
  barrier();
 }
 if(row<m && col<n)memory[c+row+col*ldc]=alpha*sum+(beta==0.?0.:beta*memory[c+row+col*ldc]);
}
'''

# A single command entry dispatches one reference kernel over its logical grid.
entry='int group_id(){return int(gl_WorkGroupID.x+gl_WorkGroupID.y*gl_NumWorkGroups.x); }\nint invocation_id(){return int(gl_GlobalInvocationID.x+gl_WorkGroupID.y*gl_NumWorkGroups.x*256u); }\nvoid reference_main(){\n blockDim_x=arguments[1];gridDim_x=arguments[2];\n int selected=specializedOperation>=0?specializedOperation:arguments[0];NORM_ENTRY\n int id=selected;bool parallel=(id>=0 && id<=42 && id!=31 && id!=32);int count=blockDim_x*gridDim_x;if(id==38)count=arguments[5];if(id==39 || id==40)count=arguments[6]*arguments[7];int first=parallel?invocation_id():0;if(!parallel && invocation_id()!=0)return;for(int gid=first;gid<count;gid+=parallel?count:1){blockIdx_x=gid/blockDim_x;threadIdx_x=gid%blockDim_x;if(id>=38 && id<=40){blockIdx_x=gid;threadIdx_x=0;blockDim_x=1;}switch(selected){\n'
pack=''
norm_entry=''
gemm_entry=''
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
 if name in team_kernels:norm_entry+='if(selected=='+str(i)+'){blockDim_x=256;blockIdx_x=group_id();threadIdx_x=int(gl_LocalInvocationID.x);'+''.join(decode)+name+'('+','.join(call)+');return;}'
 if name=='portable_gemm':gemm_entry='if(selected=='+str(i)+'){tiled_gemm('+','.join(call)+');return;}'
 entry+=f'case {i}: {{'+''.join(decode)+name+'('+','.join(call)+');break;}\n'
entry+='default:break;}}}\n'
shader+=entry.replace('NORM_ENTRY',norm_entry+gemm_entry)
for reserved in ['local','in','out','input','output','common','filter','half','sample','partition','active','packed']:
 shader=re.sub(r'\b'+reserved+r'\b',reserved+'_value',shader)
write_if_changed(OUT/'reference.comp','#version 450\nlayout(local_size_x=256)in;\nshared float normScratch[512];\nlayout(constant_id=0)const int specializedOperation=-1;\nlayout(set=0,binding=0,std430)coherent buffer Data{float memory[];};\nlayout(set=0,binding=0,std430)coherent buffer DataBits{uint integer_memory[];};\nlayout(set=0,binding=1,std430)readonly buffer Params{int arguments[];};\n'+shader+'\nvoid main(){reference_main();}\n')
# GPU host wrappers are generated from CUDA; PortableRuntime orchestration is
# maintained independently to preserve backend-specific dispatch and validation.
for family,s in sources.items():
 if family in ('Extra','Runtime'):continue
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
 s=s.replace('namespace PortableNativeOps {','namespace PortableNativeOps {\nusing namespace PortableStaging;')
 if family=='Native':
  s=s.replace('    out.clear();off=0;for(auto n:g.sizes)', '    launch("portable_copy",1,1,b.ptr,b.ptr,total);if(!finished())return false;\n    out.clear();off=0;for(auto n:g.sizes)')
  s=s.replace('#include "PortableStaging.hpp"', '#include "PortableStaging.hpp"\n#include "PortableLayerGeometry.hpp"')
  s=s.replace('const int width=l.in_features>0?l.in_features:x.size();', 'const int width=PortableLayerGeometry::rowNormWidth(l,x.size(),dy!=nullptr);')
  s=s.replace('const float* w=l.affine?l.getWeights():nullptr;const bool bias=w && l.use_bias && l.getWeightsSize()>=2ULL*width;', 'size_t weight_count=l.getWeightsSize();\n    const float* w=l.affine?l.getWeights():nullptr;\n    std::vector<float> padded;\n    if(!dy && w && weight_count<static_cast<size_t>(width)){padded.assign(width,1.f);std::copy_n(w,weight_count,padded.data());w=padded.data();weight_count=padded.size();}\n    const bool bias=w && l.use_bias && weight_count>=2ULL*width;')
  # Only row_norm uses the padded parameter count; other helpers keep their ABI.
  start=s.index('inline bool row_norm(');end=s.index('\ninline bool ',start+1)
  block=s[start:end].replace('bw.alloc(l.getWeightsSize())','bw.alloc(weight_count)').replace('bw.upload(w,l.getWeightsSize())','bw.upload(w,weight_count)')
  s=s[:start]+block+s[end:]
 write_if_changed(OUT/f'Portable{family}Ops.hpp',s)
for name,fields in geometries.items():
 pack+='inline void pack(std::vector<int>& p, const CudaOperationGeometry::'+name+'& g){'+''.join('pack(p,g.'+n+');' for t,n in fields)+'}\n'
pack+='inline int kernelId(const std::string& name){\n'+''.join(f'if(name=="{n}")return {i};\n' for i,(n,a) in enumerate(kernels))+'return -1;}\n'
pack+='inline std::vector<int> kernelWriteArguments(int id){switch(id){\n'
for i,(name,args) in enumerate(kernels):
 indices=[];offset=3
 for typ,n,original in args:
  if '*' in original and 'const' not in original:indices.append(offset)
  offset+=len(geometries[typ[1:]]) if typ.startswith('G') and typ[1:] in geometries else 1
 pack+=f'case {i}:return {{'+','.join(map(str,indices))+'};\n'
pack+='default:return {};}}\n'
write_if_changed(OUT/'PortableKernelArguments.hpp',pack)
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
   values+=['memory','threadIdx_x','blockIdx_x','blockDim_x','gridDim_x','normScratch']
   rep=called+'('+','.join(values)+')';body=body[:pos+m.start()]+rep+body[en:];pos=pos+m.start()+len(rep)
 if name=='atomicAdd':body='{volatile __global uint* address=(volatile __global uint*)(memory+at);uint old=atomic_cmpxchg(address,0u,0u);for(;;){uint expected=old;old=atomic_cmpxchg(address,expected,as_uint(as_float(expected)+value));if(old==expected)return;}}'
 if name=='tiled_gemm':body=body.replace('int(gl_WorkGroupID.x+gl_WorkGroupID.y*gl_NumWorkGroups.x)','int(get_group_id(0))')
 if name=='group_id':body='{return int(get_group_id(0));}'
 if name=='invocation_id':body='{return int(get_global_id(0));}'
 body=body.replace('gl_GlobalInvocationID.x','get_global_id(0)').replace('gl_LocalInvocationID.x','get_local_id(0)').replace('gl_WorkGroupID.x','get_group_id(0)').replace('barrier();','barrier(CLK_LOCAL_MEM_FENCE);')
 body=re.sub(r'\b(float|int|bool)\(',r'(\1)(',body)
 body=body.replace('floatBitsToInt','as_int').replace('uintBitsToFloat','as_float').replace('intBitsToFloat','as_float').replace('abs(', 'fabs(').replace('inversesqrt','rsqrt')
 body=re.sub(r'(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])',r'\1f',body)
 args=[a.replace('inout ','').rsplit(' ',1)[0]+'* '+a.split()[-1] if a.startswith('inout ') else a for a in args]
 if name=='reference_main':
  sig='__kernel void reference(__global float* memory,__global const int* arguments)'
  body=body[:1]+'__local float normScratch[512];int threadIdx_x=0,blockIdx_x=0,blockDim_x=1,gridDim_x=1;'+body[1:]
 else:
  args+=['__global float* memory','int threadIdx_x','int blockIdx_x','int blockDim_x','int gridDim_x','__local float* normScratch'];sig=rt+' '+name+'('+','.join(args)+')'
 result+=sig+body+'\n'
preamble=s[:s.index('SequenceVector sequence_new')]
result='#ifndef specializedOperation\n#define specializedOperation -1\n#endif\n#define truth(v) ((v)!=0)\n'+preamble+result
write_if_changed(Path('src/runtimes/portable/reference.cl'),result)

import subprocess, tempfile
with tempfile.TemporaryDirectory(prefix="mimir-portable-") as temporary:
    spirv_path=Path(temporary)/"reference.spv"
    subprocess.run(["glslangValidator","-V",str(OUT/"reference.comp"),"-o",str(spirv_path)],check=True)
    subprocess.run(["spirv-val",str(spirv_path)],check=True)
    from pathlib import Path
    import struct
    b=spirv_path.read_bytes();words=struct.unpack('<'+str(len(b)//4)+'I',b)
    s='#pragma once\n#include <cstdint>\nnamespace PortableStaging {\n#ifdef ENABLE_OPENCL\ninline constexpr const char* referenceOpenCL=R"MIMIRCL('+Path('src/runtimes/portable/reference.cl').read_text()+')MIMIRCL";\n#endif\n#ifdef ENABLE_VULKAN\ninline constexpr std::uint32_t referenceSpirv[]={\n'+''.join(','.join(str(x)+'u' for x in words[i:i+16])+',\n' for i in range(0,len(words),16))+'};\n#endif\n}\n';write_if_changed(Path('src/runtimes/portable/PortableShaderData.hpp'),s)
