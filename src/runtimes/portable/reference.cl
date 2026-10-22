#ifndef specializedOperation
#define specializedOperation -1
#endif
#define truth(v) ((v)!=0)
typedef struct { int steps; int input_value; int hidden; int gates; int parameters; int workspace; } GRecurrent;
typedef struct { int ed; int ked; int heads; int q; int k; bool cross; bool causal; int qbias; int obias; int mainParameters; int output_value; } GAttention;
typedef struct { int channels; int spatial; int batch; bool affine; bool bias; bool running; } GBatchNorm;
typedef struct { int mode; int ic; int oc; int h; int w; int kh; int kw; int stride; int pad; int oh; int ow; int parameters; bool bias; } GExtendedConv;
typedef struct { int mode; int dim; int vocab; int pad; int rows; int ids; int features; int second; int prefix; int output_value; int parameters; bool bias; } GProjection;
typedef struct { int c; int h; int w; int kh; int kw; int sh; int sw; int ph; int pw; int oh; int ow; bool maximum; bool token; } GPool;
typedef struct { int c; int ih; int iw; int oh; int ow; int sh; int sw; } GResize;

typedef struct { int values; int count; } SequenceVector;
typedef struct { int data; int used; int capacity; } SequenceArena;
typedef struct { int type_enum; int seq_len; int in_features; int out_features; bool use_bias; int weights; int weights_size; SequenceVector grad_weights; } SequenceLayer;
SequenceVector sequence_new(SequenceArena* arena,int n,float value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
 SequenceVector result=(SequenceVector){(*arena).data+(*arena).used,n};(*arena).used+=n;
 for(int i=0;i<n;++i)memory[result.values+i]=value;return result;
}
void sequence_assign(SequenceVector* v,int n,float value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){for(int i=0;i<n;++i)memory[(*v).values+i]=value;(*v).count=n;}
void sequence_swap(SequenceVector* a,SequenceVector* b,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){SequenceVector c=(*a);(*a)=(*b);(*b)=c;}
void sequence_copy(int first,int last,int output_value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){while(first!=last){memory[output_value++]=memory[first++];}}
float sequence_dot(int w,int x,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){float s=0;for(int i=0;i<n;++i)s+=memory[w+i]*memory[x+i];return s;}
float reduce_sum(float value,int scratch,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return value;}
float parallel_sum(float value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
 normScratch[(int)(get_local_id(0))]=value;barrier(CLK_LOCAL_MEM_FENCE);
 for(int stride=128;stride>0;stride/=2){if((int)(get_local_id(0))<stride)normScratch[(int)(get_local_id(0))]+=normScratch[(int)(get_local_id(0))+stride];barrier(CLK_LOCAL_MEM_FENCE);}
 float result=normScratch[0];barrier(CLK_LOCAL_MEM_FENCE);return result;
}
float reduce_max(float value,int scratch,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return value;}
float parallel_max(float value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
 normScratch[(int)(get_local_id(0))]=value;barrier(CLK_LOCAL_MEM_FENCE);
 for(int stride=128;stride>0;stride/=2){if((int)(get_local_id(0))<stride)normScratch[(int)(get_local_id(0))]=max(normScratch[(int)(get_local_id(0))],normScratch[(int)(get_local_id(0))+stride]);barrier(CLK_LOCAL_MEM_FENCE);}
 float result=normScratch[0];barrier(CLK_LOCAL_MEM_FENCE);return result;
}
void atomicAdd(int at,float value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){volatile __global uint* address=(volatile __global uint*)(memory+at);uint old=atomic_cmpxchg(address,0u,0u);for(;;){uint expected=old;old=atomic_cmpxchg(address,expected,as_uint(as_float(expected)+value));if(old==expected)return;}}
float round_away(float v,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return v>=0.0f?floor(v+0.5f):ceil(v-0.5f);}
float rsqrtf(float v,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return rsqrt(v);}
float log1pf(float v,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return log(1.0f+v);}
bool finite_value(float x,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return !isnan(x) && !isinf(x);}
void norm_forward(int x,int weights,int y,int normalized,int invstd,int group_size,int spatial,int channels,float eps,bool bias,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int scratch=0;
    int base = blockIdx_x * group_size;
    float sum = 0;
    for (int i = threadIdx_x; i < group_size; i +=blockDim_x) sum += memory[x+(base + i)];
    float mean = parallel_sum(sum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) / group_size;
    float variance = 0;
    for (int i = threadIdx_x; i < group_size; i +=blockDim_x) {
        float d = memory[x+(base + i)] - mean;
        variance += d * d;
    }
    float inv = rsqrtf(parallel_sum(variance,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) / group_size + eps,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    if (threadIdx_x == 0 && invstd!=0) memory[invstd+(blockIdx_x)] = inv;
    for (int i = threadIdx_x; i < group_size; i +=blockDim_x) {
        int index = base + i;
        int channel = (index / spatial) % channels;
        float value = (memory[x+(index)] - mean) * inv;
        if(truth(normalized)) memory[normalized+(index)] = value;
        memory[y+(index)] = weights!=0 ? value * memory[weights+(channel)] + ( truth(bias) ? memory[weights+(channels + channel)] : 0.f) : value;
    }
}
void norm_backward(int xhat,int invstd,int dy,int weights,int dx,int group_size,int spatial,int channels,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int scratch=0;
    int base = blockIdx_x * group_size;
    float sum = 0, product = 0;
    for (int i = threadIdx_x; i < group_size; i +=blockDim_x) {
        int index = base + i;
        float g = memory[dy+(index)] * (weights!=0 ? memory[weights+((index / spatial) % channels)] : 1.f);
        sum += g;
        product += g * memory[xhat+(index)];
    }
    float s1 = parallel_sum(sum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    float s2 = parallel_sum(product,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    for (int i = threadIdx_x; i < group_size; i +=blockDim_x) {
        int index = base + i;
        float g = memory[dy+(index)] * (weights!=0 ? memory[weights+((index / spatial) % channels)] : 1.f);
        memory[dx+(index)] = memory[invstd+(blockIdx_x)] * (g - s1 / group_size - memory[xhat+(index)] * s2 / group_size);
    }
}
void channel_gradient(int dy,int xhat,int dw,int db,int spatial,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int scratch=0;
    int channel = blockIdx_x;
    float sum = 0, product = 0;
    for (int s = threadIdx_x; s < spatial; s +=blockDim_x) {
        int index = channel * spatial + s;
        sum += memory[dy+(index)];
        if(truth(xhat)) product += memory[dy+(index)] * memory[xhat+(index)];
    }
    float bias = parallel_sum(sum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    float gamma = parallel_sum(product,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    if (threadIdx_x == 0) {
        if(truth(dw)) memory[dw+(channel)] = gamma;
        if(truth(db)) memory[db+(channel)] = bias;
    }
}
void resize_forward(int x,int y,int count,int ih,int iw,int oh,int ow,int sh,int sw,bool bilinear,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i >= count) return;
    int c = i / (oh * ow), h = (i / ow) % oh, w = i % ow;
    if (!truth(bilinear)) { memory[y+(i)] = memory[x+((c * ih + h / sh) * iw + w / sw)]; return; }
    float hf = h * ((float)(ih) / oh), wf = w * ((float)(iw) / ow);
    int h0 = (int)(floor(hf)), w0 = (int)(floor(wf));
    int h1 = min(h0 + 1, ih - 1), w1 = min(w0 + 1, iw - 1);
    float dh = hf - h0, dd = wf - w0;
    int base = c * ih * iw;
    memory[y+(i)] = memory[x+(base + h0 * iw + w0)] * (1-dh) * (1-dd) + memory[x+(base + h0 * iw + w1)] * (1-dh) * dd
         + memory[x+(base + h1 * iw + w0)] * dh * (1-dd) + memory[x+(base + h1 * iw + w1)] * dh * dd;
}
void nearest_backward(int dy,int dx,int count,int ih,int iw,int sh,int sw,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i >= count) return;
    int c = i / (ih * iw), h = (i / iw) % ih, w = i % iw;
    float sum = 0;
    for (int y = 0; y < sh; ++y) for (int x = 0; x < sw; ++x)
        sum += memory[dy+((c * ih * sh + h * sh + y) * iw * sw + w * sw + x)];
    memory[dx+(i)] = sum;
}
void bilinear_backward(int dy,int dx,int count,int ih,int iw,int oh,int ow,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i >= count) return;
    int c = i / (oh * ow), h = (i / ow) % oh, w = i % ow;
    float hf = h * ((float)(ih) / oh), wf = w * ((float)(iw) / ow);
    int h0 = (int)(floor(hf)), w0 = (int)(floor(wf));
    int h1 = min(h0+1, ih-1), w1 = min(w0+1, iw-1);
    float dh = hf-h0, dd = wf-w0, g = memory[dy+(i)];
    int base = c*ih*iw;
    atomicAdd(dx+base+h0*iw+w0,g*(1-dh)*(1-dd),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    atomicAdd(dx+base+h0*iw+w1,g*(1-dh)*dd,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    atomicAdd(dx+base+h1*iw+w0,g*dh*(1-dd),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    atomicAdd(dx+base+h1*iw+w1,g*dh*dd,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
}
void causal_mask(int scores,int rows,int columns,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i<rows*columns && i%columns>i/columns)memory[scores+(i)]=-1e9f;
}
void softmax_kernel(int x,int dy,int y,int axis_size,int inner,bool logarithmic,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int scratch=0;
    int row = blockIdx_x;
    int base = (row / inner) * axis_size * inner + row % inner;
    float maximum = -as_float(0x7f800000u);
    for (int a = threadIdx_x; a < axis_size; a +=blockDim_x) maximum = max(maximum, memory[x+(base+a*inner)]);
    maximum = parallel_max(maximum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    float sum = 0;
    for (int a = threadIdx_x; a < axis_size; a +=blockDim_x) sum += exp(memory[x+(base+a*inner)]-maximum);
    sum = parallel_sum(sum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    float reduction = 0;
    if(truth(dy)) {
        for (int a = threadIdx_x; a < axis_size; a +=blockDim_x)
            reduction += memory[dy+(base+a*inner)] * ( truth(logarithmic) ? 1.f : exp(memory[x+(base+a*inner)]-maximum)/sum);
        reduction = parallel_sum(reduction,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    }
    for (int a = threadIdx_x; a < axis_size; a +=blockDim_x) {
        int i = base+a*inner;
        float probability = exp(memory[x+(i)]-maximum)/sum;
        // Preserve the existing CPU LogSoftmax backward contract, which uses
        // the log output_value rather than its exponential in_value the reduction term.
        float log_probability = memory[x+(i)]-maximum-log(sum);
        memory[y+(i)] = dy!=0 ? ( truth(logarithmic) ? memory[dy+(i)]-log_probability*reduction : probability*(memory[dy+(i)]-reduction))
                  : ( truth(logarithmic) ? memory[x+(i)]-maximum-log(sum) : probability);
    }
}
void im2col(int x,int col,int count,int h,int w,int k,int stride,int pad,int dilation,int oh,int ow,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i >= count) return;
    int s=i%(oh*ow), row=i/(oh*ow), c=row/(k*k), kh=(row/k)%k, kw=row%k;
    int ih=(s/ow)*stride+kh*dilation-pad, iw=(s%ow)*stride+kw*dilation-pad;
    memory[col+(i)] = ih>=0 && ih<h && iw>=0 && iw<w ? memory[x+((c*h+ih)*w+iw)] : 0.f;
}
void col2im(int col,int dx,int count,int h,int w,int k,int stride,int pad,int dilation,int oh,int ow,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i >= count) return;
    int c=i/(h*w), ih=(i/w)%h, iw=i%w;
    float sum=0;
    for (int kh=0; kh<k; ++kh) for (int kw=0; kw<k; ++kw) {
        int yh=ih+pad-kh*dilation, yw=iw+pad-kw*dilation;
        if (yh<0 || yw<0 || yh%stride!=0 || yw%stride!=0) continue;
        int y=yh/stride, x=yw/stride;
        if (y<oh && x<ow) sum += memory[col+(((c*k+kh)*k+kw)*oh*ow+y*ow+x)];
    }
    memory[dx+(i)]=sum;
}
void row_norm_kernel(int x,int w,int dy,int y,int xhat,int width,float eps,bool rms,bool bias,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int scratch=0;int base=blockIdx_x*width;float sum=0;
    if(!truth(rms))for(int i=threadIdx_x;i<width;i+=blockDim_x)sum+=memory[x+(base+i)];
    float mean=truth(rms) ?0.f:parallel_sum(sum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)/width;float variance=0;
    for(int i=threadIdx_x;i<width;i+=blockDim_x){float d=memory[x+(base+i)]-mean;variance+=d*d;}
    float inv=rsqrtf(parallel_sum(variance,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)/width+eps,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    if(!truth(dy)){for(int i=threadIdx_x;i<width;i+=blockDim_x){float v=(memory[x+(base+i)]-mean)*inv;memory[y+(base+i)]=v*(w!=0 ?memory[w+(i)]:1.f)+( truth(bias) ?memory[w+(width+i)]:0.f);}return;}
    float sg=0,sgx=0;
    for(int i=threadIdx_x;i<width;i+=blockDim_x){float v=(memory[x+(base+i)]-mean)*inv;float g=memory[dy+(base+i)]*(w!=0 ?memory[w+(i)]:1.f);memory[xhat+(base+i)]=v;sg+=g;sgx+=g*v;}
    float a=parallel_sum(sg,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)/width,b=parallel_sum(sgx,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)/width;
    for(int i=threadIdx_x;i<width;i+=blockDim_x){float g=memory[dy+(base+i)]*(w!=0 ?memory[w+(i)]:1.f);memory[y+(base+i)]=inv*(g-( truth(rms) ?0.f:a)-memory[xhat+(base+i)]*b);}
}
void row_norm_grad(int dy,int xhat,int dw,int db,int rows,int width,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=width)return;float a=0,b=0;
    for(int r=0;r<rows;++r){int j=r*width+i;a+=memory[dy+(j)]*memory[xhat+(j)];b+=memory[dy+(j)];}memory[dw+(i)]=a;if(truth(db))memory[db+(i)]=b;
}
void projection_kernel(int a,int b,int w,int dy,int y,int da,int db,int dw,GProjection g,int offset_count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=g.output_value)return;
    if(g.mode==3 && o<g.prefix){if(truth(dy))memory[da+(o)]=memory[dy+(o)];else memory[y+(o)]=memory[a+(o)];return;}
    int d=(o-(g.mode==3?g.prefix:0))%g.dim,r=(o-(g.mode==3?g.prefix:0))/g.dim;
    float grad=truth(dy) ?memory[dy+(o)]:0.f;float value=0;
    if(g.mode<=2){
        int start=g.mode==1?r:0,end=g.mode==1?r+1:g.ids;
        if(g.mode==2 && offset_count!=0){start=max(0,min((int)(round_away((float)(memory[b+(r)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)),g.ids));end=max(start,min((int)(round_away((float)(memory[b+(r+1)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)),g.ids));}
        for(int i=start;i<end;++i){int id=(int)(round_away((float)(memory[a+(i)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch));if(id<0 || id>=g.vocab || id==g.pad)continue;
            int j=id*g.dim+d;if(truth(dy))atomicAdd(dw+j,grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);else value+=memory[w+(j)];}
    }else if(g.mode==3){
        float inv=rsqrtf((float)(g.features),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);int base=g.prefix+r*g.features;
        if(truth(dy))atomicAdd(dw+g.features*g.dim+d,grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);else value=memory[w+(g.features*g.dim+d)];
        for(int k=0;k<g.features;++k){int j=k*g.dim+d;if(truth(dy)){atomicAdd(da+base+k,grad*memory[w+(j)]*inv,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);atomicAdd(dw+j,grad*memory[a+(base+k)]*inv,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);}else value+=(memory[a+(base+k)]*inv)*memory[w+(j)];}
    }else{
        int base=d*g.features*g.second;
        if(truth(g.bias)){if(truth(dy))atomicAdd(dw+g.dim*g.features*g.second+d,grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);else value=memory[w+(g.dim*g.features*g.second+d)];}
        for(int i=0;i<g.features;++i)for(int j=0;j<g.second;++j){int wi=base+i*g.second+j,ai=r*g.features+i,bi=r*g.second+j;
            if(truth(dy)){atomicAdd(da+ai,grad*memory[w+(wi)]*memory[b+(bi)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);atomicAdd(db+bi,grad*memory[w+(wi)]*memory[a+(ai)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);atomicAdd(dw+wi,grad*memory[a+(ai)]*memory[b+(bi)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);}else value+=memory[a+(ai)]*memory[w+(wi)]*memory[b+(bi)];}
    }
    if(!truth(dy))memory[y+(o)]=value;
}
void geglu_kernel(int x,int dy,int y,int width,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;int a=(o/width)*2*width+o%width,b=a+width;
    float z=memory[x+(b)],th=tanh(.7978845608028654f*(z+.044715f*z*z*z)),gelu=z*.5f*(1+th);
    if(!truth(dy))memory[y+(o)]=memory[x+(a)]*gelu;else{memory[y+(a)]=memory[dy+(o)]*gelu;memory[y+(b)]=memory[dy+(o)]*memory[x+(a)]*(.5f*(1+th)+.5f*z*(1-th*th)*.7978845608028654f*(1+3*.044715f*z*z));}
}
float cubic(float x,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){x=fabs(x);float a=-.75f;if(x<=1)return ((a+2)*x-(a+3))*x*x+1;if(x<2)return (((a*x-5*a)*x+8*a)*x)-4*a;return 0;}
void bicubic_kernel(int x,int dy,int y,GResize g,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;int c=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
    float sy=(oy+.5f)*((float)(g.ih)/g.oh)-.5f,sx=(ox+.5f)*((float)(g.iw)/g.ow)-.5f;
    int iy=(int)(floor(sy)),ix=(int)(floor(sx));float sum=0,wsum=0;
    for(int h=-1;h<=2;++h)for(int w=-1;w<=2;++w){float weight=cubic(sy-(iy+h),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)*cubic(sx-(ix+w),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);wsum+=weight;if(!truth(dy))sum+=weight*memory[x+((c*g.ih+max(0,min(iy+h,g.ih-1)))*g.iw+max(0,min(ix+w,g.iw-1)))];}
    if(!truth(dy)){memory[y+(o)]=truth(wsum) ?sum/wsum:0.f;return;}if(!truth(wsum))return;
    for(int h=-1;h<=2;++h)for(int w=-1;w<=2;++w){float weight=cubic(sy-(iy+h),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)*cubic(sx-(ix+w),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);int i=(c*g.ih+max(0,min(iy+h,g.ih-1)))*g.iw+max(0,min(ix+w,g.iw-1));atomicAdd(y+i,memory[dy+(o)]*(weight/wsum),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);}
}
void pool_kernel(int x,int dy,int y,int dx,GPool g,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
 int o=blockIdx_x;if(o>=count)return;
 int c=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
 float value=g.maximum?-as_float(0x7f800000u):0.f;int valid=0,best=-1;
 for(int k=threadIdx_x;k<g.kh*g.kw;k+=blockDim_x){
  int iy=oy*g.sh+k/g.kw-g.ph,ix=ox*g.sw+k%g.kw-g.pw;
  if(iy<0 || iy>=g.h || ix<0 || ix>=g.w)continue;
  int i=g.token?iy*g.c+c:(c*g.h+iy)*g.w+ix;
  ++valid;if(g.maximum){if(memory[x+i]>value){value=memory[x+i];best=i;}}else value+=memory[x+i];
 }
 float reduced=g.maximum?parallel_max(value,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch):parallel_sum(value,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
 float valid_count=parallel_sum((float)(valid),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
 float winner=g.maximum?-parallel_max(-(best>=0 && value==reduced?(float)(best):1e30f),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch):0.f;
 if(dy==0){if(threadIdx_x==0)memory[y+o]=g.maximum?reduced:(valid_count>0.f?reduced/valid_count:0.f);return;}
 if(g.maximum){if(threadIdx_x==0 && winner<1e30f)atomicAdd(dx+(int)(winner),memory[dy+o],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}
 if(valid_count==0.f)return;
 float gradient=memory[dy+o]/valid_count;
 for(int k=threadIdx_x;k<g.kh*g.kw;k+=blockDim_x){
  int iy=oy*g.sh+k/g.kw-g.ph,ix=ox*g.sw+k%g.kw-g.pw;
  if(iy>=0 && iy<g.h && ix>=0 && ix<g.w)atomicAdd(dx+(g.token?iy*g.c+c:(c*g.h+iy)*g.w+ix),gradient,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
 }
}
void permute_kernel(int x,int y,int out_shape,int in_stride,int rank,int count,bool backward,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;int tmp=o,i=0;
    for(int d=rank-1;d>=0;--d){i+=(tmp%as_int(memory[out_shape+(d)]))*as_int(memory[in_stride+(d)]);tmp/=as_int(memory[out_shape+(d)]);}
    if(truth(backward))memory[y+(i)]=memory[x+(o)];else memory[y+(o)]=memory[x+(i)];
}
void spatial_move(int x,int y,int count,int c,int h,int w,int oh,int ow,int ph,int pw,int r,int padding,bool backward,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;
    int ch=o/(oh*ow),oy=(o/ow)%oh,ox=o%ow;int i=-1;
    if(truth(r)) {int ic=ch*r*r+(oy%r)*r+ox%r;i=(ic*h+oy/r)*w+ox/r;}
    else {
        int iy=oy-ph,ix=ox-pw;
        if(padding==2){if(iy<0)iy=-iy;if(iy>=h)iy=2*h-iy-2;if(ix<0)ix=-ix;if(ix>=w)ix=2*w-ix-2;}
        if(padding!=1){iy=max(0,min(iy,h-1));ix=max(0,min(ix,w-1));}
        if(iy>=0 && iy<h && ix>=0 && ix<w)i=(ch*h+iy)*w+ix;
    }
    if(truth(backward)){if(i>=0)atomicAdd(y+i,memory[x+(o)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);}else memory[y+(o)]=i>=0?memory[x+(i)]:0.f;
}
void extended_conv_kernel(int x,int weights,int dy,int y,int dx,int dw,GExtendedConv g,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;
    int oc=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
    float value=g.bias?memory[weights+(g.parameters+oc)]:0.f;
    float grad=truth(dy) ?memory[dy+(o)]:0.f;
    if(dy!=0 && truth(g.bias))atomicAdd(dw+g.parameters+oc,grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    int first=g.mode==2?oc:0,last=g.mode==2?oc+1:g.ic;
    for(int ic=first;ic<last;++ic)for(int kh=0;kh<g.kh;++kh)for(int kw=0;kw<g.kw;++kw){
        int iy=oy*g.stride+kh-g.pad,ix=g.mode==1?0:ox*g.stride+kw-g.pad;
        if(g.mode==3){iy=oy+g.pad-kh;ix=ox+g.pad-kw;if(iy<0 || ix<0 || iy%g.stride!=0 || ix%g.stride!=0)continue;iy/=g.stride;ix/=g.stride;}
        if(iy<0 || iy>=g.h || ix<0 || ix>=g.w)continue;
        int i=(ic*g.h+iy)*g.w+ix;
        int wi=((oc*(g.mode==2?1:g.ic)+(g.mode==2?0:ic))*g.kh+kh)*g.kw+kw;
        if(truth(dy)){atomicAdd(dx+i,grad*memory[weights+(wi)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);atomicAdd(dw+wi,grad*memory[x+(i)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);}else value+=memory[x+(i)]*memory[weights+(wi)];
    }
    if(!truth(dy))memory[y+(o)]=value;
}
void dropout_kernel(int x,int mask,int y,int n,float p,bool alpha,bool backward,bool active_value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;
    if(!truth(active_value) || p==0.f){memory[y+(i)]=memory[x+(i)];return;}if(p==1.f){memory[y+(i)]=0;return;}
    float ap=-1.6732632423543772848f*1.0507009873554804934f;
    float scale=truth(alpha) ?rsqrtf((1-p)*(1+p*ap*ap),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch):1.f/(1-p);
    if(truth(backward))memory[y+(i)]=truth(memory[mask+(i)])?memory[x+(i)]*scale:0.f;
    else memory[y+(i)]=truth(alpha) ?scale*(truth(memory[mask+(i)])?memory[x+(i)]:ap)-scale*ap*p:(truth(memory[mask+(i)])?memory[x+(i)]*scale:0.f);
}
void reparameterize_kernel(int mu,int lv,int noise,int dy,int a,int b,int n,bool training,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;float sigma=exp(.5f*max(-20.f,min(20.f,memory[lv+(i)])));
    if(!truth(dy)){memory[a+(i)]=truth(training) ?memory[mu+(i)]+sigma*memory[noise+(i)]:memory[mu+(i)];return;}
    memory[a+(i)]=memory[dy+(i)];memory[b+(i)]=0.f;
    if(noise!=0 && memory[lv+(i)]>=-20.f && memory[lv+(i)]<=20.f){float inv=sigma>1e-12f?1.f/sigma:0.f;memory[b+(i)]=memory[dy+(i)]*.5f*sigma*((memory[noise+(i)]-memory[mu+(i)])*inv);}
}
void batch_norm_kernel(int x,int w,int running,int dy,int y,int dw,GBatchNorm g,float eps,bool training,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int scratch=0;int c=blockIdx_x,N=g.batch*g.spatial;float sum=0;
    bool estimate=truth(dy) || truth(training) || !truth(g.running);
    if(truth(estimate))for(int j=threadIdx_x;j<N;j+=blockDim_x)sum+=memory[x+((j/g.spatial*g.channels+c)*g.spatial+j%g.spatial)];
    float mean=truth(estimate) ?parallel_sum(sum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)/N:memory[running+(c)];float variance=0;
    if(truth(estimate))for(int j=threadIdx_x;j<N;j+=blockDim_x){float d=memory[x+((j/g.spatial*g.channels+c)*g.spatial+j%g.spatial)]-mean;variance+=d*d;}
    float var=truth(estimate) ?parallel_sum(variance,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)/N:memory[running+(g.channels+c)],inv=rsqrtf(var+eps,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    float gamma=g.affine?memory[w+(c)]:1.f,beta=g.bias?memory[w+(g.channels+c)]:0.f;
    if(!truth(dy)){for(int j=threadIdx_x;j<N;j+=blockDim_x){int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;memory[y+(i)]=(memory[x+(i)]-mean)*inv*gamma+beta;}return;}
    float s1=0,s2=0;
    for(int j=threadIdx_x;j<N;j+=blockDim_x){int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;s1+=memory[dy+(i)];s2+=memory[dy+(i)]*(memory[x+(i)]-mean)*inv;}
    s1=parallel_sum(s1,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);s2=parallel_sum(s2,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
    if(threadIdx_x==0 && truth(g.affine)){memory[dw+(c)]=s2;if(truth(g.bias))memory[dw+(g.channels+c)]=s1;}
    for(int j=threadIdx_x;j<N;j+=blockDim_x){int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;float xhat=(memory[x+(i)]-mean)*inv;memory[y+(i)]=inv*gamma*(memory[dy+(i)]-s1/N-xhat*s2/N);}
}
void attention_add_bias(int y,int w,int n,int width,int offset,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i<n && offset>=0)memory[y+(i)]+=memory[w+(offset+i%width)];
}
void attention_project(int input_value,int w,int result,GAttention g,int part,bool backward,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;int t=i/g.ed,d=i%g.ed;
    int features=truth(part) && g.cross?g.ked:g.ed;float value=0;
    if(!truth(backward) && !truth(g.cross) && g.qbias>=0)value=memory[w+(g.qbias+part*g.ed+d)];
    for(int f=0;f<features;++f){
        int wi=!truth(g.cross)?(part*g.ed+d)*g.ed+f:(part!=0 ?g.ed*g.ed+f*2*g.ed+(part-1)*g.ed+d:f*g.ed+d);
        value+=memory[input_value+(t*features+f)]*memory[w+(wi)];
    }memory[result+(i)]=value;
}
void attention_probabilities(int q,int k,int p,GAttention g,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int scratch=0;int h=blockIdx_x/g.q,t=blockIdx_x%g.q,width=g.ed/g.heads,base=(h*g.q+t)*g.k;
    float scale=rsqrtf((float)(width),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);float maximum=-1e30f;
    for(int j=threadIdx_x;j<g.k;j+=blockDim_x){float score=-1e9f;if(!truth(g.causal) || j<=t){score=0;for(int d=0;d<width;++d)score+=memory[q+(t*g.ed+h*width+d)]*memory[k+(j*g.ed+h*width+d)];score*=scale;}memory[p+(base+j)]=score;maximum=max(maximum,score);}
    maximum=parallel_max(maximum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);float sum=0;
    for(int j=threadIdx_x;j<g.k;j+=blockDim_x){float v=exp(memory[p+(base+j)]-maximum);memory[p+(base+j)]=v;sum+=v;}
    float inv=1.f/(parallel_sum(sum,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)+1e-9f);for(int j=threadIdx_x;j<g.k;j+=blockDim_x)memory[p+(base+j)]*=inv;
}
void attention_context(int p,int v,int ctx,GAttention g,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;int t=i/g.ed,d=i%g.ed,h=d/(g.ed/g.heads);float sum=0;
    for(int j=0;j<g.k;++j)sum+=memory[p+((h*g.q+t)*g.k+j)]*memory[v+(j*g.ed+d)];memory[ctx+(i)]=sum;
}
void attention_output(int ctx,int w,int dy,int y,GAttention g,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;int t=i/g.ed,d=i%g.ed,offset=g.mainParameters-g.ed*g.ed;float sum=0;
    if(!truth(dy) && g.obias>=0)sum=memory[w+(g.obias+d)];
    for(int f=0;f<g.ed;++f)sum+=truth(dy) ?memory[dy+(t*g.ed+f)]*memory[w+(offset+d*g.ed+f)]:memory[ctx+(t*g.ed+f)]*memory[w+(offset+f*g.ed+d)];memory[y+(i)]=sum;
}
void attention_score_grad(int p,int v,int dc,int ds,GAttention g,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int scratch=0;int h=blockIdx_x/g.q,t=blockIdx_x%g.q,width=g.ed/g.heads,base=(h*g.q+t)*g.k;float dot=0;
    for(int j=threadIdx_x;j<g.k;j+=blockDim_x){float dp=0;for(int f=0;f<width;++f)dp+=memory[dc+(t*g.ed+h*width+f)]*memory[v+(j*g.ed+h*width+f)];memory[ds+(base+j)]=dp;dot+=dp*memory[p+(base+j)];}
    dot=parallel_sum(dot,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);for(int j=threadIdx_x;j<g.k;j+=blockDim_x)memory[ds+(base+j)]=g.causal && j>t?0.f:(memory[ds+(base+j)]-dot)*memory[p+(base+j)];
}
void attention_projected_grad(int q,int k,int dc,int p,int ds,int result,GAttention g,int part,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;int t=i/g.ed,d=i%g.ed,h=d/(g.ed/g.heads);float sum=0;
    if(part==0)for(int j=0;j<g.k;++j)sum+=memory[ds+((h*g.q+t)*g.k+j)]*memory[k+(j*g.ed+d)];
    else if(part==1)for(int j=0;j<g.q;++j)sum+=memory[ds+((h*g.q+j)*g.k+t)]*memory[q+(j*g.ed+d)];
    else for(int j=0;j<g.q;++j)sum+=memory[p+((h*g.q+j)*g.k+t)]*memory[dc+(j*g.ed+d)];
    memory[result+(i)]=part==2?sum:sum*rsqrtf((float)(g.ed/g.heads),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
}
void attention_weight_grad(int x,int kv,int ctx,int dy,int dq,int dk,int dv,int dw,GAttention g,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int wi=blockIdx_x*blockDim_x+threadIdx_x;if(wi>=count)return;float sum=0;int output_offset=g.mainParameters-g.ed*g.ed;
    if(wi>=output_offset){int r=(wi-output_offset)/g.ed,d=(wi-output_offset)%g.ed;for(int t=0;t<g.q;++t)sum+=memory[ctx+(t*g.ed+r)]*memory[dy+(t*g.ed+d)];}
    else if(!truth(g.cross)){int part=(wi/g.ed)/g.ed,d=(wi/g.ed)%g.ed,f=wi%g.ed;int grad=part==0?dq:(part==1?dk:dv);for(int t=0;t<g.q;++t)sum+=memory[x+(t*g.ed+f)]*memory[grad+(t*g.ed+d)];}
    else if(wi<g.ed*g.ed){int f=wi/g.ed,d=wi%g.ed;for(int t=0;t<g.q;++t)sum+=memory[x+(t*g.ed+f)]*memory[dq+(t*g.ed+d)];}
    else{int off=wi-g.ed*g.ed,f=off/(2*g.ed),col=off%(2*g.ed),d=col%g.ed;int grad=col<g.ed?dk:dv;for(int t=0;t<g.k;++t)sum+=memory[kv+(t*g.ked+f)]*memory[grad+(t*g.ed+d)];}
    memory[dw+(wi)]=sum;
}
void attention_input_grad(int w,int dq,int dk,int dv,int dx,GAttention g,bool second,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=count)return;int width=truth(second) ?g.ked:g.ed,t=i/width,f=i%width;float sum=0;
    if(!truth(g.cross))for(int d=0;d<g.ed;++d)sum+=memory[dq+(t*g.ed+d)]*memory[w+(d*g.ed+f)]+memory[dk+(t*g.ed+d)]*memory[w+((g.ed+d)*g.ed+f)]+memory[dv+(t*g.ed+d)]*memory[w+((2*g.ed+d)*g.ed+f)];
    else if(!truth(second))for(int d=0;d<g.ed;++d)sum+=memory[dq+(t*g.ed+d)]*memory[w+(f*g.ed+d)];
    else for(int d=0;d<g.ed;++d)sum+=memory[dk+(t*g.ed+d)]*memory[w+(g.ed*g.ed+f*2*g.ed+d)]+memory[dv+(t*g.ed+d)]*memory[w+(g.ed*g.ed+f*2*g.ed+g.ed+d)];
    memory[dx+(i)]=sum;
}
float sigmoid_scalar(float v,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return 1.f/(1.f+exp(-v));}
bool recurrent_forward_reference(SequenceVector x,SequenceVector* outputs,SequenceLayer* layer,SequenceArena* arena,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){

                // Pour rester fonctionnel et déterministe, on réutilise l'implémentation CPU de Model (naïve).
                int T = (*layer).seq_len;
                int I = (*layer).in_features;
                int H = (*layer).out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if ((int)(x.count) != T * I) return false;
                int W = (*layer).weights;
                if (!truth(W)) return false;

                

                if ((*layer).type_enum == 65) {
                    bool use_bias = (*layer).use_bias;
                    int Wih_sz = (int)(H) * (int)(I);
                    int Whh_sz = (int)(H) * (int)(H);
                    int bih_sz =truth(use_bias) ? (int)(H) : 0;
                    int bhh_sz =truth(use_bias) ? (int)(H) : 0;
                    int need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if ((*layer).weights_size < need) return false;

                    int Wih = W;
                    int Whh = Wih + Wih_sz;
                    int bih =truth(use_bias) ? (Whh + Whh_sz) : 0;
                    int bhh =truth(use_bias) ? (bih + bih_sz) : 0;

                    SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    
                    sequence_assign(&((*outputs)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    for (int t = 0; t < T; ++t) {
                        int xt = (x.values+((int)(t) * (int)(I)));
                        int ht = ((*outputs).values+((int)(t) * (int)(H)));
                        for (int h = 0; h < H; ++h) {
                            int wih = Wih + (int)(h) * (int)(I);
                            int whh = Whh + (int)(h) * (int)(H);
                            float sum = 0.0f;
                            for (int i = 0; i < I; ++i) sum += memory[wih+((int)(i))] * memory[xt+((int)(i))];
                            for (int k = 0; k < H; ++k) sum += memory[whh+((int)(k))] * memory[h_prev.values+((int)(k))];
                            if(truth(bih)) sum += memory[bih+((int)(h))];
                            if(truth(bhh)) sum += memory[bhh+((int)(h))];
                            memory[ht+((int)(h))] = tanh(sum);
                        }
                        sequence_copy(ht,ht + H,h_prev.values,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    }
                    return true;
                }

                if ((*layer).type_enum == 64) {
                    bool use_bias = (*layer).use_bias;
                    int Wih_sz = (int)(3 * H) * (int)(I);
                    int Whh_sz = (int)(3 * H) * (int)(H);
                    int bih_sz =truth(use_bias) ? (int)(3 * H) : 0;
                    int bhh_sz =truth(use_bias) ? (int)(3 * H) : 0;
                    int need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if ((*layer).weights_size < need) return false;

                    int Wih = W;
                    int Whh = Wih + Wih_sz;
                    int bih =truth(use_bias) ? (Whh + Whh_sz) : 0;
                    int bhh =truth(use_bias) ? (bih + bih_sz) : 0;

                    SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    SequenceVector r=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    SequenceVector z=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    SequenceVector n=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                    
                    sequence_assign(&((*outputs)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    for (int t = 0; t < T; ++t) {
                        int xt = (x.values+((int)(t) * (int)(I)));

                        for (int h = 0; h < H; ++h) {
                            int w_ir = Wih + (int)(h) * (int)(I);
                            int w_hr = Whh + (int)(h) * (int)(H);
                            float sr = 0.0f;
                            for (int i = 0; i < I; ++i) sr += memory[w_ir+((int)(i))] * memory[xt+((int)(i))];
                            for (int k = 0; k < H; ++k) sr += memory[w_hr+((int)(k))] * memory[h_prev.values+((int)(k))];
                            if(truth(bih)) sr += memory[bih+((int)(h))];
                            if(truth(bhh)) sr += memory[bhh+((int)(h))];
                            memory[r.values+((int)(h))] = sigmoid_scalar(sr,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                            int w_iz = Wih + (int)(H + h) * (int)(I);
                            int w_hz = Whh + (int)(H + h) * (int)(H);
                            float sz = 0.0f;
                            for (int i = 0; i < I; ++i) sz += memory[w_iz+((int)(i))] * memory[xt+((int)(i))];
                            for (int k = 0; k < H; ++k) sz += memory[w_hz+((int)(k))] * memory[h_prev.values+((int)(k))];
                            if(truth(bih)) sz += memory[bih+((int)(H + h))];
                            if(truth(bhh)) sz += memory[bhh+((int)(H + h))];
                            memory[z.values+((int)(h))] = sigmoid_scalar(sz,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                            int w_in = Wih + (int)(2 * H + h) * (int)(I);
                            int w_hn = Whh + (int)(2 * H + h) * (int)(H);
                            float sn = 0.0f;
                            for (int i = 0; i < I; ++i) sn += memory[w_in+((int)(i))] * memory[xt+((int)(i))];
                            for (int k = 0; k < H; ++k) sn += memory[w_hn+((int)(k))] * (memory[r.values+((int)(k))] * memory[h_prev.values+((int)(k))]);
                            if(truth(bih)) sn += memory[bih+((int)(2 * H + h))];
                            if(truth(bhh)) sn += memory[bhh+((int)(2 * H + h))];
                            memory[n.values+((int)(h))] = tanh(sn);
                        }

                        int ht = ((*outputs).values+((int)(t) * (int)(H)));
                        for (int h = 0; h < H; ++h) {
                            float zh = memory[z.values+((int)(h))];
                            float nh = memory[n.values+((int)(h))];
                            float hp = memory[h_prev.values+((int)(h))];
                            memory[ht+((int)(h))] = (1.0f - zh) * nh + zh * hp;
                        }
                        sequence_copy(ht,ht + H,h_prev.values,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    }
                    return true;
                }

                // LSTM
                {
                    bool use_bias = (*layer).use_bias;
                    int Wih_sz = (int)(4 * H) * (int)(I);
                    int Whh_sz = (int)(4 * H) * (int)(H);
                    int bih_sz =truth(use_bias) ? (int)(4 * H) : 0;
                    int bhh_sz =truth(use_bias) ? (int)(4 * H) : 0;
                    int need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if ((*layer).weights_size < need) return false;

                    int Wih = W;
                    int Whh = Wih + Wih_sz;
                    int bih =truth(use_bias) ? (Whh + Whh_sz) : 0;
                    int bhh =truth(use_bias) ? (bih + bih_sz) : 0;

                    SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    SequenceVector c_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    
                    sequence_assign(&((*outputs)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                    for (int t = 0; t < T; ++t) {
                        int xt = (x.values+((int)(t) * (int)(I)));
                        int ht = ((*outputs).values+((int)(t) * (int)(H)));

                        for (int h = 0; h < H; ++h) {
                            
                            

                            int wii = Wih + (int)(h) * (int)(I);
                            int wif = Wih + (int)(H + h) * (int)(I);
                            int wig = Wih + (int)(2 * H + h) * (int)(I);
                            int wio = Wih + (int)(3 * H + h) * (int)(I);

                            int whi = Whh + (int)(h) * (int)(H);
                            int whf = Whh + (int)(H + h) * (int)(H);
                            int whg = Whh + (int)(2 * H + h) * (int)(H);
                            int who = Whh + (int)(3 * H + h) * (int)(H);

                            float si = sequence_dot(wii,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) + sequence_dot(whi,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                            float sf = sequence_dot(wif,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) + sequence_dot(whf,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                            float sg = sequence_dot(wig,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) + sequence_dot(whg,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                            float so = sequence_dot(wio,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) + sequence_dot(who,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                            if(truth(bih)) {
                                si += memory[bih+((int)(h))];
                                sf += memory[bih+((int)(H + h))];
                                sg += memory[bih+((int)(2 * H + h))];
                                so += memory[bih+((int)(3 * H + h))];
                            }
                            if(truth(bhh)) {
                                si += memory[bhh+((int)(h))];
                                sf += memory[bhh+((int)(H + h))];
                                sg += memory[bhh+((int)(2 * H + h))];
                                so += memory[bhh+((int)(3 * H + h))];
                            }

                            float i_gate = sigmoid_scalar(si,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                            float f_gate = sigmoid_scalar(sf,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                            float g_gate = tanh(sg);
                            float o_gate = sigmoid_scalar(so,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                            float c = f_gate * memory[c_prev.values+((int)(h))] + i_gate * g_gate;
                            memory[c_prev.values+((int)(h))] = c;
                            memory[ht+((int)(h))] = o_gate * tanh(c);
                        }

                        sequence_copy(ht,ht + H,h_prev.values,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    }
                }
                return true;
            
}
bool recurrent_backward_reference(SequenceVector inputs,SequenceVector grad_outputs,SequenceVector* grad_inputs,SequenceLayer* layer,SequenceArena* arena,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    switch((*layer).type_enum){
case 63: {

                SequenceVector x = inputs;
                SequenceVector go = grad_outputs;
                int T = (*layer).seq_len;
                int I = (*layer).in_features;
                int H = (*layer).out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if ((int)(x.count) != T * I) return false;
                if ((int)(go.count) != T * H) return false;
                bool use_bias = (*layer).use_bias;
                int W = (*layer).weights;
                if (!truth(W)) return false;

                int Wih_sz = (int)(4 * H) * (int)(I);
                int Whh_sz = (int)(4 * H) * (int)(H);
                int bih_sz =truth(use_bias) ? (int)(4 * H) : 0;
                int bhh_sz =truth(use_bias) ? (int)(4 * H) : 0;
                int need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if ((*layer).weights_size < need) return false;

                int Wih = W;
                int Whh = Wih + Wih_sz;
                int bih =truth(use_bias) ? (Whh + Whh_sz) : 0;
                int bhh =truth(use_bias) ? (bih + bih_sz) : 0;

                SequenceVector i_gate=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector f_gate=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector g_gate=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector o_gate=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector c_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector h_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector c_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                for (int t = 0; t < T; ++t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    for (int h = 0; h < H; ++h) {
                        
                        
                        int wii = Wih + (int)(h) * (int)(I);
                        int wif = Wih + (int)(H + h) * (int)(I);
                        int wig = Wih + (int)(2 * H + h) * (int)(I);
                        int wio = Wih + (int)(3 * H + h) * (int)(I);
                        int whi = Whh + (int)(h) * (int)(H);
                        int whf = Whh + (int)(H + h) * (int)(H);
                        int whg = Whh + (int)(2 * H + h) * (int)(H);
                        int who = Whh + (int)(3 * H + h) * (int)(H);

                        float si = sequence_dot(wii,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) + sequence_dot(whi,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                        float sf = sequence_dot(wif,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) + sequence_dot(whf,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                        float sg = sequence_dot(wig,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) + sequence_dot(whg,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                        float so = sequence_dot(wio,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) + sequence_dot(who,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                        if(truth(bih)) {
                            si += memory[bih+((int)(h))];
                            sf += memory[bih+((int)(H + h))];
                            sg += memory[bih+((int)(2 * H + h))];
                            so += memory[bih+((int)(3 * H + h))];
                        }
                        if(truth(bhh)) {
                            si += memory[bhh+((int)(h))];
                            sf += memory[bhh+((int)(H + h))];
                            sg += memory[bhh+((int)(2 * H + h))];
                            so += memory[bhh+((int)(3 * H + h))];
                        }

                        float ig = sigmoid_scalar(si,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                        float fg = sigmoid_scalar(sf,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                        float gg = tanh(sg);
                        float og = sigmoid_scalar(so,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                        float c = fg * memory[c_prev.values+((int)(h))] + ig * gg;
                        float hv = og * tanh(c);

                        memory[i_gate.values+((int)(t) * (int)(H) + (int)(h))] = ig;
                        memory[f_gate.values+((int)(t) * (int)(H) + (int)(h))] = fg;
                        memory[g_gate.values+((int)(t) * (int)(H) + (int)(h))] = gg;
                        memory[o_gate.values+((int)(t) * (int)(H) + (int)(h))] = og;
                        memory[c_state.values+((int)(t) * (int)(H) + (int)(h))] = c;
                        memory[h_state.values+((int)(t) * (int)(H) + (int)(h))] = hv;
                        memory[c_prev.values+((int)(h))] = c;
                        memory[h_prev.values+((int)(h))] = hv;
                    }
                }

                if ((*layer).grad_weights.count != (*layer).weights_size) {
                    sequence_assign(&((*layer).grad_weights),(*layer).weights_size,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                }
                
                sequence_assign(&((*grad_inputs)),x.count,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                SequenceVector dh_next=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector dc_next=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                for (int t = T - 1; t >= 0; --t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    int hprev = (t > 0) ? (h_state.values+((int)(t - 1) * (int)(H))) : 0;
                    int cprev = (t > 0) ? (c_state.values+((int)(t - 1) * (int)(H))) : 0;

                    SequenceVector dh_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    SequenceVector dc_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                    for (int h = 0; h < H; ++h) {
                        int th = (int)(t) * (int)(H) + (int)(h);
                        float ig = memory[i_gate.values+(th)];
                        float fg = memory[f_gate.values+(th)];
                        float gg = memory[g_gate.values+(th)];
                        float og = memory[o_gate.values+(th)];
                        float c = memory[c_state.values+(th)];
                        float tanh_c = tanh(c);
                        float h_prev_v = hprev!=0 ? memory[hprev+((int)(h))] : 0.0f;
                        float c_prev_v = cprev!=0 ? memory[cprev+((int)(h))] : 0.0f;

                        float dht = memory[go.values+(th)] + memory[dh_next.values+((int)(h))];
                        float do_gate = dht * tanh_c;
                        float dc = memory[dc_next.values+((int)(h))] + dht * og * (1.0f - tanh_c * tanh_c);
                        float di_gate = dc * gg;
                        float dg_gate = dc * ig;
                        float df_gate = dc * c_prev_v;
                        memory[dc_prev.values+((int)(h))] = dc * fg;

                        float dai = di_gate * ig * (1.0f - ig);
                        float daf = df_gate * fg * (1.0f - fg);
                        float dag = dg_gate * (1.0f - gg * gg);
                        float dao = do_gate * og * (1.0f - og);

                        int row_i = (int)(h);
                        int row_f = (int)(H + h);
                        int row_g = (int)(2 * H + h);
                        int row_o = (int)(3 * H + h);

                        for (int ii = 0; ii < I; ++ii) {
                            float xv = memory[xt+((int)(ii))];
                            memory[(*layer).grad_weights.values+(row_i * (int)(I) + (int)(ii))] += dai * xv;
                            memory[(*layer).grad_weights.values+(row_f * (int)(I) + (int)(ii))] += daf * xv;
                            memory[(*layer).grad_weights.values+(row_g * (int)(I) + (int)(ii))] += dag * xv;
                            memory[(*layer).grad_weights.values+(row_o * (int)(I) + (int)(ii))] += dao * xv;

                            memory[(*grad_inputs).values+((int)(t) * (int)(I) + (int)(ii))] +=
                                dai * memory[Wih+(row_i * (int)(I) + (int)(ii))] +
                                daf * memory[Wih+(row_f * (int)(I) + (int)(ii))] +
                                dag * memory[Wih+(row_g * (int)(I) + (int)(ii))] +
                                dao * memory[Wih+(row_o * (int)(I) + (int)(ii))];
                        }

                        for (int k = 0; k < H; ++k) {
                            float hp = hprev!=0 ? memory[hprev+((int)(k))] : 0.0f;
                            memory[(*layer).grad_weights.values+(Wih_sz + row_i * (int)(H) + (int)(k))] += dai * hp;
                            memory[(*layer).grad_weights.values+(Wih_sz + row_f * (int)(H) + (int)(k))] += daf * hp;
                            memory[(*layer).grad_weights.values+(Wih_sz + row_g * (int)(H) + (int)(k))] += dag * hp;
                            memory[(*layer).grad_weights.values+(Wih_sz + row_o * (int)(H) + (int)(k))] += dao * hp;

                            memory[dh_prev.values+((int)(k))] +=
                                dai * memory[Whh+(row_i * (int)(H) + (int)(k))] +
                                daf * memory[Whh+(row_f * (int)(H) + (int)(k))] +
                                dag * memory[Whh+(row_g * (int)(H) + (int)(k))] +
                                dao * memory[Whh+(row_o * (int)(H) + (int)(k))];
                        }

                        if(truth(use_bias)) {
                            int b0 = Wih_sz + Whh_sz;
                            memory[(*layer).grad_weights.values+(b0 + row_i)] += dai;
                            memory[(*layer).grad_weights.values+(b0 + row_f)] += daf;
                            memory[(*layer).grad_weights.values+(b0 + row_g)] += dag;
                            memory[(*layer).grad_weights.values+(b0 + row_o)] += dao;
                            memory[(*layer).grad_weights.values+(b0 + bih_sz + row_i)] += dai;
                            memory[(*layer).grad_weights.values+(b0 + bih_sz + row_f)] += daf;
                            memory[(*layer).grad_weights.values+(b0 + bih_sz + row_g)] += dag;
                            memory[(*layer).grad_weights.values+(b0 + bih_sz + row_o)] += dao;
                        }
                    }

                    sequence_swap(&(dh_next),&(dh_prev),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    sequence_swap(&(dc_next),&(dc_prev),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                }
                return true;
            
}
case 64: {

                SequenceVector x = inputs;
                SequenceVector go = grad_outputs;
                int T = (*layer).seq_len;
                int I = (*layer).in_features;
                int H = (*layer).out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if ((int)(x.count) != T * I) return false;
                if ((int)(go.count) != T * H) return false;

                bool use_bias = (*layer).use_bias;
                int W = (*layer).weights;
                if (!truth(W)) return false;
                int Wih_sz = (int)(3 * H) * (int)(I);
                int Whh_sz = (int)(3 * H) * (int)(H);
                int bih_sz =truth(use_bias) ? (int)(3 * H) : 0;
                int bhh_sz =truth(use_bias) ? (int)(3 * H) : 0;
                int need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if ((*layer).weights_size < need) return false;
                int Wih = W;
                int Whh = Wih + Wih_sz;
                int bih =truth(use_bias) ? (Whh + Whh_sz) : 0;
                int bhh =truth(use_bias) ? (bih + bih_sz) : 0;

                SequenceVector h_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector r_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector z_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector n_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                for (int t = 0; t < T; ++t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    for (int h = 0; h < H; ++h) {
                        int w_ir = Wih + (int)(h) * (int)(I);
                        int w_hr = Whh + (int)(h) * (int)(H);
                        float sr = 0.0f;
                        for (int i = 0; i < I; ++i) sr += memory[w_ir+((int)(i))] * memory[xt+((int)(i))];
                        for (int k = 0; k < H; ++k) sr += memory[w_hr+((int)(k))] * memory[h_prev.values+((int)(k))];
                        if(truth(bih)) sr += memory[bih+((int)(h))];
                        if(truth(bhh)) sr += memory[bhh+((int)(h))];
                        float r = sigmoid_scalar(sr,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                        int w_iz = Wih + (int)(H + h) * (int)(I);
                        int w_hz = Whh + (int)(H + h) * (int)(H);
                        float sz = 0.0f;
                        for (int i = 0; i < I; ++i) sz += memory[w_iz+((int)(i))] * memory[xt+((int)(i))];
                        for (int k = 0; k < H; ++k) sz += memory[w_hz+((int)(k))] * memory[h_prev.values+((int)(k))];
                        if(truth(bih)) sz += memory[bih+((int)(H + h))];
                        if(truth(bhh)) sz += memory[bhh+((int)(H + h))];
                        float z = sigmoid_scalar(sz,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                        int w_in = Wih + (int)(2 * H + h) * (int)(I);
                        int w_hn = Whh + (int)(2 * H + h) * (int)(H);
                        float sn = 0.0f;
                        for (int i = 0; i < I; ++i) sn += memory[w_in+((int)(i))] * memory[xt+((int)(i))];
                        for (int k = 0; k < H; ++k) sn += memory[w_hn+((int)(k))] * (r * memory[h_prev.values+((int)(k))]);
                        if(truth(bih)) sn += memory[bih+((int)(2 * H + h))];
                        if(truth(bhh)) sn += memory[bhh+((int)(2 * H + h))];
                        float n = tanh(sn);

                        float hv = (1.0f - z) * n + z * memory[h_prev.values+((int)(h))];

                        int th = (int)(t) * (int)(H) + (int)(h);
                        memory[r_state.values+(th)] = r;
                        memory[z_state.values+(th)] = z;
                        memory[n_state.values+(th)] = n;
                        memory[h_state.values+(th)] = hv;
                    }
                    for (int h = 0; h < H; ++h) {
                        memory[h_prev.values+((int)(h))] = memory[h_state.values+((int)(t) * (int)(H) + (int)(h))];
                    }
                }

                if ((*layer).grad_weights.count != (*layer).weights_size) {
                    sequence_assign(&((*layer).grad_weights),(*layer).weights_size,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                }
                
                sequence_assign(&((*grad_inputs)),x.count,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                SequenceVector dh_next=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                for (int t = T - 1; t >= 0; --t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    int hprev = (t > 0) ? (h_state.values+((int)(t - 1) * (int)(H))) : 0;
                    SequenceVector dh_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                    SequenceVector da_r=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    SequenceVector da_z=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    SequenceVector da_n=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                    for (int h = 0; h < H; ++h) {
                        int th = (int)(t) * (int)(H) + (int)(h);
                        float r = memory[r_state.values+(th)];
                        float z = memory[z_state.values+(th)];
                        float n = memory[n_state.values+(th)];
                        float hp = hprev!=0 ? memory[hprev+((int)(h))] : 0.0f;
                        float dht = memory[go.values+(th)] + memory[dh_next.values+((int)(h))];
                        float dn = dht * (1.0f - z);
                        float dz = dht * (hp - n);
                        memory[dh_prev.values+((int)(h))] += dht * z;
                        memory[da_n.values+((int)(h))] = dn * (1.0f - n * n);
                        memory[da_z.values+((int)(h))] = dz * z * (1.0f - z);
                        
                    }

                    SequenceVector drh=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    for (int h = 0; h < H; ++h) {
                        int row_n = (int)(2 * H + h);
                        for (int i = 0; i < I; ++i) {
                            float xv = memory[xt+((int)(i))];
                            memory[(*layer).grad_weights.values+(row_n * (int)(I) + (int)(i))] += memory[da_n.values+((int)(h))] * xv;
                            memory[(*grad_inputs).values+((int)(t) * (int)(I) + (int)(i))] +=
                                memory[da_n.values+((int)(h))] * memory[Wih+(row_n * (int)(I) + (int)(i))];
                        }
                        for (int k = 0; k < H; ++k) {
                            float hp = hprev!=0 ? memory[hprev+((int)(k))] : 0.0f;
                            float r = memory[r_state.values+((int)(t) * (int)(H) + (int)(k))];
                            memory[(*layer).grad_weights.values+(Wih_sz + row_n * (int)(H) + (int)(k))] += memory[da_n.values+((int)(h))] * (r * hp);
                            memory[drh.values+((int)(k))] += memory[da_n.values+((int)(h))] * memory[Whh+(row_n * (int)(H) + (int)(k))];
                        }
                        if(truth(use_bias)) {
                            int b0 = Wih_sz + Whh_sz;
                            memory[(*layer).grad_weights.values+(b0 + row_n)] += memory[da_n.values+((int)(h))];
                            memory[(*layer).grad_weights.values+(b0 + bih_sz + row_n)] += memory[da_n.values+((int)(h))];
                        }
                    }

                    for (int k = 0; k < H; ++k) {
                        float hp = hprev!=0 ? memory[hprev+((int)(k))] : 0.0f;
                        float r = memory[r_state.values+((int)(t) * (int)(H) + (int)(k))];
                        float dr = memory[drh.values+((int)(k))] * hp;
                        memory[dh_prev.values+((int)(k))] += memory[drh.values+((int)(k))] * r;
                        memory[da_r.values+((int)(k))] = dr * r * (1.0f - r);
                    }

                    for (int h = 0; h < H; ++h) {
                        int row_r = (int)(h);
                        int row_z = (int)(H + h);

                        for (int i = 0; i < I; ++i) {
                            float xv = memory[xt+((int)(i))];
                            memory[(*layer).grad_weights.values+(row_r * (int)(I) + (int)(i))] += memory[da_r.values+((int)(h))] * xv;
                            memory[(*layer).grad_weights.values+(row_z * (int)(I) + (int)(i))] += memory[da_z.values+((int)(h))] * xv;
                            memory[(*grad_inputs).values+((int)(t) * (int)(I) + (int)(i))] +=
                                memory[da_r.values+((int)(h))] * memory[Wih+(row_r * (int)(I) + (int)(i))] +
                                memory[da_z.values+((int)(h))] * memory[Wih+(row_z * (int)(I) + (int)(i))];
                        }

                        for (int k = 0; k < H; ++k) {
                            float hp = hprev!=0 ? memory[hprev+((int)(k))] : 0.0f;
                            memory[(*layer).grad_weights.values+(Wih_sz + row_r * (int)(H) + (int)(k))] += memory[da_r.values+((int)(h))] * hp;
                            memory[(*layer).grad_weights.values+(Wih_sz + row_z * (int)(H) + (int)(k))] += memory[da_z.values+((int)(h))] * hp;

                            memory[dh_prev.values+((int)(k))] +=
                                memory[da_r.values+((int)(h))] * memory[Whh+(row_r * (int)(H) + (int)(k))] +
                                memory[da_z.values+((int)(h))] * memory[Whh+(row_z * (int)(H) + (int)(k))];
                        }

                        if(truth(use_bias)) {
                            int b0 = Wih_sz + Whh_sz;
                            memory[(*layer).grad_weights.values+(b0 + row_r)] += memory[da_r.values+((int)(h))];
                            memory[(*layer).grad_weights.values+(b0 + row_z)] += memory[da_z.values+((int)(h))];
                            memory[(*layer).grad_weights.values+(b0 + bih_sz + row_r)] += memory[da_r.values+((int)(h))];
                            memory[(*layer).grad_weights.values+(b0 + bih_sz + row_z)] += memory[da_z.values+((int)(h))];
                        }
                    }

                    sequence_swap(&(dh_next),&(dh_prev),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                }
                return true;
            
}
case 65: {

                SequenceVector x = inputs;
                SequenceVector go = grad_outputs;
                int T = (*layer).seq_len;
                int I = (*layer).in_features;
                int H = (*layer).out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if ((int)(x.count) != T * I) return false;
                if ((int)(go.count) != T * H) return false;
                bool use_bias = (*layer).use_bias;
                int W = (*layer).weights;
                if (!truth(W)) return false;
                int Wih_sz = (int)(H) * (int)(I);
                int Whh_sz = (int)(H) * (int)(H);
                int bih_sz =truth(use_bias) ? (int)(H) : 0;
                int bhh_sz =truth(use_bias) ? (int)(H) : 0;
                int need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if ((*layer).weights_size < need) return false;

                int Wih = W;
                int Whh = W + Wih_sz;
                int bih =truth(use_bias) ? (Whh + Whh_sz) : 0;
                int bhh =truth(use_bias) ? (bih + bih_sz) : 0;

                SequenceVector h_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                for (int t = 0; t < T; ++t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    int ht = (h_state.values+((int)(t) * (int)(H)));
                    for (int h = 0; h < H; ++h) {
                        float sum = 0.0f;
                        int wih = Wih + (int)(h) * (int)(I);
                        int whh = Whh + (int)(h) * (int)(H);
                        for (int i = 0; i < I; ++i) sum += memory[wih+((int)(i))] * memory[xt+((int)(i))];
                        for (int k = 0; k < H; ++k) sum += memory[whh+((int)(k))] * memory[h_prev.values+((int)(k))];
                        if(truth(bih)) sum += memory[bih+((int)(h))];
                        if(truth(bhh)) sum += memory[bhh+((int)(h))];
                        memory[ht+((int)(h))] = tanh(sum);
                    }
                    for (int h = 0; h < H; ++h) memory[h_prev.values+((int)(h))] = memory[ht+((int)(h))];
                }

                if ((*layer).grad_weights.count != (*layer).weights_size) {
                    sequence_assign(&((*layer).grad_weights),(*layer).weights_size,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                }

                
                sequence_assign(&((*grad_inputs)),x.count,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                SequenceVector dh_next=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);

                for (int t = T - 1; t >= 0; --t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    int hprev = (t > 0) ? (h_state.values+((int)(t - 1) * (int)(H))) : 0;
                    SequenceVector dh_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                    for (int h = 0; h < H; ++h) {
                        float h_t = memory[h_state.values+((int)(t) * (int)(H) + (int)(h))];
                        float dht = memory[go.values+((int)(t) * (int)(H) + (int)(h))] + memory[dh_next.values+((int)(h))];
                        float dpre = dht * (1.0f - h_t * h_t);

                        int wih = Wih + (int)(h) * (int)(I);
                        int whh = Whh + (int)(h) * (int)(H);

                        for (int i = 0; i < I; ++i) {
                            memory[(*layer).grad_weights.values+((int)(h) * (int)(I) + (int)(i))] += dpre * memory[xt+((int)(i))];
                            memory[(*grad_inputs).values+((int)(t) * (int)(I) + (int)(i))] += dpre * memory[wih+((int)(i))];
                        }
                        for (int k = 0; k < H; ++k) {
                            float hp = hprev!=0 ? memory[hprev+((int)(k))] : 0.0f;
                            memory[(*layer).grad_weights.values+(Wih_sz + (int)(h) * (int)(H) + (int)(k))] += dpre * hp;
                            memory[dh_prev.values+((int)(k))] += dpre * memory[whh+((int)(k))];
                        }
                        if(truth(use_bias)) {
                            memory[(*layer).grad_weights.values+(Wih_sz + Whh_sz + (int)(h))] += dpre;
                            memory[(*layer).grad_weights.values+(Wih_sz + Whh_sz + bih_sz + (int)(h))] += dpre;
                        }
                    }
                    sequence_swap(&(dh_next),&(dh_prev),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
                }
                return true;
            
}

    default:return false;
    }
}
void recurrent_kernel(int x,int w,int dy,int y,int dw,int workspace,int status,int type,GRecurrent g,bool bias,int weights_size,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    if(blockIdx_x!=0 || threadIdx_x!=0)return;
    SequenceArena arena=(SequenceArena){workspace,0,g.workspace};SequenceVector xv=(SequenceVector){x,(int)(g.steps)*g.input_value};SequenceVector yv=(SequenceVector){y,dy!=0 ?(int)(g.steps)*g.input_value:(int)(g.steps)*g.hidden};SequenceVector gv=(SequenceVector){dy,(int)(g.steps)*g.hidden};
    SequenceLayer layer=(SequenceLayer){type,g.steps,g.input_value,g.hidden,bias,w,weights_size,(SequenceVector){dw,weights_size}};
    
    memory[status]=as_float((int)( truth(dy) ?recurrent_backward_reference(xv,gv,&(yv),&(layer),&(arena),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch):recurrent_forward_reference(xv,&(yv),&(layer),&(arena),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)));
}
float box_iou(int boxes,int a,int b,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    a*=4;b*=4;
    float aa=max(0.f,memory[boxes+(a+2)]-memory[boxes+(a)])*max(0.f,memory[boxes+(a+3)]-memory[boxes+(a+1)]);
    float ab=max(0.f,memory[boxes+(b+2)]-memory[boxes+(b)])*max(0.f,memory[boxes+(b+3)]-memory[boxes+(b+1)]);
    float iw=max(0.f,min(memory[boxes+(a+2)],memory[boxes+(b+2)])-max(memory[boxes+(a)],memory[boxes+(b)]));
    float ih=max(0.f,min(memory[boxes+(a+3)],memory[boxes+(b+3)])-max(memory[boxes+(a+1)],memory[boxes+(b+1)]));
    float intersection=iw*ih,area=aa+ab-intersection;return area>0?intersection/area:0.f;
}
void nms_kernel(int boxes,int scores,int classes,int visited,int output_value,int count,int n,float score_threshold,float iou_threshold,int maximum,bool agnostic,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    if(blockIdx_x!=0 || threadIdx_x!=0)return;memory[count]=as_float((int)(0));
    for(int i=0;i<n;++i){bool valid=finite_value(memory[scores+(i)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch) && memory[scores+(i)]>=score_threshold;
        for(int d=0;d<4;++d)valid=truth(valid) && finite_value(memory[boxes+(i*4+d)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);if(truth(classes))valid=truth(valid) && finite_value(memory[classes+(i)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);memory[visited+(i)]=truth(valid) ?0.f:1.f;}
    for(int pass=0;pass<n;++pass){
        int best=-1;for(int i=0;i<n;++i)if(!truth(memory[visited+(i)]) && (best<0 || memory[scores+(i)]>memory[scores+(best)]))best=i;
        if(best<0)break;memory[visited+(best)]=1;bool suppress=false;
        int cls=truth(classes) ?(int)(round_away((float)(memory[classes+(best)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)):0;
        for(int j=0;j<as_int(memory[count]);++j){int accepted=(int)(memory[output_value+(j)]);
            bool same=truth(agnostic) || !truth(classes) || cls==(int)(round_away((float)(memory[classes+(accepted)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch));
            if(same && box_iou(boxes,best,accepted,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)>iou_threshold){suppress=true;break;}}
        if(truth(suppress))continue;memory[output_value+(as_int(memory[count]))]=(float)(best);memory[count]=as_float(as_int(memory[count])+1);if(maximum>0 && as_int(memory[count])>=maximum)break;
    }
}
void nms_stop_gradient_kernel(int gradient,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    for(int i=blockIdx_x*blockDim_x+threadIdx_x;i<count;i+=blockDim_x*gridDim_x)memory[gradient+(i)]=0.f;
}
void cuda_unary_kernel(int in_value,int out_value,int n,int op,float alpha,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = (int)(blockIdx_x * blockDim_x + threadIdx_x);
    if (i >= n) return;
    float x = memory[in_value+(i)];
    switch (op) {
        case (int)(0):
            memory[out_value+(i)] = x > 0.0f ? x : 0.0f;
            break;
        case (int)(1):
            memory[out_value+(i)] = x > 0.0f ? x : alpha * x;
            break;
        case (int)(2):
            memory[out_value+(i)] = 1.0f / (1.0f + exp(-x));
            break;
        case (int)(3):
            memory[out_value+(i)] = tanh(x);
            break;
        case (int)(4): {
            float s = 1.0f / (1.0f + exp(-x));
            memory[out_value+(i)] = x * s;
            break;
        }
        case (int)(5): {
            float c = 0.7978845608f;
            float x3 = x * x * x;
            memory[out_value+(i)] = 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x3)));
            break;
        }
        case (int)(6):
            memory[out_value+(i)] = x > 20.0f ? x : log1pf(exp(x),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
            break;
        case (int)(7): {
            float sp = x > 20.0f ? x : log1pf(exp(x),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
            memory[out_value+(i)] = x * tanh(sp);
            break;
        }
        case (int)(8): {
            float hs = (x + 3.0f) / 6.0f;
            memory[out_value+(i)] = min(1.0f, max(0.0f, hs));
            break;
        }
        case (int)(9): {
            float hs = min(1.0f, max(0.0f, (x + 3.0f) / 6.0f));
            memory[out_value+(i)] = x * hs;
            break;
        }
        default:
            memory[out_value+(i)] = x;
            break;
    }
}
void cuda_binary_kernel(int a,int b,int out_value,int n,int op,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = (int)(blockIdx_x * blockDim_x + threadIdx_x);
    if (i >= n) return;
    switch (op) {
        case (int)(0):
            memory[out_value+(i)] = memory[a+(i)] + memory[b+(i)];
            break;
        case (int)(1):
            memory[out_value+(i)] = memory[a+(i)] - memory[b+(i)];
            break;
        case (int)(2):
            memory[out_value+(i)] = memory[a+(i)] * memory[b+(i)];
            break;
        case (int)(3): {
            float d = memory[b+(i)];
            memory[out_value+(i)] = (fabs(d) > 1e-12f) ? (memory[a+(i)] / d) : 0.0f;
            break;
        }
        default:
            memory[out_value+(i)] = memory[a+(i)];
            break;
    }
}
void cuda_unary_backward_kernel(int in_value,int grad_out,int grad_in,int n,int op,float alpha,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = (int)(blockIdx_x * blockDim_x + threadIdx_x);
    if (i >= n) return;

    float x = memory[in_value+(i)];
    float go = memory[grad_out+(i)];
    float d = 1.0f;

    switch (op) {
        case (int)(0):
            d = x > 0.0f ? 1.0f : 0.0f;
            break;
        case (int)(1):
            d = x > 0.0f ? 1.0f : alpha;
            break;
        case (int)(2): {
            float s = 1.0f / (1.0f + exp(-x));
            d = s * (1.0f - s);
            break;
        }
        case (int)(3): {
            float t = tanh(x);
            d = 1.0f - t * t;
            break;
        }
        case (int)(4): {
            float s = 1.0f / (1.0f + exp(-x));
            d = s + x * s * (1.0f - s);
            break;
        }
        case (int)(5): {
            float c = 0.7978845608f;
            float x2 = x * x;
            float x3 = x2 * x;
            float u = c * (x + 0.044715f * x3);
            float t = tanh(u);
            float sech2 = 1.0f - t * t;
            float du = c * (1.0f + 0.134145f * x2);
            d = 0.5f * (1.0f + t) + 0.5f * x * sech2 * du;
            break;
        }
        case (int)(6):
            d = 1.0f / (1.0f + exp(-x));
            break;
        case (int)(7): {
            float sp = x > 20.0f ? x : log1pf(exp(x),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
            float t = tanh(sp);
            float s = 1.0f / (1.0f + exp(-x));
            d = t + x * s * (1.0f - t * t);
            break;
        }
        case (int)(8):
            d = (x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f;
            break;
        case (int)(9):
            if (x <= -3.0f) d = 0.0f;
            else if (x >= 3.0f) d = 1.0f;
            else d = x / 3.0f + 0.5f;
            break;
        default:
            d = 1.0f;
            break;
    }

    memory[grad_in+(i)] = go * d;
}
void cuda_binary_backward_kernel(int a,int b,int grad_out,int grad_a,int grad_b,int n,int op,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
    int i = (int)(blockIdx_x * blockDim_x + threadIdx_x);
    if (i >= n) return;

    float av = memory[a+(i)];
    float bv = memory[b+(i)];
    float go = memory[grad_out+(i)];
    float kEps = 1e-12f;

    switch (op) {
        case (int)(0):
            memory[grad_a+(i)] = go;
            memory[grad_b+(i)] = go;
            break;
        case (int)(1):
            memory[grad_a+(i)] = go;
            memory[grad_b+(i)] = -go;
            break;
        case (int)(2):
            memory[grad_a+(i)] = go * bv;
            memory[grad_b+(i)] = go * av;
            break;
        case (int)(3): {
            if (fabs(bv) > kEps) {
                memory[grad_a+(i)] = go / bv;
                memory[grad_b+(i)] = -go * av / (bv * bv);
            } else {
                memory[grad_a+(i)] = 0.0f;
                memory[grad_b+(i)] = 0.0f;
            }
            break;
        }
        default:
            memory[grad_a+(i)] = go;
            memory[grad_b+(i)] = 0.0f;
            break;
    }
}
void portable_copy(int source,int dest,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){int i=blockIdx_x*blockDim_x+threadIdx_x;if(i<n)memory[dest+(i)]=memory[source+(i)];}
void portable_gemm(int a,int b,int c,int m,int n,int k,int lda,int ldb,int ldc,bool ta,bool tb,float alpha,float beta,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
 int index=blockIdx_x*blockDim_x+threadIdx_x;int i=index%m,j=index/m;if(j<n){float sum=0;for(int f=0;f<k;++f)sum+=memory[a+( truth(ta) ?f+i*lda:i+f*lda)]*memory[b+( truth(tb) ?j+f*ldb:f+j*ldb)];memory[c+(i+j*ldc)]=alpha*sum+beta*memory[c+(i+j*ldc)];}}
void portable_ger(int x,int y,int a,int m,int n,int sx,int sy,int lda,float alpha,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){int index=blockIdx_x*blockDim_x+threadIdx_x;int i=index%m,j=index/m;if(j<n)memory[a+(i+j*lda)]+=alpha*memory[x+(i*sx)]*memory[y+(j*sy)];}
void portable_sequence_constant_forward(int source,int rows,int result,int width,int base_rows,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
 int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=count)return;
 int row=i/width,c=i%width,a=(int)(memory[rows+(2*row)]),b=a+1<base_rows?a+1:base_rows-1;
 float t=memory[rows+(2*row+1)];memory[result+(i)]=(1.0f-t)*memory[source+(a*width+c)]+t*memory[source+(b*width+c)];
}
void portable_sequence_constant_backward(int source,int rows,int result,int width,int base_rows,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
 int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=count)return;
 int row=i/width,c=i%width,a=(int)(memory[rows+(2*row)]),b=a+1<base_rows?a+1:base_rows-1;
 float t=memory[rows+(2*row+1)];atomicAdd(result+a*width+c,(1.0f-t)*memory[source+(i)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);atomicAdd(result+b*width+c,t*memory[source+(i)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);
}
void tiled_gemm(int a,int b,int c,int m,int n,int k,int lda,int ldb,int ldc,bool ta,bool tb,float alpha,float beta,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){
 int local_value=(int)(get_local_id(0)),group=(int)(get_group_id(0));
 int tiles=(m+15)/16,r=local_value%16,d=local_value/16,row=(group%tiles)*16+r,col=(group/tiles)*16+d;
 float sum=0;
 for(int base=0;base<k;base+=16){
  int fa=base+d,fb=base+r;
  normScratch[local_value]=row<m && fa<k?memory[a+(ta?fa+row*lda:row+fa*lda)]:0.f;
  normScratch[256+local_value]=col<n && fb<k?memory[b+(tb?col+fb*ldb:fb+col*ldb)]:0.f;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int f=0;f<16;++f)sum+=normScratch[r+f*16]*normScratch[256+f+d*16];
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 if(row<m && col<n)memory[c+row+col*ldc]=alpha*sum+(beta==0.f?0.f:beta*memory[c+row+col*ldc]);
}
int group_id(__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return (int)(get_group_id(0));}
int invocation_id(__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x,__local float* normScratch){return (int)(get_global_id(0));}
__kernel void reference(__global float* memory,__global const int* arguments){__local float normScratch[512];int threadIdx_x=0,blockIdx_x=0,blockDim_x=1,gridDim_x=1;
 blockDim_x=arguments[1];gridDim_x=arguments[2];
 int selected=specializedOperation>=0?specializedOperation:arguments[0];if(selected==0){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));norm_forward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],as_float(arguments[11]),arguments[12]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==1){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));norm_backward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==2){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));channel_gradient(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==7){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));softmax_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==10){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));row_norm_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],as_float(arguments[9]),arguments[10]!=0,arguments[11]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==15){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));GPool g=(GPool){arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16],arguments[17],arguments[18]!=0,arguments[19]!=0};pool_kernel(arguments[3],arguments[4],arguments[5],arguments[6],g,arguments[20],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==21){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));GBatchNorm g=(GBatchNorm){arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,arguments[14]!=0};batch_norm_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],g,as_float(arguments[15]),arguments[16]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==24){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));GAttention g=(GAttention){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11]!=0,arguments[12]!=0,arguments[13],arguments[14],arguments[15],arguments[16]};attention_probabilities(arguments[3],arguments[4],arguments[5],g,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==27){blockDim_x=256;blockIdx_x=group_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);threadIdx_x=(int)(get_local_id(0));GAttention g=(GAttention){arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,arguments[14],arguments[15],arguments[16],arguments[17]};attention_score_grad(arguments[3],arguments[4],arguments[5],arguments[6],g,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}if(selected==39){tiled_gemm(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,as_float(arguments[14]),as_float(arguments[15]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);return;}
 int id=selected;bool parallel=(id>=0 && id<=42 && id!=31 && id!=32);int count=blockDim_x*gridDim_x;if(id==38)count=arguments[5];if(id==39 || id==40)count=arguments[6]*arguments[7];int first=parallel?invocation_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch):0;if(!parallel && invocation_id(memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch)!=0)return;for(int gid=first;gid<count;gid+=parallel?count:1){blockIdx_x=gid/blockDim_x;threadIdx_x=gid%blockDim_x;if(id>=38 && id<=40){blockIdx_x=gid;threadIdx_x=0;blockDim_x=1;}switch(selected){
case 0: {norm_forward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],as_float(arguments[11]),arguments[12]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 1: {norm_backward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 2: {channel_gradient(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 3: {resize_forward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 4: {nearest_backward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 5: {bilinear_backward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 6: {causal_mask(arguments[3],arguments[4],arguments[5],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 7: {softmax_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 8: {im2col(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 9: {col2im(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 10: {row_norm_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],as_float(arguments[9]),arguments[10]!=0,arguments[11]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 11: {row_norm_grad(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 12: {GProjection g=(GProjection){arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16],arguments[17],arguments[18],arguments[19],arguments[20],arguments[21],arguments[22]!=0};projection_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],g,arguments[23],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 13: {geglu_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 14: {GResize g=(GResize){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]};bicubic_kernel(arguments[3],arguments[4],arguments[5],g,arguments[13],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 15: {GPool g=(GPool){arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16],arguments[17],arguments[18]!=0,arguments[19]!=0};pool_kernel(arguments[3],arguments[4],arguments[5],arguments[6],g,arguments[20],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 16: {permute_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 17: {spatial_move(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14],arguments[15]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 18: {GExtendedConv g=(GExtendedConv){arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16],arguments[17],arguments[18],arguments[19],arguments[20],arguments[21]!=0};extended_conv_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],g,arguments[22],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 19: {dropout_kernel(arguments[3],arguments[4],arguments[5],arguments[6],as_float(arguments[7]),arguments[8]!=0,arguments[9]!=0,arguments[10]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 20: {reparameterize_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 21: {GBatchNorm g=(GBatchNorm){arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,arguments[14]!=0};batch_norm_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],g,as_float(arguments[15]),arguments[16]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 22: {attention_add_bias(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 23: {GAttention g=(GAttention){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11]!=0,arguments[12]!=0,arguments[13],arguments[14],arguments[15],arguments[16]};attention_project(arguments[3],arguments[4],arguments[5],g,arguments[17],arguments[18]!=0,arguments[19],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 24: {GAttention g=(GAttention){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11]!=0,arguments[12]!=0,arguments[13],arguments[14],arguments[15],arguments[16]};attention_probabilities(arguments[3],arguments[4],arguments[5],g,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 25: {GAttention g=(GAttention){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11]!=0,arguments[12]!=0,arguments[13],arguments[14],arguments[15],arguments[16]};attention_context(arguments[3],arguments[4],arguments[5],g,arguments[17],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 26: {GAttention g=(GAttention){arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,arguments[14],arguments[15],arguments[16],arguments[17]};attention_output(arguments[3],arguments[4],arguments[5],arguments[6],g,arguments[18],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 27: {GAttention g=(GAttention){arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,arguments[14],arguments[15],arguments[16],arguments[17]};attention_score_grad(arguments[3],arguments[4],arguments[5],arguments[6],g,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 28: {GAttention g=(GAttention){arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14]!=0,arguments[15]!=0,arguments[16],arguments[17],arguments[18],arguments[19]};attention_projected_grad(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],g,arguments[20],arguments[21],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 29: {GAttention g=(GAttention){arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16]!=0,arguments[17]!=0,arguments[18],arguments[19],arguments[20],arguments[21]};attention_weight_grad(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],g,arguments[22],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 30: {GAttention g=(GAttention){arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13]!=0,arguments[14]!=0,arguments[15],arguments[16],arguments[17],arguments[18]};attention_input_grad(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],g,arguments[19]!=0,arguments[20],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 31: {GRecurrent g=(GRecurrent){arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16]};recurrent_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],g,arguments[17]!=0,arguments[18],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 32: {nms_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],as_float(arguments[10]),as_float(arguments[11]),arguments[12],arguments[13]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 33: {nms_stop_gradient_kernel(arguments[3],arguments[4],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 34: {cuda_unary_kernel(arguments[3],arguments[4],arguments[5],arguments[6],as_float(arguments[7]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 35: {cuda_binary_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 36: {cuda_unary_backward_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],as_float(arguments[8]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 37: {cuda_binary_backward_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 38: {portable_copy(arguments[3],arguments[4],arguments[5],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 39: {portable_gemm(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,as_float(arguments[14]),as_float(arguments[15]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 40: {portable_ger(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],as_float(arguments[11]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 41: {portable_sequence_constant_forward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
case 42: {portable_sequence_constant_backward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x,normScratch);break;}
default:break;}}}
