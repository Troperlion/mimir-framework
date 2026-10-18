#pragma once
#include <cstdint>
namespace PortableStaging {
#ifdef ENABLE_OPENCL
inline constexpr const char* referenceOpenCL=R"MIMIRCL(#define truth(v) ((v)!=0)
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
SequenceVector sequence_new(SequenceArena* arena,int n,float value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
 SequenceVector result=(SequenceVector){(*arena).data+(*arena).used,n};(*arena).used+=n;
 for(int i=0;i<n;++i)memory[result.values+i]=value;return result;
}
void sequence_assign(SequenceVector* v,int n,float value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){for(int i=0;i<n;++i)memory[(*v).values+i]=value;(*v).count=n;}
void sequence_swap(SequenceVector* a,SequenceVector* b,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){SequenceVector c=(*a);(*a)=(*b);(*b)=c;}
void sequence_copy(int first,int last,int output_value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){while(first!=last){memory[output_value++]=memory[first++];}}
float sequence_dot(int w,int x,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){float s=0;for(int i=0;i<n;++i)s+=memory[w+i]*memory[x+i];return s;}
float reduce_sum(float value,int scratch,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){return value;}
float reduce_max(float value,int scratch,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){return value;}
void atomicAdd(int at,float value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){memory[at]+=value;}
float round_away(float v,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){return v>=0.0f?floor(v+0.5f):ceil(v-0.5f);}
float rsqrtf(float v,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){return rsqrt(v);}
float log1pf(float v,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){return log(1.0f+v);}
bool finite_value(float x,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){return !isnan(x) && !isinf(x);}
void norm_forward(int x,int weights,int y,int normalized,int invstd,int group_size,int spatial,int channels,float eps,bool bias,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int scratch=0;
    int base = blockIdx_x * group_size;
    float sum = 0;
    for (int i = threadIdx_x; i < group_size; i += 1) sum += memory[x+(base + i)];
    float mean = reduce_sum(sum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) / group_size;
    float variance = 0;
    for (int i = threadIdx_x; i < group_size; i += 1) {
        float d = memory[x+(base + i)] - mean;
        variance += d * d;
    }
    float inv = rsqrtf(reduce_sum(variance,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) / group_size + eps,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    if (threadIdx_x == 0 && invstd!=0) memory[invstd+(blockIdx_x)] = inv;
    for (int i = threadIdx_x; i < group_size; i += 1) {
        int index = base + i;
        int channel = (index / spatial) % channels;
        float value = (memory[x+(index)] - mean) * inv;
        if(truth(normalized)) memory[normalized+(index)] = value;
        memory[y+(index)] = weights!=0 ? value * memory[weights+(channel)] + ( truth(bias) ? memory[weights+(channels + channel)] : 0.f) : value;
    }
}
void norm_backward(int xhat,int invstd,int dy,int weights,int dx,int group_size,int spatial,int channels,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int scratch=0;
    int base = blockIdx_x * group_size;
    float sum = 0, product = 0;
    for (int i = threadIdx_x; i < group_size; i += 1) {
        int index = base + i;
        float g = memory[dy+(index)] * (weights!=0 ? memory[weights+((index / spatial) % channels)] : 1.f);
        sum += g;
        product += g * memory[xhat+(index)];
    }
    float s1 = reduce_sum(sum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    float s2 = reduce_sum(product,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    for (int i = threadIdx_x; i < group_size; i += 1) {
        int index = base + i;
        float g = memory[dy+(index)] * (weights!=0 ? memory[weights+((index / spatial) % channels)] : 1.f);
        memory[dx+(index)] = memory[invstd+(blockIdx_x)] * (g - s1 / group_size - memory[xhat+(index)] * s2 / group_size);
    }
}
void channel_gradient(int dy,int xhat,int dw,int db,int spatial,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int scratch=0;
    int channel = blockIdx_x;
    float sum = 0, product = 0;
    for (int s = threadIdx_x; s < spatial; s += 1) {
        int index = channel * spatial + s;
        sum += memory[dy+(index)];
        if(truth(xhat)) product += memory[dy+(index)] * memory[xhat+(index)];
    }
    float bias = reduce_sum(sum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    float gamma = reduce_sum(product,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    if (threadIdx_x == 0) {
        if(truth(dw)) memory[dw+(channel)] = gamma;
        if(truth(db)) memory[db+(channel)] = bias;
    }
}
void resize_forward(int x,int y,int count,int ih,int iw,int oh,int ow,int sh,int sw,bool bilinear,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
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
void nearest_backward(int dy,int dx,int count,int ih,int iw,int sh,int sw,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i >= count) return;
    int c = i / (ih * iw), h = (i / iw) % ih, w = i % iw;
    float sum = 0;
    for (int y = 0; y < sh; ++y) for (int x = 0; x < sw; ++x)
        sum += memory[dy+((c * ih * sh + h * sh + y) * iw * sw + w * sw + x)];
    memory[dx+(i)] = sum;
}
void bilinear_backward(int dy,int dx,int count,int ih,int iw,int oh,int ow,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i >= count) return;
    int c = i / (oh * ow), h = (i / ow) % oh, w = i % ow;
    float hf = h * ((float)(ih) / oh), wf = w * ((float)(iw) / ow);
    int h0 = (int)(floor(hf)), w0 = (int)(floor(wf));
    int h1 = min(h0+1, ih-1), w1 = min(w0+1, iw-1);
    float dh = hf-h0, dd = wf-w0, g = memory[dy+(i)];
    int base = c*ih*iw;
    atomicAdd(dx+base+h0*iw+w0,g*(1-dh)*(1-dd),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    atomicAdd(dx+base+h0*iw+w1,g*(1-dh)*dd,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    atomicAdd(dx+base+h1*iw+w0,g*dh*(1-dd),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    atomicAdd(dx+base+h1*iw+w1,g*dh*dd,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
}
void causal_mask(int scores,int rows,int columns,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i<rows*columns && i%columns>i/columns)memory[scores+(i)]=-1e9f;
}
void softmax_kernel(int x,int dy,int y,int axis_size,int inner,bool logarithmic,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int scratch=0;
    int row = blockIdx_x;
    int base = (row / inner) * axis_size * inner + row % inner;
    float maximum = -as_float(0x7f800000u);
    for (int a = threadIdx_x; a < axis_size; a += 1) maximum = max(maximum, memory[x+(base+a*inner)]);
    maximum = reduce_max(maximum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    float sum = 0;
    for (int a = threadIdx_x; a < axis_size; a += 1) sum += exp(memory[x+(base+a*inner)]-maximum);
    sum = reduce_sum(sum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    float reduction = 0;
    if(truth(dy)) {
        for (int a = threadIdx_x; a < axis_size; a += 1)
            reduction += memory[dy+(base+a*inner)] * ( truth(logarithmic) ? 1.f : exp(memory[x+(base+a*inner)]-maximum)/sum);
        reduction = reduce_sum(reduction,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    }
    for (int a = threadIdx_x; a < axis_size; a += 1) {
        int i = base+a*inner;
        float probability = exp(memory[x+(i)]-maximum)/sum;
        // Preserve the existing CPU LogSoftmax backward contract, which uses
        // the log output_value rather than its exponential in_value the reduction term.
        float log_probability = memory[x+(i)]-maximum-log(sum);
        memory[y+(i)] = dy!=0 ? ( truth(logarithmic) ? memory[dy+(i)]-log_probability*reduction : probability*(memory[dy+(i)]-reduction))
                  : ( truth(logarithmic) ? memory[x+(i)]-maximum-log(sum) : probability);
    }
}
void im2col(int x,int col,int count,int h,int w,int k,int stride,int pad,int dilation,int oh,int ow,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i >= count) return;
    int s=i%(oh*ow), row=i/(oh*ow), c=row/(k*k), kh=(row/k)%k, kw=row%k;
    int ih=(s/ow)*stride+kh*dilation-pad, iw=(s%ow)*stride+kw*dilation-pad;
    memory[col+(i)] = ih>=0 && ih<h && iw>=0 && iw<w ? memory[x+((c*h+ih)*w+iw)] : 0.f;
}
void col2im(int col,int dx,int count,int h,int w,int k,int stride,int pad,int dilation,int oh,int ow,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
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
void row_norm_kernel(int x,int w,int dy,int y,int xhat,int width,float eps,bool rms,bool bias,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int scratch=0;int base=blockIdx_x*width;float sum=0;
    if(!truth(rms))for(int i=threadIdx_x;i<width;i+=1)sum+=memory[x+(base+i)];
    float mean=truth(rms) ?0.f:reduce_sum(sum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)/width;float variance=0;
    for(int i=threadIdx_x;i<width;i+=1){float d=memory[x+(base+i)]-mean;variance+=d*d;}
    float inv=rsqrtf(reduce_sum(variance,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)/width+eps,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    if(!truth(dy)){for(int i=threadIdx_x;i<width;i+=1){float v=(memory[x+(base+i)]-mean)*inv;memory[y+(base+i)]=v*(w!=0 ?memory[w+(i)]:1.f)+( truth(bias) ?memory[w+(width+i)]:0.f);}return;}
    float sg=0,sgx=0;
    for(int i=threadIdx_x;i<width;i+=1){float v=(memory[x+(base+i)]-mean)*inv;float g=memory[dy+(base+i)]*(w!=0 ?memory[w+(i)]:1.f);memory[xhat+(base+i)]=v;sg+=g;sgx+=g*v;}
    float a=reduce_sum(sg,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)/width,b=reduce_sum(sgx,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)/width;
    for(int i=threadIdx_x;i<width;i+=1){float g=memory[dy+(base+i)]*(w!=0 ?memory[w+(i)]:1.f);memory[y+(base+i)]=inv*(g-( truth(rms) ?0.f:a)-memory[xhat+(base+i)]*b);}
}
void row_norm_grad(int dy,int xhat,int dw,int db,int rows,int width,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=width)return;float a=0,b=0;
    for(int r=0;r<rows;++r){int j=r*width+i;a+=memory[dy+(j)]*memory[xhat+(j)];b+=memory[dy+(j)];}memory[dw+(i)]=a;if(truth(db))memory[db+(i)]=b;
}
void projection_kernel(int a,int b,int w,int dy,int y,int da,int db,int dw,GProjection g,int offset_count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=g.output_value)return;
    if(g.mode==3 && o<g.prefix){if(truth(dy))memory[da+(o)]=memory[dy+(o)];else memory[y+(o)]=memory[a+(o)];return;}
    int d=(o-(g.mode==3?g.prefix:0))%g.dim,r=(o-(g.mode==3?g.prefix:0))/g.dim;
    float grad=truth(dy) ?memory[dy+(o)]:0.f;float value=0;
    if(g.mode<=2){
        int start=g.mode==1?r:0,end=g.mode==1?r+1:g.ids;
        if(g.mode==2 && offset_count!=0){start=max(0,min((int)(round_away((float)(memory[b+(r)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)),g.ids));end=max(start,min((int)(round_away((float)(memory[b+(r+1)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)),g.ids));}
        for(int i=start;i<end;++i){int id=(int)(round_away((float)(memory[a+(i)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x));if(id<0 || id>=g.vocab || id==g.pad)continue;
            int j=id*g.dim+d;if(truth(dy))atomicAdd(dw+j,grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);else value+=memory[w+(j)];}
    }else if(g.mode==3){
        float inv=rsqrtf((float)(g.features),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);int base=g.prefix+r*g.features;
        if(truth(dy))atomicAdd(dw+g.features*g.dim+d,grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);else value=memory[w+(g.features*g.dim+d)];
        for(int k=0;k<g.features;++k){int j=k*g.dim+d;if(truth(dy)){atomicAdd(da+base+k,grad*memory[w+(j)]*inv,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);atomicAdd(dw+j,grad*memory[a+(base+k)]*inv,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);}else value+=(memory[a+(base+k)]*inv)*memory[w+(j)];}
    }else{
        int base=d*g.features*g.second;
        if(truth(g.bias)){if(truth(dy))atomicAdd(dw+g.dim*g.features*g.second+d,grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);else value=memory[w+(g.dim*g.features*g.second+d)];}
        for(int i=0;i<g.features;++i)for(int j=0;j<g.second;++j){int wi=base+i*g.second+j,ai=r*g.features+i,bi=r*g.second+j;
            if(truth(dy)){atomicAdd(da+ai,grad*memory[w+(wi)]*memory[b+(bi)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);atomicAdd(db+bi,grad*memory[w+(wi)]*memory[a+(ai)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);atomicAdd(dw+wi,grad*memory[a+(ai)]*memory[b+(bi)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);}else value+=memory[a+(ai)]*memory[w+(wi)]*memory[b+(bi)];}
    }
    if(!truth(dy))memory[y+(o)]=value;
}
void geglu_kernel(int x,int dy,int y,int width,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;int a=(o/width)*2*width+o%width,b=a+width;
    float z=memory[x+(b)],th=tanh(.7978845608028654f*(z+.044715f*z*z*z)),gelu=z*.5f*(1+th);
    if(!truth(dy))memory[y+(o)]=memory[x+(a)]*gelu;else{memory[y+(a)]=memory[dy+(o)]*gelu;memory[y+(b)]=memory[dy+(o)]*memory[x+(a)]*(.5f*(1+th)+.5f*z*(1-th*th)*.7978845608028654f*(1+3*.044715f*z*z));}
}
float cubic(float x,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){x=fabs(x);float a=-.75f;if(x<=1)return ((a+2)*x-(a+3))*x*x+1;if(x<2)return (((a*x-5*a)*x+8*a)*x)-4*a;return 0;}
void bicubic_kernel(int x,int dy,int y,GResize g,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;int c=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
    float sy=(oy+.5f)*((float)(g.ih)/g.oh)-.5f,sx=(ox+.5f)*((float)(g.iw)/g.ow)-.5f;
    int iy=(int)(floor(sy)),ix=(int)(floor(sx));float sum=0,wsum=0;
    for(int h=-1;h<=2;++h)for(int w=-1;w<=2;++w){float weight=cubic(sy-(iy+h),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)*cubic(sx-(ix+w),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);wsum+=weight;if(!truth(dy))sum+=weight*memory[x+((c*g.ih+max(0,min(iy+h,g.ih-1)))*g.iw+max(0,min(ix+w,g.iw-1)))];}
    if(!truth(dy)){memory[y+(o)]=truth(wsum) ?sum/wsum:0.f;return;}if(!truth(wsum))return;
    for(int h=-1;h<=2;++h)for(int w=-1;w<=2;++w){float weight=cubic(sy-(iy+h),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)*cubic(sx-(ix+w),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);int i=(c*g.ih+max(0,min(iy+h,g.ih-1)))*g.iw+max(0,min(ix+w,g.iw-1));atomicAdd(y+i,memory[dy+(o)]*(weight/wsum),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);}
}
void pool_kernel(int x,int dy,int y,int dx,GPool g,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;
    int c=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
    float value=g.maximum?-as_float(0x7f800000u):0.f;int valid=0,best=-1;
    for(int ky=0;ky<g.kh;++ky)for(int kx=0;kx<g.kw;++kx) {
        int iy=oy*g.sh+ky-g.ph,ix=ox*g.sw+kx-g.pw;
        if(iy<0 || iy>=g.h || ix<0 || ix>=g.w)continue;
        int i=g.token?iy*g.c+c:(c*g.h+iy)*g.w+ix;
        ++valid;if(truth(g.maximum)){if(memory[x+(i)]>value){value=memory[x+(i)];best=i;}}else value+=memory[x+(i)];
    }
    if(!truth(dy)){memory[y+(o)]=g.maximum?value:( truth(valid) ?value/valid:0.f);return;}
    if(truth(g.maximum)){if(best>=0)atomicAdd(dx+best,memory[dy+(o)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);return;}
    if(!truth(valid))return;float grad=memory[dy+(o)]/valid;
    for(int ky=0;ky<g.kh;++ky)for(int kx=0;kx<g.kw;++kx){
        int iy=oy*g.sh+ky-g.ph,ix=ox*g.sw+kx-g.pw;
        if(iy>=0 && iy<g.h && ix>=0 && ix<g.w)atomicAdd(dx+( truth(g.token) ?iy*g.c+c:(c*g.h+iy)*g.w+ix),grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    }
}
void permute_kernel(int x,int y,int out_shape,int in_stride,int rank,int count,bool backward,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;int tmp=o,i=0;
    for(int d=rank-1;d>=0;--d){i+=(tmp%as_int(memory[out_shape+(d)]))*as_int(memory[in_stride+(d)]);tmp/=as_int(memory[out_shape+(d)]);}
    if(truth(backward))memory[y+(i)]=memory[x+(o)];else memory[y+(o)]=memory[x+(i)];
}
void spatial_move(int x,int y,int count,int c,int h,int w,int oh,int ow,int ph,int pw,int r,int padding,bool backward,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;
    int ch=o/(oh*ow),oy=(o/ow)%oh,ox=o%ow;int i=-1;
    if(truth(r)) {int ic=ch*r*r+(oy%r)*r+ox%r;i=(ic*h+oy/r)*w+ox/r;}
    else {
        int iy=oy-ph,ix=ox-pw;
        if(padding==2){if(iy<0)iy=-iy;if(iy>=h)iy=2*h-iy-2;if(ix<0)ix=-ix;if(ix>=w)ix=2*w-ix-2;}
        if(padding!=1){iy=max(0,min(iy,h-1));ix=max(0,min(ix,w-1));}
        if(iy>=0 && iy<h && ix>=0 && ix<w)i=(ch*h+iy)*w+ix;
    }
    if(truth(backward)){if(i>=0)atomicAdd(y+i,memory[x+(o)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);}else memory[y+(o)]=i>=0?memory[x+(i)]:0.f;
}
void extended_conv_kernel(int x,int weights,int dy,int y,int dx,int dw,GExtendedConv g,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int o=blockIdx_x*blockDim_x+threadIdx_x;if(o>=count)return;
    int oc=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
    float value=g.bias?memory[weights+(g.parameters+oc)]:0.f;
    float grad=truth(dy) ?memory[dy+(o)]:0.f;
    if(dy!=0 && truth(g.bias))atomicAdd(dw+g.parameters+oc,grad,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    int first=g.mode==2?oc:0,last=g.mode==2?oc+1:g.ic;
    for(int ic=first;ic<last;++ic)for(int kh=0;kh<g.kh;++kh)for(int kw=0;kw<g.kw;++kw){
        int iy=oy*g.stride+kh-g.pad,ix=g.mode==1?0:ox*g.stride+kw-g.pad;
        if(g.mode==3){iy=oy+g.pad-kh;ix=ox+g.pad-kw;if(iy<0 || ix<0 || iy%g.stride!=0 || ix%g.stride!=0)continue;iy/=g.stride;ix/=g.stride;}
        if(iy<0 || iy>=g.h || ix<0 || ix>=g.w)continue;
        int i=(ic*g.h+iy)*g.w+ix;
        int wi=((oc*(g.mode==2?1:g.ic)+(g.mode==2?0:ic))*g.kh+kh)*g.kw+kw;
        if(truth(dy)){atomicAdd(dx+i,grad*memory[weights+(wi)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);atomicAdd(dw+wi,grad*memory[x+(i)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);}else value+=memory[x+(i)]*memory[weights+(wi)];
    }
    if(!truth(dy))memory[y+(o)]=value;
}
void dropout_kernel(int x,int mask,int y,int n,float p,bool alpha,bool backward,bool active_value,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;
    if(!truth(active_value) || p==0.f){memory[y+(i)]=memory[x+(i)];return;}if(p==1.f){memory[y+(i)]=0;return;}
    float ap=-1.6732632423543772848f*1.0507009873554804934f;
    float scale=truth(alpha) ?rsqrtf((1-p)*(1+p*ap*ap),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x):1.f/(1-p);
    if(truth(backward))memory[y+(i)]=truth(memory[mask+(i)])?memory[x+(i)]*scale:0.f;
    else memory[y+(i)]=truth(alpha) ?scale*(truth(memory[mask+(i)])?memory[x+(i)]:ap)-scale*ap*p:(truth(memory[mask+(i)])?memory[x+(i)]*scale:0.f);
}
void reparameterize_kernel(int mu,int lv,int noise,int dy,int a,int b,int n,bool training,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;float sigma=exp(.5f*max(-20.f,min(20.f,memory[lv+(i)])));
    if(!truth(dy)){memory[a+(i)]=truth(training) ?memory[mu+(i)]+sigma*memory[noise+(i)]:memory[mu+(i)];return;}
    memory[a+(i)]=memory[dy+(i)];memory[b+(i)]=0.f;
    if(noise!=0 && memory[lv+(i)]>=-20.f && memory[lv+(i)]<=20.f){float inv=sigma>1e-12f?1.f/sigma:0.f;memory[b+(i)]=memory[dy+(i)]*.5f*sigma*((memory[noise+(i)]-memory[mu+(i)])*inv);}
}
void batch_norm_kernel(int x,int w,int running,int dy,int y,int dw,GBatchNorm g,float eps,bool training,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int scratch=0;int c=blockIdx_x,N=g.batch*g.spatial;float sum=0;
    bool estimate=truth(dy) || truth(training) || !truth(g.running);
    if(truth(estimate))for(int j=threadIdx_x;j<N;j+=1)sum+=memory[x+((j/g.spatial*g.channels+c)*g.spatial+j%g.spatial)];
    float mean=truth(estimate) ?reduce_sum(sum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)/N:memory[running+(c)];float variance=0;
    if(truth(estimate))for(int j=threadIdx_x;j<N;j+=1){float d=memory[x+((j/g.spatial*g.channels+c)*g.spatial+j%g.spatial)]-mean;variance+=d*d;}
    float var=truth(estimate) ?reduce_sum(variance,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)/N:memory[running+(g.channels+c)],inv=rsqrtf(var+eps,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    float gamma=g.affine?memory[w+(c)]:1.f,beta=g.bias?memory[w+(g.channels+c)]:0.f;
    if(!truth(dy)){for(int j=threadIdx_x;j<N;j+=1){int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;memory[y+(i)]=(memory[x+(i)]-mean)*inv*gamma+beta;}return;}
    float s1=0,s2=0;
    for(int j=threadIdx_x;j<N;j+=1){int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;s1+=memory[dy+(i)];s2+=memory[dy+(i)]*(memory[x+(i)]-mean)*inv;}
    s1=reduce_sum(s1,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);s2=reduce_sum(s2,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
    if(threadIdx_x==0 && truth(g.affine)){memory[dw+(c)]=s2;if(truth(g.bias))memory[dw+(g.channels+c)]=s1;}
    for(int j=threadIdx_x;j<N;j+=1){int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;float xhat=(memory[x+(i)]-mean)*inv;memory[y+(i)]=inv*gamma*(memory[dy+(i)]-s1/N-xhat*s2/N);}
}
void attention_add_bias(int y,int w,int n,int width,int offset,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i<n && offset>=0)memory[y+(i)]+=memory[w+(offset+i%width)];
}
void attention_project(int input_value,int w,int result,GAttention g,int part,bool backward,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;int t=i/g.ed,d=i%g.ed;
    int features=truth(part) && g.cross?g.ked:g.ed;float value=0;
    if(!truth(backward) && !truth(g.cross) && g.qbias>=0)value=memory[w+(g.qbias+part*g.ed+d)];
    for(int f=0;f<features;++f){
        int wi=!truth(g.cross)?(part*g.ed+d)*g.ed+f:(part!=0 ?g.ed*g.ed+f*2*g.ed+(part-1)*g.ed+d:f*g.ed+d);
        value+=memory[input_value+(t*features+f)]*memory[w+(wi)];
    }memory[result+(i)]=value;
}
void attention_probabilities(int q,int k,int p,GAttention g,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int scratch=0;int h=blockIdx_x/g.q,t=blockIdx_x%g.q,width=g.ed/g.heads,base=(h*g.q+t)*g.k;
    float scale=rsqrtf((float)(width),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);float maximum=-1e30f;
    for(int j=threadIdx_x;j<g.k;j+=1){float score=-1e9f;if(!truth(g.causal) || j<=t){score=0;for(int d=0;d<width;++d)score+=memory[q+(t*g.ed+h*width+d)]*memory[k+(j*g.ed+h*width+d)];score*=scale;}memory[p+(base+j)]=score;maximum=max(maximum,score);}
    maximum=reduce_max(maximum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);float sum=0;
    for(int j=threadIdx_x;j<g.k;j+=1){float v=exp(memory[p+(base+j)]-maximum);memory[p+(base+j)]=v;sum+=v;}
    float inv=1.f/(reduce_sum(sum,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)+1e-9f);for(int j=threadIdx_x;j<g.k;j+=1)memory[p+(base+j)]*=inv;
}
void attention_context(int p,int v,int ctx,GAttention g,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;int t=i/g.ed,d=i%g.ed,h=d/(g.ed/g.heads);float sum=0;
    for(int j=0;j<g.k;++j)sum+=memory[p+((h*g.q+t)*g.k+j)]*memory[v+(j*g.ed+d)];memory[ctx+(i)]=sum;
}
void attention_output(int ctx,int w,int dy,int y,GAttention g,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;int t=i/g.ed,d=i%g.ed,offset=g.mainParameters-g.ed*g.ed;float sum=0;
    if(!truth(dy) && g.obias>=0)sum=memory[w+(g.obias+d)];
    for(int f=0;f<g.ed;++f)sum+=truth(dy) ?memory[dy+(t*g.ed+f)]*memory[w+(offset+d*g.ed+f)]:memory[ctx+(t*g.ed+f)]*memory[w+(offset+f*g.ed+d)];memory[y+(i)]=sum;
}
void attention_score_grad(int p,int v,int dc,int ds,GAttention g,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int scratch=0;int h=blockIdx_x/g.q,t=blockIdx_x%g.q,width=g.ed/g.heads,base=(h*g.q+t)*g.k;float dot=0;
    for(int j=threadIdx_x;j<g.k;j+=1){float dp=0;for(int f=0;f<width;++f)dp+=memory[dc+(t*g.ed+h*width+f)]*memory[v+(j*g.ed+h*width+f)];memory[ds+(base+j)]=dp;dot+=dp*memory[p+(base+j)];}
    dot=reduce_sum(dot,scratch,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);for(int j=threadIdx_x;j<g.k;j+=1)memory[ds+(base+j)]=g.causal && j>t?0.f:(memory[ds+(base+j)]-dot)*memory[p+(base+j)];
}
void attention_projected_grad(int q,int k,int dc,int p,int ds,int result,GAttention g,int part,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=n)return;int t=i/g.ed,d=i%g.ed,h=d/(g.ed/g.heads);float sum=0;
    if(part==0)for(int j=0;j<g.k;++j)sum+=memory[ds+((h*g.q+t)*g.k+j)]*memory[k+(j*g.ed+d)];
    else if(part==1)for(int j=0;j<g.q;++j)sum+=memory[ds+((h*g.q+j)*g.k+t)]*memory[q+(j*g.ed+d)];
    else for(int j=0;j<g.q;++j)sum+=memory[p+((h*g.q+j)*g.k+t)]*memory[dc+(j*g.ed+d)];
    memory[result+(i)]=part==2?sum:sum*rsqrtf((float)(g.ed/g.heads),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
}
void attention_weight_grad(int x,int kv,int ctx,int dy,int dq,int dk,int dv,int dw,GAttention g,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int wi=blockIdx_x*blockDim_x+threadIdx_x;if(wi>=count)return;float sum=0;int output_offset=g.mainParameters-g.ed*g.ed;
    if(wi>=output_offset){int r=(wi-output_offset)/g.ed,d=(wi-output_offset)%g.ed;for(int t=0;t<g.q;++t)sum+=memory[ctx+(t*g.ed+r)]*memory[dy+(t*g.ed+d)];}
    else if(!truth(g.cross)){int part=(wi/g.ed)/g.ed,d=(wi/g.ed)%g.ed,f=wi%g.ed;int grad=part==0?dq:(part==1?dk:dv);for(int t=0;t<g.q;++t)sum+=memory[x+(t*g.ed+f)]*memory[grad+(t*g.ed+d)];}
    else if(wi<g.ed*g.ed){int f=wi/g.ed,d=wi%g.ed;for(int t=0;t<g.q;++t)sum+=memory[x+(t*g.ed+f)]*memory[dq+(t*g.ed+d)];}
    else{int off=wi-g.ed*g.ed,f=off/(2*g.ed),col=off%(2*g.ed),d=col%g.ed;int grad=col<g.ed?dk:dv;for(int t=0;t<g.k;++t)sum+=memory[kv+(t*g.ked+f)]*memory[grad+(t*g.ed+d)];}
    memory[dw+(wi)]=sum;
}
void attention_input_grad(int w,int dq,int dk,int dv,int dx,GAttention g,bool second,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    int i=blockIdx_x*blockDim_x+threadIdx_x;if(i>=count)return;int width=truth(second) ?g.ked:g.ed,t=i/width,f=i%width;float sum=0;
    if(!truth(g.cross))for(int d=0;d<g.ed;++d)sum+=memory[dq+(t*g.ed+d)]*memory[w+(d*g.ed+f)]+memory[dk+(t*g.ed+d)]*memory[w+((g.ed+d)*g.ed+f)]+memory[dv+(t*g.ed+d)]*memory[w+((2*g.ed+d)*g.ed+f)];
    else if(!truth(second))for(int d=0;d<g.ed;++d)sum+=memory[dq+(t*g.ed+d)]*memory[w+(f*g.ed+d)];
    else for(int d=0;d<g.ed;++d)sum+=memory[dk+(t*g.ed+d)]*memory[w+(g.ed*g.ed+f*2*g.ed+d)]+memory[dv+(t*g.ed+d)]*memory[w+(g.ed*g.ed+f*2*g.ed+g.ed+d)];
    memory[dx+(i)]=sum;
}
float sigmoid_scalar(float v,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){return 1.f/(1.f+exp(-v));}
bool recurrent_forward_reference(SequenceVector x,SequenceVector* outputs,SequenceLayer* layer,SequenceArena* arena,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){

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

                    SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    
                    sequence_assign(&((*outputs)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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
                        sequence_copy(ht,ht + H,h_prev.values,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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

                    SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    SequenceVector r=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    SequenceVector z=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    SequenceVector n=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                    
                    sequence_assign(&((*outputs)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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
                            memory[r.values+((int)(h))] = sigmoid_scalar(sr,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                            int w_iz = Wih + (int)(H + h) * (int)(I);
                            int w_hz = Whh + (int)(H + h) * (int)(H);
                            float sz = 0.0f;
                            for (int i = 0; i < I; ++i) sz += memory[w_iz+((int)(i))] * memory[xt+((int)(i))];
                            for (int k = 0; k < H; ++k) sz += memory[w_hz+((int)(k))] * memory[h_prev.values+((int)(k))];
                            if(truth(bih)) sz += memory[bih+((int)(H + h))];
                            if(truth(bhh)) sz += memory[bhh+((int)(H + h))];
                            memory[z.values+((int)(h))] = sigmoid_scalar(sz,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

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
                        sequence_copy(ht,ht + H,h_prev.values,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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

                    SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    SequenceVector c_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    
                    sequence_assign(&((*outputs)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

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

                            float si = sequence_dot(wii,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) + sequence_dot(whi,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                            float sf = sequence_dot(wif,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) + sequence_dot(whf,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                            float sg = sequence_dot(wig,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) + sequence_dot(whg,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                            float so = sequence_dot(wio,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) + sequence_dot(who,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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

                            float i_gate = sigmoid_scalar(si,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                            float f_gate = sigmoid_scalar(sf,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                            float g_gate = tanh(sg);
                            float o_gate = sigmoid_scalar(so,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                            float c = f_gate * memory[c_prev.values+((int)(h))] + i_gate * g_gate;
                            memory[c_prev.values+((int)(h))] = c;
                            memory[ht+((int)(h))] = o_gate * tanh(c);
                        }

                        sequence_copy(ht,ht + H,h_prev.values,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    }
                }
                return true;
            
}
bool recurrent_backward_reference(SequenceVector inputs,SequenceVector grad_outputs,SequenceVector* grad_inputs,SequenceLayer* layer,SequenceArena* arena,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
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

                SequenceVector i_gate=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector f_gate=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector g_gate=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector o_gate=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector c_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector h_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector c_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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

                        float si = sequence_dot(wii,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) + sequence_dot(whi,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                        float sf = sequence_dot(wif,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) + sequence_dot(whf,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                        float sg = sequence_dot(wig,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) + sequence_dot(whg,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                        float so = sequence_dot(wio,xt,I,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) + sequence_dot(who,h_prev.values,H,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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

                        float ig = sigmoid_scalar(si,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                        float fg = sigmoid_scalar(sf,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                        float gg = tanh(sg);
                        float og = sigmoid_scalar(so,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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
                    sequence_assign(&((*layer).grad_weights),(*layer).weights_size,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                }
                
                sequence_assign(&((*grad_inputs)),x.count,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                SequenceVector dh_next=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector dc_next=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                for (int t = T - 1; t >= 0; --t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    int hprev = (t > 0) ? (h_state.values+((int)(t - 1) * (int)(H))) : 0;
                    int cprev = (t > 0) ? (c_state.values+((int)(t - 1) * (int)(H))) : 0;

                    SequenceVector dh_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    SequenceVector dc_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

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

                    sequence_swap(&(dh_next),&(dh_prev),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    sequence_swap(&(dc_next),&(dc_prev),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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

                SequenceVector h_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector r_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector z_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector n_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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
                        float r = sigmoid_scalar(sr,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                        int w_iz = Wih + (int)(H + h) * (int)(I);
                        int w_hz = Whh + (int)(H + h) * (int)(H);
                        float sz = 0.0f;
                        for (int i = 0; i < I; ++i) sz += memory[w_iz+((int)(i))] * memory[xt+((int)(i))];
                        for (int k = 0; k < H; ++k) sz += memory[w_hz+((int)(k))] * memory[h_prev.values+((int)(k))];
                        if(truth(bih)) sz += memory[bih+((int)(H + h))];
                        if(truth(bhh)) sz += memory[bhh+((int)(H + h))];
                        float z = sigmoid_scalar(sz,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

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
                    sequence_assign(&((*layer).grad_weights),(*layer).weights_size,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                }
                
                sequence_assign(&((*grad_inputs)),x.count,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                SequenceVector dh_next=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                for (int t = T - 1; t >= 0; --t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    int hprev = (t > 0) ? (h_state.values+((int)(t - 1) * (int)(H))) : 0;
                    SequenceVector dh_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                    SequenceVector da_r=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    SequenceVector da_z=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                    SequenceVector da_n=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

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

                    SequenceVector drh=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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

                    sequence_swap(&(dh_next),&(dh_prev),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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

                SequenceVector h_state=sequence_new(&((*arena)),(int)(T) * (int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector h_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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
                    sequence_assign(&((*layer).grad_weights),(*layer).weights_size,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                }

                
                sequence_assign(&((*grad_inputs)),x.count,0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                SequenceVector dh_next=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);

                for (int t = T - 1; t >= 0; --t) {
                    int xt = (x.values+((int)(t) * (int)(I)));
                    int hprev = (t > 0) ? (h_state.values+((int)(t - 1) * (int)(H))) : 0;
                    SequenceVector dh_prev=sequence_new(&((*arena)),(int)(H),0.0f,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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
                    sequence_swap(&(dh_next),&(dh_prev),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
                }
                return true;
            
}

    default:return false;
    }
}
void recurrent_kernel(int x,int w,int dy,int y,int dw,int workspace,int status,int type,GRecurrent g,bool bias,int weights_size,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    if(blockIdx_x!=0 || threadIdx_x!=0)return;
    SequenceArena arena=(SequenceArena){workspace,0,g.workspace};SequenceVector xv=(SequenceVector){x,(int)(g.steps)*g.input_value};SequenceVector yv=(SequenceVector){y,dy!=0 ?(int)(g.steps)*g.input_value:(int)(g.steps)*g.hidden};SequenceVector gv=(SequenceVector){dy,(int)(g.steps)*g.hidden};
    SequenceLayer layer=(SequenceLayer){type,g.steps,g.input_value,g.hidden,bias,w,weights_size,(SequenceVector){dw,weights_size}};
    
    memory[status]=as_float((int)( truth(dy) ?recurrent_backward_reference(xv,gv,&(yv),&(layer),&(arena),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x):recurrent_forward_reference(xv,&(yv),&(layer),&(arena),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)));
}
float box_iou(int boxes,int a,int b,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    a*=4;b*=4;
    float aa=max(0.f,memory[boxes+(a+2)]-memory[boxes+(a)])*max(0.f,memory[boxes+(a+3)]-memory[boxes+(a+1)]);
    float ab=max(0.f,memory[boxes+(b+2)]-memory[boxes+(b)])*max(0.f,memory[boxes+(b+3)]-memory[boxes+(b+1)]);
    float iw=max(0.f,min(memory[boxes+(a+2)],memory[boxes+(b+2)])-max(memory[boxes+(a)],memory[boxes+(b)]));
    float ih=max(0.f,min(memory[boxes+(a+3)],memory[boxes+(b+3)])-max(memory[boxes+(a+1)],memory[boxes+(b+1)]));
    float intersection=iw*ih,area=aa+ab-intersection;return area>0?intersection/area:0.f;
}
void nms_kernel(int boxes,int scores,int classes,int visited,int output_value,int count,int n,float score_threshold,float iou_threshold,int maximum,bool agnostic,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    if(blockIdx_x!=0 || threadIdx_x!=0)return;memory[count]=as_float((int)(0));
    for(int i=0;i<n;++i){bool valid=finite_value(memory[scores+(i)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x) && memory[scores+(i)]>=score_threshold;
        for(int d=0;d<4;++d)valid=truth(valid) && finite_value(memory[boxes+(i*4+d)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);if(truth(classes))valid=truth(valid) && finite_value(memory[classes+(i)],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);memory[visited+(i)]=truth(valid) ?0.f:1.f;}
    for(int pass=0;pass<n;++pass){
        int best=-1;for(int i=0;i<n;++i)if(!truth(memory[visited+(i)]) && (best<0 || memory[scores+(i)]>memory[scores+(best)]))best=i;
        if(best<0)break;memory[visited+(best)]=1;bool suppress=false;
        int cls=truth(classes) ?(int)(round_away((float)(memory[classes+(best)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)):0;
        for(int j=0;j<as_int(memory[count]);++j){int accepted=(int)(memory[output_value+(j)]);
            bool same=truth(agnostic) || !truth(classes) || cls==(int)(round_away((float)(memory[classes+(accepted)]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x));
            if(same && box_iou(boxes,best,accepted,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x)>iou_threshold){suppress=true;break;}}
        if(truth(suppress))continue;memory[output_value+(as_int(memory[count]))]=(float)(best);memory[count]=as_float(as_int(memory[count])+1);if(maximum>0 && as_int(memory[count])>=maximum)break;
    }
}
void nms_stop_gradient_kernel(int gradient,int count,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
    for(int i=blockIdx_x*blockDim_x+threadIdx_x;i<count;i+=blockDim_x*gridDim_x)memory[gradient+(i)]=0.f;
}
void cuda_unary_kernel(int in_value,int out_value,int n,int op,float alpha,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
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
            memory[out_value+(i)] = x > 20.0f ? x : log1pf(exp(x),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
            break;
        case (int)(7): {
            float sp = x > 20.0f ? x : log1pf(exp(x),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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
void cuda_binary_kernel(int a,int b,int out_value,int n,int op,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
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
void cuda_unary_backward_kernel(int in_value,int grad_out,int grad_in,int n,int op,float alpha,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
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
            float sp = x > 20.0f ? x : log1pf(exp(x),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);
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
void cuda_binary_backward_kernel(int a,int b,int grad_out,int grad_a,int grad_b,int n,int op,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
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
void portable_copy(int source,int dest,int n,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){for(int i=0;i<n;++i)memory[dest+(i)]=memory[source+(i)];}
void portable_gemm(int a,int b,int c,int m,int n,int k,int lda,int ldb,int ldc,bool ta,bool tb,float alpha,float beta,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){
 for(int j=0;j<n;++j)for(int i=0;i<m;++i){float sum=0;for(int f=0;f<k;++f)sum+=memory[a+( truth(ta) ?f+i*lda:i+f*lda)]*memory[b+( truth(tb) ?j+f*ldb:f+j*ldb)];memory[c+(i+j*ldc)]=alpha*sum+beta*memory[c+(i+j*ldc)];}}
void portable_ger(int x,int y,int a,int m,int n,int sx,int sy,int lda,float alpha,__global float* memory,int threadIdx_x,int blockIdx_x,int blockDim_x,int gridDim_x){for(int j=0;j<n;++j)for(int i=0;i<m;++i)memory[a+(i+j*lda)]+=alpha*memory[x+(i*sx)]*memory[y+(j*sy)];}
__kernel void reference(__global float* memory,__global const int* arguments){int threadIdx_x=0,blockIdx_x=0,blockDim_x=1,gridDim_x=1;
 blockDim_x=arguments[1];gridDim_x=arguments[2];
 for(int gid=0;gid<blockDim_x*gridDim_x;++gid){blockIdx_x=gid/blockDim_x;threadIdx_x=gid%blockDim_x;switch(arguments[0]){
case 0: {norm_forward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],as_float(arguments[11]),arguments[12]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 1: {norm_backward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 2: {channel_gradient(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 3: {resize_forward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 4: {nearest_backward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 5: {bilinear_backward(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 6: {causal_mask(arguments[3],arguments[4],arguments[5],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 7: {softmax_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 8: {im2col(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 9: {col2im(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 10: {row_norm_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],as_float(arguments[9]),arguments[10]!=0,arguments[11]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 11: {row_norm_grad(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 12: {GProjection g=(GProjection){arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16],arguments[17],arguments[18],arguments[19],arguments[20],arguments[21],arguments[22]!=0};projection_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],g,arguments[23],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 13: {geglu_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 14: {GResize g=(GResize){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]};bicubic_kernel(arguments[3],arguments[4],arguments[5],g,arguments[13],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 15: {GPool g=(GPool){arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16],arguments[17],arguments[18]!=0,arguments[19]!=0};pool_kernel(arguments[3],arguments[4],arguments[5],arguments[6],g,arguments[20],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 16: {permute_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 17: {spatial_move(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14],arguments[15]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 18: {GExtendedConv g=(GExtendedConv){arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16],arguments[17],arguments[18],arguments[19],arguments[20],arguments[21]!=0};extended_conv_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],g,arguments[22],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 19: {dropout_kernel(arguments[3],arguments[4],arguments[5],arguments[6],as_float(arguments[7]),arguments[8]!=0,arguments[9]!=0,arguments[10]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 20: {reparameterize_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 21: {GBatchNorm g=(GBatchNorm){arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,arguments[14]!=0};batch_norm_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],g,as_float(arguments[15]),arguments[16]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 22: {attention_add_bias(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 23: {GAttention g=(GAttention){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11]!=0,arguments[12]!=0,arguments[13],arguments[14],arguments[15],arguments[16]};attention_project(arguments[3],arguments[4],arguments[5],g,arguments[17],arguments[18]!=0,arguments[19],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 24: {GAttention g=(GAttention){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11]!=0,arguments[12]!=0,arguments[13],arguments[14],arguments[15],arguments[16]};attention_probabilities(arguments[3],arguments[4],arguments[5],g,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 25: {GAttention g=(GAttention){arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11]!=0,arguments[12]!=0,arguments[13],arguments[14],arguments[15],arguments[16]};attention_context(arguments[3],arguments[4],arguments[5],g,arguments[17],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 26: {GAttention g=(GAttention){arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,arguments[14],arguments[15],arguments[16],arguments[17]};attention_output(arguments[3],arguments[4],arguments[5],arguments[6],g,arguments[18],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 27: {GAttention g=(GAttention){arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,arguments[14],arguments[15],arguments[16],arguments[17]};attention_score_grad(arguments[3],arguments[4],arguments[5],arguments[6],g,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 28: {GAttention g=(GAttention){arguments[9],arguments[10],arguments[11],arguments[12],arguments[13],arguments[14]!=0,arguments[15]!=0,arguments[16],arguments[17],arguments[18],arguments[19]};attention_projected_grad(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],g,arguments[20],arguments[21],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 29: {GAttention g=(GAttention){arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16]!=0,arguments[17]!=0,arguments[18],arguments[19],arguments[20],arguments[21]};attention_weight_grad(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],g,arguments[22],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 30: {GAttention g=(GAttention){arguments[8],arguments[9],arguments[10],arguments[11],arguments[12],arguments[13]!=0,arguments[14]!=0,arguments[15],arguments[16],arguments[17],arguments[18]};attention_input_grad(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],g,arguments[19]!=0,arguments[20],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 31: {GRecurrent g=(GRecurrent){arguments[11],arguments[12],arguments[13],arguments[14],arguments[15],arguments[16]};recurrent_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],g,arguments[17]!=0,arguments[18],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 32: {nms_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],as_float(arguments[10]),as_float(arguments[11]),arguments[12],arguments[13]!=0,memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 33: {nms_stop_gradient_kernel(arguments[3],arguments[4],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 34: {cuda_unary_kernel(arguments[3],arguments[4],arguments[5],arguments[6],as_float(arguments[7]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 35: {cuda_binary_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 36: {cuda_unary_backward_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],as_float(arguments[8]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 37: {cuda_binary_backward_kernel(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 38: {portable_copy(arguments[3],arguments[4],arguments[5],memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 39: {portable_gemm(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],arguments[11],arguments[12]!=0,arguments[13]!=0,as_float(arguments[14]),as_float(arguments[15]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
case 40: {portable_ger(arguments[3],arguments[4],arguments[5],arguments[6],arguments[7],arguments[8],arguments[9],arguments[10],as_float(arguments[11]),memory,threadIdx_x,blockIdx_x,blockDim_x,gridDim_x);break;}
default:break;}}}
)MIMIRCL";
#endif
#ifdef ENABLE_VULKAN
inline constexpr std::uint32_t referenceSpirv[]={
119734787u,65536u,524299u,13617u,0u,131089u,1u,393227u,1u,1280527431u,1685353262u,808793134u,0u,196622u,0u,1u,
327695u,5u,4u,1852399981u,0u,393232u,4u,17u,1u,1u,1u,196611u,2u,450u,262149u,4u,
1852399981u,0u,393221u,7u,1970365779u,1701015141u,1852142145u,97u,327686u,7u,0u,1635017060u,0u,327686u,7u,1u,
1684370293u,0u,393222u,7u,2u,1634754915u,2037672291u,0u,393221u,12u,1970365779u,1701015141u,1952671062u,29295u,327686u,12u,
0u,1970037110u,29541u,327686u,12u,1u,1853189987u,116u,983045u,17u,1970365811u,1701015141u,2003136095u,1920234280u,762602357u,1970365779u,
1701015141u,1852142145u,828976481u,758212909u,993079657u,1715155305u,15153u,262149u,14u,1852142177u,97u,196613u,15u,110u,262149u,16u,
1970037110u,101u,983045u,24u,1970365811u,1701015141u,1936941407u,678324073u,1970435187u,1395487843u,1702195557u,1449485166u,1869898597u,828976498u,825321773u,993093947u,
3879270u,196613u,21u,118u,196613u,22u,110u,262149u,23u,1970037110u,101u,1376261u,29u,1970365811u,1701015141u,1635218271u,
1953704048u,1952675186u,1902465837u,1668179317u,1667585637u,762474356u,1764569449u,1933259057u,1668641396u,1699949940u,1852142961u,1700160867u,1919906915u,758212909u,993079657u,0u,
196613u,27u,97u,196613u,28u,98u,524293u,35u,1970365811u,1701015141u,1886348127u,828975225u,993093947u,3879273u,262149u,32u,
1936877926u,116u,262149u,33u,1953718636u,0u,393221u,34u,1886680431u,1985967221u,1702194273u,0u,524293u,41u,1970365811u,1701015141u,
1953457247u,993093928u,1765486953u,15153u,196613u,38u,119u,196613u,39u,120u,196613u,40u,110u,458757u,46u,1969513842u,
1935631715u,1713925493u,828980017u,59u,262149u,44u,1970037110u,101u,262149u,45u,1634886515u,6841204u,458757u,50u,1969513842u,1834968419u,
1713928289u,828980017u,59u,262149u,48u,1970037110u,101u,262149u,49u,1634886515u,6841204u,458757u,55u,1836020833u,1682006889u,828975204u,
993093179u,0u,196613u,53u,29793u,262149u,54u,1970037110u,101u,327685u,61u,1953854068u,828516456u,59u,196613u,60u,
118u,327685u,65u,1953854068u,828975208u,59u,196613u,64u,118u,327685u,69u,1953854068u,828778600u,59u,196613u,68u,
118u,393221u,73u,1853190002u,2002870116u,1713928545u,15153u,196613u,72u,118u,327685u,76u,1920037746u,1713923700u,15153u,196613u,
75u,118u,327685u,79u,828862316u,1713923696u,15153u,196613u,78u,118u,458757u,82u,1768843622u,1985963380u,1702194273u,993093160u,
0u,196613u,81u,120u,851973u,95u,1836216174u,1919903327u,1685217655u,993093928u,1765486953u,828980017u,993093947u,1765486953u,828980017u,993093179u,
3879266u,196613u,85u,120u,262149u,86u,1734960503u,7566440u,196613u,87u,121u,327685u,88u,1836216174u,2053729377u,25701u,
262149u,89u,1937141353u,25716u,327685u,90u,1970238055u,1769168752u,25978u,262149u,91u,1952542835u,7102825u,327685u,92u,1851877475u,
1936483694u,0u,196613u,93u,7565413u,262149u,94u,1935763810u,0u,786437u,106u,1836216174u,1667326559u,1918990187u,828975204u,993093947u,
1765486953u,828980017u,993093947u,1765486953u,15153u,262149u,98u,1952540792u,0u,262149u,99u,1937141353u,25716u,196613u,100u,31076u,
262149u,101u,1734960503u,7566440u,196613u,102u,30820u,327685u,103u,1970238055u,1769168752u,25978u,262149u,104u,1952542835u,7102825u,
327685u,105u,1851877475u,1936483694u,0u,720901u,114u,1851877475u,1600939374u,1684107879u,1953391977u,993093928u,1765486953u,828980017u,993093947u,0u,
196613u,109u,31076u,262149u,110u,1952540792u,0u,196613u,111u,30564u,196613u,112u,25188u,262149u,113u,1952542835u,
7102825u,917509u,127u,1769170290u,1717527930u,1635218031u,1764254834u,828980017u,993093947u,1765486953u,828980017u,993093947u,1765486953u,828521265u,59u,196613u,
117u,120u,196613u,118u,121u,262149u,119u,1853189987u,116u,196613u,120u,26729u,196613u,121u,30569u,196613u,
122u,26735u,196613u,123u,30575u,196613u,124u,26739u,196613u,125u,30579u,327685u,126u,1768712546u,1918985582u,0u,
786437u,137u,1918985582u,1601467237u,1801675106u,1685217655u,993093928u,1765486953u,828980017u,993093947u,1765486953u,15153u,196613u,130u,31076u,196613u,
131u,30820u,262149u,132u,1853189987u,116u,196613u,133u,26729u,196613u,134u,30569u,196613u,135u,26739u,196613u,
136u,30579u,786437u,146u,1768712546u,1918985582u,1667326559u,1918990187u,828975204u,993093947u,1765486953u,828980017u,993093947u,3879273u,196613u,139u,
31076u,196613u,140u,30820u,262149u,141u,1853189987u,116u,196613u,142u,26729u,196613u,143u,30569u,196613u,144u,
26735u,196613u,145u,30575u,524293u,151u,1937072483u,1834970209u,678130529u,1765486953u,828980017u,59u,262149u,148u,1919902579u,29541u,
262149u,149u,1937207154u,0u,262149u,150u,1970040675u,7564909u,720901u,160u,1952870259u,1601724781u,1852990827u,1764256869u,828980017u,993093947u,
1765486953u,828521265u,59u,196613u,154u,120u,196613u,155u,31076u,196613u,156u,121u,327685u,157u,1936291937u,2053731167u,
101u,262149u,158u,1701736041u,114u,327685u,159u,1634168684u,1752459634u,6515053u,851973u,174u,1664249193u,1764256879u,828980017u,993093947u,
1765486953u,828980017u,993093947u,1765486953u,828980017u,993093947u,0u,196613u,163u,120u,196613u,164u,7106403u,262149u,165u,1853189987u,
116u,196613u,166u,104u,196613u,167u,119u,196613u,168u,107u,262149u,169u,1769108595u,25956u,196613u,170u,
6578544u,327685u,171u,1634494820u,1852795252u,0u,196613u,172u,26735u,196613u,173u,30575u,851973u,187u,845967203u,1764257129u,
828980017u,993093947u,1765486953u,828980017u,993093947u,1765486953u,828980017u,993093947u,0u,196613u,176u,7106403u,196613u,177u,30820u,262149u,
178u,1853189987u,116u,196613u,179u,104u,196613u,180u,119u,196613u,181u,107u,262149u,182u,1769108595u,25956u,
196613u,183u,6578544u,327685u,184u,1634494820u,1852795252u,0u,196613u,185u,26735u,196613u,186u,30575u,851973u,199u,
1601662834u,1836216174u,1919249247u,678192494u,1765486953u,828980017u,993093947u,1765486953u,828783409u,993092155u,3879266u,196613u,190u,120u,196613u,191u,
119u,196613u,192u,31076u,196613u,193u,121u,262149u,194u,1952540792u,0u,262149u,195u,1952737655u,104u,196613u,
196u,7565413u,196613u,197u,7564658u,262149u,198u,1935763810u,0u,720901u,208u,1601662834u,1836216174u,1634887519u,828975204u,993093947u,
1765486953u,828980017u,993093947u,0u,196613u,202u,31076u,262149u,203u,1952540792u,0u,196613u,204u,30564u,196613u,205u,
25188u,262149u,206u,1937207154u,0u,262149u,207u,1952737655u,104u,327685u,210u,1869762631u,1952671082u,7237481u,327686u,210u,
0u,1701080941u,0u,262150u,210u,1u,7170404u,327686u,210u,2u,1633906550u,98u,262150u,210u,3u,6578544u,
327686u,210u,4u,1937207154u,0u,262150u,210u,5u,7562345u,393222u,210u,6u,1952540006u,1936028277u,0u,327686u,
210u,7u,1868785011u,25710u,327686u,210u,8u,1717924464u,30825u,458758u,210u,9u,1886680431u,1985967221u,1702194273u,0u,
393222u,210u,10u,1634886000u,1702126957u,29554u,327686u,210u,11u,1935763810u,0u,1835013u,223u,1785688688u,1769235301u,1801416303u,
1701737061u,828975212u,993093947u,1765486953u,828980017u,993093947u,1765486953u,1953708849u,1952675186u,1917863725u,1667590767u,1852795252u,758212909u,1764569449u,828976433u,758212909u,
1764569449u,828976433u,758212909u,1764569449u,828517681u,828980017u,59u,196613u,213u,97u,196613u,214u,98u,196613u,215u,119u,
196613u,216u,31076u,196613u,217u,121u,196613u,218u,24932u,196613u,219u,25188u,196613u,220u,30564u,196613u,
221u,103u,393221u,222u,1936090735u,1667200101u,1953396079u,0u,655365u,230u,1818715495u,1701535605u,1818586738u,993093928u,1765486953u,828980017u,
993093947u,0u,196613u,225u,120u,196613u,226u,31076u,196613u,227u,121u,262149u,228u,1952737655u,104u,262149u,
229u,1853189987u,116u,327685u,233u,1768060259u,828778595u,59u,196613u,232u,120u,262149u,235u,1936020039u,6650473u,262150u,
235u,0u,99u,262150u,235u,1u,26729u,262150u,235u,2u,30569u,262150u,235u,3u,26735u,262150u,
235u,4u,30575u,262150u,235u,5u,26739u,262150u,235u,6u,30579u,1245189u,243u,1969449314u,1600350562u,1852990827u,
1764256869u,828980017u,993093947u,1970435187u,1194161251u,1769170258u,1764582778u,828976433u,758212909u,1764569449u,828976433u,825321773u,993093947u,0u,196613u,238u,
120u,196613u,239u,31076u,196613u,240u,121u,196613u,241u,103u,262149u,242u,1853189987u,116u,262149u,245u,
1869566023u,108u,262150u,245u,0u,99u,262150u,245u,1u,104u,262150u,245u,2u,119u,262150u,245u,
3u,26731u,262150u,245u,4u,30571u,262150u,245u,5u,26739u,262150u,245u,6u,30579u,262150u,245u,
7u,26736u,262150u,245u,8u,30576u,262150u,245u,9u,26735u,262150u,245u,10u,30575u,327686u,245u,
11u,1769496941u,7173485u,327686u,245u,12u,1701539700u,110u,1507333u,254u,1819242352u,1919249247u,678192494u,1765486953u,828980017u,993093947u,
1970435187u,1194161251u,1819242320u,758212909u,1764569449u,828976433u,758212909u,1764569449u,828976433u,758212909u,1764569449u,828517681u,825319981u,993093947u,0u,196613u,
248u,120u,196613u,249u,31076u,196613u,250u,121u,196613u,251u,30820u,196613u,252u,103u,262149u,253u,
1853189987u,116u,786437u,264u,1836213616u,1600484469u,1852990827u,1764256869u,828980017u,993093947u,1765486953u,828980017u,993092155u,0u,196613u,257u,
120u,196613u,258u,121u,327685u,259u,1601467759u,1885431923u,101u,327685u,260u,1935634025u,1684632180u,101u,262149u,261u,
1802396018u,0u,262149u,262u,1853189987u,116u,327685u,263u,1801675106u,1685217655u,0u,1048581u,280u,1952542835u,1600938345u,1702260589u,
993093928u,1765486953u,828980017u,993093947u,1765486953u,828980017u,993093947u,1765486953u,828980017u,993092155u,0u,196613u,267u,120u,196613u,268u,
121u,262149u,269u,1853189987u,116u,196613u,270u,99u,196613u,271u,104u,196613u,272u,119u,196613u,273u,
26735u,196613u,274u,30575u,196613u,275u,26736u,196613u,276u,30576u,196613u,277u,114u,262149u,278u,1684300144u,
6778473u,327685u,279u,1801675106u,1685217655u,0u,393221u,282u,1954039111u,1701080677u,1852785508u,118u,327686u,282u,0u,1701080941u,
0u,262150u,282u,1u,25449u,262150u,282u,2u,25455u,262150u,282u,3u,104u,262150u,282u,4u,
119u,262150u,282u,5u,26731u,262150u,282u,6u,30571u,327686u,282u,7u,1769108595u,25956u,262150u,282u,
8u,6578544u,262150u,282u,9u,26735u,262150u,282u,10u,30575u,393222u,282u,11u,1634886000u,1702126957u,29554u,
327686u,282u,12u,1935763810u,0u,1835013u,293u,1702131813u,1684366446u,1852793695u,1701535606u,1818586738u,993093928u,1765486953u,828980017u,993093947u,
1933259113u,1668641396u,1162292596u,1852142712u,1130653028u,762736239u,1764569449u,828976433u,758212909u,1764569449u,828976433u,758212909u,1764569449u,828976433u,758212909u,993079650u,
3879273u,196613u,285u,120u,262149u,286u,1734960503u,7566440u,196613u,287u,31076u,196613u,288u,121u,196613u,289u,
30820u,196613u,290u,30564u,196613u,291u,103u,262149u,292u,1853189987u,116u,786437u,304u,1886351972u,1601467759u,1852990827u,
1764256869u,828980017u,993093947u,1715155305u,828521265u,993092155u,3879266u,196613u,296u,120u,262149u,297u,1802723693u,0u,196613u,298u,
121u,196613u,299u,110u,196613u,300u,112u,262149u,301u,1752198241u,97u,327685u,302u,1801675106u,1685217655u,0u,
393221u,303u,1769235297u,1985963382u,1702194273u,0u,917509u,315u,1634755954u,1701667186u,1769104756u,1801414010u,1701737061u,828975212u,993093947u,1765486953u,
828980017u,993093947u,1648046441u,15153u,196613u,307u,30061u,196613u,308u,30316u,262149u,309u,1936289646u,101u,196613u,310u,
31076u,196613u,311u,97u,196613u,312u,98u,196613u,313u,110u,327685u,314u,1767993972u,1735289198u,0u,327685u,
317u,1952531015u,1867409507u,28018u,393222u,317u,0u,1851877475u,1936483694u,0u,327686u,317u,1u,1952542835u,7102825u,327686u,
317u,2u,1668571490u,104u,327686u,317u,3u,1768318561u,25966u,327686u,317u,4u,1935763810u,0u,327686u,317u,
5u,1852732786u,6778473u,1441797u,329u,1668571490u,1869504360u,1801416050u,1701737061u,828975212u,993093947u,1765486953u,828980017u,993093947u,1970435187u,1194161251u,
1668571458u,1919897192u,828976493u,758212909u,1647128937u,828517681u,825319981u,993093179u,3879266u,196613u,320u,120u,196613u,321u,119u,262149u,
322u,1852732786u,6778473u,196613u,323u,31076u,196613u,324u,121u,196613u,325u,30564u,196613u,326u,103u,196613u,
327u,7565413u,327685u,328u,1767993972u,1735289198u,0u,720901u,336u,1702130785u,1869182062u,1684103022u,1768054628u,1764258657u,828980017u,993093947u,
1765486953u,15153u,196613u,331u,121u,196613u,332u,119u,196613u,333u,110u,262149u,334u,1952737655u,104u,262149u,
335u,1936090735u,29797u,327685u,338u,1953775943u,1769238117u,28271u,262150u,338u,0u,25701u,262150u,338u,1u,6579563u,
327686u,338u,2u,1684104552u,115u,262150u,338u,3u,113u,262150u,338u,4u,107u,327686u,338u,5u,
1936683619u,115u,327686u,338u,6u,1937072483u,27745u,327686u,338u,7u,1634296433u,115u,327686u,338u,8u,1634296431u,
115u,458758u,338u,9u,1852399981u,1634885968u,1702126957u,29554u,458758u,338u,10u,1886680431u,1985967221u,1702194273u,0u,1638405u,
348u,1702130785u,1869182062u,1919967086u,1667590767u,828975220u,993093947u,1933259113u,1668641396u,1095183732u,1852142708u,1852795252u,758212909u,1764569449u,828976433u,758212909u,
1647128930u,828976433u,758212909u,1764569449u,1765486897u,828521265u,993093947u,0u,327685u,341u,1970302569u,1635147636u,6649196u,196613u,342u,119u,
262149u,343u,1970496882u,29804u,196613u,344u,103u,262149u,345u,1953653104u,0u,327685u,346u,1801675106u,1685217655u,0u,
196613u,347u,110u,1572869u,355u,1702130785u,1869182062u,1919967086u,1650549359u,1953066089u,678651241u,1765486953u,828980017u,1920234299u,762602357u,1953775943u,
1769238117u,1764585071u,828976433u,758212909u,1764569449u,828517681u,758211117u,1764569449u,828976433u,825321773u,59u,196613u,351u,113u,196613u,352u,
107u,196613u,353u,112u,196613u,354u,103u,1507333u,363u,1702130785u,1869182062u,1868783470u,2019914862u,828975220u,993093947u,1933259113u,
1668641396u,1095183732u,1852142708u,1852795252u,758212909u,1764569449u,828976433u,758212909u,1647128930u,828976433u,758212909u,1764569449u,1765486897u,15153u,196613u,358u,
112u,196613u,359u,118u,196613u,360u,7894115u,196613u,361u,103u,196613u,362u,110u,1572869u,372u,1702130785u,
1869182062u,1970233198u,1953853556u,993093928u,1765486953u,828980017u,1920234299u,762602357u,1953775943u,1769238117u,1764585071u,828976433u,758212909u,1764569449u,828517681u,758211117u,
1764569449u,828976433u,825321773u,993093947u,0u,196613u,366u,7894115u,196613u,367u,119u,196613u,368u,31076u,196613u,369u,
121u,196613u,370u,103u,196613u,371u,110u,1572869u,380u,1702130785u,1869182062u,1668505454u,1600483951u,1684107879u,993093928u,1765486953u,
828980017u,1920234299u,762602357u,1953775943u,1769238117u,1764585071u,828976433u,758212909u,1764569449u,828517681u,758211117u,1764569449u,828976433u,825321773u,59u,196613u,
375u,112u,196613u,376u,118u,196613u,377u,25444u,196613u,378u,29540u,196613u,379u,103u,1835013u,392u,
1702130785u,1869182062u,1919967086u,1667590767u,1600415092u,1684107879u,993093928u,1765486953u,828980017u,993093947u,1933259113u,1668641396u,1095183732u,1852142708u,1852795252u,758212909u,
1764569449u,828976433u,758212909u,1647128930u,828976433u,758212909u,1764569449u,1765486897u,828980017u,59u,196613u,383u,113u,196613u,384u,107u,
196613u,385u,25444u,196613u,386u,112u,196613u,387u,29540u,262149u,388u,1970496882u,29804u,196613u,389u,103u,
262149u,390u,1953653104u,0u,196613u,391u,110u,1835013u,405u,1702130785u,1869182062u,1702322030u,1952999273u,1634887519u,828975204u,993093947u,
1765486953u,828980017u,993093947u,1765486953u,1953708849u,1952675186u,1950435117u,1953391988u,762212201u,1764569449u,828976433u,758212909u,1647128937u,828517681u,758212909u,1764569449u,
828976433u,828980017u,59u,196613u,395u,120u,196613u,396u,30315u,196613u,397u,7894115u,196613u,398u,31076u,196613u,
399u,29028u,196613u,400u,27492u,196613u,401u,30308u,196613u,402u,30564u,196613u,403u,103u,262149u,404u,
1853189987u,116u,1703941u,416u,1702130785u,1869182062u,1852399470u,1601467760u,1684107879u,993093928u,1765486953u,828980017u,993093947u,1970435187u,1194161251u,1702130753u,
1869182062u,828976494u,758212909u,1764569449u,828976433u,758211117u,1764569442u,828976433u,758212909u,993079657u,1765486946u,15153u,196613u,408u,119u,196613u,
409u,29028u,196613u,410u,27492u,196613u,411u,30308u,196613u,412u,30820u,196613u,413u,103u,262149u,414u,
1868785011u,25710u,262149u,415u,1853189987u,116u,458757u,419u,1835493747u,1600416111u,1818321779u,1713926753u,15153u,196613u,418u,118u,
393221u,421u,1970365779u,1701015141u,1702453580u,114u,393222u,421u,0u,1701869940u,1970169183u,109u,327686u,421u,1u,1601267059u,
7234924u,393222u,421u,2u,1717530217u,1970561381u,7562610u,458758u,421u,3u,1601467759u,1952540006u,1936028277u,0u,393222u,421u,
4u,1600484213u,1935763810u,0u,327686u,421u,5u,1734960503u,7566440u,458758u,421u,6u,1734960503u,1601401960u,1702521203u,0u,
458758u,421u,7u,1684107879u,1768257375u,1937008743u,0u,3276805u,428u,1969448306u,1852142194u,1868980084u,1918990194u,1701994340u,1701995878u,677733230u,
1970435187u,1395487843u,1702195557u,1449485166u,1869898597u,828976498u,825321773u,1920234299u,762602357u,1970365779u,1701015141u,1952671062u,1764586095u,828976433u,1953708849u,1952675186u,
1902465837u,1668179317u,2036419685u,1764586085u,828976433u,758212909u,1647128937u,828976433u,758212909u,1970435187u,1395487843u,1702195557u,1449485166u,1869898597u,828976498u,825321773u,
1953708849u,1952675186u,1902465837u,1668179317u,1701986661u,1764581742u,828976433u,825321773u,59u,196613u,424u,120u,262149u,425u,1886680431u,7566453u,
262149u,426u,1702453612u,114u,262149u,427u,1852142177u,97u,3735557u,436u,1969448306u,1852142194u,1633836916u,1635216227u,1918854258u,1919247973u,
1701015141u,1920234280u,762602357u,1970365779u,1701015141u,1952671062u,1764586095u,828976433u,1953708849u,1952675186u,1902465837u,1668179317u,1667585637u,762474356u,1764569449u,1933259057u,
1668641396u,1699949940u,1852142961u,1700160867u,1919906915u,758212909u,993079657u,1970435187u,1395487843u,1702195557u,1281713006u,1919252833u,758212909u,1764569449u,828976433u,758211117u,
1764569449u,1953705265u,1952675186u,1902465837u,1668179317u,1667585637u,762474356u,1764569449u,993079601u,1970435187u,1395487843u,1702195557u,1097163630u,1634624882u,758212909u,1764569449u,
3879217u,262149u,431u,1970302569u,29556u,393221u,432u,1684107879u,1953853279u,1937012080u,0u,327685u,433u,1684107879u,1886284127u,7566453u,
262149u,434u,1702453612u,114u,262149u,435u,1852142177u,97u,327685u,438u,1667584583u,1701999221u,29806u,327686u,438u,0u,
1885697139u,115u,393222u,438u,1u,1970302569u,1635147636u,6649196u,327686u,438u,2u,1684302184u,28261u,327686u,438u,3u,
1702125927u,115u,393222u,438u,4u,1634886000u,1702126957u,29554u,393222u,438u,5u,1802661751u,1667330163u,101u,1572869u,452u,
1969448306u,1852142194u,1701535604u,1818586738u,993093928u,1765486953u,828980017u,993093947u,1765486953u,828980017u,1920234299u,762602357u,1667584583u,1701999221u,1764586606u,828976433u,
758212909u,1764569449u,828976433u,828521265u,993093947u,0u,196613u,441u,120u,196613u,442u,119u,196613u,443u,31076u,196613u,
444u,121u,196613u,445u,30564u,327685u,446u,1802661751u,1667330163u,101u,262149u,447u,1952543859u,29557u,262149u,448u,
1701869940u,0u,196613u,449u,103u,262149u,450u,1935763810u,0u,393221u,451u,1734960503u,1601401960u,1702521203u,0u,458757u,
457u,1601728354u,678784873u,1765486953u,828980017u,59u,262149u,454u,1702391650u,115u,196613u,455u,97u,196613u,456u,98u,
917509u,471u,1601400174u,1852990827u,1764256869u,828980017u,993093947u,1765486953u,828980017u,993093947u,1715155302u,828980017u,993092155u,0u,262149u,460u,
1702391650u,115u,262149u,461u,1919902579u,29541u,262149u,462u,1935764579u,7562611u,262149u,463u,1769171318u,6579572u,393221u,464u,
1886680431u,1985967221u,1702194273u,0u,262149u,465u,1853189987u,116u,196613u,466u,110u,393221u,467u,1919902579u,1752457061u,1752393074u,
6581359u,393221u,468u,1601531753u,1701996660u,1819240563u,100u,262149u,469u,1769496941u,7173485u,327685u,470u,1869506401u,1667855475u,0u,
655365u,476u,1601400174u,1886352499u,1634887519u,1852139876u,1701535604u,1818586738u,993093928u,3879273u,327685u,474u,1684107879u,1953391977u,0u,262149u,
475u,1853189987u,116u,720901u,484u,1633973603u,1634628959u,1801419122u,1701737061u,828975212u,993093947u,1765486953u,828783409u,59u,327685u,479u,
1985965673u,1702194273u,0u,327685u,480u,1601467759u,1970037110u,101u,196613u,481u,110u,196613u,482u,28783u,262149u,483u,
1752198241u,97u,720901u,491u,1633973603u,1852400223u,1601794657u,1852990827u,1764256869u,828980017u,993093947u,1765486953u,15153u,196613u,486u,97u,
196613u,487u,98u,327685u,488u,1601467759u,1970037110u,101u,196613u,489u,110u,196613u,490u,28783u,917509u,500u,
1633973603u,1634628959u,1650424178u,2003526497u,1600418401u,1852990827u,1764256869u,828980017u,993093947u,1765486953u,828783409u,59u,327685u,494u,1985965673u,1702194273u,
0u,327685u,495u,1684107879u,1953853279u,0u,262149u,496u,1684107879u,7235935u,196613u,497u,110u,196613u,498u,28783u,
262149u,499u,1752198241u,97u,983045u,509u,1633973603u,1852400223u,1601794657u,1801675106u,1685217655u,1919249247u,678192494u,1765486953u,828980017u,993093947u,
1765486953u,828980017u,59u,196613u,502u,97u,196613u,503u,98u,327685u,504u,1684107879u,1953853279u,0u,262149u,505u,
1684107879u,24927u,262149u,506u,1684107879u,25183u,196613u,507u,110u,196613u,508u,28783u,524293u,514u,1953656688u,1701601889u,
1886348127u,828975225u,993093947u,3879273u,262149u,511u,1920298867u,25955u,262149u,512u,1953719652u,0u,196613u,513u,110u,1048581u,
530u,1953656688u,1701601889u,1835362143u,828975213u,993093947u,1765486953u,828980017u,993093947u,1765486953u,828980017u,993092155u,1715155298u,828783409u,59u,196613u,
517u,97u,196613u,518u,98u,196613u,519u,99u,196613u,520u,109u,196613u,521u,110u,196613u,522u,
107u,196613u,523u,6382700u,196613u,524u,6448236u,196613u,525u,6513772u,196613u,526u,24948u,196613u,527u,25204u,
262149u,528u,1752198241u,97u,262149u,529u,1635018082u,0u,851973u,542u,1953656688u,1701601889u,1919248223u,993093928u,1765486953u,828980017u,
993093947u,1765486953u,828980017u,993093179u,0u,196613u,533u,120u,196613u,534u,121u,196613u,535u,97u,196613u,536u,
109u,196613u,537u,110u,196613u,538u,30835u,196613u,539u,31091u,196613u,540u,6382700u,262149u,541u,1752198241u,
97u,393221u,544u,1701209458u,1668179314u,1634557797u,2649705u,327685u,547u,1701996660u,1682531425u,7888760u,327685u,549u,1668246626u,2019838315u,
30815u,327685u,550u,1668246626u,1835615339u,30815u,327685u,552u,1684632167u,1601005892u,120u,262149u,553u,1970496882u,29804u,196613u,
566u,105u,262149u,576u,1635017028u,0u,327686u,576u,0u,1869440365u,31090u,196613u,578u,0u,196613u,591u,
105u,196613u,610u,99u,196613u,629u,115u,196613u,631u,105u,262149u,718u,1634886515u,6841204u,262149u,719u,
1702060386u,0u,196613u,723u,7173491u,196613u,724u,105u,262149u,745u,1851876717u,0u,262149u,746u,1634886000u,109u,
262149u,748u,1634886000u,109u,327685u,754u,1769103734u,1701015137u,0u,196613u,755u,105u,196613u,765u,100u,196613u,
782u,7761513u,262149u,783u,1634886000u,109u,262149u,785u,1634886000u,109u,262149u,793u,1634886000u,109u,196613u,807u,
105u,262149u,817u,1701080681u,120u,262149u,821u,1851877475u,7103854u,262149u,827u,1970037110u,101u,262149u,837u,1634886000u,
109u,262149u,862u,1634886000u,109u,262149u,884u,1634886515u,6841204u,262149u,885u,1702060386u,0u,196613u,889u,7173491u,
262149u,890u,1685025392u,7627637u,196613u,891u,105u,262149u,901u,1701080681u,120u,196613u,905u,103u,196613u,942u,
12659u,262149u,943u,1634886000u,109u,262149u,945u,1634886000u,109u,196613u,948u,12915u,262149u,949u,1634886000u,109u,
262149u,951u,1634886000u,109u,196613u,954u,105u,262149u,964u,1701080681u,120u,196613u,968u,103u,262149u,1020u,
1634886515u,6841204u,262149u,1021u,1851877475u,7103854u,196613u,1023u,7173491u,262149u,1024u,1685025392u,7627637u,196613u,1025u,115u,
262149u,1035u,1701080681u,120u,262149u,1048u,1634886000u,109u,262149u,1068u,1935763810u,0u,262149u,1069u,1634886000u,109u,
262149u,1071u,1634886000u,109u,262149u,1074u,1835884903u,97u,262149u,1075u,1634886000u,109u,262149u,1077u,1634886000u,109u,
262149u,1084u,1634886000u,109u,262149u,1094u,1634886000u,109u,196613u,1104u,105u,196613u,1116u,99u,196613u,1122u,
104u,196613u,1128u,119u,262149u,1132u,1634886000u,109u,196613u,1160u,26216u,196613u,1169u,26231u,196613u,1178u,
12392u,196613u,1182u,12407u,196613u,1186u,12648u,196613u,1192u,12663u,196613u,1198u,26724u,196613u,1203u,25700u,
262149u,1208u,1702060386u,0u,196613u,1285u,105u,196613u,1297u,99u,196613u,1303u,104u,196613u,1309u,119u,
196613u,1313u,7173491u,196613u,1314u,121u,196613u,1323u,120u,196613u,1368u,105u,196613u,1380u,99u,196613u,
1386u,104u,196613u,1392u,119u,196613u,1396u,26216u,196613u,1405u,26231u,196613u,1414u,12392u,196613u,1418u,
12407u,196613u,1422u,12648u,196613u,1428u,12663u,196613u,1434u,26724u,196613u,1439u,25700u,196613u,1444u,103u,
262149u,1450u,1702060386u,0u,262149u,1472u,1634886000u,109u,262149u,1473u,1634886000u,109u,262149u,1490u,1634886000u,109u,
262149u,1491u,1634886000u,109u,262149u,1508u,1634886000u,109u,262149u,1509u,1634886000u,109u,262149u,1525u,1634886000u,109u,
262149u,1526u,1634886000u,109u,196613u,1528u,105u,262149u,1556u,1634886515u,6841204u,196613u,1557u,7827314u,262149u,1559u,
1702060386u,0u,262149u,1571u,1769496941u,7173485u,196613u,1576u,97u,262149u,1599u,1634886000u,109u,262149u,1601u,1634886000u,
109u,196613u,1604u,7173491u,196613u,1605u,97u,262149u,1631u,1634886000u,109u,262149u,1633u,1634886000u,109u,327685u,
1636u,1969513842u,1869182051u,110u,262149u,1637u,1634886000u,109u,196613u,1642u,97u,262149u,1661u,1634886000u,109u,262149u,
1688u,1634886000u,109u,262149u,1690u,1634886000u,109u,196613u,1693u,97u,196613u,1703u,105u,327685u,1709u,1651470960u,
1818845793u,7959657u,393221u,1720u,1600614252u,1651470960u,1818845793u,7959657u,262149u,1739u,1634886000u,109u,262149u,1766u,1634886000u,109u,
196613u,1789u,105u,196613u,1801u,115u,196613u,1807u,7827314u,196613u,1813u,99u,196613u,1819u,26731u,196613u,
1825u,30571u,196613u,1829u,26729u,196613u,1841u,30569u,196613u,1888u,105u,196613u,1900u,99u,196613u,1906u,
26729u,196613u,1912u,30569u,196613u,1916u,7173491u,196613u,1917u,26731u,196613u,1926u,30571u,196613u,1935u,26745u,
196613u,1943u,30585u,196613u,1975u,121u,196613u,1979u,120u,262149u,2026u,1634886515u,6841204u,262149u,2027u,1702060386u,
0u,196613u,2031u,7173491u,262149u,2032u,1634886000u,109u,196613u,2038u,105u,262149u,2059u,1851876717u,0u,262149u,
2060u,1634886000u,109u,262149u,2067u,1634886000u,109u,262149u,2069u,1634886000u,109u,327685u,2076u,1769103734u,1701015137u,0u,
196613u,2077u,105u,196613u,2087u,100u,196613u,2104u,7761513u,262149u,2105u,1634886000u,109u,262149u,2107u,1634886000u,
109u,262149u,2115u,1634886000u,109u,262149u,2117u,1634886000u,109u,196613u,2123u,105u,196613u,2133u,118u,262149u,
2164u,1634886000u,109u,196613u,2184u,26483u,196613u,2185u,7890803u,196613u,2186u,105u,196613u,2196u,118u,196613u,
2208u,103u,196613u,2246u,97u,262149u,2247u,1634886000u,109u,262149u,2249u,1634886000u,109u,196613u,2255u,98u,
262149u,2256u,1634886000u,109u,262149u,2258u,1634886000u,109u,196613u,2264u,105u,196613u,2274u,103u,262149u,2302u,
1634886000u,109u,196613u,2322u,105u,196613u,2334u,97u,196613u,2335u,98u,196613u,2336u,114u,196613u,2345u,
106u,262149u,2378u,1634886000u,109u,196613u,2388u,111u,262149u,2416u,1634886000u,109u,196613u,2441u,100u,196613u,
2457u,114u,262149u,2473u,1684107879u,0u,262149u,2474u,1634886000u,109u,262149u,2487u,1970037110u,101u,262149u,2494u,
1918989427u,116u,196613u,2500u,6581861u,262149u,2527u,1634886000u,109u,262149u,2541u,1634886000u,109u,196613u,2548u,105u,
196613u,2558u,25705u,262149u,2564u,1634886000u,109u,196613u,2588u,106u,262149u,2595u,1634886000u,109u,262149u,2603u,
1634886000u,109u,262149u,2604u,1634886000u,109u,196613u,2623u,7761513u,262149u,2628u,1634886000u,109u,262149u,2630u,1702060386u,
0u,262149u,2638u,1634886000u,109u,262149u,2652u,1634886000u,109u,262149u,2653u,1634886000u,109u,196613u,2668u,107u,
196613u,2678u,106u,262149u,2685u,1634886000u,109u,262149u,2704u,1634886000u,109u,262149u,2705u,1634886000u,109u,262149u,
2721u,1634886000u,109u,262149u,2722u,1634886000u,109u,262149u,2745u,1702060386u,0u,262149u,2755u,1634886000u,109u,262149u,
2761u,1634886000u,109u,262149u,2778u,1634886000u,109u,262149u,2779u,1634886000u,109u,196613u,2797u,105u,196613u,2807u,
106u,196613u,2817u,26999u,196613u,2826u,26977u,196613u,2833u,26978u,262149u,2840u,1634886000u,109u,262149u,2861u,
1634886000u,109u,262149u,2862u,1634886000u,109u,262149u,2880u,1634886000u,109u,262149u,2881u,1634886000u,109u,262149u,2899u,
1634886000u,109u,262149u,2900u,1634886000u,109u,262149u,2926u,1634886000u,109u,196613u,2937u,111u,196613u,2949u,97u,
196613u,2960u,98u,196613u,2964u,122u,196613u,2970u,26740u,262149u,2983u,1970038119u,0u,262149u,2989u,1634886000u,
109u,196613u,3055u,97u,196613u,3102u,111u,196613u,3114u,99u,196613u,3123u,31087u,196613u,3131u,30831u,
196613u,3136u,31091u,196613u,3149u,30835u,196613u,3162u,31081u,196613u,3166u,30825u,196613u,3170u,7173491u,262149u,
3171u,1836413815u,0u,196613u,3172u,104u,196613u,3181u,119u,262149u,3189u,1734960503u,29800u,262149u,3196u,1634886000u,
109u,262149u,3204u,1634886000u,109u,262149u,3210u,1634886000u,109u,262149u,3253u,1634886000u,109u,262149u,3262u,1634886000u,
109u,262149u,3275u,1634886000u,109u,196613u,3282u,104u,196613u,3290u,119u,262149u,3298u,1734960503u,29800u,262149u,
3305u,1634886000u,109u,262149u,3313u,1634886000u,109u,196613u,3316u,105u,262149u,3354u,1634886000u,109u,262149u,3355u,
1634886000u,109u,196613u,3361u,111u,196613u,3373u,99u,196613u,3382u,31087u,196613u,3390u,30831u,262149u,3395u,
1970037110u,101u,262149u,3405u,1768710518u,100u,262149u,3406u,1953719650u,0u,196613u,3407u,31083u,196613u,3417u,30827u,
196613u,3427u,31081u,196613u,3437u,30825u,196613u,3471u,105u,262149u,3499u,1634886000u,109u,262149u,3532u,1634886000u,
109u,262149u,3548u,1634886000u,109u,262149u,3563u,1634886000u,109u,262149u,3579u,1634886000u,109u,262149u,3580u,1634886000u,
109u,262149u,3585u,1634886000u,109u,262149u,3592u,1684107879u,0u,196613u,3601u,31083u,196613u,3611u,30827u,196613u,
3621u,31081u,196613u,3631u,30825u,262149u,3663u,1634886000u,109u,262149u,3690u,1634886000u,109u,262149u,3691u,1634886000u,
109u,196613u,3698u,111u,196613u,3710u,7368052u,196613u,3712u,105u,196613u,3713u,100u,262149u,3750u,1634886000u,
109u,196613u,3774u,111u,196613u,3786u,26723u,196613u,3792u,31087u,196613u,3798u,30831u,196613u,3802u,105u,
262149u,3803u,1634886000u,109u,196613u,3808u,25449u,196613u,3838u,31081u,196613u,3842u,30825u,262149u,3920u,1634886000u,
109u,262149u,3935u,1634886000u,109u,262149u,3936u,1634886000u,109u,196613u,3957u,111u,196613u,3969u,25455u,196613u,
3977u,31087u,196613u,3985u,30831u,262149u,3990u,1970037110u,101u,262149u,4006u,1684107879u,0u,262149u,4007u,1634886000u,
109u,262149u,4024u,1634886000u,109u,262149u,4037u,1634886000u,109u,262149u,4038u,1634886000u,109u,262149u,4041u,1936877926u,
116u,262149u,4047u,1953718636u,0u,196613u,4060u,25449u,196613u,4070u,26731u,196613u,4080u,30571u,196613u,4090u,
31081u,196613u,4100u,30825u,196613u,4193u,105u,196613u,4205u,26999u,262149u,4234u,1634886000u,109u,262149u,4249u,
1634886000u,109u,262149u,4250u,1634886000u,109u,262149u,4262u,1634886000u,109u,262149u,4263u,1634886000u,109u,262149u,4285u,
1634886000u,109u,196613u,4296u,105u,262149u,4308u,1634886000u,109u,196613u,4336u,28769u,262149u,4338u,1818321779u,101u,
262149u,4339u,1634886000u,109u,262149u,4354u,1634886000u,109u,262149u,4361u,1634886000u,109u,262149u,4372u,1634886000u,109u,
262149u,4393u,1634886000u,109u,262149u,4403u,1634886000u,109u,262149u,4429u,1634886000u,109u,196613u,4447u,105u,262149u,
4459u,1835493747u,97u,262149u,4471u,1634886000u,109u,262149u,4480u,1634886000u,109u,196613u,4543u,7761513u,262149u,4580u,
1634886515u,6841204u,196613u,4581u,99u,196613u,4583u,78u,196613u,4589u,7173491u,327685u,4590u,1769239397u,1702125933u,0u,
262149u,4591u,1634886000u,109u,262149u,4597u,1634886000u,109u,262149u,4604u,1634886000u,109u,262149u,4610u,1634886000u,109u,
196613u,4615u,106u,262149u,4650u,1851876717u,0u,262149u,4651u,1634886000u,109u,262149u,4657u,1634886000u,109u,262149u,
4659u,1634886000u,109u,327685u,4672u,1769103734u,1701015137u,0u,262149u,4673u,1634886000u,109u,196613u,4678u,106u,196613u,
4688u,100u,196613u,4719u,7496054u,262149u,4720u,1634886000u,109u,262149u,4726u,1634886000u,109u,262149u,4728u,1634886000u,
109u,196613u,4744u,7761513u,262149u,4748u,1634886000u,109u,262149u,4750u,1835884903u,97u,262149u,4763u,1635018082u,0u,
262149u,4779u,1634886000u,109u,196613u,4785u,106u,196613u,4795u,105u,196613u,4833u,12659u,196613u,4834u,12915u,
196613u,4835u,106u,196613u,4845u,105u,262149u,4889u,1634886000u,109u,262149u,4891u,1634886000u,109u,262149u,4894u,
1634886000u,109u,262149u,4896u,1634886000u,109u,262149u,4903u,1634886000u,109u,262149u,4915u,1634886000u,109u,196613u,4929u,
106u,196613u,4939u,105u,262149u,4957u,1952540792u,0u,196613u,4994u,105u,196613u,5024u,105u,196613u,5036u,
116u,196613u,5041u,100u,327685u,5046u,1952540006u,1936028277u,0u,262149u,5047u,1634886000u,109u,262149u,5062u,1970037110u,
101u,262149u,5063u,1634886000u,109u,262149u,5069u,1634886000u,109u,196613u,5096u,102u,196613u,5105u,26999u,262149u,
5106u,1634886000u,109u,262149u,5183u,1634886515u,6841204u,196613u,5184u,104u,196613u,5189u,116u,262149u,5194u,1952737655u,
104u,262149u,5200u,1702060386u,0u,262149u,5210u,1818321779u,101u,262149u,5213u,1634886000u,109u,262149u,5215u,1769496941u,
7173485u,196613u,5217u,106u,262149u,5228u,1919902579u,101u,262149u,5229u,1634886000u,109u,196613u,5240u,100u,262149u,
5297u,1634886000u,109u,262149u,5299u,1634886000u,109u,196613u,5302u,7173491u,196613u,5303u,106u,196613u,5314u,118u,
196613u,5337u,7761513u,262149u,5338u,1634886000u,109u,262149u,5340u,1634886000u,109u,196613u,5346u,106u,196613u,5369u,
105u,196613u,5381u,116u,196613u,5386u,100u,196613u,5391u,104u,196613u,5399u,7173491u,196613u,5400u,106u,
196613u,5445u,105u,196613u,5457u,116u,196613u,5462u,100u,262149u,5467u,1936090735u,29797u,196613u,5476u,7173491u,
262149u,5477u,1634886000u,109u,196613u,5497u,102u,262149u,5507u,1634886000u,109u,262149u,5570u,1634886515u,6841204u,196613u,
5571u,104u,196613u,5576u,116u,262149u,5581u,1952737655u,104u,262149u,5587u,1702060386u,0u,196613u,5597u,7630692u,
196613u,5598u,106u,196613u,5609u,28772u,196613u,5610u,102u,262149u,5672u,1634886000u,109u,262149u,5674u,1634886000u,
109u,196613u,5677u,106u,196613u,5724u,105u,196613u,5736u,116u,196613u,5741u,100u,196613u,5746u,104u,
196613u,5754u,7173491u,196613u,5759u,106u,196613u,5804u,106u,196613u,5845u,106u,262149u,5902u,1634886000u,109u,
196613u,5907u,26999u,196613u,5919u,7173491u,393221u,5920u,1886680431u,1868526709u,1702061670u,116u,196613u,5934u,114u,196613u,
5941u,100u,196613u,5948u,116u,262149u,5984u,1634886000u,109u,262149u,5991u,1953653104u,0u,196613u,5999u,100u,
196613u,6007u,102u,262149u,6012u,1684107879u,0u,196613u,6026u,116u,196613u,6071u,102u,196613u,6076u,100u,
196613u,6081u,116u,196613u,6117u,6710895u,196613u,6125u,102u,196613u,6131u,7106403u,196613u,6137u,100u,262149u,
6142u,1684107879u,0u,196613u,6150u,116u,196613u,6190u,105u,262149u,6202u,1952737655u,104u,262149u,6203u,1634886000u,
109u,196613u,6215u,116u,196613u,6219u,102u,196613u,6223u,7173491u,262149u,6224u,1634886000u,109u,196613u,6231u,
100u,262149u,6318u,1634886000u,109u,196613u,6324u,100u,196613u,6360u,100u,196613u,6446u,84u,196613u,6449u,
73u,196613u,6452u,72u,196613u,6476u,87u,262149u,6479u,1634886000u,109u,327685u,6492u,1600484213u,1935763810u,0u,
262149u,6495u,1600678231u,31347u,262149u,6499u,1600677975u,31347u,262149u,6503u,1600678242u,31347u,262149u,6504u,1634886000u,109u,
262149u,6513u,1600677986u,31347u,262149u,6514u,1634886000u,109u,262149u,6523u,1684366702u,0u,196613u,6538u,6842711u,196613u,
6540u,6842455u,196613u,6544u,6842722u,262149u,6545u,1634886000u,109u,196613u,6556u,6842466u,262149u,6557u,1634886000u,109u,
262149u,6568u,1919967080u,30309u,262149u,6570u,1634886000u,109u,262149u,6572u,1634886000u,109u,262149u,6573u,1634886000u,109u,
262149u,6579u,1634886000u,109u,262149u,6581u,1634886000u,109u,262149u,6582u,1634886000u,109u,196613u,6585u,116u,196613u,
6594u,29816u,196613u,6601u,29800u,196613u,6608u,104u,196613u,6617u,6842743u,196613u,6623u,6842487u,196613u,6629u,
7173491u,196613u,6630u,105u,196613u,6654u,107u,262149u,6679u,1634886000u,109u,262149u,6691u,1634886000u,109u,262149u,
6714u,1634886000u,109u,262149u,6716u,1634886000u,109u,262149u,6717u,1634886000u,109u,327685u,6731u,1600484213u,1935763810u,0u,
262149u,6734u,1600678231u,31347u,262149u,6739u,1600677975u,31347u,262149u,6744u,1600678242u,31347u,262149u,6745u,1634886000u,109u,
262149u,6755u,1600677986u,31347u,262149u,6756u,1634886000u,109u,262149u,6766u,1684366702u,0u,196613u,6781u,6842711u,196613u,
6783u,6842455u,196613u,6787u,6842722u,262149u,6788u,1634886000u,109u,196613u,6799u,6842466u,262149u,6800u,1634886000u,109u,
262149u,6811u,1919967080u,30309u,262149u,6813u,1634886000u,109u,262149u,6815u,1634886000u,109u,262149u,6816u,1634886000u,109u,
196613u,6819u,114u,262149u,6821u,1634886000u,109u,262149u,6823u,1634886000u,109u,262149u,6824u,1634886000u,109u,196613u,
6827u,122u,262149u,6829u,1634886000u,109u,262149u,6831u,1634886000u,109u,262149u,6832u,1634886000u,109u,196613u,6835u,
110u,262149u,6837u,1634886000u,109u,262149u,6839u,1634886000u,109u,262149u,6840u,1634886000u,109u,262149u,6846u,1634886000u,
109u,262149u,6848u,1634886000u,109u,262149u,6849u,1634886000u,109u,196613u,6852u,116u,196613u,6861u,29816u,196613u,
6868u,104u,262149u,6877u,1919508343u,0u,262149u,6883u,1919442807u,0u,196613u,6889u,29299u,196613u,6890u,105u,
196613u,6914u,107u,262149u,6939u,1634886000u,109u,262149u,6951u,1634886000u,109u,262149u,6967u,1634886000u,109u,262149u,
6971u,2053726071u,0u,262149u,6979u,2053660535u,0u,196613u,6987u,31347u,196613u,6988u,105u,196613u,7012u,107u,
262149u,7037u,1634886000u,109u,262149u,7051u,1634886000u,109u,262149u,7069u,1634886000u,109u,262149u,7073u,1852399479u,0u,
262149u,7082u,1852333943u,0u,196613u,7091u,28275u,196613u,7092u,105u,196613u,7116u,107u,262149u,7148u,1634886000u,
109u,262149u,7163u,1634886000u,109u,196613u,7187u,29800u,196613u,7194u,104u,196613u,7203u,26746u,196613u,7210u,
26734u,196613u,7217u,28776u,262149u,7241u,1634886000u,109u,262149u,7243u,1634886000u,109u,262149u,7244u,1634886000u,109u,
327685u,7251u,1600484213u,1935763810u,0u,262149u,7254u,1600678231u,31347u,262149u,7259u,1600677975u,31347u,262149u,7264u,1600678242u,
31347u,262149u,7265u,1634886000u,109u,262149u,7275u,1600677986u,31347u,262149u,7276u,1634886000u,109u,262149u,7286u,1684366702u,
0u,196613u,7301u,6842711u,196613u,7303u,6842455u,196613u,7307u,6842722u,262149u,7308u,1634886000u,109u,196613u,7319u,
6842466u,262149u,7320u,1634886000u,109u,262149u,7331u,1919967080u,30309u,262149u,7333u,1634886000u,109u,262149u,7335u,1634886000u,
109u,262149u,7336u,1634886000u,109u,262149u,7339u,1919967075u,30309u,262149u,7341u,1634886000u,109u,262149u,7343u,1634886000u,
109u,262149u,7344u,1634886000u,109u,262149u,7350u,1634886000u,109u,262149u,7352u,1634886000u,109u,262149u,7353u,1634886000u,
109u,196613u,7356u,116u,196613u,7365u,29816u,196613u,7372u,29800u,196613u,7379u,104u,196613u,7388u,6908279u,
196613u,7394u,6711671u,196613u,7402u,6777207u,196613u,7411u,7301495u,196613u,7420u,6908023u,196613u,7426u,6711415u,196613u,
7434u,6776951u,196613u,7443u,7301239u,196613u,7452u,26995u,262149u,7453u,1634886000u,109u,262149u,7455u,1634886000u,109u,
262149u,7457u,1634886000u,109u,262149u,7460u,1634886000u,109u,262149u,7462u,1634886000u,109u,262149u,7465u,1634886000u,109u,
196613u,7469u,26227u,262149u,7470u,1634886000u,109u,262149u,7472u,1634886000u,109u,262149u,7474u,1634886000u,109u,262149u,
7477u,1634886000u,109u,262149u,7479u,1634886000u,109u,262149u,7482u,1634886000u,109u,196613u,7486u,26483u,262149u,7487u,
1634886000u,109u,262149u,7489u,1634886000u,109u,262149u,7491u,1634886000u,109u,262149u,7494u,1634886000u,109u,262149u,7496u,
1634886000u,109u,262149u,7499u,1634886000u,109u,196613u,7503u,28531u,262149u,7504u,1634886000u,109u,262149u,7506u,1634886000u,
109u,262149u,7508u,1634886000u,109u,262149u,7511u,1634886000u,109u,262149u,7513u,1634886000u,109u,262149u,7516u,1634886000u,
109u,262149u,7520u,1634886000u,109u,262149u,7561u,1634886000u,109u,262149u,7602u,1634164585u,25972u,262149u,7603u,1634886000u,
109u,262149u,7606u,1634164582u,25972u,262149u,7607u,1634886000u,109u,262149u,7610u,1634164583u,25972u,262149u,7613u,1634164591u,
25972u,262149u,7614u,1634886000u,109u,196613u,7617u,99u,262149u,7649u,1634886000u,109u,262149u,7651u,1634886000u,109u,
262149u,7652u,1634886000u,109u,196613u,7667u,120u,196613u,7669u,28519u,196613u,7671u,84u,196613u,7674u,73u,
196613u,7677u,72u,327685u,7709u,1600484213u,1935763810u,0u,196613u,7712u,87u,262149u,7715u,1634886000u,109u,262149u,
7722u,1600678231u,31347u,262149u,7727u,1600677975u,31347u,262149u,7732u,1600678242u,31347u,262149u,7733u,1634886000u,109u,262149u,
7743u,1600677986u,31347u,262149u,7744u,1634886000u,109u,262149u,7754u,1684366702u,0u,196613u,7769u,6842711u,196613u,7771u,
6842455u,196613u,7775u,6842722u,262149u,7776u,1634886000u,109u,196613u,7787u,6842466u,262149u,7788u,1634886000u,109u,262149u,
7799u,1634164585u,25972u,262149u,7803u,1634886000u,109u,262149u,7805u,1634886000u,109u,262149u,7806u,1634886000u,109u,262149u,
7809u,1634164582u,25972u,262149u,7813u,1634886000u,109u,262149u,7815u,1634886000u,109u,262149u,7816u,1634886000u,109u,262149u,
7819u,1634164583u,25972u,262149u,7823u,1634886000u,109u,262149u,7825u,1634886000u,109u,262149u,7826u,1634886000u,109u,262149u,
7829u,1634164591u,25972u,262149u,7833u,1634886000u,109u,262149u,7835u,1634886000u,109u,262149u,7836u,1634886000u,109u,262149u,
7839u,1953718115u,6648929u,262149u,7843u,1634886000u,109u,262149u,7845u,1634886000u,109u,262149u,7846u,1634886000u,109u,262149u,
7849u,1953718120u,6648929u,262149u,7853u,1634886000u,109u,262149u,7855u,1634886000u,109u,262149u,7856u,1634886000u,109u,262149u,
7859u,1919967080u,30309u,262149u,7861u,1634886000u,109u,262149u,7863u,1634886000u,109u,262149u,7864u,1634886000u,109u,262149u,
7867u,1919967075u,30309u,262149u,7869u,1634886000u,109u,262149u,7871u,1634886000u,109u,262149u,7872u,1634886000u,109u,196613u,
7875u,116u,196613u,7884u,29816u,196613u,7891u,104u,196613u,7900u,6908279u,196613u,7906u,6711671u,196613u,7914u,
6777207u,196613u,7923u,7301495u,196613u,7932u,6908023u,196613u,7938u,6711415u,196613u,7946u,6776951u,196613u,7955u,7301239u,
196613u,7964u,26995u,262149u,7965u,1634886000u,109u,262149u,7967u,1634886000u,109u,262149u,7969u,1634886000u,109u,262149u,
7972u,1634886000u,109u,262149u,7974u,1634886000u,109u,262149u,7977u,1634886000u,109u,196613u,7981u,26227u,262149u,7982u,
1634886000u,109u,262149u,7984u,1634886000u,109u,262149u,7986u,1634886000u,109u,262149u,7989u,1634886000u,109u,262149u,7991u,
1634886000u,109u,262149u,7994u,1634886000u,109u,196613u,7998u,26483u,262149u,7999u,1634886000u,109u,262149u,8001u,1634886000u,
109u,262149u,8003u,1634886000u,109u,262149u,8006u,1634886000u,109u,262149u,8008u,1634886000u,109u,262149u,8011u,1634886000u,
109u,196613u,8015u,28531u,262149u,8016u,1634886000u,109u,262149u,8018u,1634886000u,109u,262149u,8020u,1634886000u,109u,
262149u,8023u,1634886000u,109u,262149u,8025u,1634886000u,109u,262149u,8028u,1634886000u,109u,262149u,8032u,1634886000u,109u,
262149u,8073u,1634886000u,109u,196613u,8114u,26473u,262149u,8115u,1634886000u,109u,196613u,8118u,26470u,262149u,8119u,
1634886000u,109u,196613u,8122u,26471u,196613u,8125u,26479u,262149u,8126u,1634886000u,109u,196613u,8129u,99u,196613u,
8142u,30312u,262149u,8230u,1634886000u,109u,262149u,8233u,1634886000u,109u,262149u,8236u,1634886000u,109u,262149u,8240u,
1634886000u,109u,262149u,8242u,1634886000u,109u,262149u,8245u,1634886000u,109u,262149u,8248u,1851746404u,7632997u,262149u,8250u,
1634886000u,109u,262149u,8252u,1634886000u,109u,262149u,8253u,1634886000u,109u,262149u,8256u,1851745124u,7632997u,262149u,8258u,
1634886000u,109u,262149u,8260u,1634886000u,109u,262149u,8261u,1634886000u,109u,196613u,8264u,116u,196613u,8274u,29816u,
262149u,8281u,1701998696u,118u,262149u,8296u,1701998691u,118u,262149u,8311u,1885300836u,7759218u,262149u,8313u,1634886000u,109u,
262149u,8315u,1634886000u,109u,262149u,8316u,1634886000u,109u,262149u,8319u,1885299556u,7759218u,262149u,8321u,1634886000u,109u,
262149u,8323u,1634886000u,109u,262149u,8324u,1634886000u,109u,196613u,8327u,104u,196613u,8336u,26740u,196613u,8342u,
26473u,196613u,8349u,26470u,196613u,8356u,26471u,196613u,8363u,26479u,196613u,8370u,99u,262149u,8377u,1752064372u,
25439u,327685u,8380u,1919967080u,1985967717u,0u,327685u,8393u,1919967075u,1985967717u,0u,196613u,8406u,7628900u,262149u,8420u,
1734307684u,6648929u,196613u,8424u,25444u,262149u,8440u,1734306148u,6648929u,262149u,8444u,1734305636u,6648929u,262149u,8448u,1734305380u,
6648929u,196613u,8460u,6906212u,196613u,8467u,6709604u,196613u,8474u,6775140u,196613u,8481u,7299428u,262149u,8488u,1601662834u,
105u,262149u,8490u,1601662834u,102u,262149u,8494u,1601662834u,103u,262149u,8499u,1601662834u,111u,196613u,8504u,26985u,
196613u,8513u,30328u,196613u,8640u,107u,196613u,8649u,28776u,262149u,8787u,1634886000u,109u,196613u,8792u,12386u,
262149u,8894u,1634886000u,109u,262149u,8896u,1634886000u,109u,262149u,8901u,1634886000u,109u,262149u,8903u,1634886000u,109u,
196613u,8911u,120u,196613u,8913u,28519u,196613u,8915u,84u,196613u,8918u,73u,196613u,8921u,72u,327685u,
8953u,1600484213u,1935763810u,0u,196613u,8956u,87u,262149u,8959u,1634886000u,109u,262149u,8966u,1600678231u,31347u,262149u,
8971u,1600677975u,31347u,262149u,8976u,1600678242u,31347u,262149u,8977u,1634886000u,109u,262149u,8987u,1600677986u,31347u,262149u,
8988u,1634886000u,109u,262149u,8998u,1684366702u,0u,196613u,9013u,6842711u,196613u,9015u,6842455u,196613u,9019u,6842722u,
262149u,9020u,1634886000u,109u,196613u,9031u,6842466u,262149u,9032u,1634886000u,109u,262149u,9043u,1953718120u,6648929u,262149u,
9047u,1634886000u,109u,262149u,9049u,1634886000u,109u,262149u,9050u,1634886000u,109u,262149u,9053u,1953718130u,6648929u,262149u,
9057u,1634886000u,109u,262149u,9059u,1634886000u,109u,262149u,9060u,1634886000u,109u,262149u,9063u,1953718138u,6648929u,262149u,
9067u,1634886000u,109u,262149u,9069u,1634886000u,109u,262149u,9070u,1634886000u,109u,262149u,9073u,1953718126u,6648929u,262149u,
9077u,1634886000u,109u,262149u,9079u,1634886000u,109u,262149u,9080u,1634886000u,109u,262149u,9083u,1919967080u,30309u,262149u,
9085u,1634886000u,109u,262149u,9087u,1634886000u,109u,262149u,9088u,1634886000u,109u,196613u,9091u,116u,196613u,9100u,
29816u,196613u,9107u,104u,262149u,9116u,1919508343u,0u,262149u,9122u,1919442807u,0u,196613u,9128u,29299u,196613u,
9129u,105u,196613u,9153u,107u,262149u,9178u,1634886000u,109u,262149u,9190u,1634886000u,109u,196613u,9202u,114u,
262149u,9203u,1634886000u,109u,262149u,9206u,2053726071u,0u,262149u,9214u,2053660535u,0u,196613u,9222u,31347u,196613u,
9223u,105u,196613u,9247u,107u,262149u,9272u,1634886000u,109u,262149u,9286u,1634886000u,109u,196613u,9300u,122u,
262149u,9301u,1634886000u,109u,262149u,9304u,1852399479u,0u,262149u,9313u,1852333943u,0u,196613u,9322u,28275u,196613u,
9323u,105u,196613u,9347u,107u,262149u,9374u,1634886000u,109u,262149u,9389u,1634886000u,109u,196613u,9404u,110u,
196613u,9407u,30312u,196613u,9421u,26740u,196613u,9453u,104u,262149u,9488u,1634886000u,109u,262149u,9491u,1634886000u,
109u,262149u,9494u,1634886000u,109u,262149u,9498u,1634886000u,109u,262149u,9500u,1634886000u,109u,262149u,9503u,1634886000u,
109u,262149u,9506u,1851746404u,7632997u,262149u,9508u,1634886000u,109u,262149u,9510u,1634886000u,109u,262149u,9511u,1634886000u,
109u,196613u,9514u,116u,196613u,9524u,29816u,262149u,9531u,1701998696u,118u,262149u,9546u,1885300836u,7759218u,262149u,
9548u,1634886000u,109u,262149u,9550u,1634886000u,109u,262149u,9551u,1634886000u,109u,262149u,9554u,1918853476u,0u,262149u,
9556u,1634886000u,109u,262149u,9558u,1634886000u,109u,262149u,9559u,1634886000u,109u,262149u,9562u,2053071204u,0u,262149u,
9564u,1634886000u,109u,262149u,9566u,1634886000u,109u,262149u,9567u,1634886000u,109u,262149u,9570u,1851744612u,0u,262149u,
9572u,1634886000u,109u,262149u,9574u,1634886000u,109u,262149u,9575u,1634886000u,109u,196613u,9578u,104u,196613u,9587u,
26740u,196613u,9593u,114u,196613u,9600u,122u,196613u,9607u,110u,196613u,9614u,28776u,196613u,9627u,7628900u,
196613u,9641u,28260u,196613u,9646u,31332u,196613u,9687u,6845028u,262149u,9689u,1634886000u,109u,262149u,9691u,1634886000u,
109u,262149u,9692u,1634886000u,109u,196613u,9695u,104u,262149u,9704u,1601662834u,110u,196613u,9709u,105u,196613u,
9718u,30328u,196613u,9774u,107u,196613u,9783u,28776u,196613u,9796u,114u,262149u,9857u,1634886000u,109u,196613u,
9862u,12386u,196613u,9902u,107u,196613u,9911u,28776u,196613u,9924u,114u,196613u,9935u,29284u,196613u,9973u,
104u,262149u,9982u,1601662834u,114u,262149u,9984u,1601662834u,122u,196613u,9988u,105u,196613u,9997u,30328u,196613u,
10090u,107u,196613u,10099u,28776u,262149u,10199u,1634886000u,109u,196613u,10204u,12386u,262149u,10278u,1634886000u,109u,
262149u,10280u,1634886000u,109u,196613u,10288u,120u,196613u,10290u,28519u,196613u,10292u,84u,196613u,10295u,73u,
196613u,10298u,72u,327685u,10330u,1600484213u,1935763810u,0u,196613u,10333u,87u,262149u,10336u,1634886000u,109u,262149u,
10343u,1600678231u,31347u,262149u,10347u,1600677975u,31347u,262149u,10351u,1600678242u,31347u,262149u,10352u,1634886000u,109u,262149u,
10361u,1600677986u,31347u,262149u,10362u,1634886000u,109u,262149u,10371u,1684366702u,0u,196613u,10386u,6842711u,196613u,10388u,
6842455u,196613u,10392u,6842722u,262149u,10393u,1634886000u,109u,196613u,10404u,6842466u,262149u,10405u,1634886000u,109u,262149u,
10416u,1953718120u,6648929u,262149u,10420u,1634886000u,109u,262149u,10422u,1634886000u,109u,262149u,10423u,1634886000u,109u,262149u,
10426u,1919967080u,30309u,262149u,10428u,1634886000u,109u,262149u,10430u,1634886000u,109u,262149u,10431u,1634886000u,109u,196613u,
10434u,116u,196613u,10443u,29816u,196613u,10450u,29800u,196613u,10457u,104u,196613u,10466u,7173491u,196613u,10467u,
6842743u,196613u,10473u,6842487u,196613u,10479u,105u,196613u,10503u,107u,262149u,10528u,1634886000u,109u,262149u,10540u,
1634886000u,109u,196613u,10560u,104u,262149u,10590u,1634886000u,109u,262149u,10593u,1634886000u,109u,262149u,10596u,1634886000u,
109u,262149u,10600u,1634886000u,109u,262149u,10602u,1634886000u,109u,262149u,10605u,1634886000u,109u,262149u,10608u,1851746404u,
7632997u,262149u,10610u,1634886000u,109u,262149u,10612u,1634886000u,109u,262149u,10613u,1634886000u,109u,196613u,10616u,116u,
196613u,10626u,29816u,262149u,10633u,1701998696u,118u,262149u,10648u,1885300836u,7759218u,262149u,10650u,1634886000u,109u,262149u,
10652u,1634886000u,109u,262149u,10653u,1634886000u,109u,196613u,10656u,104u,196613u,10665u,7626600u,196613u,10676u,7628900u,
262149u,10694u,1701998692u,0u,196613u,10701u,6842743u,196613u,10707u,6842487u,196613u,10713u,105u,196613u,10762u,107u,
196613u,10771u,28776u,262149u,10818u,1634886000u,109u,262149u,10853u,1634886000u,109u,262149u,10855u,1634886000u,109u,262149u,
10874u,1852142177u,97u,196613u,10879u,30328u,196613u,10887u,30329u,196613u,10907u,30311u,262149u,10915u,1702453612u,114u,
262149u,10931u,1634886000u,109u,262149u,10937u,1634886000u,109u,262149u,10939u,1634886000u,109u,262149u,10941u,1634886000u,109u,
262149u,10943u,1634886000u,109u,262149u,10945u,1634886000u,109u,262149u,10952u,1634886000u,109u,262149u,10954u,1634886000u,109u,
262149u,10956u,1634886000u,109u,262149u,10958u,1634886000u,109u,196613u,10972u,24929u,196613u,11001u,25185u,196613u,11030u,
30569u,196613u,11057u,26729u,393221u,11086u,1702129257u,1667593074u,1852795252u,0u,262149u,11090u,1634038369u,0u,196613u,11119u,
105u,262149u,11128u,1768710518u,100u,262149u,11132u,1634886000u,109u,196613u,11146u,100u,262149u,11154u,1634886000u,109u,
262149u,11165u,1634886000u,109u,262149u,11172u,1634886000u,109u,262149u,11177u,1634886000u,109u,262149u,11185u,1634886000u,109u,
262149u,11193u,1634886000u,109u,262149u,11200u,1936941424u,0u,262149u,11209u,1953719650u,0u,196613u,11210u,105u,262149u,
11222u,1634886000u,109u,327685u,11261u,1886418291u,1936942450u,0u,196613u,11262u,7564387u,262149u,11263u,1634886000u,109u,262149u,
11274u,1634886000u,109u,196613u,11279u,106u,327685u,11291u,1701012321u,1684370544u,0u,262149u,11298u,1701667187u,0u,262149u,
11299u,1634886000u,109u,262149u,11305u,1634886000u,109u,262149u,11319u,1634886000u,109u,262149u,11327u,1634886000u,109u,262149u,
11329u,1634886000u,109u,262149u,11331u,1634886000u,109u,262149u,11342u,1634886000u,109u,196613u,11381u,105u,196613u,11404u,
105u,196613u,11416u,120u,196613u,11477u,115u,196613u,11491u,99u,196613u,11492u,13176u,262149u,11526u,1634886000u,
109u,196613u,11531u,28787u,262149u,11541u,1634886000u,109u,196613u,11553u,29544u,196613u,11566u,29544u,196613u,11587u,
105u,196613u,11654u,100u,196613u,11691u,105u,196613u,11703u,120u,196613u,11709u,28519u,196613u,11715u,100u,
196613u,11738u,115u,196613u,11749u,116u,196613u,11757u,115u,196613u,11772u,99u,196613u,11773u,12920u,196613u,
11777u,13176u,196613u,11781u,117u,196613u,11788u,116u,262149u,11791u,1751344499u,50u,196613u,11796u,30052u,196613u,
11819u,28787u,262149u,11829u,1634886000u,109u,196613u,11832u,116u,196613u,11835u,115u,196613u,11884u,105u,196613u,
11896u,30305u,196613u,11902u,30306u,196613u,11908u,28519u,262149u,11914u,1936737643u,0u,196613u,12006u,105u,196613u,
12026u,106u,196613u,12035u,105u,196613u,12044u,7173491u,196613u,12045u,102u,262149u,12055u,1634886000u,109u,262149u,
12077u,1634886000u,109u,196613u,12130u,106u,196613u,12139u,105u,262149u,12181u,1634885968u,29549u,393222u,12181u,0u,
1969713761u,1953391981u,115u,196613u,12183u,0u,196613u,12189u,6580583u,262149u,12257u,1634886000u,109u,262149u,12260u,1634886000u,
109u,262149u,12263u,1634886000u,109u,262149u,12266u,1634886000u,109u,262149u,12269u,1634886000u,109u,262149u,12272u,1634886000u,
109u,262149u,12275u,1634886000u,109u,262149u,12278u,1634886000u,109u,262149u,12281u,1634886000u,109u,262149u,12282u,1634886000u,
109u,262149u,12285u,1634886000u,109u,262149u,12288u,1634886000u,109u,262149u,12291u,1634886000u,109u,262149u,12294u,1634886000u,
109u,262149u,12297u,1634886000u,109u,262149u,12300u,1634886000u,109u,262149u,12303u,1634886000u,109u,262149u,12306u,1634886000u,
109u,262149u,12311u,1634886000u,109u,262149u,12314u,1634886000u,109u,262149u,12317u,1634886000u,109u,262149u,12320u,1634886000u,
109u,262149u,12323u,1634886000u,109u,262149u,12331u,1634886000u,109u,262149u,12334u,1634886000u,109u,262149u,12337u,1634886000u,
109u,262149u,12340u,1634886000u,109u,262149u,12343u,1634886000u,109u,262149u,12346u,1634886000u,109u,262149u,12349u,1634886000u,
109u,262149u,12352u,1634886000u,109u,262149u,12355u,1634886000u,109u,262149u,12358u,1634886000u,109u,262149u,12361u,1634886000u,
109u,262149u,12364u,1634886000u,109u,262149u,12367u,1634886000u,109u,262149u,12370u,1634886000u,109u,262149u,12373u,1634886000u,
109u,262149u,12376u,1634886000u,109u,262149u,12379u,1634886000u,109u,262149u,12384u,1634886000u,109u,262149u,12387u,1634886000u,
109u,262149u,12390u,1634886000u,109u,262149u,12393u,1634886000u,109u,262149u,12396u,1634886000u,109u,262149u,12399u,1634886000u,
109u,262149u,12402u,1634886000u,109u,262149u,12407u,1634886000u,109u,262149u,12410u,1634886000u,109u,262149u,12413u,1634886000u,
109u,262149u,12421u,1634886000u,109u,262149u,12424u,1634886000u,109u,262149u,12427u,1634886000u,109u,262149u,12430u,1634886000u,
109u,262149u,12433u,1634886000u,109u,262149u,12436u,1634886000u,109u,262149u,12440u,1634886000u,109u,262149u,12443u,1634886000u,
109u,262149u,12446u,1634886000u,109u,262149u,12449u,1634886000u,109u,262149u,12452u,1634886000u,109u,262149u,12455u,1634886000u,
109u,262149u,12458u,1634886000u,109u,262149u,12461u,1634886000u,109u,262149u,12464u,1634886000u,109u,262149u,12467u,1634886000u,
109u,262149u,12470u,1634886000u,109u,262149u,12475u,1634886000u,109u,262149u,12478u,1634886000u,109u,262149u,12481u,1634886000u,
109u,262149u,12484u,1634886000u,109u,262149u,12487u,1634886000u,109u,262149u,12490u,1634886000u,109u,262149u,12493u,1634886000u,
109u,262149u,12496u,1634886000u,109u,262149u,12499u,1634886000u,109u,262149u,12502u,1634886000u,109u,262149u,12505u,1634886000u,
109u,262149u,12519u,1634886000u,109u,262149u,12522u,1634886000u,109u,262149u,12525u,1634886000u,109u,262149u,12528u,1634886000u,
109u,262149u,12531u,1634886000u,109u,262149u,12534u,1634886000u,109u,262149u,12537u,1634886000u,109u,262149u,12538u,1634886000u,
109u,262149u,12539u,1634886000u,109u,262149u,12542u,1634886000u,109u,262149u,12545u,1634886000u,109u,262149u,12548u,1634886000u,
109u,262149u,12551u,1634886000u,109u,262149u,12554u,1634886000u,109u,262149u,12557u,1634886000u,109u,196613u,12562u,103u,
262149u,12599u,1634886000u,109u,262149u,12602u,1634886000u,109u,262149u,12605u,1634886000u,109u,262149u,12608u,1634886000u,109u,
262149u,12611u,1634886000u,109u,262149u,12614u,1634886000u,109u,262149u,12617u,1634886000u,109u,262149u,12620u,1634886000u,109u,
262149u,12623u,1634886000u,109u,262149u,12625u,1634886000u,109u,262149u,12630u,1634886000u,109u,262149u,12633u,1634886000u,109u,
262149u,12636u,1634886000u,109u,262149u,12639u,1634886000u,109u,262149u,12642u,1634886000u,109u,196613u,12647u,103u,262149u,
12663u,1634886000u,109u,262149u,12666u,1634886000u,109u,262149u,12669u,1634886000u,109u,262149u,12672u,1634886000u,109u,262149u,
12674u,1634886000u,109u,196613u,12679u,103u,262149u,12709u,1634886000u,109u,262149u,12712u,1634886000u,109u,262149u,12715u,
1634886000u,109u,262149u,12718u,1634886000u,109u,262149u,12721u,1634886000u,109u,262149u,12723u,1634886000u,109u,262149u,12731u,
1634886000u,109u,262149u,12734u,1634886000u,109u,262149u,12737u,1634886000u,109u,262149u,12740u,1634886000u,109u,262149u,12743u,
1634886000u,109u,262149u,12746u,1634886000u,109u,262149u,12749u,1634886000u,109u,262149u,12755u,1634886000u,109u,262149u,12758u,
1634886000u,109u,262149u,12761u,1634886000u,109u,262149u,12764u,1634886000u,109u,262149u,12767u,1634886000u,109u,262149u,12770u,
1634886000u,109u,262149u,12773u,1634886000u,109u,262149u,12776u,1634886000u,109u,262149u,12779u,1634886000u,109u,262149u,12782u,
1634886000u,109u,262149u,12785u,1634886000u,109u,262149u,12788u,1634886000u,109u,262149u,12791u,1634886000u,109u,196613u,12794u,
103u,262149u,12823u,1634886000u,109u,262149u,12826u,1634886000u,109u,262149u,12829u,1634886000u,109u,262149u,12832u,1634886000u,
109u,262149u,12835u,1634886000u,109u,262149u,12838u,1634886000u,109u,262149u,12841u,1634886000u,109u,262149u,12843u,1634886000u,
109u,262149u,12860u,1634886000u,109u,262149u,12863u,1634886000u,109u,262149u,12866u,1634886000u,109u,262149u,12869u,1634886000u,
109u,262149u,12872u,1634886000u,109u,262149u,12873u,1634886000u,109u,262149u,12874u,1634886000u,109u,262149u,12875u,1634886000u,
109u,262149u,12881u,1634886000u,109u,262149u,12884u,1634886000u,109u,262149u,12887u,1634886000u,109u,262149u,12890u,1634886000u,
109u,262149u,12893u,1634886000u,109u,262149u,12896u,1634886000u,109u,262149u,12899u,1634886000u,109u,262149u,12902u,1634886000u,
109u,196613u,12905u,103u,262149u,12928u,1634886000u,109u,262149u,12931u,1634886000u,109u,262149u,12934u,1634886000u,109u,
262149u,12937u,1634886000u,109u,262149u,12940u,1634886000u,109u,262149u,12943u,1634886000u,109u,262149u,12946u,1634886000u,109u,
262149u,12948u,1634886000u,109u,262149u,12949u,1634886000u,109u,262149u,12952u,1634886000u,109u,262149u,12955u,1634886000u,109u,
262149u,12958u,1634886000u,109u,262149u,12961u,1634886000u,109u,262149u,12964u,1634886000u,109u,196613u,12969u,103u,262149u,
12998u,1634886000u,109u,262149u,13001u,1634886000u,109u,262149u,13004u,1634886000u,109u,262149u,13007u,1634886000u,109u,262149u,
13009u,1634886000u,109u,262149u,13012u,1634886000u,109u,262149u,13013u,1634886000u,109u,196613u,13018u,103u,262149u,13044u,
1634886000u,109u,262149u,13047u,1634886000u,109u,262149u,13050u,1634886000u,109u,262149u,13053u,1634886000u,109u,196613u,13057u,
103u,262149u,13083u,1634886000u,109u,262149u,13086u,1634886000u,109u,262149u,13089u,1634886000u,109u,262149u,13092u,1634886000u,
109u,262149u,13094u,1634886000u,109u,196613u,13099u,103u,262149u,13125u,1634886000u,109u,262149u,13128u,1634886000u,109u,
262149u,13131u,1634886000u,109u,262149u,13134u,1634886000u,109u,262149u,13137u,1634886000u,109u,262149u,13139u,1634886000u,109u,
196613u,13144u,103u,262149u,13170u,1634886000u,109u,262149u,13173u,1634886000u,109u,262149u,13176u,1634886000u,109u,262149u,
13179u,1634886000u,109u,262149u,13182u,1634886000u,109u,196613u,13186u,103u,262149u,13212u,1634886000u,109u,262149u,13215u,
1634886000u,109u,262149u,13218u,1634886000u,109u,262149u,13221u,1634886000u,109u,262149u,13224u,1634886000u,109u,262149u,13227u,
1634886000u,109u,262149u,13230u,1634886000u,109u,262149u,13232u,1634886000u,109u,262149u,13235u,1634886000u,109u,196613u,13240u,
103u,262149u,13266u,1634886000u,109u,262149u,13269u,1634886000u,109u,262149u,13272u,1634886000u,109u,262149u,13275u,1634886000u,
109u,262149u,13278u,1634886000u,109u,262149u,13281u,1634886000u,109u,262149u,13284u,1634886000u,109u,262149u,13287u,1634886000u,
109u,262149u,13290u,1634886000u,109u,262149u,13292u,1634886000u,109u,196613u,13297u,103u,262149u,13326u,1634886000u,109u,
262149u,13329u,1634886000u,109u,262149u,13332u,1634886000u,109u,262149u,13335u,1634886000u,109u,262149u,13338u,1634886000u,109u,
262149u,13341u,1634886000u,109u,262149u,13343u,1634886000u,109u,262149u,13344u,1634886000u,109u,196613u,13349u,103u,262149u,
13366u,1634886000u,109u,262149u,13369u,1634886000u,109u,262149u,13372u,1634886000u,109u,262149u,13375u,1634886000u,109u,262149u,
13378u,1634886000u,109u,262149u,13381u,1634886000u,109u,262149u,13384u,1634886000u,109u,262149u,13387u,1634886000u,109u,262149u,
13390u,1634886000u,109u,262149u,13392u,1634886000u,109u,262149u,13393u,1634886000u,109u,262149u,13407u,1634886000u,109u,262149u,
13410u,1634886000u,109u,262149u,13413u,1634886000u,109u,262149u,13416u,1634886000u,109u,262149u,13419u,1634886000u,109u,262149u,
13422u,1634886000u,109u,262149u,13425u,1634886000u,109u,262149u,13428u,1634886000u,109u,262149u,13429u,1634886000u,109u,262149u,
13430u,1634886000u,109u,262149u,13433u,1634886000u,109u,262149u,13436u,1634886000u,109u,262149u,13439u,1634886000u,109u,262149u,
13447u,1634886000u,109u,262149u,13450u,1634886000u,109u,262149u,13453u,1634886000u,109u,262149u,13456u,1634886000u,109u,262149u,
13459u,1634886000u,109u,262149u,13462u,1634886000u,109u,262149u,13465u,1634886000u,109u,262149u,13468u,1634886000u,109u,262149u,
13471u,1634886000u,109u,262149u,13474u,1634886000u,109u,262149u,13482u,1634886000u,109u,262149u,13485u,1634886000u,109u,262149u,
13488u,1634886000u,109u,262149u,13491u,1634886000u,109u,262149u,13494u,1634886000u,109u,262149u,13497u,1634886000u,109u,262149u,
13500u,1634886000u,109u,262149u,13503u,1634886000u,109u,262149u,13506u,1634886000u,109u,262149u,13509u,1634886000u,109u,262149u,
13512u,1634886000u,109u,262149u,13515u,1634886000u,109u,262149u,13518u,1634886000u,109u,262149u,13523u,1634886000u,109u,262149u,
13526u,1634886000u,109u,262149u,13529u,1634886000u,109u,262149u,13546u,1634886000u,109u,262149u,13549u,1634886000u,109u,262149u,
13552u,1634886000u,109u,262149u,13555u,1634886000u,109u,262149u,13558u,1634886000u,109u,262149u,13561u,1634886000u,109u,262149u,
13564u,1634886000u,109u,262149u,13567u,1634886000u,109u,262149u,13570u,1634886000u,109u,262149u,13573u,1634886000u,109u,262149u,
13574u,1634886000u,109u,262149u,13575u,1634886000u,109u,262149u,13576u,1634886000u,109u,262149u,13582u,1634886000u,109u,262149u,
13585u,1634886000u,109u,262149u,13588u,1634886000u,109u,262149u,13591u,1634886000u,109u,262149u,13594u,1634886000u,109u,262149u,
13597u,1634886000u,109u,262149u,13600u,1634886000u,109u,262149u,13603u,1634886000u,109u,262149u,13606u,1634886000u,109u,262215u,
575u,6u,4u,196679u,576u,3u,327752u,576u,0u,35u,0u,262215u,578u,33u,0u,262215u,
578u,34u,0u,262215u,12180u,6u,4u,196679u,12181u,3u,262216u,12181u,0u,24u,327752u,12181u,
0u,35u,0u,196679u,12183u,24u,262215u,12183u,33u,1u,262215u,12183u,34u,0u,262215u,13616u,
11u,25u,131091u,2u,196641u,3u,2u,262165u,6u,32u,1u,327710u,7u,6u,6u,6u,
262176u,8u,7u,7u,262176u,9u,7u,6u,196630u,10u,32u,262176u,11u,7u,10u,262174u,
12u,6u,6u,393249u,13u,12u,8u,9u,11u,262176u,19u,7u,12u,393249u,20u,2u,
19u,9u,11u,327713u,26u,2u,19u,19u,393249u,31u,2u,9u,9u,9u,393249u,37u,
10u,9u,9u,9u,327713u,43u,10u,11u,9u,327713u,52u,2u,9u,11u,131092u,57u,
262176u,58u,7u,57u,262177u,59u,57u,58u,262177u,63u,57u,9u,262177u,67u,57u,11u,
262177u,71u,10u,11u,852001u,84u,2u,9u,9u,9u,9u,9u,9u,9u,9u,11u,
58u,720929u,97u,2u,9u,9u,9u,9u,9u,9u,9u,9u,524321u,108u,2u,9u,
9u,9u,9u,9u,852001u,116u,2u,9u,9u,9u,9u,9u,9u,9u,9u,9u,
58u,655393u,129u,2u,9u,9u,9u,9u,9u,9u,9u,589857u,153u,2u,9u,9u,
9u,9u,9u,58u,917537u,162u,2u,9u,9u,9u,9u,9u,9u,9u,9u,9u,
9u,9u,786465u,189u,2u,9u,9u,9u,9u,9u,9u,11u,58u,58u,589857u,201u,
2u,9u,9u,9u,9u,9u,9u,917534u,210u,6u,6u,6u,6u,6u,6u,6u,
6u,6u,6u,6u,57u,262176u,211u,7u,210u,852001u,212u,2u,9u,9u,9u,9u,
9u,9u,9u,9u,211u,9u,589854u,235u,6u,6u,6u,6u,6u,6u,6u,262176u,
236u,7u,235u,524321u,237u,2u,9u,9u,9u,236u,9u,983070u,245u,6u,6u,6u,
6u,6u,6u,6u,6u,6u,6u,6u,57u,57u,262176u,246u,7u,245u,589857u,247u,
2u,9u,9u,9u,9u,246u,9u,655393u,256u,2u,9u,9u,9u,9u,9u,9u,
58u,1048609u,266u,2u,9u,9u,9u,9u,9u,9u,9u,9u,9u,9u,9u,9u,
58u,983070u,282u,6u,6u,6u,6u,6u,6u,6u,6u,6u,6u,6u,6u,57u,
262176u,283u,7u,282u,720929u,284u,2u,9u,9u,9u,9u,9u,9u,283u,9u,720929u,
295u,2u,9u,9u,9u,9u,11u,58u,58u,58u,720929u,306u,2u,9u,9u,9u,
9u,9u,9u,9u,58u,524318u,317u,6u,6u,6u,57u,57u,57u,262176u,318u,7u,
317u,786465u,319u,2u,9u,9u,9u,9u,9u,9u,318u,11u,58u,851998u,338u,6u,
6u,6u,6u,6u,57u,57u,6u,6u,6u,6u,262176u,339u,7u,338u,655393u,340u,
2u,9u,9u,9u,339u,9u,58u,9u,458785u,350u,2u,9u,9u,9u,339u,524321u,
357u,2u,9u,9u,9u,339u,9u,589857u,365u,2u,9u,9u,9u,9u,339u,9u,
524321u,374u,2u,9u,9u,9u,9u,339u,786465u,382u,2u,9u,9u,9u,9u,9u,
9u,339u,9u,9u,852001u,394u,2u,9u,9u,9u,9u,9u,9u,9u,9u,339u,
9u,720929u,407u,2u,9u,9u,9u,9u,9u,339u,58u,9u,655390u,421u,6u,6u,
6u,6u,57u,6u,6u,12u,262176u,422u,7u,421u,458785u,423u,57u,19u,19u,422u,
8u,524321u,430u,57u,19u,19u,19u,422u,8u,524318u,438u,6u,6u,6u,6u,6u,
6u,262176u,439u,7u,438u,917537u,440u,2u,9u,9u,9u,9u,9u,9u,9u,9u,
439u,58u,9u,917537u,459u,2u,9u,9u,9u,9u,9u,9u,9u,11u,11u,9u,
58u,327713u,473u,2u,9u,9u,524321u,478u,2u,9u,9u,9u,9u,11u,589857u,493u,
2u,9u,9u,9u,9u,9u,11u,1048609u,516u,2u,9u,9u,9u,9u,9u,9u,
9u,9u,9u,58u,58u,11u,11u,786465u,532u,2u,9u,9u,9u,9u,9u,9u,
9u,9u,11u,262176u,546u,6u,6u,262203u,546u,547u,6u,262187u,6u,548u,0u,262203u,
546u,549u,6u,262203u,546u,550u,6u,262187u,6u,551u,1u,262203u,546u,552u,6u,196637u,
575u,10u,196638u,576u,575u,262176u,577u,2u,576u,262203u,577u,578u,2u,262176u,584u,2u,
10u,262187u,10u,630u,0u,262187u,10u,687u,1056964608u,262187u,10u,701u,1065353216u,262187u,10u,1554u,
3463342888u,262165u,1572u,32u,0u,262187u,1572u,1573u,2139095040u,262187u,6u,2395u,9u,262187u,6u,2404u,
3u,262187u,6u,2409u,8u,262187u,6u,2490u,2u,262187u,6u,2510u,5u,262187u,6u,2624u,
6u,262187u,6u,2750u,7u,262187u,6u,2754u,11u,262187u,10u,2971u,1061962282u,262187u,10u,2973u,
1027024659u,262187u,10u,3043u,1040801103u,262187u,10u,3056u,3208642560u,262187u,10u,3062u,1073741824u,262187u,10u,3067u,
1077936128u,262187u,10u,3083u,1084227584u,262187u,10u,3089u,1090519040u,262187u,10u,3095u,1082130432u,262187u,6u,3118u,
4u,262187u,6u,3173u,4294967295u,262187u,6u,3377u,10u,262187u,6u,3472u,12u,262187u,10u,4337u,
3219196262u,262187u,10u,4460u,3248488448u,262187u,10u,4461u,1101004800u,262187u,10u,4545u,730643660u,262187u,10u,5216u,
4048155338u,262187u,10u,5343u,814313567u,196650u,57u,6465u,262187u,6u,6488u,65u,196649u,57u,6723u,262187u,
6u,6727u,64u,262187u,10u,11556u,1086324736u,262187u,10u,11853u,3225419776u,262187u,10u,11858u,1042983595u,196637u,
12180u,6u,196638u,12181u,12180u,262176u,12182u,2u,12181u,262203u,12182u,12183u,2u,262176u,12184u,2u,
6u,262187u,6u,12439u,13u,262187u,6u,12569u,14u,262187u,6u,12572u,15u,262187u,6u,12575u,
16u,262187u,6u,12578u,17u,262187u,6u,12581u,18u,262187u,6u,12584u,19u,262187u,6u,12587u,
20u,262187u,6u,12590u,21u,262187u,6u,12593u,22u,262187u,6u,12598u,23u,262167u,13614u,1572u,
3u,262187u,1572u,13615u,1u,393260u,13614u,13616u,13615u,13615u,13615u,327734u,2u,4u,0u,3u,
131320u,5u,196670u,547u,548u,196670u,549u,548u,196670u,550u,551u,196670u,552u,551u,262201u,2u,
13613u,544u,65789u,65592u,327734u,12u,17u,0u,13u,196663u,8u,14u,196663u,9u,15u,196663u,
11u,16u,131320u,18u,262203u,19u,553u,7u,262203u,9u,566u,7u,327745u,9u,554u,14u,
548u,262205u,6u,555u,554u,327745u,9u,556u,14u,551u,262205u,6u,557u,556u,327808u,6u,
558u,555u,557u,262205u,6u,559u,15u,327760u,12u,560u,558u,559u,196670u,553u,560u,262205u,
6u,561u,15u,327745u,9u,562u,14u,551u,262205u,6u,563u,562u,327808u,6u,564u,563u,
561u,327745u,9u,565u,14u,551u,196670u,565u,564u,196670u,566u,548u,131321u,567u,131320u,567u,
262390u,569u,570u,0u,131321u,571u,131320u,571u,262205u,6u,572u,566u,262205u,6u,573u,15u,
327857u,57u,574u,572u,573u,262394u,574u,568u,569u,131320u,568u,327745u,9u,579u,553u,548u,
262205u,6u,580u,579u,262205u,6u,581u,566u,327808u,6u,582u,580u,581u,262205u,10u,583u,
16u,393281u,584u,585u,578u,548u,582u,196670u,585u,583u,131321u,570u,131320u,570u,262205u,6u,
586u,566u,327808u,6u,587u,586u,551u,196670u,566u,587u,131321u,567u,131320u,569u,262205u,12u,
588u,553u,131326u,588u,65592u,327734u,2u,24u,0u,20u,196663u,19u,21u,196663u,9u,22u,
196663u,11u,23u,131320u,25u,262203u,9u,591u,7u,196670u,591u,548u,131321u,592u,131320u,592u,
262390u,594u,595u,0u,131321u,596u,131320u,596u,262205u,6u,597u,591u,262205u,6u,598u,22u,
327857u,57u,599u,597u,598u,262394u,599u,593u,594u,131320u,593u,327745u,9u,600u,21u,548u,
262205u,6u,601u,600u,262205u,6u,602u,591u,327808u,6u,603u,601u,602u,262205u,10u,604u,
23u,393281u,584u,605u,578u,548u,603u,196670u,605u,604u,131321u,595u,131320u,595u,262205u,6u,
606u,591u,327808u,6u,607u,606u,551u,196670u,591u,607u,131321u,592u,131320u,594u,262205u,6u,
608u,22u,327745u,9u,609u,21u,551u,196670u,609u,608u,65789u,65592u,327734u,2u,29u,0u,
26u,196663u,19u,27u,196663u,19u,28u,131320u,30u,262203u,19u,610u,7u,262205u,12u,611u,
27u,196670u,610u,611u,262205u,12u,612u,28u,196670u,27u,612u,262205u,12u,613u,610u,196670u,
28u,613u,65789u,65592u,327734u,2u,35u,0u,31u,196663u,9u,32u,196663u,9u,33u,196663u,
9u,34u,131320u,36u,131321u,614u,131320u,614u,262390u,616u,617u,0u,131321u,618u,131320u,618u,
262205u,6u,619u,32u,262205u,6u,620u,33u,327851u,57u,621u,619u,620u,262394u,621u,615u,
616u,131320u,615u,262205u,6u,622u,34u,327808u,6u,623u,622u,551u,196670u,34u,623u,262205u,
6u,624u,32u,327808u,6u,625u,624u,551u,196670u,32u,625u,393281u,584u,626u,578u,548u,
624u,262205u,10u,627u,626u,393281u,584u,628u,578u,548u,622u,196670u,628u,627u,131321u,617u,
131320u,617u,131321u,614u,131320u,616u,65789u,65592u,327734u,10u,41u,0u,37u,196663u,9u,38u,
196663u,9u,39u,196663u,9u,40u,131320u,42u,262203u,11u,629u,7u,262203u,9u,631u,7u,
196670u,629u,630u,196670u,631u,548u,131321u,632u,131320u,632u,262390u,634u,635u,0u,131321u,636u,
131320u,636u,262205u,6u,637u,631u,262205u,6u,638u,40u,327857u,57u,639u,637u,638u,262394u,
639u,633u,634u,131320u,633u,262205u,6u,640u,38u,262205u,6u,641u,631u,327808u,6u,642u,
640u,641u,393281u,584u,643u,578u,548u,642u,262205u,10u,644u,643u,262205u,6u,645u,39u,
262205u,6u,646u,631u,327808u,6u,647u,645u,646u,393281u,584u,648u,578u,548u,647u,262205u,
10u,649u,648u,327813u,10u,650u,644u,649u,262205u,10u,651u,629u,327809u,10u,652u,651u,
650u,196670u,629u,652u,131321u,635u,131320u,635u,262205u,6u,653u,631u,327808u,6u,654u,653u,
551u,196670u,631u,654u,131321u,632u,131320u,634u,262205u,10u,655u,629u,131326u,655u,65592u,327734u,
10u,46u,0u,43u,196663u,11u,44u,196663u,9u,45u,131320u,47u,262205u,10u,658u,44u,
131326u,658u,65592u,327734u,10u,50u,0u,43u,196663u,11u,48u,196663u,9u,49u,131320u,51u,
262205u,10u,661u,48u,131326u,661u,65592u,327734u,2u,55u,0u,52u,196663u,9u,53u,196663u,
11u,54u,131320u,56u,262205u,6u,664u,53u,262205u,10u,665u,54u,393281u,584u,666u,578u,
548u,664u,262205u,10u,667u,666u,327809u,10u,668u,667u,665u,393281u,584u,669u,578u,548u,
664u,196670u,669u,668u,65789u,65592u,327734u,57u,61u,0u,59u,196663u,58u,60u,131320u,62u,
262205u,57u,670u,60u,131326u,670u,65592u,327734u,57u,65u,0u,63u,196663u,9u,64u,131320u,
66u,262205u,6u,673u,64u,327851u,57u,674u,673u,548u,131326u,674u,65592u,327734u,57u,69u,
0u,67u,196663u,11u,68u,131320u,70u,262205u,10u,677u,68u,327863u,57u,678u,677u,630u,
131326u,678u,65592u,327734u,10u,73u,0u,71u,196663u,11u,72u,131320u,74u,262203u,11u,683u,
7u,262205u,10u,681u,72u,327870u,57u,682u,681u,630u,196855u,685u,0u,262394u,682u,684u,
690u,131320u,684u,262205u,10u,686u,72u,327809u,10u,688u,686u,687u,393228u,10u,689u,1u,
8u,688u,196670u,683u,689u,131321u,685u,131320u,690u,262205u,10u,691u,72u,327811u,10u,692u,
691u,687u,393228u,10u,693u,1u,9u,692u,196670u,683u,693u,131321u,685u,131320u,685u,262205u,
10u,694u,683u,131326u,694u,65592u,327734u,10u,76u,0u,71u,196663u,11u,75u,131320u,77u,
262205u,10u,697u,75u,393228u,10u,698u,1u,32u,697u,131326u,698u,65592u,327734u,10u,79u,
0u,71u,196663u,11u,78u,131320u,80u,262205u,10u,702u,78u,327809u,10u,703u,701u,702u,
393228u,10u,704u,1u,28u,703u,131326u,704u,65592u,327734u,57u,82u,0u,67u,196663u,11u,
81u,131320u,83u,262205u,10u,707u,81u,262300u,57u,708u,707u,262312u,57u,709u,708u,196855u,
711u,0u,262394u,709u,710u,711u,131320u,710u,262205u,10u,712u,81u,262301u,57u,713u,712u,
262312u,57u,714u,713u,131321u,711u,131320u,711u,458997u,57u,715u,709u,83u,714u,710u,131326u,
715u,65592u,327734u,2u,95u,0u,84u,196663u,9u,85u,196663u,9u,86u,196663u,9u,87u,
196663u,9u,88u,196663u,9u,89u,196663u,9u,90u,196663u,9u,91u,196663u,9u,92u,196663u,
11u,93u,196663u,58u,94u,131320u,96u,262203u,9u,718u,7u,262203u,9u,719u,7u,262203u,
11u,723u,7u,262203u,9u,724u,7u,262203u,11u,745u,7u,262203u,11u,746u,7u,262203u,
9u,748u,7u,262203u,11u,754u,7u,262203u,9u,755u,7u,262203u,11u,765u,7u,262203u,
11u,782u,7u,262203u,11u,783u,7u,262203u,9u,785u,7u,262203u,11u,793u,7u,262203u,
9u,807u,7u,262203u,9u,817u,7u,262203u,9u,821u,7u,262203u,11u,827u,7u,262203u,
9u,837u,7u,262203u,11u,852u,7u,262203u,58u,862u,7u,262203u,11u,865u,7u,196670u,
718u,548u,262205u,6u,720u,549u,262205u,6u,721u,90u,327812u,6u,722u,720u,721u,196670u,
719u,722u,196670u,723u,630u,262205u,6u,725u,547u,196670u,724u,725u,131321u,726u,131320u,726u,
262390u,728u,729u,0u,131321u,730u,131320u,730u,262205u,6u,731u,724u,262205u,6u,732u,90u,
327857u,57u,733u,731u,732u,262394u,733u,727u,728u,131320u,727u,262205u,6u,734u,85u,262205u,
6u,735u,719u,262205u,6u,736u,724u,327808u,6u,737u,735u,736u,327808u,6u,738u,734u,
737u,393281u,584u,739u,578u,548u,738u,262205u,10u,740u,739u,262205u,10u,741u,723u,327809u,
10u,742u,741u,740u,196670u,723u,742u,131321u,729u,131320u,729u,262205u,6u,743u,724u,327808u,
6u,744u,743u,551u,196670u,724u,744u,131321u,726u,131320u,728u,262205u,10u,747u,723u,196670u,
746u,747u,262205u,6u,749u,718u,196670u,748u,749u,393273u,10u,750u,46u,746u,748u,262205u,
6u,751u,90u,262255u,10u,752u,751u,327816u,10u,753u,750u,752u,196670u,745u,753u,196670u,
754u,630u,262205u,6u,756u,547u,196670u,755u,756u,131321u,757u,131320u,757u,262390u,759u,760u,
0u,131321u,761u,131320u,761u,262205u,6u,762u,755u,262205u,6u,763u,90u,327857u,57u,764u,
762u,763u,262394u,764u,758u,759u,131320u,758u,262205u,6u,766u,85u,262205u,6u,767u,719u,
262205u,6u,768u,755u,327808u,6u,769u,767u,768u,327808u,6u,770u,766u,769u,393281u,584u,
771u,578u,548u,770u,262205u,10u,772u,771u,262205u,10u,773u,745u,327811u,10u,774u,772u,
773u,196670u,765u,774u,262205u,10u,775u,765u,262205u,10u,776u,765u,327813u,10u,777u,775u,
776u,262205u,10u,778u,754u,327809u,10u,779u,778u,777u,196670u,754u,779u,131321u,760u,131320u,
760u,262205u,6u,780u,755u,327808u,6u,781u,780u,551u,196670u,755u,781u,131321u,757u,131320u,
759u,262205u,10u,784u,754u,196670u,783u,784u,262205u,6u,786u,718u,196670u,785u,786u,393273u,
10u,787u,46u,783u,785u,262205u,6u,788u,90u,262255u,10u,789u,788u,327816u,10u,790u,
787u,789u,262205u,10u,791u,93u,327809u,10u,792u,790u,791u,196670u,793u,792u,327737u,10u,
794u,76u,793u,196670u,782u,794u,262205u,6u,795u,547u,327850u,57u,796u,795u,548u,262205u,
6u,797u,89u,327851u,57u,798u,797u,548u,327847u,57u,799u,796u,798u,196855u,801u,0u,
262394u,799u,800u,801u,131320u,800u,262205u,6u,802u,89u,262205u,6u,803u,549u,327808u,6u,
804u,802u,803u,262205u,10u,805u,782u,393281u,584u,806u,578u,548u,804u,196670u,806u,805u,
131321u,801u,131320u,801u,262205u,6u,808u,547u,196670u,807u,808u,131321u,809u,131320u,809u,262390u,
811u,812u,0u,131321u,813u,131320u,813u,262205u,6u,814u,807u,262205u,6u,815u,90u,327857u,
57u,816u,814u,815u,262394u,816u,810u,811u,131320u,810u,262205u,6u,818u,719u,262205u,6u,
819u,807u,327808u,6u,820u,818u,819u,196670u,817u,820u,262205u,6u,822u,817u,262205u,6u,
823u,91u,327815u,6u,824u,822u,823u,262205u,6u,825u,92u,327819u,6u,826u,824u,825u,
196670u,821u,826u,262205u,6u,828u,85u,262205u,6u,829u,817u,327808u,6u,830u,828u,829u,
393281u,584u,831u,578u,548u,830u,262205u,10u,832u,831u,262205u,10u,833u,745u,327811u,10u,
834u,832u,833u,262205u,10u,835u,782u,327813u,10u,836u,834u,835u,196670u,827u,836u,262205u,
6u,838u,88u,196670u,837u,838u,327737u,57u,839u,65u,837u,196855u,841u,0u,262394u,839u,
840u,841u,131320u,840u,262205u,6u,842u,88u,262205u,6u,843u,817u,327808u,6u,844u,842u,
843u,262205u,10u,845u,827u,393281u,584u,846u,578u,548u,844u,196670u,846u,845u,131321u,841u,
131320u,841u,262205u,6u,847u,87u,262205u,6u,848u,817u,327808u,6u,849u,847u,848u,262205u,
6u,850u,86u,327851u,57u,851u,850u,548u,196855u,854u,0u,262394u,851u,853u,878u,131320u,
853u,262205u,10u,855u,827u,262205u,6u,856u,86u,262205u,6u,857u,821u,327808u,6u,858u,
856u,857u,393281u,584u,859u,578u,548u,858u,262205u,10u,860u,859u,327813u,10u,861u,855u,
860u,262205u,57u,863u,94u,196670u,862u,863u,327737u,57u,864u,61u,862u,196855u,867u,0u,
262394u,864u,866u,875u,131320u,866u,262205u,6u,868u,86u,262205u,6u,869u,92u,262205u,6u,
870u,821u,327808u,6u,871u,869u,870u,327808u,6u,872u,868u,871u,393281u,584u,873u,578u,
548u,872u,262205u,10u,874u,873u,196670u,865u,874u,131321u,867u,131320u,875u,196670u,865u,630u,
131321u,867u,131320u,867u,262205u,10u,876u,865u,327809u,10u,877u,861u,876u,196670u,852u,877u,
131321u,854u,131320u,878u,262205u,10u,879u,827u,196670u,852u,879u,131321u,854u,131320u,854u,262205u,
10u,880u,852u,393281u,584u,881u,578u,548u,849u,196670u,881u,880u,131321u,812u,131320u,812u,
262205u,6u,882u,807u,327808u,6u,883u,882u,551u,196670u,807u,883u,131321u,809u,131320u,811u,
65789u,65592u,327734u,2u,106u,0u,97u,196663u,9u,98u,196663u,9u,99u,196663u,9u,100u,
196663u,9u,101u,196663u,9u,102u,196663u,9u,103u,196663u,9u,104u,196663u,9u,105u,131320u,
107u,262203u,9u,884u,7u,262203u,9u,885u,7u,262203u,11u,889u,7u,262203u,11u,890u,
7u,262203u,9u,891u,7u,262203u,9u,901u,7u,262203u,11u,905u,7u,262203u,11u,913u,
7u,262203u,11u,942u,7u,262203u,11u,943u,7u,262203u,9u,945u,7u,262203u,11u,948u,
7u,262203u,11u,949u,7u,262203u,9u,951u,7u,262203u,9u,954u,7u,262203u,9u,964u,
7u,262203u,11u,968u,7u,262203u,11u,976u,7u,196670u,884u,548u,262205u,6u,886u,549u,
262205u,6u,887u,103u,327812u,6u,888u,886u,887u,196670u,885u,888u,196670u,889u,630u,196670u,
890u,630u,262205u,6u,892u,547u,196670u,891u,892u,131321u,893u,131320u,893u,262390u,895u,896u,
0u,131321u,897u,131320u,897u,262205u,6u,898u,891u,262205u,6u,899u,103u,327857u,57u,900u,
898u,899u,262394u,900u,894u,895u,131320u,894u,262205u,6u,902u,885u,262205u,6u,903u,891u,
327808u,6u,904u,902u,903u,196670u,901u,904u,262205u,6u,906u,100u,262205u,6u,907u,901u,
327808u,6u,908u,906u,907u,393281u,584u,909u,578u,548u,908u,262205u,10u,910u,909u,262205u,
6u,911u,101u,327851u,57u,912u,911u,548u,196855u,915u,0u,262394u,912u,914u,925u,131320u,
914u,262205u,6u,916u,101u,262205u,6u,917u,901u,262205u,6u,918u,104u,327815u,6u,919u,
917u,918u,262205u,6u,920u,105u,327819u,6u,921u,919u,920u,327808u,6u,922u,916u,921u,
393281u,584u,923u,578u,548u,922u,262205u,10u,924u,923u,196670u,913u,924u,131321u,915u,131320u,
925u,196670u,913u,701u,131321u,915u,131320u,915u,262205u,10u,926u,913u,327813u,10u,927u,910u,
926u,196670u,905u,927u,262205u,10u,928u,905u,262205u,10u,929u,889u,327809u,10u,930u,929u,
928u,196670u,889u,930u,262205u,10u,931u,905u,262205u,6u,932u,98u,262205u,6u,933u,901u,
327808u,6u,934u,932u,933u,393281u,584u,935u,578u,548u,934u,262205u,10u,936u,935u,327813u,
10u,937u,931u,936u,262205u,10u,938u,890u,327809u,10u,939u,938u,937u,196670u,890u,939u,
131321u,896u,131320u,896u,262205u,6u,940u,891u,327808u,6u,941u,940u,551u,196670u,891u,941u,
131321u,893u,131320u,895u,262205u,10u,944u,889u,196670u,943u,944u,262205u,6u,946u,884u,196670u,
945u,946u,393273u,10u,947u,46u,943u,945u,196670u,942u,947u,262205u,10u,950u,890u,196670u,
949u,950u,262205u,6u,952u,884u,196670u,951u,952u,393273u,10u,953u,46u,949u,951u,196670u,
948u,953u,262205u,6u,955u,547u,196670u,954u,955u,131321u,956u,131320u,956u,262390u,958u,959u,
0u,131321u,960u,131320u,960u,262205u,6u,961u,954u,262205u,6u,962u,103u,327857u,57u,963u,
961u,962u,262394u,963u,957u,958u,131320u,957u,262205u,6u,965u,885u,262205u,6u,966u,954u,
327808u,6u,967u,965u,966u,196670u,964u,967u,262205u,6u,969u,100u,262205u,6u,970u,964u,
327808u,6u,971u,969u,970u,393281u,584u,972u,578u,548u,971u,262205u,10u,973u,972u,262205u,
6u,974u,101u,327851u,57u,975u,974u,548u,196855u,978u,0u,262394u,975u,977u,988u,131320u,
977u,262205u,6u,979u,101u,262205u,6u,980u,964u,262205u,6u,981u,104u,327815u,6u,982u,
980u,981u,262205u,6u,983u,105u,327819u,6u,984u,982u,983u,327808u,6u,985u,979u,984u,
393281u,584u,986u,578u,548u,985u,262205u,10u,987u,986u,196670u,976u,987u,131321u,978u,131320u,
988u,196670u,976u,701u,131321u,978u,131320u,978u,262205u,10u,989u,976u,327813u,10u,990u,973u,
989u,196670u,968u,990u,262205u,6u,991u,102u,262205u,6u,992u,964u,327808u,6u,993u,991u,
992u,262205u,6u,994u,99u,262205u,6u,995u,549u,327808u,6u,996u,994u,995u,393281u,584u,
997u,578u,548u,996u,262205u,10u,998u,997u,262205u,10u,999u,968u,262205u,10u,1000u,942u,
262205u,6u,1001u,103u,262255u,10u,1002u,1001u,327816u,10u,1003u,1000u,1002u,327811u,10u,1004u,
999u,1003u,262205u,6u,1005u,98u,262205u,6u,1006u,964u,327808u,6u,1007u,1005u,1006u,393281u,
584u,1008u,578u,548u,1007u,262205u,10u,1009u,1008u,262205u,10u,1010u,948u,327813u,10u,1011u,
1009u,1010u,262205u,6u,1012u,103u,262255u,10u,1013u,1012u,327816u,10u,1014u,1011u,1013u,327811u,
10u,1015u,1004u,1014u,327813u,10u,1016u,998u,1015u,393281u,584u,1017u,578u,548u,993u,196670u,
1017u,1016u,131321u,959u,131320u,959u,262205u,6u,1018u,954u,327808u,6u,1019u,1018u,551u,196670u,
954u,1019u,131321u,956u,131320u,958u,65789u,65592u,327734u,2u,114u,0u,108u,196663u,9u,109u,
196663u,9u,110u,196663u,9u,111u,196663u,9u,112u,196663u,9u,113u,131320u,115u,262203u,9u,
1020u,7u,262203u,9u,1021u,7u,262203u,11u,1023u,7u,262203u,11u,1024u,7u,262203u,9u,
1025u,7u,262203u,9u,1035u,7u,262203u,9u,1048u,7u,262203u,11u,1068u,7u,262203u,11u,
1069u,7u,262203u,9u,1071u,7u,262203u,11u,1074u,7u,262203u,11u,1075u,7u,262203u,9u,
1077u,7u,262203u,9u,1084u,7u,262203u,9u,1094u,7u,196670u,1020u,548u,262205u,6u,1022u,
549u,196670u,1021u,1022u,196670u,1023u,630u,196670u,1024u,630u,262205u,6u,1026u,547u,196670u,1025u,
1026u,131321u,1027u,131320u,1027u,262390u,1029u,1030u,0u,131321u,1031u,131320u,1031u,262205u,6u,1032u,
1025u,262205u,6u,1033u,113u,327857u,57u,1034u,1032u,1033u,262394u,1034u,1028u,1029u,131320u,1028u,
262205u,6u,1036u,1021u,262205u,6u,1037u,113u,327812u,6u,1038u,1036u,1037u,262205u,6u,1039u,
1025u,327808u,6u,1040u,1038u,1039u,196670u,1035u,1040u,262205u,6u,1041u,109u,262205u,6u,1042u,
1035u,327808u,6u,1043u,1041u,1042u,393281u,584u,1044u,578u,548u,1043u,262205u,10u,1045u,1044u,
262205u,10u,1046u,1023u,327809u,10u,1047u,1046u,1045u,196670u,1023u,1047u,262205u,6u,1049u,110u,
196670u,1048u,1049u,327737u,57u,1050u,65u,1048u,196855u,1052u,0u,262394u,1050u,1051u,1052u,131320u,
1051u,262205u,6u,1053u,109u,262205u,6u,1054u,1035u,327808u,6u,1055u,1053u,1054u,393281u,584u,
1056u,578u,548u,1055u,262205u,10u,1057u,1056u,262205u,6u,1058u,110u,262205u,6u,1059u,1035u,
327808u,6u,1060u,1058u,1059u,393281u,584u,1061u,578u,548u,1060u,262205u,10u,1062u,1061u,327813u,
10u,1063u,1057u,1062u,262205u,10u,1064u,1024u,327809u,10u,1065u,1064u,1063u,196670u,1024u,1065u,
131321u,1052u,131320u,1052u,131321u,1030u,131320u,1030u,262205u,6u,1066u,1025u,327808u,6u,1067u,1066u,
551u,196670u,1025u,1067u,131321u,1027u,131320u,1029u,262205u,10u,1070u,1023u,196670u,1069u,1070u,262205u,
6u,1072u,1020u,196670u,1071u,1072u,393273u,10u,1073u,46u,1069u,1071u,196670u,1068u,1073u,262205u,
10u,1076u,1024u,196670u,1075u,1076u,262205u,6u,1078u,1020u,196670u,1077u,1078u,393273u,10u,1079u,
46u,1075u,1077u,196670u,1074u,1079u,262205u,6u,1080u,547u,327850u,57u,1081u,1080u,548u,196855u,
1083u,0u,262394u,1081u,1082u,1083u,131320u,1082u,262205u,6u,1085u,111u,196670u,1084u,1085u,327737u,
57u,1086u,65u,1084u,196855u,1088u,0u,262394u,1086u,1087u,1088u,131320u,1087u,262205u,6u,1089u,
111u,262205u,6u,1090u,1021u,327808u,6u,1091u,1089u,1090u,262205u,10u,1092u,1074u,393281u,584u,
1093u,578u,548u,1091u,196670u,1093u,1092u,131321u,1088u,131320u,1088u,262205u,6u,1095u,112u,196670u,
1094u,1095u,327737u,57u,1096u,65u,1094u,196855u,1098u,0u,262394u,1096u,1097u,1098u,131320u,1097u,
262205u,6u,1099u,112u,262205u,6u,1100u,1021u,327808u,6u,1101u,1099u,1100u,262205u,10u,1102u,
1068u,393281u,584u,1103u,578u,548u,1101u,196670u,1103u,1102u,131321u,1098u,131320u,1098u,131321u,1083u,
131320u,1083u,65789u,65592u,327734u,2u,127u,0u,116u,196663u,9u,117u,196663u,9u,118u,196663u,
9u,119u,196663u,9u,120u,196663u,9u,121u,196663u,9u,122u,196663u,9u,123u,196663u,9u,
124u,196663u,9u,125u,196663u,58u,126u,131320u,128u,262203u,9u,1104u,7u,262203u,9u,1116u,
7u,262203u,9u,1122u,7u,262203u,9u,1128u,7u,262203u,58u,1132u,7u,262203u,11u,1160u,
7u,262203u,11u,1169u,7u,262203u,9u,1178u,7u,262203u,9u,1182u,7u,262203u,9u,1186u,
7u,262203u,9u,1192u,7u,262203u,11u,1198u,7u,262203u,11u,1203u,7u,262203u,9u,1208u,
7u,262205u,6u,1105u,549u,262205u,6u,1106u,550u,327812u,6u,1107u,1105u,1106u,262205u,6u,
1108u,547u,327808u,6u,1109u,1107u,1108u,196670u,1104u,1109u,262205u,6u,1110u,1104u,262205u,6u,
1111u,119u,327855u,57u,1112u,1110u,1111u,196855u,1114u,0u,262394u,1112u,1113u,1114u,131320u,1113u,
65789u,131320u,1114u,262205u,6u,1117u,1104u,262205u,6u,1118u,122u,262205u,6u,1119u,123u,327812u,
6u,1120u,1118u,1119u,327815u,6u,1121u,1117u,1120u,196670u,1116u,1121u,262205u,6u,1123u,1104u,
262205u,6u,1124u,123u,327815u,6u,1125u,1123u,1124u,262205u,6u,1126u,122u,327819u,6u,1127u,
1125u,1126u,196670u,1122u,1127u,262205u,6u,1129u,1104u,262205u,6u,1130u,123u,327819u,6u,1131u,
1129u,1130u,196670u,1128u,1131u,262205u,57u,1133u,126u,196670u,1132u,1133u,327737u,57u,1134u,61u,
1132u,262312u,57u,1135u,1134u,196855u,1137u,0u,262394u,1135u,1136u,1137u,131320u,1136u,262205u,6u,
1138u,118u,262205u,6u,1139u,1104u,327808u,6u,1140u,1138u,1139u,262205u,6u,1141u,117u,262205u,
6u,1142u,1116u,262205u,6u,1143u,120u,327812u,6u,1144u,1142u,1143u,262205u,6u,1145u,1122u,
262205u,6u,1146u,124u,327815u,6u,1147u,1145u,1146u,327808u,6u,1148u,1144u,1147u,262205u,6u,
1149u,121u,327812u,6u,1150u,1148u,1149u,262205u,6u,1151u,1128u,262205u,6u,1152u,125u,327815u,
6u,1153u,1151u,1152u,327808u,6u,1154u,1150u,1153u,327808u,6u,1155u,1141u,1154u,393281u,584u,
1156u,578u,548u,1155u,262205u,10u,1157u,1156u,393281u,584u,1158u,578u,548u,1140u,196670u,1158u,
1157u,65789u,131320u,1137u,262205u,6u,1161u,1122u,262255u,10u,1162u,1161u,262205u,6u,1163u,120u,
262255u,10u,1164u,1163u,262205u,6u,1165u,122u,262255u,10u,1166u,1165u,327816u,10u,1167u,1164u,
1166u,327813u,10u,1168u,1162u,1167u,196670u,1160u,1168u,262205u,6u,1170u,1128u,262255u,10u,1171u,
1170u,262205u,6u,1172u,121u,262255u,10u,1173u,1172u,262205u,6u,1174u,123u,262255u,10u,1175u,
1174u,327816u,10u,1176u,1173u,1175u,327813u,10u,1177u,1171u,1176u,196670u,1169u,1177u,262205u,10u,
1179u,1160u,393228u,10u,1180u,1u,8u,1179u,262254u,6u,1181u,1180u,196670u,1178u,1181u,262205u,
10u,1183u,1169u,393228u,10u,1184u,1u,8u,1183u,262254u,6u,1185u,1184u,196670u,1182u,1185u,
262205u,6u,1187u,1178u,327808u,6u,1188u,1187u,551u,262205u,6u,1189u,120u,327810u,6u,1190u,
1189u,551u,458764u,6u,1191u,1u,39u,1188u,1190u,196670u,1186u,1191u,262205u,6u,1193u,1182u,
327808u,6u,1194u,1193u,551u,262205u,6u,1195u,121u,327810u,6u,1196u,1195u,551u,458764u,6u,
1197u,1u,39u,1194u,1196u,196670u,1192u,1197u,262205u,10u,1199u,1160u,262205u,6u,1200u,1178u,
262255u,10u,1201u,1200u,327811u,10u,1202u,1199u,1201u,196670u,1198u,1202u,262205u,10u,1204u,1169u,
262205u,6u,1205u,1182u,262255u,10u,1206u,1205u,327811u,10u,1207u,1204u,1206u,196670u,1203u,1207u,
262205u,6u,1209u,1116u,262205u,6u,1210u,120u,327812u,6u,1211u,1209u,1210u,262205u,6u,1212u,
121u,327812u,6u,1213u,1211u,1212u,196670u,1208u,1213u,262205u,6u,1214u,118u,262205u,6u,1215u,
1104u,327808u,6u,1216u,1214u,1215u,262205u,6u,1217u,117u,262205u,6u,1218u,1208u,262205u,6u,
1219u,1178u,262205u,6u,1220u,121u,327812u,6u,1221u,1219u,1220u,327808u,6u,1222u,1218u,1221u,
262205u,6u,1223u,1182u,327808u,6u,1224u,1222u,1223u,327808u,6u,1225u,1217u,1224u,393281u,584u,
1226u,578u,548u,1225u,262205u,10u,1227u,1226u,262205u,10u,1228u,1198u,327811u,10u,1229u,701u,
1228u,327813u,10u,1230u,1227u,1229u,262205u,10u,1231u,1203u,327811u,10u,1232u,701u,1231u,327813u,
10u,1233u,1230u,1232u,262205u,6u,1234u,117u,262205u,6u,1235u,1208u,262205u,6u,1236u,1178u,
262205u,6u,1237u,121u,327812u,6u,1238u,1236u,1237u,327808u,6u,1239u,1235u,1238u,262205u,6u,
1240u,1192u,327808u,6u,1241u,1239u,1240u,327808u,6u,1242u,1234u,1241u,393281u,584u,1243u,578u,
548u,1242u,262205u,10u,1244u,1243u,262205u,10u,1245u,1198u,327811u,10u,1246u,701u,1245u,327813u,
10u,1247u,1244u,1246u,262205u,10u,1248u,1203u,327813u,10u,1249u,1247u,1248u,327809u,10u,1250u,
1233u,1249u,262205u,6u,1251u,117u,262205u,6u,1252u,1208u,262205u,6u,1253u,1186u,262205u,6u,
1254u,121u,327812u,6u,1255u,1253u,1254u,327808u,6u,1256u,1252u,1255u,262205u,6u,1257u,1182u,
327808u,6u,1258u,1256u,1257u,327808u,6u,1259u,1251u,1258u,393281u,584u,1260u,578u,548u,1259u,
262205u,10u,1261u,1260u,262205u,10u,1262u,1198u,327813u,10u,1263u,1261u,1262u,262205u,10u,1264u,
1203u,327811u,10u,1265u,701u,1264u,327813u,10u,1266u,1263u,1265u,327809u,10u,1267u,1250u,1266u,
262205u,6u,1268u,117u,262205u,6u,1269u,1208u,262205u,6u,1270u,1186u,262205u,6u,1271u,121u,
327812u,6u,1272u,1270u,1271u,327808u,6u,1273u,1269u,1272u,262205u,6u,1274u,1192u,327808u,6u,
1275u,1273u,1274u,327808u,6u,1276u,1268u,1275u,393281u,584u,1277u,578u,548u,1276u,262205u,10u,
1278u,1277u,262205u,10u,1279u,1198u,327813u,10u,1280u,1278u,1279u,262205u,10u,1281u,1203u,327813u,
10u,1282u,1280u,1281u,327809u,10u,1283u,1267u,1282u,393281u,584u,1284u,578u,548u,1216u,196670u,
1284u,1283u,65789u,65592u,327734u,2u,137u,0u,129u,196663u,9u,130u,196663u,9u,131u,196663u,
9u,132u,196663u,9u,133u,196663u,9u,134u,196663u,9u,135u,196663u,9u,136u,131320u,138u,
262203u,9u,1285u,7u,262203u,9u,1297u,7u,262203u,9u,1303u,7u,262203u,9u,1309u,7u,
262203u,11u,1313u,7u,262203u,9u,1314u,7u,262203u,9u,1323u,7u,262205u,6u,1286u,549u,
262205u,6u,1287u,550u,327812u,6u,1288u,1286u,1287u,262205u,6u,1289u,547u,327808u,6u,1290u,
1288u,1289u,196670u,1285u,1290u,262205u,6u,1291u,1285u,262205u,6u,1292u,132u,327855u,57u,1293u,
1291u,1292u,196855u,1295u,0u,262394u,1293u,1294u,1295u,131320u,1294u,65789u,131320u,1295u,262205u,6u,
1298u,1285u,262205u,6u,1299u,133u,262205u,6u,1300u,134u,327812u,6u,1301u,1299u,1300u,327815u,
6u,1302u,1298u,1301u,196670u,1297u,1302u,262205u,6u,1304u,1285u,262205u,6u,1305u,134u,327815u,
6u,1306u,1304u,1305u,262205u,6u,1307u,133u,327819u,6u,1308u,1306u,1307u,196670u,1303u,1308u,
262205u,6u,1310u,1285u,262205u,6u,1311u,134u,327819u,6u,1312u,1310u,1311u,196670u,1309u,1312u,
196670u,1313u,630u,196670u,1314u,548u,131321u,1315u,131320u,1315u,262390u,1317u,1318u,0u,131321u,1319u,
131320u,1319u,262205u,6u,1320u,1314u,262205u,6u,1321u,135u,327857u,57u,1322u,1320u,1321u,262394u,
1322u,1316u,1317u,131320u,1316u,196670u,1323u,548u,131321u,1324u,131320u,1324u,262390u,1326u,1327u,0u,
131321u,1328u,131320u,1328u,262205u,6u,1329u,1323u,262205u,6u,1330u,136u,327857u,57u,1331u,1329u,
1330u,262394u,1331u,1325u,1326u,131320u,1325u,262205u,6u,1332u,130u,262205u,6u,1333u,1297u,262205u,
6u,1334u,133u,327812u,6u,1335u,1333u,1334u,262205u,6u,1336u,135u,327812u,6u,1337u,1335u,
1336u,262205u,6u,1338u,1303u,262205u,6u,1339u,135u,327812u,6u,1340u,1338u,1339u,327808u,6u,
1341u,1337u,1340u,262205u,6u,1342u,1314u,327808u,6u,1343u,1341u,1342u,262205u,6u,1344u,134u,
327812u,6u,1345u,1343u,1344u,262205u,6u,1346u,136u,327812u,6u,1347u,1345u,1346u,262205u,6u,
1348u,1309u,262205u,6u,1349u,136u,327812u,6u,1350u,1348u,1349u,327808u,6u,1351u,1347u,1350u,
262205u,6u,1352u,1323u,327808u,6u,1353u,1351u,1352u,327808u,6u,1354u,1332u,1353u,393281u,584u,
1355u,578u,548u,1354u,262205u,10u,1356u,1355u,262205u,10u,1357u,1313u,327809u,10u,1358u,1357u,
1356u,196670u,1313u,1358u,131321u,1327u,131320u,1327u,262205u,6u,1359u,1323u,327808u,6u,1360u,1359u,
551u,196670u,1323u,1360u,131321u,1324u,131320u,1326u,131321u,1318u,131320u,1318u,262205u,6u,1361u,1314u,
327808u,6u,1362u,1361u,551u,196670u,1314u,1362u,131321u,1315u,131320u,1317u,262205u,6u,1363u,131u,
262205u,6u,1364u,1285u,327808u,6u,1365u,1363u,1364u,262205u,10u,1366u,1313u,393281u,584u,1367u,
578u,548u,1365u,196670u,1367u,1366u,65789u,65592u,327734u,2u,146u,0u,129u,196663u,9u,139u,
196663u,9u,140u,196663u,9u,141u,196663u,9u,142u,196663u,9u,143u,196663u,9u,144u,196663u,
9u,145u,131320u,147u,262203u,9u,1368u,7u,262203u,9u,1380u,7u,262203u,9u,1386u,7u,
262203u,9u,1392u,7u,262203u,11u,1396u,7u,262203u,11u,1405u,7u,262203u,9u,1414u,7u,
262203u,9u,1418u,7u,262203u,9u,1422u,7u,262203u,9u,1428u,7u,262203u,11u,1434u,7u,
262203u,11u,1439u,7u,262203u,11u,1444u,7u,262203u,9u,1450u,7u,262203u,9u,1472u,7u,
262203u,11u,1473u,7u,262203u,9u,1490u,7u,262203u,11u,1491u,7u,262203u,9u,1508u,7u,
262203u,11u,1509u,7u,262203u,9u,1525u,7u,262203u,11u,1526u,7u,262205u,6u,1369u,549u,
262205u,6u,1370u,550u,327812u,6u,1371u,1369u,1370u,262205u,6u,1372u,547u,327808u,6u,1373u,
1371u,1372u,196670u,1368u,1373u,262205u,6u,1374u,1368u,262205u,6u,1375u,141u,327855u,57u,1376u,
1374u,1375u,196855u,1378u,0u,262394u,1376u,1377u,1378u,131320u,1377u,65789u,131320u,1378u,262205u,6u,
1381u,1368u,262205u,6u,1382u,144u,262205u,6u,1383u,145u,327812u,6u,1384u,1382u,1383u,327815u,
6u,1385u,1381u,1384u,196670u,1380u,1385u,262205u,6u,1387u,1368u,262205u,6u,1388u,145u,327815u,
6u,1389u,1387u,1388u,262205u,6u,1390u,144u,327819u,6u,1391u,1389u,1390u,196670u,1386u,1391u,
262205u,6u,1393u,1368u,262205u,6u,1394u,145u,327819u,6u,1395u,1393u,1394u,196670u,1392u,1395u,
262205u,6u,1397u,1386u,262255u,10u,1398u,1397u,262205u,6u,1399u,142u,262255u,10u,1400u,1399u,
262205u,6u,1401u,144u,262255u,10u,1402u,1401u,327816u,10u,1403u,1400u,1402u,327813u,10u,1404u,
1398u,1403u,196670u,1396u,1404u,262205u,6u,1406u,1392u,262255u,10u,1407u,1406u,262205u,6u,1408u,
143u,262255u,10u,1409u,1408u,262205u,6u,1410u,145u,262255u,10u,1411u,1410u,327816u,10u,1412u,
1409u,1411u,327813u,10u,1413u,1407u,1412u,196670u,1405u,1413u,262205u,10u,1415u,1396u,393228u,10u,
1416u,1u,8u,1415u,262254u,6u,1417u,1416u,196670u,1414u,1417u,262205u,10u,1419u,1405u,393228u,
10u,1420u,1u,8u,1419u,262254u,6u,1421u,1420u,196670u,1418u,1421u,262205u,6u,1423u,1414u,
327808u,6u,1424u,1423u,551u,262205u,6u,1425u,142u,327810u,6u,1426u,1425u,551u,458764u,6u,
1427u,1u,39u,1424u,1426u,196670u,1422u,1427u,262205u,6u,1429u,1418u,327808u,6u,1430u,1429u,
551u,262205u,6u,1431u,143u,327810u,6u,1432u,1431u,551u,458764u,6u,1433u,1u,39u,1430u,
1432u,196670u,1428u,1433u,262205u,10u,1435u,1396u,262205u,6u,1436u,1414u,262255u,10u,1437u,1436u,
327811u,10u,1438u,1435u,1437u,196670u,1434u,1438u,262205u,10u,1440u,1405u,262205u,6u,1441u,1418u,
262255u,10u,1442u,1441u,327811u,10u,1443u,1440u,1442u,196670u,1439u,1443u,262205u,6u,1445u,139u,
262205u,6u,1446u,1368u,327808u,6u,1447u,1445u,1446u,393281u,584u,1448u,578u,548u,1447u,262205u,
10u,1449u,1448u,196670u,1444u,1449u,262205u,6u,1451u,1380u,262205u,6u,1452u,142u,327812u,6u,
1453u,1451u,1452u,262205u,6u,1454u,143u,327812u,6u,1455u,1453u,1454u,196670u,1450u,1455u,262205u,
6u,1456u,140u,262205u,6u,1457u,1450u,327808u,6u,1458u,1456u,1457u,262205u,6u,1459u,1414u,
262205u,6u,1460u,143u,327812u,6u,1461u,1459u,1460u,327808u,6u,1462u,1458u,1461u,262205u,6u,
1463u,1418u,327808u,6u,1464u,1462u,1463u,262205u,10u,1465u,1444u,262205u,10u,1466u,1434u,327811u,
10u,1467u,701u,1466u,327813u,10u,1468u,1465u,1467u,262205u,10u,1469u,1439u,327811u,10u,1470u,
701u,1469u,327813u,10u,1471u,1468u,1470u,196670u,1472u,1464u,196670u,1473u,1471u,393273u,2u,1474u,
55u,1472u,1473u,262205u,6u,1475u,140u,262205u,6u,1476u,1450u,327808u,6u,1477u,1475u,1476u,
262205u,6u,1478u,1414u,262205u,6u,1479u,143u,327812u,6u,1480u,1478u,1479u,327808u,6u,1481u,
1477u,1480u,262205u,6u,1482u,1428u,327808u,6u,1483u,1481u,1482u,262205u,10u,1484u,1444u,262205u,
10u,1485u,1434u,327811u,10u,1486u,701u,1485u,327813u,10u,1487u,1484u,1486u,262205u,10u,1488u,
1439u,327813u,10u,1489u,1487u,1488u,196670u,1490u,1483u,196670u,1491u,1489u,393273u,2u,1492u,55u,
1490u,1491u,262205u,6u,1493u,140u,262205u,6u,1494u,1450u,327808u,6u,1495u,1493u,1494u,262205u,
6u,1496u,1422u,262205u,6u,1497u,143u,327812u,6u,1498u,1496u,1497u,327808u,6u,1499u,1495u,
1498u,262205u,6u,1500u,1418u,327808u,6u,1501u,1499u,1500u,262205u,10u,1502u,1444u,262205u,10u,
1503u,1434u,327813u,10u,1504u,1502u,1503u,262205u,10u,1505u,1439u,327811u,10u,1506u,701u,1505u,
327813u,10u,1507u,1504u,1506u,196670u,1508u,1501u,196670u,1509u,1507u,393273u,2u,1510u,55u,1508u,
1509u,262205u,6u,1511u,140u,262205u,6u,1512u,1450u,327808u,6u,1513u,1511u,1512u,262205u,6u,
1514u,1422u,262205u,6u,1515u,143u,327812u,6u,1516u,1514u,1515u,327808u,6u,1517u,1513u,1516u,
262205u,6u,1518u,1428u,327808u,6u,1519u,1517u,1518u,262205u,10u,1520u,1444u,262205u,10u,1521u,
1434u,327813u,10u,1522u,1520u,1521u,262205u,10u,1523u,1439u,327813u,10u,1524u,1522u,1523u,196670u,
1525u,1519u,196670u,1526u,1524u,393273u,2u,1527u,55u,1525u,1526u,65789u,65592u,327734u,2u,151u,
0u,31u,196663u,9u,148u,196663u,9u,149u,196663u,9u,150u,131320u,152u,262203u,9u,1528u,
7u,262205u,6u,1529u,549u,262205u,6u,1530u,550u,327812u,6u,1531u,1529u,1530u,262205u,6u,
1532u,547u,327808u,6u,1533u,1531u,1532u,196670u,1528u,1533u,262205u,6u,1534u,1528u,262205u,6u,
1535u,149u,262205u,6u,1536u,150u,327812u,6u,1537u,1535u,1536u,327857u,57u,1538u,1534u,1537u,
196855u,1540u,0u,262394u,1538u,1539u,1540u,131320u,1539u,262205u,6u,1541u,1528u,262205u,6u,1542u,
150u,327819u,6u,1543u,1541u,1542u,262205u,6u,1544u,1528u,262205u,6u,1545u,150u,327815u,6u,
1546u,1544u,1545u,327853u,57u,1547u,1543u,1546u,131321u,1540u,131320u,1540u,458997u,57u,1548u,1538u,
152u,1547u,1539u,196855u,1550u,0u,262394u,1548u,1549u,1550u,131320u,1549u,262205u,6u,1551u,148u,
262205u,6u,1552u,1528u,327808u,6u,1553u,1551u,1552u,393281u,584u,1555u,578u,548u,1553u,196670u,
1555u,1554u,131321u,1550u,131320u,1550u,65789u,65592u,327734u,2u,160u,0u,153u,196663u,9u,154u,
196663u,9u,155u,196663u,9u,156u,196663u,9u,157u,196663u,9u,158u,196663u,58u,159u,131320u,
161u,262203u,9u,1556u,7u,262203u,9u,1557u,7u,262203u,9u,1559u,7u,262203u,11u,1571u,
7u,262203u,9u,1576u,7u,262203u,11u,1599u,7u,262203u,9u,1601u,7u,262203u,11u,1604u,
7u,262203u,9u,1605u,7u,262203u,11u,1631u,7u,262203u,9u,1633u,7u,262203u,11u,1636u,
7u,262203u,9u,1637u,7u,262203u,9u,1642u,7u,262203u,58u,1661u,7u,262203u,11u,1664u,
7u,262203u,11u,1688u,7u,262203u,9u,1690u,7u,262203u,9u,1693u,7u,262203u,9u,1703u,
7u,262203u,11u,1709u,7u,262203u,11u,1720u,7u,262203u,11u,1736u,7u,262203u,58u,1739u,
7u,262203u,11u,1742u,7u,262203u,58u,1766u,7u,262203u,11u,1769u,7u,196670u,1556u,548u,
262205u,6u,1558u,549u,196670u,1557u,1558u,262205u,6u,1560u,1557u,262205u,6u,1561u,158u,327815u,
6u,1562u,1560u,1561u,262205u,6u,1563u,157u,327812u,6u,1564u,1562u,1563u,262205u,6u,1565u,
158u,327812u,6u,1566u,1564u,1565u,262205u,6u,1567u,1557u,262205u,6u,1568u,158u,327819u,6u,
1569u,1567u,1568u,327808u,6u,1570u,1566u,1569u,196670u,1559u,1570u,262268u,10u,1574u,1573u,262271u,
10u,1575u,1574u,196670u,1571u,1575u,262205u,6u,1577u,547u,196670u,1576u,1577u,131321u,1578u,131320u,
1578u,262390u,1580u,1581u,0u,131321u,1582u,131320u,1582u,262205u,6u,1583u,1576u,262205u,6u,1584u,
157u,327857u,57u,1585u,1583u,1584u,262394u,1585u,1579u,1580u,131320u,1579u,262205u,10u,1586u,1571u,
262205u,6u,1587u,154u,262205u,6u,1588u,1559u,262205u,6u,1589u,1576u,262205u,6u,1590u,158u,
327812u,6u,1591u,1589u,1590u,327808u,6u,1592u,1588u,1591u,327808u,6u,1593u,1587u,1592u,393281u,
584u,1594u,578u,548u,1593u,262205u,10u,1595u,1594u,458764u,10u,1596u,1u,40u,1586u,1595u,
196670u,1571u,1596u,131321u,1581u,131320u,1581u,262205u,6u,1597u,1576u,327808u,6u,1598u,1597u,551u,
196670u,1576u,1598u,131321u,1578u,131320u,1580u,262205u,10u,1600u,1571u,196670u,1599u,1600u,262205u,6u,
1602u,1556u,196670u,1601u,1602u,393273u,10u,1603u,50u,1599u,1601u,196670u,1571u,1603u,196670u,1604u,
630u,262205u,6u,1606u,547u,196670u,1605u,1606u,131321u,1607u,131320u,1607u,262390u,1609u,1610u,0u,
131321u,1611u,131320u,1611u,262205u,6u,1612u,1605u,262205u,6u,1613u,157u,327857u,57u,1614u,1612u,
1613u,262394u,1614u,1608u,1609u,131320u,1608u,262205u,6u,1615u,154u,262205u,6u,1616u,1559u,262205u,
6u,1617u,1605u,262205u,6u,1618u,158u,327812u,6u,1619u,1617u,1618u,327808u,6u,1620u,1616u,
1619u,327808u,6u,1621u,1615u,1620u,393281u,584u,1622u,578u,548u,1621u,262205u,10u,1623u,1622u,
262205u,10u,1624u,1571u,327811u,10u,1625u,1623u,1624u,393228u,10u,1626u,1u,27u,1625u,262205u,
10u,1627u,1604u,327809u,10u,1628u,1627u,1626u,196670u,1604u,1628u,131321u,1610u,131320u,1610u,262205u,
6u,1629u,1605u,327808u,6u,1630u,1629u,551u,196670u,1605u,1630u,131321u,1607u,131320u,1609u,262205u,
10u,1632u,1604u,196670u,1631u,1632u,262205u,6u,1634u,1556u,196670u,1633u,1634u,393273u,10u,1635u,
46u,1631u,1633u,196670u,1604u,1635u,196670u,1636u,630u,262205u,6u,1638u,155u,196670u,1637u,1638u,
327737u,57u,1639u,65u,1637u,196855u,1641u,0u,262394u,1639u,1640u,1641u,131320u,1640u,262205u,6u,
1643u,547u,196670u,1642u,1643u,131321u,1644u,131320u,1644u,262390u,1646u,1647u,0u,131321u,1648u,131320u,
1648u,262205u,6u,1649u,1642u,262205u,6u,1650u,157u,327857u,57u,1651u,1649u,1650u,262394u,1651u,
1645u,1646u,131320u,1645u,262205u,6u,1652u,155u,262205u,6u,1653u,1559u,262205u,6u,1654u,1642u,
262205u,6u,1655u,158u,327812u,6u,1656u,1654u,1655u,327808u,6u,1657u,1653u,1656u,327808u,6u,
1658u,1652u,1657u,393281u,584u,1659u,578u,548u,1658u,262205u,10u,1660u,1659u,262205u,57u,1662u,
159u,196670u,1661u,1662u,327737u,57u,1663u,61u,1661u,196855u,1666u,0u,262394u,1663u,1665u,1667u,
131320u,1665u,196670u,1664u,701u,131321u,1666u,131320u,1667u,262205u,6u,1668u,154u,262205u,6u,1669u,
1559u,262205u,6u,1670u,1642u,262205u,6u,1671u,158u,327812u,6u,1672u,1670u,1671u,327808u,6u,
1673u,1669u,1672u,327808u,6u,1674u,1668u,1673u,393281u,584u,1675u,578u,548u,1674u,262205u,10u,
1676u,1675u,262205u,10u,1677u,1571u,327811u,10u,1678u,1676u,1677u,393228u,10u,1679u,1u,27u,
1678u,262205u,10u,1680u,1604u,327816u,10u,1681u,1679u,1680u,196670u,1664u,1681u,131321u,1666u,131320u,
1666u,262205u,10u,1682u,1664u,327813u,10u,1683u,1660u,1682u,262205u,10u,1684u,1636u,327809u,10u,
1685u,1684u,1683u,196670u,1636u,1685u,131321u,1647u,131320u,1647u,262205u,6u,1686u,1642u,327808u,6u,
1687u,1686u,551u,196670u,1642u,1687u,131321u,1644u,131320u,1646u,262205u,10u,1689u,1636u,196670u,1688u,
1689u,262205u,6u,1691u,1556u,196670u,1690u,1691u,393273u,10u,1692u,46u,1688u,1690u,196670u,1636u,
1692u,131321u,1641u,131320u,1641u,262205u,6u,1694u,547u,196670u,1693u,1694u,131321u,1695u,131320u,1695u,
262390u,1697u,1698u,0u,131321u,1699u,131320u,1699u,262205u,6u,1700u,1693u,262205u,6u,1701u,157u,
327857u,57u,1702u,1700u,1701u,262394u,1702u,1696u,1697u,131320u,1696u,262205u,6u,1704u,1559u,262205u,
6u,1705u,1693u,262205u,6u,1706u,158u,327812u,6u,1707u,1705u,1706u,327808u,6u,1708u,1704u,
1707u,196670u,1703u,1708u,262205u,6u,1710u,154u,262205u,6u,1711u,1703u,327808u,6u,1712u,1710u,
1711u,393281u,584u,1713u,578u,548u,1712u,262205u,10u,1714u,1713u,262205u,10u,1715u,1571u,327811u,
10u,1716u,1714u,1715u,393228u,10u,1717u,1u,27u,1716u,262205u,10u,1718u,1604u,327816u,10u,
1719u,1717u,1718u,196670u,1709u,1719u,262205u,6u,1721u,154u,262205u,6u,1722u,1703u,327808u,6u,
1723u,1721u,1722u,393281u,584u,1724u,578u,548u,1723u,262205u,10u,1725u,1724u,262205u,10u,1726u,
1571u,327811u,10u,1727u,1725u,1726u,262205u,10u,1728u,1604u,393228u,10u,1729u,1u,28u,1728u,
327811u,10u,1730u,1727u,1729u,196670u,1720u,1730u,262205u,6u,1731u,156u,262205u,6u,1732u,1703u,
327808u,6u,1733u,1731u,1732u,262205u,6u,1734u,155u,327851u,57u,1735u,1734u,548u,196855u,1738u,
0u,262394u,1735u,1737u,1765u,131320u,1737u,262205u,57u,1740u,159u,196670u,1739u,1740u,327737u,57u,
1741u,61u,1739u,196855u,1744u,0u,262394u,1741u,1743u,1754u,131320u,1743u,262205u,6u,1745u,155u,
262205u,6u,1746u,1703u,327808u,6u,1747u,1745u,1746u,393281u,584u,1748u,578u,548u,1747u,262205u,
10u,1749u,1748u,262205u,10u,1750u,1720u,262205u,10u,1751u,1636u,327813u,10u,1752u,1750u,1751u,
327811u,10u,1753u,1749u,1752u,196670u,1742u,1753u,131321u,1744u,131320u,1754u,262205u,10u,1755u,1709u,
262205u,6u,1756u,155u,262205u,6u,1757u,1703u,327808u,6u,1758u,1756u,1757u,393281u,584u,1759u,
578u,548u,1758u,262205u,10u,1760u,1759u,262205u,10u,1761u,1636u,327811u,10u,1762u,1760u,1761u,
327813u,10u,1763u,1755u,1762u,196670u,1742u,1763u,131321u,1744u,131320u,1744u,262205u,10u,1764u,1742u,
196670u,1736u,1764u,131321u,1738u,131320u,1765u,262205u,57u,1767u,159u,196670u,1766u,1767u,327737u,57u,
1768u,61u,1766u,196855u,1771u,0u,262394u,1768u,1770u,1782u,131320u,1770u,262205u,6u,1772u,154u,
262205u,6u,1773u,1703u,327808u,6u,1774u,1772u,1773u,393281u,584u,1775u,578u,548u,1774u,262205u,
10u,1776u,1775u,262205u,10u,1777u,1571u,327811u,10u,1778u,1776u,1777u,262205u,10u,1779u,1604u,
393228u,10u,1780u,1u,28u,1779u,327811u,10u,1781u,1778u,1780u,196670u,1769u,1781u,131321u,1771u,
131320u,1782u,262205u,10u,1783u,1709u,196670u,1769u,1783u,131321u,1771u,131320u,1771u,262205u,10u,1784u,
1769u,196670u,1736u,1784u,131321u,1738u,131320u,1738u,262205u,10u,1785u,1736u,393281u,584u,1786u,578u,
548u,1733u,196670u,1786u,1785u,131321u,1698u,131320u,1698u,262205u,6u,1787u,1693u,327808u,6u,1788u,
1787u,551u,196670u,1693u,1788u,131321u,1695u,131320u,1697u,65789u,65592u,327734u,2u,174u,0u,162u,
196663u,9u,163u,196663u,9u,164u,196663u,9u,165u,196663u,9u,166u,196663u,9u,167u,196663u,
9u,168u,196663u,9u,169u,196663u,9u,170u,196663u,9u,171u,196663u,9u,172u,196663u,9u,
173u,131320u,175u,262203u,9u,1789u,7u,262203u,9u,1801u,7u,262203u,9u,1807u,7u,262203u,
9u,1813u,7u,262203u,9u,1819u,7u,262203u,9u,1825u,7u,262203u,9u,1829u,7u,262203u,
9u,1841u,7u,262203u,11u,1869u,7u,262205u,6u,1790u,549u,262205u,6u,1791u,550u,327812u,
6u,1792u,1790u,1791u,262205u,6u,1793u,547u,327808u,6u,1794u,1792u,1793u,196670u,1789u,1794u,
262205u,6u,1795u,1789u,262205u,6u,1796u,165u,327855u,57u,1797u,1795u,1796u,196855u,1799u,0u,
262394u,1797u,1798u,1799u,131320u,1798u,65789u,131320u,1799u,262205u,6u,1802u,1789u,262205u,6u,1803u,
172u,262205u,6u,1804u,173u,327812u,6u,1805u,1803u,1804u,327819u,6u,1806u,1802u,1805u,196670u,
1801u,1806u,262205u,6u,1808u,1789u,262205u,6u,1809u,172u,262205u,6u,1810u,173u,327812u,6u,
1811u,1809u,1810u,327815u,6u,1812u,1808u,1811u,196670u,1807u,1812u,262205u,6u,1814u,1807u,262205u,
6u,1815u,168u,262205u,6u,1816u,168u,327812u,6u,1817u,1815u,1816u,327815u,6u,1818u,1814u,
1817u,196670u,1813u,1818u,262205u,6u,1820u,1807u,262205u,6u,1821u,168u,327815u,6u,1822u,1820u,
1821u,262205u,6u,1823u,168u,327819u,6u,1824u,1822u,1823u,196670u,1819u,1824u,262205u,6u,1826u,
1807u,262205u,6u,1827u,168u,327819u,6u,1828u,1826u,1827u,196670u,1825u,1828u,262205u,6u,1830u,
1801u,262205u,6u,1831u,173u,327815u,6u,1832u,1830u,1831u,262205u,6u,1833u,169u,327812u,6u,
1834u,1832u,1833u,262205u,6u,1835u,1819u,262205u,6u,1836u,171u,327812u,6u,1837u,1835u,1836u,
327808u,6u,1838u,1834u,1837u,262205u,6u,1839u,170u,327810u,6u,1840u,1838u,1839u,196670u,1829u,
1840u,262205u,6u,1842u,1801u,262205u,6u,1843u,173u,327819u,6u,1844u,1842u,1843u,262205u,6u,
1845u,169u,327812u,6u,1846u,1844u,1845u,262205u,6u,1847u,1825u,262205u,6u,1848u,171u,327812u,
6u,1849u,1847u,1848u,327808u,6u,1850u,1846u,1849u,262205u,6u,1851u,170u,327810u,6u,1852u,
1850u,1851u,196670u,1841u,1852u,262205u,6u,1853u,164u,262205u,6u,1854u,1789u,327808u,6u,1855u,
1853u,1854u,262205u,6u,1856u,1829u,327855u,57u,1857u,1856u,548u,262205u,6u,1858u,1829u,262205u,
6u,1859u,166u,327857u,57u,1860u,1858u,1859u,327847u,57u,1861u,1857u,1860u,262205u,6u,1862u,
1841u,327855u,57u,1863u,1862u,548u,327847u,57u,1864u,1861u,1863u,262205u,6u,1865u,1841u,262205u,
6u,1866u,167u,327857u,57u,1867u,1865u,1866u,327847u,57u,1868u,1864u,1867u,196855u,1871u,0u,
262394u,1868u,1870u,1885u,131320u,1870u,262205u,6u,1872u,163u,262205u,6u,1873u,1813u,262205u,6u,
1874u,166u,327812u,6u,1875u,1873u,1874u,262205u,6u,1876u,1829u,327808u,6u,1877u,1875u,1876u,
262205u,6u,1878u,167u,327812u,6u,1879u,1877u,1878u,262205u,6u,1880u,1841u,327808u,6u,1881u,
1879u,1880u,327808u,6u,1882u,1872u,1881u,393281u,584u,1883u,578u,548u,1882u,262205u,10u,1884u,
1883u,196670u,1869u,1884u,131321u,1871u,131320u,1885u,196670u,1869u,630u,131321u,1871u,131320u,1871u,262205u,
10u,1886u,1869u,393281u,584u,1887u,578u,548u,1855u,196670u,1887u,1886u,65789u,65592u,327734u,2u,
187u,0u,162u,196663u,9u,176u,196663u,9u,177u,196663u,9u,178u,196663u,9u,179u,196663u,
9u,180u,196663u,9u,181u,196663u,9u,182u,196663u,9u,183u,196663u,9u,184u,196663u,9u,
185u,196663u,9u,186u,131320u,188u,262203u,9u,1888u,7u,262203u,9u,1900u,7u,262203u,9u,
1906u,7u,262203u,9u,1912u,7u,262203u,11u,1916u,7u,262203u,9u,1917u,7u,262203u,9u,
1926u,7u,262203u,9u,1935u,7u,262203u,9u,1943u,7u,262203u,9u,1975u,7u,262203u,9u,
1979u,7u,262205u,6u,1889u,549u,262205u,6u,1890u,550u,327812u,6u,1891u,1889u,1890u,262205u,
6u,1892u,547u,327808u,6u,1893u,1891u,1892u,196670u,1888u,1893u,262205u,6u,1894u,1888u,262205u,
6u,1895u,178u,327855u,57u,1896u,1894u,1895u,196855u,1898u,0u,262394u,1896u,1897u,1898u,131320u,
1897u,65789u,131320u,1898u,262205u,6u,1901u,1888u,262205u,6u,1902u,179u,262205u,6u,1903u,180u,
327812u,6u,1904u,1902u,1903u,327815u,6u,1905u,1901u,1904u,196670u,1900u,1905u,262205u,6u,1907u,
1888u,262205u,6u,1908u,180u,327815u,6u,1909u,1907u,1908u,262205u,6u,1910u,179u,327819u,6u,
1911u,1909u,1910u,196670u,1906u,1911u,262205u,6u,1913u,1888u,262205u,6u,1914u,180u,327819u,6u,
1915u,1913u,1914u,196670u,1912u,1915u,196670u,1916u,630u,196670u,1917u,548u,131321u,1918u,131320u,1918u,
262390u,1920u,1921u,0u,131321u,1922u,131320u,1922u,262205u,6u,1923u,1917u,262205u,6u,1924u,181u,
327857u,57u,1925u,1923u,1924u,262394u,1925u,1919u,1920u,131320u,1919u,196670u,1926u,548u,131321u,1927u,
131320u,1927u,262390u,1929u,1930u,0u,131321u,1931u,131320u,1931u,262205u,6u,1932u,1926u,262205u,6u,
1933u,181u,327857u,57u,1934u,1932u,1933u,262394u,1934u,1928u,1929u,131320u,1928u,262205u,6u,1936u,
1906u,262205u,6u,1937u,183u,327808u,6u,1938u,1936u,1937u,262205u,6u,1939u,1917u,262205u,6u,
1940u,184u,327812u,6u,1941u,1939u,1940u,327810u,6u,1942u,1938u,1941u,196670u,1935u,1942u,262205u,
6u,1944u,1912u,262205u,6u,1945u,183u,327808u,6u,1946u,1944u,1945u,262205u,6u,1947u,1926u,
262205u,6u,1948u,184u,327812u,6u,1949u,1947u,1948u,327810u,6u,1950u,1946u,1949u,196670u,1943u,
1950u,262205u,6u,1951u,1935u,327857u,57u,1952u,1951u,548u,262205u,6u,1953u,1943u,327857u,57u,
1954u,1953u,548u,327846u,57u,1955u,1952u,1954u,262312u,57u,1956u,1955u,196855u,1958u,0u,262394u,
1956u,1957u,1958u,131320u,1957u,262205u,6u,1959u,1935u,262205u,6u,1960u,182u,327819u,6u,1961u,
1959u,1960u,327851u,57u,1962u,1961u,548u,131321u,1958u,131320u,1958u,458997u,57u,1963u,1955u,1928u,
1962u,1957u,262312u,57u,1964u,1963u,196855u,1966u,0u,262394u,1964u,1965u,1966u,131320u,1965u,262205u,
6u,1967u,1943u,262205u,6u,1968u,182u,327819u,6u,1969u,1967u,1968u,327851u,57u,1970u,1969u,
548u,131321u,1966u,131320u,1966u,458997u,57u,1971u,1963u,1958u,1970u,1965u,196855u,1973u,0u,262394u,
1971u,1972u,1973u,131320u,1972u,131321u,1930u,131320u,1973u,262205u,6u,1976u,1935u,262205u,6u,1977u,
182u,327815u,6u,1978u,1976u,1977u,196670u,1975u,1978u,262205u,6u,1980u,1943u,262205u,6u,1981u,
182u,327815u,6u,1982u,1980u,1981u,196670u,1979u,1982u,262205u,6u,1983u,1975u,262205u,6u,1984u,
185u,327857u,57u,1985u,1983u,1984u,262205u,6u,1986u,1979u,262205u,6u,1987u,186u,327857u,57u,
1988u,1986u,1987u,327847u,57u,1989u,1985u,1988u,196855u,1991u,0u,262394u,1989u,1990u,1991u,131320u,
1990u,262205u,6u,1992u,176u,262205u,6u,1993u,1900u,262205u,6u,1994u,181u,327812u,6u,1995u,
1993u,1994u,262205u,6u,1996u,1917u,327808u,6u,1997u,1995u,1996u,262205u,6u,1998u,181u,327812u,
6u,1999u,1997u,1998u,262205u,6u,2000u,1926u,327808u,6u,2001u,1999u,2000u,262205u,6u,2002u,
185u,327812u,6u,2003u,2001u,2002u,262205u,6u,2004u,186u,327812u,6u,2005u,2003u,2004u,262205u,
6u,2006u,1975u,262205u,6u,2007u,186u,327812u,6u,2008u,2006u,2007u,327808u,6u,2009u,2005u,
2008u,262205u,6u,2010u,1979u,327808u,6u,2011u,2009u,2010u,327808u,6u,2012u,1992u,2011u,393281u,
584u,2013u,578u,548u,2012u,262205u,10u,2014u,2013u,262205u,10u,2015u,1916u,327809u,10u,2016u,
2015u,2014u,196670u,1916u,2016u,131321u,1991u,131320u,1991u,131321u,1930u,131320u,1930u,262205u,6u,2017u,
1926u,327808u,6u,2018u,2017u,551u,196670u,1926u,2018u,131321u,1927u,131320u,1929u,131321u,1921u,131320u,
1921u,262205u,6u,2019u,1917u,327808u,6u,2020u,2019u,551u,196670u,1917u,2020u,131321u,1918u,131320u,
1920u,262205u,6u,2021u,177u,262205u,6u,2022u,1888u,327808u,6u,2023u,2021u,2022u,262205u,10u,
2024u,1916u,393281u,584u,2025u,578u,548u,2023u,196670u,2025u,2024u,65789u,65592u,327734u,2u,199u,
0u,189u,196663u,9u,190u,196663u,9u,191u,196663u,9u,192u,196663u,9u,193u,196663u,9u,
194u,196663u,9u,195u,196663u,11u,196u,196663u,58u,197u,196663u,58u,198u,131320u,200u,262203u,
9u,2026u,7u,262203u,9u,2027u,7u,262203u,11u,2031u,7u,262203u,58u,2032u,7u,262203u,
9u,2038u,7u,262203u,11u,2059u,7u,262203u,58u,2060u,7u,262203u,11u,2063u,7u,262203u,
11u,2067u,7u,262203u,9u,2069u,7u,262203u,11u,2076u,7u,262203u,9u,2077u,7u,262203u,
11u,2087u,7u,262203u,11u,2104u,7u,262203u,11u,2105u,7u,262203u,9u,2107u,7u,262203u,
11u,2115u,7u,262203u,9u,2117u,7u,262203u,9u,2123u,7u,262203u,11u,2133u,7u,262203u,
11u,2153u,7u,262203u,58u,2164u,7u,262203u,11u,2167u,7u,262203u,11u,2184u,7u,262203u,
11u,2185u,7u,262203u,9u,2186u,7u,262203u,11u,2196u,7u,262203u,11u,2208u,7u,262203u,
11u,2218u,7u,262203u,11u,2246u,7u,262203u,11u,2247u,7u,262203u,9u,2249u,7u,262203u,
11u,2255u,7u,262203u,11u,2256u,7u,262203u,9u,2258u,7u,262203u,9u,2264u,7u,262203u,
11u,2274u,7u,262203u,11u,2284u,7u,262203u,58u,2302u,7u,196670u,2026u,548u,262205u,6u,
2028u,549u,262205u,6u,2029u,195u,327812u,6u,2030u,2028u,2029u,196670u,2027u,2030u,196670u,2031u,
630u,262205u,57u,2033u,197u,196670u,2032u,2033u,327737u,57u,2034u,61u,2032u,262312u,57u,2035u,
2034u,196855u,2037u,0u,262394u,2035u,2036u,2037u,131320u,2036u,262205u,6u,2039u,547u,196670u,2038u,
2039u,131321u,2040u,131320u,2040u,262390u,2042u,2043u,0u,131321u,2044u,131320u,2044u,262205u,6u,2045u,
2038u,262205u,6u,2046u,195u,327857u,57u,2047u,2045u,2046u,262394u,2047u,2041u,2042u,131320u,2041u,
262205u,6u,2048u,190u,262205u,6u,2049u,2027u,262205u,6u,2050u,2038u,327808u,6u,2051u,2049u,
2050u,327808u,6u,2052u,2048u,2051u,393281u,584u,2053u,578u,548u,2052u,262205u,10u,2054u,2053u,
262205u,10u,2055u,2031u,327809u,10u,2056u,2055u,2054u,196670u,2031u,2056u,131321u,2043u,131320u,2043u,
262205u,6u,2057u,2038u,327808u,6u,2058u,2057u,551u,196670u,2038u,2058u,131321u,2040u,131320u,2042u,
131321u,2037u,131320u,2037u,262205u,57u,2061u,197u,196670u,2060u,2061u,327737u,57u,2062u,61u,2060u,
196855u,2065u,0u,262394u,2062u,2064u,2066u,131320u,2064u,196670u,2063u,630u,131321u,2065u,131320u,2066u,
262205u,10u,2068u,2031u,196670u,2067u,2068u,262205u,6u,2070u,2026u,196670u,2069u,2070u,393273u,10u,
2071u,46u,2067u,2069u,262205u,6u,2072u,195u,262255u,10u,2073u,2072u,327816u,10u,2074u,2071u,
2073u,196670u,2063u,2074u,131321u,2065u,131320u,2065u,262205u,10u,2075u,2063u,196670u,2059u,2075u,196670u,
2076u,630u,262205u,6u,2078u,547u,196670u,2077u,2078u,131321u,2079u,131320u,2079u,262390u,2081u,2082u,
0u,131321u,2083u,131320u,2083u,262205u,6u,2084u,2077u,262205u,6u,2085u,195u,327857u,57u,2086u,
2084u,2085u,262394u,2086u,2080u,2081u,131320u,2080u,262205u,6u,2088u,190u,262205u,6u,2089u,2027u,
262205u,6u,2090u,2077u,327808u,6u,2091u,2089u,2090u,327808u,6u,2092u,2088u,2091u,393281u,584u,
2093u,578u,548u,2092u,262205u,10u,2094u,2093u,262205u,10u,2095u,2059u,327811u,10u,2096u,2094u,
2095u,196670u,2087u,2096u,262205u,10u,2097u,2087u,262205u,10u,2098u,2087u,327813u,10u,2099u,2097u,
2098u,262205u,10u,2100u,2076u,327809u,10u,2101u,2100u,2099u,196670u,2076u,2101u,131321u,2082u,131320u,
2082u,262205u,6u,2102u,2077u,327808u,6u,2103u,2102u,551u,196670u,2077u,2103u,131321u,2079u,131320u,
2081u,262205u,10u,2106u,2076u,196670u,2105u,2106u,262205u,6u,2108u,2026u,196670u,2107u,2108u,393273u,
10u,2109u,46u,2105u,2107u,262205u,6u,2110u,195u,262255u,10u,2111u,2110u,327816u,10u,2112u,
2109u,2111u,262205u,10u,2113u,196u,327809u,10u,2114u,2112u,2113u,196670u,2115u,2114u,327737u,10u,
2116u,76u,2115u,196670u,2104u,2116u,262205u,6u,2118u,192u,196670u,2117u,2118u,327737u,57u,2119u,
65u,2117u,262312u,57u,2120u,2119u,196855u,2122u,0u,262394u,2120u,2121u,2122u,131320u,2121u,262205u,
6u,2124u,547u,196670u,2123u,2124u,131321u,2125u,131320u,2125u,262390u,2127u,2128u,0u,131321u,2129u,
131320u,2129u,262205u,6u,2130u,2123u,262205u,6u,2131u,195u,327857u,57u,2132u,2130u,2131u,262394u,
2132u,2126u,2127u,131320u,2126u,262205u,6u,2134u,190u,262205u,6u,2135u,2027u,262205u,6u,2136u,
2123u,327808u,6u,2137u,2135u,2136u,327808u,6u,2138u,2134u,2137u,393281u,584u,2139u,578u,548u,
2138u,262205u,10u,2140u,2139u,262205u,10u,2141u,2059u,327811u,10u,2142u,2140u,2141u,262205u,10u,
2143u,2104u,327813u,10u,2144u,2142u,2143u,196670u,2133u,2144u,262205u,6u,2145u,193u,262205u,6u,
2146u,2027u,262205u,6u,2147u,2123u,327808u,6u,2148u,2146u,2147u,327808u,6u,2149u,2145u,2148u,
262205u,10u,2150u,2133u,262205u,6u,2151u,191u,327851u,57u,2152u,2151u,548u,196855u,2155u,0u,
262394u,2152u,2154u,2161u,131320u,2154u,262205u,6u,2156u,191u,262205u,6u,2157u,2123u,327808u,6u,
2158u,2156u,2157u,393281u,584u,2159u,578u,548u,2158u,262205u,10u,2160u,2159u,196670u,2153u,2160u,
131321u,2155u,131320u,2161u,196670u,2153u,701u,131321u,2155u,131320u,2155u,262205u,10u,2162u,2153u,327813u,
10u,2163u,2150u,2162u,262205u,57u,2165u,198u,196670u,2164u,2165u,327737u,57u,2166u,61u,2164u,
196855u,2169u,0u,262394u,2166u,2168u,2177u,131320u,2168u,262205u,6u,2170u,191u,262205u,6u,2171u,
195u,262205u,6u,2172u,2123u,327808u,6u,2173u,2171u,2172u,327808u,6u,2174u,2170u,2173u,393281u,
584u,2175u,578u,548u,2174u,262205u,10u,2176u,2175u,196670u,2167u,2176u,131321u,2169u,131320u,2177u,
196670u,2167u,630u,131321u,2169u,131320u,2169u,262205u,10u,2178u,2167u,327809u,10u,2179u,2163u,2178u,
393281u,584u,2180u,578u,548u,2149u,196670u,2180u,2179u,131321u,2128u,131320u,2128u,262205u,6u,2181u,
2123u,327808u,6u,2182u,2181u,551u,196670u,2123u,2182u,131321u,2125u,131320u,2127u,65789u,131320u,2122u,
196670u,2184u,630u,196670u,2185u,630u,262205u,6u,2187u,547u,196670u,2186u,2187u,131321u,2188u,131320u,
2188u,262390u,2190u,2191u,0u,131321u,2192u,131320u,2192u,262205u,6u,2193u,2186u,262205u,6u,2194u,
195u,327857u,57u,2195u,2193u,2194u,262394u,2195u,2189u,2190u,131320u,2189u,262205u,6u,2197u,190u,
262205u,6u,2198u,2027u,262205u,6u,2199u,2186u,327808u,6u,2200u,2198u,2199u,327808u,6u,2201u,
2197u,2200u,393281u,584u,2202u,578u,548u,2201u,262205u,10u,2203u,2202u,262205u,10u,2204u,2059u,
327811u,10u,2205u,2203u,2204u,262205u,10u,2206u,2104u,327813u,10u,2207u,2205u,2206u,196670u,2196u,
2207u,262205u,6u,2209u,192u,262205u,6u,2210u,2027u,262205u,6u,2211u,2186u,327808u,6u,2212u,
2210u,2211u,327808u,6u,2213u,2209u,2212u,393281u,584u,2214u,578u,548u,2213u,262205u,10u,2215u,
2214u,262205u,6u,2216u,191u,327851u,57u,2217u,2216u,548u,196855u,2220u,0u,262394u,2217u,2219u,
2226u,131320u,2219u,262205u,6u,2221u,191u,262205u,6u,2222u,2186u,327808u,6u,2223u,2221u,2222u,
393281u,584u,2224u,578u,548u,2223u,262205u,10u,2225u,2224u,196670u,2218u,2225u,131321u,2220u,131320u,
2226u,196670u,2218u,701u,131321u,2220u,131320u,2220u,262205u,10u,2227u,2218u,327813u,10u,2228u,2215u,
2227u,196670u,2208u,2228u,262205u,6u,2229u,194u,262205u,6u,2230u,2027u,262205u,6u,2231u,2186u,
327808u,6u,2232u,2230u,2231u,327808u,6u,2233u,2229u,2232u,262205u,10u,2234u,2196u,393281u,584u,
2235u,578u,548u,2233u,196670u,2235u,2234u,262205u,10u,2236u,2208u,262205u,10u,2237u,2184u,327809u,
10u,2238u,2237u,2236u,196670u,2184u,2238u,262205u,10u,2239u,2208u,262205u,10u,2240u,2196u,327813u,
10u,2241u,2239u,2240u,262205u,10u,2242u,2185u,327809u,10u,2243u,2242u,2241u,196670u,2185u,2243u,
131321u,2191u,131320u,2191u,262205u,6u,2244u,2186u,327808u,6u,2245u,2244u,551u,196670u,2186u,2245u,
131321u,2188u,131320u,2190u,262205u,10u,2248u,2184u,196670u,2247u,2248u,262205u,6u,2250u,2026u,196670u,
2249u,2250u,393273u,10u,2251u,46u,2247u,2249u,262205u,6u,2252u,195u,262255u,10u,2253u,2252u,
327816u,10u,2254u,2251u,2253u,196670u,2246u,2254u,262205u,10u,2257u,2185u,196670u,2256u,2257u,262205u,
6u,2259u,2026u,196670u,2258u,2259u,393273u,10u,2260u,46u,2256u,2258u,262205u,6u,2261u,195u,
262255u,10u,2262u,2261u,327816u,10u,2263u,2260u,2262u,196670u,2255u,2263u,262205u,6u,2265u,547u,
196670u,2264u,2265u,131321u,2266u,131320u,2266u,262390u,2268u,2269u,0u,131321u,2270u,131320u,2270u,262205u,
6u,2271u,2264u,262205u,6u,2272u,195u,327857u,57u,2273u,2271u,2272u,262394u,2273u,2267u,2268u,
131320u,2267u,262205u,6u,2275u,192u,262205u,6u,2276u,2027u,262205u,6u,2277u,2264u,327808u,6u,
2278u,2276u,2277u,327808u,6u,2279u,2275u,2278u,393281u,584u,2280u,578u,548u,2279u,262205u,10u,
2281u,2280u,262205u,6u,2282u,191u,327851u,57u,2283u,2282u,548u,196855u,2286u,0u,262394u,2283u,
2285u,2292u,131320u,2285u,262205u,6u,2287u,191u,262205u,6u,2288u,2264u,327808u,6u,2289u,2287u,
2288u,393281u,584u,2290u,578u,548u,2289u,262205u,10u,2291u,2290u,196670u,2284u,2291u,131321u,2286u,
131320u,2292u,196670u,2284u,701u,131321u,2286u,131320u,2286u,262205u,10u,2293u,2284u,327813u,10u,2294u,
2281u,2293u,196670u,2274u,2294u,262205u,6u,2295u,193u,262205u,6u,2296u,2027u,262205u,6u,2297u,
2264u,327808u,6u,2298u,2296u,2297u,327808u,6u,2299u,2295u,2298u,262205u,10u,2300u,2104u,262205u,
10u,2301u,2274u,262205u,57u,2303u,197u,196670u,2302u,2303u,327737u,57u,2304u,61u,2302u,262205u,
10u,2305u,2246u,393385u,10u,2306u,2304u,630u,2305u,327811u,10u,2307u,2301u,2306u,262205u,6u,
2308u,194u,262205u,6u,2309u,2027u,262205u,6u,2310u,2264u,327808u,6u,2311u,2309u,2310u,327808u,
6u,2312u,2308u,2311u,393281u,584u,2313u,578u,548u,2312u,262205u,10u,2314u,2313u,262205u,10u,
2315u,2255u,327813u,10u,2316u,2314u,2315u,327811u,10u,2317u,2307u,2316u,327813u,10u,2318u,2300u,
2317u,393281u,584u,2319u,578u,548u,2299u,196670u,2319u,2318u,131321u,2269u,131320u,2269u,262205u,6u,
2320u,2264u,327808u,6u,2321u,2320u,551u,196670u,2264u,2321u,131321u,2266u,131320u,2268u,65789u,65592u,
327734u,2u,208u,0u,201u,196663u,9u,202u,196663u,9u,203u,196663u,9u,204u,196663u,9u,
205u,196663u,9u,206u,196663u,9u,207u,131320u,209u,262203u,9u,2322u,7u,262203u,11u,2334u,
7u,262203u,11u,2335u,7u,262203u,9u,2336u,7u,262203u,9u,2345u,7u,262203u,9u,2378u,
7u,262205u,6u,2323u,549u,262205u,6u,2324u,550u,327812u,6u,2325u,2323u,2324u,262205u,6u,
2326u,547u,327808u,6u,2327u,2325u,2326u,196670u,2322u,2327u,262205u,6u,2328u,2322u,262205u,6u,
2329u,207u,327855u,57u,2330u,2328u,2329u,196855u,2332u,0u,262394u,2330u,2331u,2332u,131320u,2331u,
65789u,131320u,2332u,196670u,2334u,630u,196670u,2335u,630u,196670u,2336u,548u,131321u,2337u,131320u,2337u,
262390u,2339u,2340u,0u,131321u,2341u,131320u,2341u,262205u,6u,2342u,2336u,262205u,6u,2343u,206u,
327857u,57u,2344u,2342u,2343u,262394u,2344u,2338u,2339u,131320u,2338u,262205u,6u,2346u,2336u,262205u,
6u,2347u,207u,327812u,6u,2348u,2346u,2347u,262205u,6u,2349u,2322u,327808u,6u,2350u,2348u,
2349u,196670u,2345u,2350u,262205u,6u,2351u,202u,262205u,6u,2352u,2345u,327808u,6u,2353u,2351u,
2352u,393281u,584u,2354u,578u,548u,2353u,262205u,10u,2355u,2354u,262205u,6u,2356u,203u,262205u,
6u,2357u,2345u,327808u,6u,2358u,2356u,2357u,393281u,584u,2359u,578u,548u,2358u,262205u,10u,
2360u,2359u,327813u,10u,2361u,2355u,2360u,262205u,10u,2362u,2334u,327809u,10u,2363u,2362u,2361u,
196670u,2334u,2363u,262205u,6u,2364u,202u,262205u,6u,2365u,2345u,327808u,6u,2366u,2364u,2365u,
393281u,584u,2367u,578u,548u,2366u,262205u,10u,2368u,2367u,262205u,10u,2369u,2335u,327809u,10u,
2370u,2369u,2368u,196670u,2335u,2370u,131321u,2340u,131320u,2340u,262205u,6u,2371u,2336u,327808u,6u,
2372u,2371u,551u,196670u,2336u,2372u,131321u,2337u,131320u,2339u,262205u,6u,2373u,204u,262205u,6u,
2374u,2322u,327808u,6u,2375u,2373u,2374u,262205u,10u,2376u,2334u,393281u,584u,2377u,578u,548u,
2375u,196670u,2377u,2376u,262205u,6u,2379u,205u,196670u,2378u,2379u,327737u,57u,2380u,65u,2378u,
196855u,2382u,0u,262394u,2380u,2381u,2382u,131320u,2381u,262205u,6u,2383u,205u,262205u,6u,2384u,
2322u,327808u,6u,2385u,2383u,2384u,262205u,10u,2386u,2335u,393281u,584u,2387u,578u,548u,2385u,
196670u,2387u,2386u,131321u,2382u,131320u,2382u,65789u,65592u,327734u,2u,223u,0u,212u,196663u,9u,
213u,196663u,9u,214u,196663u,9u,215u,196663u,9u,216u,196663u,9u,217u,196663u,9u,218u,
196663u,9u,219u,196663u,9u,220u,196663u,211u,221u,196663u,9u,222u,131320u,224u,262203u,9u,
2388u,7u,262203u,9u,2416u,7u,262203u,9u,2441u,7u,262203u,9u,2446u,7u,262203u,9u,
2457u,7u,262203u,9u,2462u,7u,262203u,11u,2473u,7u,262203u,9u,2474u,7u,262203u,11u,
2477u,7u,262203u,11u,2487u,7u,262203u,9u,2494u,7u,262203u,9u,2500u,7u,262203u,9u,
2504u,7u,262203u,11u,2527u,7u,262203u,11u,2541u,7u,262203u,9u,2548u,7u,262203u,9u,
2558u,7u,262203u,11u,2564u,7u,262203u,9u,2588u,7u,262203u,9u,2595u,7u,262203u,9u,
2603u,7u,262203u,11u,2604u,7u,262203u,11u,2623u,7u,262203u,11u,2628u,7u,262203u,9u,
2630u,7u,262203u,9u,2638u,7u,262203u,9u,2652u,7u,262203u,11u,2653u,7u,262203u,9u,
2668u,7u,262203u,9u,2678u,7u,262203u,9u,2685u,7u,262203u,9u,2704u,7u,262203u,11u,
2705u,7u,262203u,9u,2721u,7u,262203u,11u,2722u,7u,262203u,9u,2745u,7u,262203u,58u,
2755u,7u,262203u,9u,2761u,7u,262203u,9u,2778u,7u,262203u,11u,2779u,7u,262203u,9u,
2797u,7u,262203u,9u,2807u,7u,262203u,9u,2817u,7u,262203u,9u,2826u,7u,262203u,9u,
2833u,7u,262203u,9u,2840u,7u,262203u,9u,2861u,7u,262203u,11u,2862u,7u,262203u,9u,
2880u,7u,262203u,11u,2881u,7u,262203u,9u,2899u,7u,262203u,11u,2900u,7u,262203u,9u,
2926u,7u,262205u,6u,2389u,549u,262205u,6u,2390u,550u,327812u,6u,2391u,2389u,2390u,262205u,
6u,2392u,547u,327808u,6u,2393u,2391u,2392u,196670u,2388u,2393u,262205u,6u,2394u,2388u,327745u,
9u,2396u,221u,2395u,262205u,6u,2397u,2396u,327855u,57u,2398u,2394u,2397u,196855u,2400u,0u,
262394u,2398u,2399u,2400u,131320u,2399u,65789u,131320u,2400u,327745u,9u,2402u,221u,548u,262205u,6u,
2403u,2402u,327850u,57u,2405u,2403u,2404u,196855u,2407u,0u,262394u,2405u,2406u,2407u,131320u,2406u,
262205u,6u,2408u,2388u,327745u,9u,2410u,221u,2409u,262205u,6u,2411u,2410u,327857u,57u,2412u,
2408u,2411u,131321u,2407u,131320u,2407u,458997u,57u,2413u,2405u,2400u,2412u,2406u,196855u,2415u,0u,
262394u,2413u,2414u,2415u,131320u,2414u,262205u,6u,2417u,216u,196670u,2416u,2417u,327737u,57u,2418u,
65u,2416u,196855u,2420u,0u,262394u,2418u,2419u,2430u,131320u,2419u,262205u,6u,2421u,218u,262205u,
6u,2422u,2388u,327808u,6u,2423u,2421u,2422u,262205u,6u,2424u,216u,262205u,6u,2425u,2388u,
327808u,6u,2426u,2424u,2425u,393281u,584u,2427u,578u,548u,2426u,262205u,10u,2428u,2427u,393281u,
584u,2429u,578u,548u,2423u,196670u,2429u,2428u,131321u,2420u,131320u,2430u,262205u,6u,2431u,217u,
262205u,6u,2432u,2388u,327808u,6u,2433u,2431u,2432u,262205u,6u,2434u,213u,262205u,6u,2435u,
2388u,327808u,6u,2436u,2434u,2435u,393281u,584u,2437u,578u,548u,2436u,262205u,10u,2438u,2437u,
393281u,584u,2439u,578u,548u,2433u,196670u,2439u,2438u,131321u,2420u,131320u,2420u,65789u,131320u,2415u,
262205u,6u,2442u,2388u,327745u,9u,2443u,221u,548u,262205u,6u,2444u,2443u,327850u,57u,2445u,
2444u,2404u,196855u,2448u,0u,262394u,2445u,2447u,2451u,131320u,2447u,327745u,9u,2449u,221u,2409u,
262205u,6u,2450u,2449u,196670u,2446u,2450u,131321u,2448u,131320u,2451u,196670u,2446u,548u,131321u,2448u,
131320u,2448u,262205u,6u,2452u,2446u,327810u,6u,2453u,2442u,2452u,327745u,9u,2454u,221u,551u,
262205u,6u,2455u,2454u,327819u,6u,2456u,2453u,2455u,196670u,2441u,2456u,262205u,6u,2458u,2388u,
327745u,9u,2459u,221u,548u,262205u,6u,2460u,2459u,327850u,57u,2461u,2460u,2404u,196855u,2464u,
0u,262394u,2461u,2463u,2467u,131320u,2463u,327745u,9u,2465u,221u,2409u,262205u,6u,2466u,2465u,
196670u,2462u,2466u,131321u,2464u,131320u,2467u,196670u,2462u,548u,131321u,2464u,131320u,2464u,262205u,6u,
2468u,2462u,327810u,6u,2469u,2458u,2468u,327745u,9u,2470u,221u,551u,262205u,6u,2471u,2470u,
327815u,6u,2472u,2469u,2471u,196670u,2457u,2472u,262205u,6u,2475u,216u,196670u,2474u,2475u,327737u,
57u,2476u,65u,2474u,196855u,2479u,0u,262394u,2476u,2478u,2485u,131320u,2478u,262205u,6u,2480u,
216u,262205u,6u,2481u,2388u,327808u,6u,2482u,2480u,2481u,393281u,584u,2483u,578u,548u,2482u,
262205u,10u,2484u,2483u,196670u,2477u,2484u,131321u,2479u,131320u,2485u,196670u,2477u,630u,131321u,2479u,
131320u,2479u,262205u,10u,2486u,2477u,196670u,2473u,2486u,196670u,2487u,630u,327745u,9u,2488u,221u,
548u,262205u,6u,2489u,2488u,327859u,57u,2491u,2489u,2490u,196855u,2493u,0u,262394u,2491u,2492u,
2617u,131320u,2492u,327745u,9u,2495u,221u,548u,262205u,6u,2496u,2495u,327850u,57u,2497u,2496u,
551u,262205u,6u,2498u,2457u,393385u,6u,2499u,2497u,2498u,548u,196670u,2494u,2499u,327745u,9u,
2501u,221u,548u,262205u,6u,2502u,2501u,327850u,57u,2503u,2502u,551u,196855u,2506u,0u,262394u,
2503u,2505u,2509u,131320u,2505u,262205u,6u,2507u,2457u,327808u,6u,2508u,2507u,551u,196670u,2504u,
2508u,131321u,2506u,131320u,2509u,327745u,9u,2511u,221u,2510u,262205u,6u,2512u,2511u,196670u,2504u,
2512u,131321u,2506u,131320u,2506u,262205u,6u,2513u,2504u,196670u,2500u,2513u,327745u,9u,2514u,221u,
548u,262205u,6u,2515u,2514u,327850u,57u,2516u,2515u,2490u,262205u,6u,2517u,222u,327851u,57u,
2518u,2517u,548u,327847u,57u,2519u,2516u,2518u,196855u,2521u,0u,262394u,2519u,2520u,2521u,131320u,
2520u,262205u,6u,2522u,214u,262205u,6u,2523u,2457u,327808u,6u,2524u,2522u,2523u,393281u,584u,
2525u,578u,548u,2524u,262205u,10u,2526u,2525u,196670u,2527u,2526u,327737u,10u,2528u,73u,2527u,
262254u,6u,2529u,2528u,327745u,9u,2530u,221u,2510u,262205u,6u,2531u,2530u,458764u,6u,2532u,
1u,39u,2529u,2531u,458764u,6u,2533u,1u,42u,548u,2532u,196670u,2494u,2533u,262205u,6u,
2534u,2494u,262205u,6u,2535u,214u,262205u,6u,2536u,2457u,327808u,6u,2537u,2536u,551u,327808u,
6u,2538u,2535u,2537u,393281u,584u,2539u,578u,548u,2538u,262205u,10u,2540u,2539u,196670u,2541u,
2540u,327737u,10u,2542u,73u,2541u,262254u,6u,2543u,2542u,327745u,9u,2544u,221u,2510u,262205u,
6u,2545u,2544u,458764u,6u,2546u,1u,39u,2543u,2545u,458764u,6u,2547u,1u,42u,2534u,
2546u,196670u,2500u,2547u,131321u,2521u,131320u,2521u,262205u,6u,2549u,2494u,196670u,2548u,2549u,131321u,
2550u,131320u,2550u,262390u,2552u,2553u,0u,131321u,2554u,131320u,2554u,262205u,6u,2555u,2548u,262205u,
6u,2556u,2500u,327857u,57u,2557u,2555u,2556u,262394u,2557u,2551u,2552u,131320u,2551u,262205u,6u,
2559u,213u,262205u,6u,2560u,2548u,327808u,6u,2561u,2559u,2560u,393281u,584u,2562u,578u,548u,
2561u,262205u,10u,2563u,2562u,196670u,2564u,2563u,327737u,10u,2565u,73u,2564u,262254u,6u,2566u,
2565u,196670u,2558u,2566u,262205u,6u,2567u,2558u,327857u,57u,2568u,2567u,548u,262312u,57u,2569u,
2568u,196855u,2571u,0u,262394u,2569u,2570u,2571u,131320u,2570u,262205u,6u,2572u,2558u,327745u,9u,
2573u,221u,2490u,262205u,6u,2574u,2573u,327855u,57u,2575u,2572u,2574u,131321u,2571u,131320u,2571u,
458997u,57u,2576u,2568u,2551u,2575u,2570u,262312u,57u,2577u,2576u,196855u,2579u,0u,262394u,2577u,
2578u,2579u,131320u,2578u,262205u,6u,2580u,2558u,327745u,9u,2581u,221u,2404u,262205u,6u,2582u,
2581u,327850u,57u,2583u,2580u,2582u,131321u,2579u,131320u,2579u,458997u,57u,2584u,2576u,2571u,2583u,
2578u,196855u,2586u,0u,262394u,2584u,2585u,2586u,131320u,2585u,131321u,2553u,131320u,2586u,262205u,6u,
2589u,2558u,327745u,9u,2590u,221u,551u,262205u,6u,2591u,2590u,327812u,6u,2592u,2589u,2591u,
262205u,6u,2593u,2441u,327808u,6u,2594u,2592u,2593u,196670u,2588u,2594u,262205u,6u,2596u,216u,
196670u,2595u,2596u,327737u,57u,2597u,65u,2595u,196855u,2599u,0u,262394u,2597u,2598u,2607u,131320u,
2598u,262205u,6u,2600u,220u,262205u,6u,2601u,2588u,327808u,6u,2602u,2600u,2601u,196670u,2603u,
2602u,262205u,10u,2605u,2473u,196670u,2604u,2605u,393273u,2u,2606u,55u,2603u,2604u,131321u,2599u,
131320u,2607u,262205u,6u,2608u,215u,262205u,6u,2609u,2588u,327808u,6u,2610u,2608u,2609u,393281u,
584u,2611u,578u,548u,2610u,262205u,10u,2612u,2611u,262205u,10u,2613u,2487u,327809u,10u,2614u,
2613u,2612u,196670u,2487u,2614u,131321u,2599u,131320u,2599u,131321u,2553u,131320u,2553u,262205u,6u,2615u,
2548u,327808u,6u,2616u,2615u,551u,196670u,2548u,2616u,131321u,2550u,131320u,2552u,131321u,2493u,131320u,
2617u,327745u,9u,2618u,221u,548u,262205u,6u,2619u,2618u,327850u,57u,2620u,2619u,2404u,196855u,
2622u,0u,262394u,2620u,2621u,2744u,131320u,2621u,327745u,9u,2625u,221u,2624u,262205u,6u,2626u,
2625u,262255u,10u,2627u,2626u,196670u,2628u,2627u,327737u,10u,2629u,76u,2628u,196670u,2623u,2629u,
327745u,9u,2631u,221u,2409u,262205u,6u,2632u,2631u,262205u,6u,2633u,2457u,327745u,9u,2634u,
221u,2624u,262205u,6u,2635u,2634u,327812u,6u,2636u,2633u,2635u,327808u,6u,2637u,2632u,2636u,
196670u,2630u,2637u,262205u,6u,2639u,216u,196670u,2638u,2639u,327737u,57u,2640u,65u,2638u,196855u,
2642u,0u,262394u,2640u,2641u,2656u,131320u,2641u,262205u,6u,2643u,220u,327745u,9u,2644u,221u,
2624u,262205u,6u,2645u,2644u,327745u,9u,2646u,221u,551u,262205u,6u,2647u,2646u,327812u,6u,
2648u,2645u,2647u,327808u,6u,2649u,2643u,2648u,262205u,6u,2650u,2441u,327808u,6u,2651u,2649u,
2650u,196670u,2652u,2651u,262205u,10u,2654u,2473u,196670u,2653u,2654u,393273u,2u,2655u,55u,2652u,
2653u,131321u,2642u,131320u,2656u,262205u,6u,2657u,215u,327745u,9u,2658u,221u,2624u,262205u,6u,
2659u,2658u,327745u,9u,2660u,221u,551u,262205u,6u,2661u,2660u,327812u,6u,2662u,2659u,2661u,
262205u,6u,2663u,2441u,327808u,6u,2664u,2662u,2663u,327808u,6u,2665u,2657u,2664u,393281u,584u,
2666u,578u,548u,2665u,262205u,10u,2667u,2666u,196670u,2487u,2667u,131321u,2642u,131320u,2642u,196670u,
2668u,548u,131321u,2669u,131320u,2669u,262390u,2671u,2672u,0u,131321u,2673u,131320u,2673u,262205u,6u,
2674u,2668u,327745u,9u,2675u,221u,2624u,262205u,6u,2676u,2675u,327857u,57u,2677u,2674u,2676u,
262394u,2677u,2670u,2671u,131320u,2670u,262205u,6u,2679u,2668u,327745u,9u,2680u,221u,551u,262205u,
6u,2681u,2680u,327812u,6u,2682u,2679u,2681u,262205u,6u,2683u,2441u,327808u,6u,2684u,2682u,
2683u,196670u,2678u,2684u,262205u,6u,2686u,216u,196670u,2685u,2686u,327737u,57u,2687u,65u,2685u,
196855u,2689u,0u,262394u,2687u,2688u,2724u,131320u,2688u,262205u,6u,2690u,218u,262205u,6u,2691u,
2630u,327808u,6u,2692u,2690u,2691u,262205u,6u,2693u,2668u,327808u,6u,2694u,2692u,2693u,262205u,
10u,2695u,2473u,262205u,6u,2696u,215u,262205u,6u,2697u,2678u,327808u,6u,2698u,2696u,2697u,
393281u,584u,2699u,578u,548u,2698u,262205u,10u,2700u,2699u,327813u,10u,2701u,2695u,2700u,262205u,
10u,2702u,2623u,327813u,10u,2703u,2701u,2702u,196670u,2704u,2694u,196670u,2705u,2703u,393273u,2u,
2706u,55u,2704u,2705u,262205u,6u,2707u,220u,262205u,6u,2708u,2678u,327808u,6u,2709u,2707u,
2708u,262205u,10u,2710u,2473u,262205u,6u,2711u,213u,262205u,6u,2712u,2630u,262205u,6u,2713u,
2668u,327808u,6u,2714u,2712u,2713u,327808u,6u,2715u,2711u,2714u,393281u,584u,2716u,578u,548u,
2715u,262205u,10u,2717u,2716u,327813u,10u,2718u,2710u,2717u,262205u,10u,2719u,2623u,327813u,10u,
2720u,2718u,2719u,196670u,2721u,2709u,196670u,2722u,2720u,393273u,2u,2723u,55u,2721u,2722u,131321u,
2689u,131320u,2724u,262205u,6u,2725u,213u,262205u,6u,2726u,2630u,262205u,6u,2727u,2668u,327808u,
6u,2728u,2726u,2727u,327808u,6u,2729u,2725u,2728u,393281u,584u,2730u,578u,548u,2729u,262205u,
10u,2731u,2730u,262205u,10u,2732u,2623u,327813u,10u,2733u,2731u,2732u,262205u,6u,2734u,215u,
262205u,6u,2735u,2678u,327808u,6u,2736u,2734u,2735u,393281u,584u,2737u,578u,548u,2736u,262205u,
10u,2738u,2737u,327813u,10u,2739u,2733u,2738u,262205u,10u,2740u,2487u,327809u,10u,2741u,2740u,
2739u,196670u,2487u,2741u,131321u,2689u,131320u,2689u,131321u,2672u,131320u,2672u,262205u,6u,2742u,2668u,
327808u,6u,2743u,2742u,551u,196670u,2668u,2743u,131321u,2669u,131320u,2671u,131321u,2622u,131320u,2744u,
262205u,6u,2746u,2441u,327745u,9u,2747u,221u,2624u,262205u,6u,2748u,2747u,327812u,6u,2749u,
2746u,2748u,327745u,9u,2751u,221u,2750u,262205u,6u,2752u,2751u,327812u,6u,2753u,2749u,2752u,
196670u,2745u,2753u,327745u,58u,2756u,221u,2754u,262205u,57u,2757u,2756u,196670u,2755u,2757u,327737u,
57u,2758u,61u,2755u,196855u,2760u,0u,262394u,2758u,2759u,2760u,131320u,2759u,262205u,6u,2762u,
216u,196670u,2761u,2762u,327737u,57u,2763u,65u,2761u,196855u,2765u,0u,262394u,2763u,2764u,2782u,
131320u,2764u,262205u,6u,2766u,220u,327745u,9u,2767u,221u,551u,262205u,6u,2768u,2767u,327745u,
9u,2769u,221u,2624u,262205u,6u,2770u,2769u,327812u,6u,2771u,2768u,2770u,327745u,9u,2772u,
221u,2750u,262205u,6u,2773u,2772u,327812u,6u,2774u,2771u,2773u,327808u,6u,2775u,2766u,2774u,
262205u,6u,2776u,2441u,327808u,6u,2777u,2775u,2776u,196670u,2778u,2777u,262205u,10u,2780u,2473u,
196670u,2779u,2780u,393273u,2u,2781u,55u,2778u,2779u,131321u,2765u,131320u,2782u,262205u,6u,2783u,
215u,327745u,9u,2784u,221u,551u,262205u,6u,2785u,2784u,327745u,9u,2786u,221u,2624u,262205u,
6u,2787u,2786u,327812u,6u,2788u,2785u,2787u,327745u,9u,2789u,221u,2750u,262205u,6u,2790u,
2789u,327812u,6u,2791u,2788u,2790u,262205u,6u,2792u,2441u,327808u,6u,2793u,2791u,2792u,327808u,
6u,2794u,2783u,2793u,393281u,584u,2795u,578u,548u,2794u,262205u,10u,2796u,2795u,196670u,2487u,
2796u,131321u,2765u,131320u,2765u,131321u,2760u,131320u,2760u,196670u,2797u,548u,131321u,2798u,131320u,2798u,
262390u,2800u,2801u,0u,131321u,2802u,131320u,2802u,262205u,6u,2803u,2797u,327745u,9u,2804u,221u,
2624u,262205u,6u,2805u,2804u,327857u,57u,2806u,2803u,2805u,262394u,2806u,2799u,2800u,131320u,2799u,
196670u,2807u,548u,131321u,2808u,131320u,2808u,262390u,2810u,2811u,0u,131321u,2812u,131320u,2812u,262205u,
6u,2813u,2807u,327745u,9u,2814u,221u,2750u,262205u,6u,2815u,2814u,327857u,57u,2816u,2813u,
2815u,262394u,2816u,2809u,2810u,131320u,2809u,262205u,6u,2818u,2745u,262205u,6u,2819u,2797u,327745u,
9u,2820u,221u,2750u,262205u,6u,2821u,2820u,327812u,6u,2822u,2819u,2821u,327808u,6u,2823u,
2818u,2822u,262205u,6u,2824u,2807u,327808u,6u,2825u,2823u,2824u,196670u,2817u,2825u,262205u,6u,
2827u,2457u,327745u,9u,2828u,221u,2624u,262205u,6u,2829u,2828u,327812u,6u,2830u,2827u,2829u,
262205u,6u,2831u,2797u,327808u,6u,2832u,2830u,2831u,196670u,2826u,2832u,262205u,6u,2834u,2457u,
327745u,9u,2835u,221u,2750u,262205u,6u,2836u,2835u,327812u,6u,2837u,2834u,2836u,262205u,6u,
2838u,2807u,327808u,6u,2839u,2837u,2838u,196670u,2833u,2839u,262205u,6u,2841u,216u,196670u,2840u,
2841u,327737u,57u,2842u,65u,2840u,196855u,2844u,0u,262394u,2842u,2843u,2902u,131320u,2843u,262205u,
6u,2845u,218u,262205u,6u,2846u,2826u,327808u,6u,2847u,2845u,2846u,262205u,10u,2848u,2473u,
262205u,6u,2849u,215u,262205u,6u,2850u,2817u,327808u,6u,2851u,2849u,2850u,393281u,584u,2852u,
578u,548u,2851u,262205u,10u,2853u,2852u,327813u,10u,2854u,2848u,2853u,262205u,6u,2855u,214u,
262205u,6u,2856u,2833u,327808u,6u,2857u,2855u,2856u,393281u,584u,2858u,578u,548u,2857u,262205u,
10u,2859u,2858u,327813u,10u,2860u,2854u,2859u,196670u,2861u,2847u,196670u,2862u,2860u,393273u,2u,
2863u,55u,2861u,2862u,262205u,6u,2864u,219u,262205u,6u,2865u,2833u,327808u,6u,2866u,2864u,
2865u,262205u,10u,2867u,2473u,262205u,6u,2868u,215u,262205u,6u,2869u,2817u,327808u,6u,2870u,
2868u,2869u,393281u,584u,2871u,578u,548u,2870u,262205u,10u,2872u,2871u,327813u,10u,2873u,2867u,
2872u,262205u,6u,2874u,213u,262205u,6u,2875u,2826u,327808u,6u,2876u,2874u,2875u,393281u,584u,
2877u,578u,548u,2876u,262205u,10u,2878u,2877u,327813u,10u,2879u,2873u,2878u,196670u,2880u,2866u,
196670u,2881u,2879u,393273u,2u,2882u,55u,2880u,2881u,262205u,6u,2883u,220u,262205u,6u,2884u,
2817u,327808u,6u,2885u,2883u,2884u,262205u,10u,2886u,2473u,262205u,6u,2887u,213u,262205u,6u,
2888u,2826u,327808u,6u,2889u,2887u,2888u,393281u,584u,2890u,578u,548u,2889u,262205u,10u,2891u,
2890u,327813u,10u,2892u,2886u,2891u,262205u,6u,2893u,214u,262205u,6u,2894u,2833u,327808u,6u,
2895u,2893u,2894u,393281u,584u,2896u,578u,548u,2895u,262205u,10u,2897u,2896u,327813u,10u,2898u,
2892u,2897u,196670u,2899u,2885u,196670u,2900u,2898u,393273u,2u,2901u,55u,2899u,2900u,131321u,2844u,
131320u,2902u,262205u,6u,2903u,213u,262205u,6u,2904u,2826u,327808u,6u,2905u,2903u,2904u,393281u,
584u,2906u,578u,548u,2905u,262205u,10u,2907u,2906u,262205u,6u,2908u,215u,262205u,6u,2909u,
2817u,327808u,6u,2910u,2908u,2909u,393281u,584u,2911u,578u,548u,2910u,262205u,10u,2912u,2911u,
327813u,10u,2913u,2907u,2912u,262205u,6u,2914u,214u,262205u,6u,2915u,2833u,327808u,6u,2916u,
2914u,2915u,393281u,584u,2917u,578u,548u,2916u,262205u,10u,2918u,2917u,327813u,10u,2919u,2913u,
2918u,262205u,10u,2920u,2487u,327809u,10u,2921u,2920u,2919u,196670u,2487u,2921u,131321u,2844u,131320u,
2844u,131321u,2811u,131320u,2811u,262205u,6u,2922u,2807u,327808u,6u,2923u,2922u,551u,196670u,2807u,
2923u,131321u,2808u,131320u,2810u,131321u,2801u,131320u,2801u,262205u,6u,2924u,2797u,327808u,6u,2925u,
2924u,551u,196670u,2797u,2925u,131321u,2798u,131320u,2800u,131321u,2622u,131320u,2622u,131321u,2493u,131320u,
2493u,262205u,6u,2927u,216u,196670u,2926u,2927u,327737u,57u,2928u,65u,2926u,262312u,57u,2929u,
2928u,196855u,2931u,0u,262394u,2929u,2930u,2931u,131320u,2930u,262205u,6u,2932u,217u,262205u,6u,
2933u,2388u,327808u,6u,2934u,2932u,2933u,262205u,10u,2935u,2487u,393281u,584u,2936u,578u,548u,
2934u,196670u,2936u,2935u,131321u,2931u,131320u,2931u,65789u,65592u,327734u,2u,230u,0u,108u,196663u,
9u,225u,196663u,9u,226u,196663u,9u,227u,196663u,9u,228u,196663u,9u,229u,131320u,231u,
262203u,9u,2937u,7u,262203u,9u,2949u,7u,262203u,9u,2960u,7u,262203u,11u,2964u,7u,
262203u,11u,2970u,7u,262203u,11u,2983u,7u,262203u,9u,2989u,7u,262205u,6u,2938u,549u,
262205u,6u,2939u,550u,327812u,6u,2940u,2938u,2939u,262205u,6u,2941u,547u,327808u,6u,2942u,
2940u,2941u,196670u,2937u,2942u,262205u,6u,2943u,2937u,262205u,6u,2944u,229u,327855u,57u,2945u,
2943u,2944u,196855u,2947u,0u,262394u,2945u,2946u,2947u,131320u,2946u,65789u,131320u,2947u,262205u,6u,
2950u,2937u,262205u,6u,2951u,228u,327815u,6u,2952u,2950u,2951u,327812u,6u,2953u,2952u,2490u,
262205u,6u,2954u,228u,327812u,6u,2955u,2953u,2954u,262205u,6u,2956u,2937u,262205u,6u,2957u,
228u,327819u,6u,2958u,2956u,2957u,327808u,6u,2959u,2955u,2958u,196670u,2949u,2959u,262205u,6u,
2961u,2949u,262205u,6u,2962u,228u,327808u,6u,2963u,2961u,2962u,196670u,2960u,2963u,262205u,6u,
2965u,225u,262205u,6u,2966u,2960u,327808u,6u,2967u,2965u,2966u,393281u,584u,2968u,578u,548u,
2967u,262205u,10u,2969u,2968u,196670u,2964u,2969u,262205u,10u,2972u,2964u,262205u,10u,2974u,2964u,
327813u,10u,2975u,2973u,2974u,262205u,10u,2976u,2964u,327813u,10u,2977u,2975u,2976u,262205u,10u,
2978u,2964u,327813u,10u,2979u,2977u,2978u,327809u,10u,2980u,2972u,2979u,327813u,10u,2981u,2971u,
2980u,393228u,10u,2982u,1u,21u,2981u,196670u,2970u,2982u,262205u,10u,2984u,2964u,327813u,10u,
2985u,2984u,687u,262205u,10u,2986u,2970u,327809u,10u,2987u,701u,2986u,327813u,10u,2988u,2985u,
2987u,196670u,2983u,2988u,262205u,6u,2990u,226u,196670u,2989u,2990u,327737u,57u,2991u,65u,2989u,
262312u,57u,2992u,2991u,196855u,2994u,0u,262394u,2992u,2993u,3006u,131320u,2993u,262205u,6u,2995u,
227u,262205u,6u,2996u,2937u,327808u,6u,2997u,2995u,2996u,262205u,6u,2998u,225u,262205u,6u,
2999u,2949u,327808u,6u,3000u,2998u,2999u,393281u,584u,3001u,578u,548u,3000u,262205u,10u,3002u,
3001u,262205u,10u,3003u,2983u,327813u,10u,3004u,3002u,3003u,393281u,584u,3005u,578u,548u,2997u,
196670u,3005u,3004u,131321u,2994u,131320u,3006u,262205u,6u,3007u,227u,262205u,6u,3008u,2949u,327808u,
6u,3009u,3007u,3008u,262205u,6u,3010u,226u,262205u,6u,3011u,2937u,327808u,6u,3012u,3010u,
3011u,393281u,584u,3013u,578u,548u,3012u,262205u,10u,3014u,3013u,262205u,10u,3015u,2983u,327813u,
10u,3016u,3014u,3015u,393281u,584u,3017u,578u,548u,3009u,196670u,3017u,3016u,262205u,6u,3018u,
227u,262205u,6u,3019u,2960u,327808u,6u,3020u,3018u,3019u,262205u,6u,3021u,226u,262205u,6u,
3022u,2937u,327808u,6u,3023u,3021u,3022u,393281u,584u,3024u,578u,548u,3023u,262205u,10u,3025u,
3024u,262205u,6u,3026u,225u,262205u,6u,3027u,2949u,327808u,6u,3028u,3026u,3027u,393281u,584u,
3029u,578u,548u,3028u,262205u,10u,3030u,3029u,327813u,10u,3031u,3025u,3030u,262205u,10u,3032u,
2970u,327809u,10u,3033u,701u,3032u,327813u,10u,3034u,687u,3033u,262205u,10u,3035u,2964u,327813u,
10u,3036u,687u,3035u,262205u,10u,3037u,2970u,262205u,10u,3038u,2970u,327813u,10u,3039u,3037u,
3038u,327811u,10u,3040u,701u,3039u,327813u,10u,3041u,3036u,3040u,327813u,10u,3042u,3041u,2971u,
262205u,10u,3044u,2964u,327813u,10u,3045u,3043u,3044u,262205u,10u,3046u,2964u,327813u,10u,3047u,
3045u,3046u,327809u,10u,3048u,701u,3047u,327813u,10u,3049u,3042u,3048u,327809u,10u,3050u,3034u,
3049u,327813u,10u,3051u,3031u,3050u,393281u,584u,3052u,578u,548u,3020u,196670u,3052u,3051u,131321u,
2994u,131320u,2994u,65789u,65592u,327734u,10u,233u,0u,71u,196663u,11u,232u,131320u,234u,262203u,
11u,3055u,7u,262205u,10u,3053u,232u,393228u,10u,3054u,1u,4u,3053u,196670u,232u,3054u,
196670u,3055u,3056u,262205u,10u,3057u,232u,327868u,57u,3058u,3057u,701u,196855u,3060u,0u,262394u,
3058u,3059u,3060u,131320u,3059u,262205u,10u,3061u,3055u,327809u,10u,3063u,3061u,3062u,262205u,10u,
3064u,232u,327813u,10u,3065u,3063u,3064u,262205u,10u,3066u,3055u,327809u,10u,3068u,3066u,3067u,
327811u,10u,3069u,3065u,3068u,262205u,10u,3070u,232u,327813u,10u,3071u,3069u,3070u,262205u,10u,
3072u,232u,327813u,10u,3073u,3071u,3072u,327809u,10u,3074u,3073u,701u,131326u,3074u,131320u,3060u,
262205u,10u,3076u,232u,327864u,57u,3077u,3076u,3062u,196855u,3079u,0u,262394u,3077u,3078u,3079u,
131320u,3078u,262205u,10u,3080u,3055u,262205u,10u,3081u,232u,327813u,10u,3082u,3080u,3081u,262205u,
10u,3084u,3055u,327813u,10u,3085u,3083u,3084u,327811u,10u,3086u,3082u,3085u,262205u,10u,3087u,
232u,327813u,10u,3088u,3086u,3087u,262205u,10u,3090u,3055u,327813u,10u,3091u,3089u,3090u,327809u,
10u,3092u,3088u,3091u,262205u,10u,3093u,232u,327813u,10u,3094u,3092u,3093u,262205u,10u,3096u,
3055u,327813u,10u,3097u,3095u,3096u,327811u,10u,3098u,3094u,3097u,131326u,3098u,131320u,3079u,131326u,
630u,65592u,327734u,2u,243u,0u,237u,196663u,9u,238u,196663u,9u,239u,196663u,9u,240u,
196663u,236u,241u,196663u,9u,242u,131320u,244u,262203u,9u,3102u,7u,262203u,9u,3114u,7u,
262203u,9u,3123u,7u,262203u,9u,3131u,7u,262203u,11u,3136u,7u,262203u,11u,3149u,7u,
262203u,9u,3162u,7u,262203u,9u,3166u,7u,262203u,11u,3170u,7u,262203u,11u,3171u,7u,
262203u,9u,3172u,7u,262203u,9u,3181u,7u,262203u,11u,3189u,7u,262203u,11u,3196u,7u,
262203u,11u,3204u,7u,262203u,9u,3210u,7u,262203u,9u,3253u,7u,262203u,11u,3262u,7u,
262203u,11u,3265u,7u,262203u,11u,3275u,7u,262203u,9u,3282u,7u,262203u,9u,3290u,7u,
262203u,11u,3298u,7u,262203u,11u,3305u,7u,262203u,11u,3313u,7u,262203u,9u,3316u,7u,
262203u,9u,3354u,7u,262203u,11u,3355u,7u,262205u,6u,3103u,549u,262205u,6u,3104u,550u,
327812u,6u,3105u,3103u,3104u,262205u,6u,3106u,547u,327808u,6u,3107u,3105u,3106u,196670u,3102u,
3107u,262205u,6u,3108u,3102u,262205u,6u,3109u,242u,327855u,57u,3110u,3108u,3109u,196855u,3112u,
0u,262394u,3110u,3111u,3112u,131320u,3111u,65789u,131320u,3112u,262205u,6u,3115u,3102u,327745u,9u,
3116u,241u,2404u,262205u,6u,3117u,3116u,327745u,9u,3119u,241u,3118u,262205u,6u,3120u,3119u,
327812u,6u,3121u,3117u,3120u,327815u,6u,3122u,3115u,3121u,196670u,3114u,3122u,262205u,6u,3124u,
3102u,327745u,9u,3125u,241u,3118u,262205u,6u,3126u,3125u,327815u,6u,3127u,3124u,3126u,327745u,
9u,3128u,241u,2404u,262205u,6u,3129u,3128u,327819u,6u,3130u,3127u,3129u,196670u,3123u,3130u,
262205u,6u,3132u,3102u,327745u,9u,3133u,241u,3118u,262205u,6u,3134u,3133u,327819u,6u,3135u,
3132u,3134u,196670u,3131u,3135u,262205u,6u,3137u,3123u,262255u,10u,3138u,3137u,327809u,10u,3139u,
3138u,687u,327745u,9u,3140u,241u,551u,262205u,6u,3141u,3140u,262255u,10u,3142u,3141u,327745u,
9u,3143u,241u,2404u,262205u,6u,3144u,3143u,262255u,10u,3145u,3144u,327816u,10u,3146u,3142u,
3145u,327813u,10u,3147u,3139u,3146u,327811u,10u,3148u,3147u,687u,196670u,3136u,3148u,262205u,6u,
3150u,3131u,262255u,10u,3151u,3150u,327809u,10u,3152u,3151u,687u,327745u,9u,3153u,241u,2490u,
262205u,6u,3154u,3153u,262255u,10u,3155u,3154u,327745u,9u,3156u,241u,3118u,262205u,6u,3157u,
3156u,262255u,10u,3158u,3157u,327816u,10u,3159u,3155u,3158u,327813u,10u,3160u,3152u,3159u,327811u,
10u,3161u,3160u,687u,196670u,3149u,3161u,262205u,10u,3163u,3136u,393228u,10u,3164u,1u,8u,
3163u,262254u,6u,3165u,3164u,196670u,3162u,3165u,262205u,10u,3167u,3149u,393228u,10u,3168u,1u,
8u,3167u,262254u,6u,3169u,3168u,196670u,3166u,3169u,196670u,3170u,630u,196670u,3171u,630u,196670u,
3172u,3173u,131321u,3174u,131320u,3174u,262390u,3176u,3177u,0u,131321u,3178u,131320u,3178u,262205u,6u,
3179u,3172u,327859u,57u,3180u,3179u,2490u,262394u,3180u,3175u,3176u,131320u,3175u,196670u,3181u,3173u,
131321u,3182u,131320u,3182u,262390u,3184u,3185u,0u,131321u,3186u,131320u,3186u,262205u,6u,3187u,3181u,
327859u,57u,3188u,3187u,2490u,262394u,3188u,3183u,3184u,131320u,3183u,262205u,10u,3190u,3136u,262205u,
6u,3191u,3162u,262205u,6u,3192u,3172u,327808u,6u,3193u,3191u,3192u,262255u,10u,3194u,3193u,
327811u,10u,3195u,3190u,3194u,196670u,3196u,3195u,327737u,10u,3197u,233u,3196u,262205u,10u,3198u,
3149u,262205u,6u,3199u,3166u,262205u,6u,3200u,3181u,327808u,6u,3201u,3199u,3200u,262255u,10u,
3202u,3201u,327811u,10u,3203u,3198u,3202u,196670u,3204u,3203u,327737u,10u,3205u,233u,3204u,327813u,
10u,3206u,3197u,3205u,196670u,3189u,3206u,262205u,10u,3207u,3189u,262205u,10u,3208u,3171u,327809u,
10u,3209u,3208u,3207u,196670u,3171u,3209u,262205u,6u,3211u,239u,196670u,3210u,3211u,327737u,57u,
3212u,65u,3210u,262312u,57u,3213u,3212u,196855u,3215u,0u,262394u,3213u,3214u,3215u,131320u,3214u,
262205u,10u,3216u,3189u,262205u,6u,3217u,238u,262205u,6u,3218u,3114u,327745u,9u,3219u,241u,
551u,262205u,6u,3220u,3219u,327812u,6u,3221u,3218u,3220u,262205u,6u,3222u,3162u,262205u,6u,
3223u,3172u,327808u,6u,3224u,3222u,3223u,327745u,9u,3225u,241u,551u,262205u,6u,3226u,3225u,
327810u,6u,3227u,3226u,551u,458764u,6u,3228u,1u,39u,3224u,3227u,458764u,6u,3229u,1u,
42u,548u,3228u,327808u,6u,3230u,3221u,3229u,327745u,9u,3231u,241u,2490u,262205u,6u,3232u,
3231u,327812u,6u,3233u,3230u,3232u,262205u,6u,3234u,3166u,262205u,6u,3235u,3181u,327808u,6u,
3236u,3234u,3235u,327745u,9u,3237u,241u,2490u,262205u,6u,3238u,3237u,327810u,6u,3239u,3238u,
551u,458764u,6u,3240u,1u,39u,3236u,3239u,458764u,6u,3241u,1u,42u,548u,3240u,327808u,
6u,3242u,3233u,3241u,327808u,6u,3243u,3217u,3242u,393281u,584u,3244u,578u,548u,3243u,262205u,
10u,3245u,3244u,327813u,10u,3246u,3216u,3245u,262205u,10u,3247u,3170u,327809u,10u,3248u,3247u,
3246u,196670u,3170u,3248u,131321u,3215u,131320u,3215u,131321u,3185u,131320u,3185u,262205u,6u,3249u,3181u,
327808u,6u,3250u,3249u,551u,196670u,3181u,3250u,131321u,3182u,131320u,3184u,131321u,3177u,131320u,3177u,
262205u,6u,3251u,3172u,327808u,6u,3252u,3251u,551u,196670u,3172u,3252u,131321u,3174u,131320u,3176u,
262205u,6u,3254u,239u,196670u,3253u,3254u,327737u,57u,3255u,65u,3253u,262312u,57u,3256u,3255u,
196855u,3258u,0u,262394u,3256u,3257u,3258u,131320u,3257u,262205u,6u,3259u,240u,262205u,6u,3260u,
3102u,327808u,6u,3261u,3259u,3260u,262205u,10u,3263u,3171u,196670u,3262u,3263u,327737u,57u,3264u,
69u,3262u,196855u,3267u,0u,262394u,3264u,3266u,3271u,131320u,3266u,262205u,10u,3268u,3170u,262205u,
10u,3269u,3171u,327816u,10u,3270u,3268u,3269u,196670u,3265u,3270u,131321u,3267u,131320u,3271u,196670u,
3265u,630u,131321u,3267u,131320u,3267u,262205u,10u,3272u,3265u,393281u,584u,3273u,578u,548u,3261u,
196670u,3273u,3272u,65789u,131320u,3258u,262205u,10u,3276u,3171u,196670u,3275u,3276u,327737u,57u,3277u,
69u,3275u,262312u,57u,3278u,3277u,196855u,3280u,0u,262394u,3278u,3279u,3280u,131320u,3279u,65789u,
131320u,3280u,196670u,3282u,3173u,131321u,3283u,131320u,3283u,262390u,3285u,3286u,0u,131321u,3287u,131320u,
3287u,262205u,6u,3288u,3282u,327859u,57u,3289u,3288u,2490u,262394u,3289u,3284u,3285u,131320u,3284u,
196670u,3290u,3173u,131321u,3291u,131320u,3291u,262390u,3293u,3294u,0u,131321u,3295u,131320u,3295u,262205u,
6u,3296u,3290u,327859u,57u,3297u,3296u,2490u,262394u,3297u,3292u,3293u,131320u,3292u,262205u,10u,
3299u,3136u,262205u,6u,3300u,3162u,262205u,6u,3301u,3282u,327808u,6u,3302u,3300u,3301u,262255u,
10u,3303u,3302u,327811u,10u,3304u,3299u,3303u,196670u,3305u,3304u,327737u,10u,3306u,233u,3305u,
262205u,10u,3307u,3149u,262205u,6u,3308u,3166u,262205u,6u,3309u,3290u,327808u,6u,3310u,3308u,
3309u,262255u,10u,3311u,3310u,327811u,10u,3312u,3307u,3311u,196670u,3313u,3312u,327737u,10u,3314u,
233u,3313u,327813u,10u,3315u,3306u,3314u,196670u,3298u,3315u,262205u,6u,3317u,3114u,327745u,9u,
3318u,241u,551u,262205u,6u,3319u,3318u,327812u,6u,3320u,3317u,3319u,262205u,6u,3321u,3162u,
262205u,6u,3322u,3282u,327808u,6u,3323u,3321u,3322u,327745u,9u,3324u,241u,551u,262205u,6u,
3325u,3324u,327810u,6u,3326u,3325u,551u,458764u,6u,3327u,1u,39u,3323u,3326u,458764u,6u,
3328u,1u,42u,548u,3327u,327808u,6u,3329u,3320u,3328u,327745u,9u,3330u,241u,2490u,262205u,
6u,3331u,3330u,327812u,6u,3332u,3329u,3331u,262205u,6u,3333u,3166u,262205u,6u,3334u,3290u,
327808u,6u,3335u,3333u,3334u,327745u,9u,3336u,241u,2490u,262205u,6u,3337u,3336u,327810u,6u,
3338u,3337u,551u,458764u,6u,3339u,1u,39u,3335u,3338u,458764u,6u,3340u,1u,42u,548u,
3339u,327808u,6u,3341u,3332u,3340u,196670u,3316u,3341u,262205u,6u,3342u,240u,262205u,6u,3343u,
3316u,327808u,6u,3344u,3342u,3343u,262205u,6u,3345u,239u,262205u,6u,3346u,3102u,327808u,6u,
3347u,3345u,3346u,393281u,584u,3348u,578u,548u,3347u,262205u,10u,3349u,3348u,262205u,10u,3350u,
3298u,262205u,10u,3351u,3171u,327816u,10u,3352u,3350u,3351u,327813u,10u,3353u,3349u,3352u,196670u,
3354u,3344u,196670u,3355u,3353u,393273u,2u,3356u,55u,3354u,3355u,131321u,3294u,131320u,3294u,262205u,
6u,3357u,3290u,327808u,6u,3358u,3357u,551u,196670u,3290u,3358u,131321u,3291u,131320u,3293u,131321u,
3286u,131320u,3286u,262205u,6u,3359u,3282u,327808u,6u,3360u,3359u,551u,196670u,3282u,3360u,131321u,
3283u,131320u,3285u,65789u,65592u,327734u,2u,254u,0u,247u,196663u,9u,248u,196663u,9u,249u,
196663u,9u,250u,196663u,9u,251u,196663u,246u,252u,196663u,9u,253u,131320u,255u,262203u,9u,
3361u,7u,262203u,9u,3373u,7u,262203u,9u,3382u,7u,262203u,9u,3390u,7u,262203u,11u,
3395u,7u,262203u,11u,3398u,7u,262203u,9u,3405u,7u,262203u,9u,3406u,7u,262203u,9u,
3407u,7u,262203u,9u,3417u,7u,262203u,9u,3427u,7u,262203u,9u,3437u,7u,262203u,9u,
3471u,7u,262203u,9u,3475u,7u,262203u,58u,3499u,7u,262203u,9u,3532u,7u,262203u,11u,
3543u,7u,262203u,9u,3548u,7u,262203u,11u,3551u,7u,262203u,58u,3563u,7u,262203u,9u,
3579u,7u,262203u,11u,3580u,7u,262203u,9u,3585u,7u,262203u,11u,3592u,7u,262203u,9u,
3601u,7u,262203u,9u,3611u,7u,262203u,9u,3621u,7u,262203u,9u,3631u,7u,262203u,58u,
3663u,7u,262203u,9u,3667u,7u,262203u,9u,3690u,7u,262203u,11u,3691u,7u,262205u,6u,
3362u,549u,262205u,6u,3363u,550u,327812u,6u,3364u,3362u,3363u,262205u,6u,3365u,547u,327808u,
6u,3366u,3364u,3365u,196670u,3361u,3366u,262205u,6u,3367u,3361u,262205u,6u,3368u,253u,327855u,
57u,3369u,3367u,3368u,196855u,3371u,0u,262394u,3369u,3370u,3371u,131320u,3370u,65789u,131320u,3371u,
262205u,6u,3374u,3361u,327745u,9u,3375u,252u,2395u,262205u,6u,3376u,3375u,327745u,9u,3378u,
252u,3377u,262205u,6u,3379u,3378u,327812u,6u,3380u,3376u,3379u,327815u,6u,3381u,3374u,3380u,
196670u,3373u,3381u,262205u,6u,3383u,3361u,327745u,9u,3384u,252u,3377u,262205u,6u,3385u,3384u,
327815u,6u,3386u,3383u,3385u,327745u,9u,3387u,252u,2395u,262205u,6u,3388u,3387u,327819u,6u,
3389u,3386u,3388u,196670u,3382u,3389u,262205u,6u,3391u,3361u,327745u,9u,3392u,252u,3377u,262205u,
6u,3393u,3392u,327819u,6u,3394u,3391u,3393u,196670u,3390u,3394u,327745u,58u,3396u,252u,2754u,
262205u,57u,3397u,3396u,196855u,3400u,0u,262394u,3397u,3399u,3403u,131320u,3399u,262268u,10u,3401u,
1573u,262271u,10u,3402u,3401u,196670u,3398u,3402u,131321u,3400u,131320u,3403u,196670u,3398u,630u,131321u,
3400u,131320u,3400u,262205u,10u,3404u,3398u,196670u,3395u,3404u,196670u,3405u,548u,196670u,3406u,3173u,
196670u,3407u,548u,131321u,3408u,131320u,3408u,262390u,3410u,3411u,0u,131321u,3412u,131320u,3412u,262205u,
6u,3413u,3407u,327745u,9u,3414u,252u,2404u,262205u,6u,3415u,3414u,327857u,57u,3416u,3413u,
3415u,262394u,3416u,3409u,3410u,131320u,3409u,196670u,3417u,548u,131321u,3418u,131320u,3418u,262390u,3420u,
3421u,0u,131321u,3422u,131320u,3422u,262205u,6u,3423u,3417u,327745u,9u,3424u,252u,3118u,262205u,
6u,3425u,3424u,327857u,57u,3426u,3423u,3425u,262394u,3426u,3419u,3420u,131320u,3419u,262205u,6u,
3428u,3382u,327745u,9u,3429u,252u,2510u,262205u,6u,3430u,3429u,327812u,6u,3431u,3428u,3430u,
262205u,6u,3432u,3407u,327808u,6u,3433u,3431u,3432u,327745u,9u,3434u,252u,2750u,262205u,6u,
3435u,3434u,327810u,6u,3436u,3433u,3435u,196670u,3427u,3436u,262205u,6u,3438u,3390u,327745u,9u,
3439u,252u,2624u,262205u,6u,3440u,3439u,327812u,6u,3441u,3438u,3440u,262205u,6u,3442u,3417u,
327808u,6u,3443u,3441u,3442u,327745u,9u,3444u,252u,2409u,262205u,6u,3445u,3444u,327810u,6u,
3446u,3443u,3445u,196670u,3437u,3446u,262205u,6u,3447u,3427u,327857u,57u,3448u,3447u,548u,262312u,
57u,3449u,3448u,196855u,3451u,0u,262394u,3449u,3450u,3451u,131320u,3450u,262205u,6u,3452u,3427u,
327745u,9u,3453u,252u,551u,262205u,6u,3454u,3453u,327855u,57u,3455u,3452u,3454u,131321u,3451u,
131320u,3451u,458997u,57u,3456u,3448u,3419u,3455u,3450u,262205u,6u,3457u,3437u,327857u,57u,3458u,
3457u,548u,327846u,57u,3459u,3456u,3458u,262312u,57u,3460u,3459u,196855u,3462u,0u,262394u,3460u,
3461u,3462u,131320u,3461u,262205u,6u,3463u,3437u,327745u,9u,3464u,252u,2490u,262205u,6u,3465u,
3464u,327855u,57u,3466u,3463u,3465u,131321u,3462u,131320u,3462u,458997u,57u,3467u,3459u,3451u,3466u,
3461u,196855u,3469u,0u,262394u,3467u,3468u,3469u,131320u,3468u,131321u,3421u,131320u,3469u,327745u,58u,
3473u,252u,3472u,262205u,57u,3474u,3473u,196855u,3477u,0u,262394u,3474u,3476u,3484u,131320u,3476u,
262205u,6u,3478u,3427u,327745u,9u,3479u,252u,548u,262205u,6u,3480u,3479u,327812u,6u,3481u,
3478u,3480u,262205u,6u,3482u,3373u,327808u,6u,3483u,3481u,3482u,196670u,3475u,3483u,131321u,3477u,
131320u,3484u,262205u,6u,3485u,3373u,327745u,9u,3486u,252u,551u,262205u,6u,3487u,3486u,327812u,
6u,3488u,3485u,3487u,262205u,6u,3489u,3427u,327808u,6u,3490u,3488u,3489u,327745u,9u,3491u,
252u,2490u,262205u,6u,3492u,3491u,327812u,6u,3493u,3490u,3492u,262205u,6u,3494u,3437u,327808u,
6u,3495u,3493u,3494u,196670u,3475u,3495u,131321u,3477u,131320u,3477u,262205u,6u,3496u,3475u,196670u,
3471u,3496u,262205u,6u,3497u,3405u,327808u,6u,3498u,3497u,551u,196670u,3405u,3498u,327745u,58u,
3500u,252u,2754u,262205u,57u,3501u,3500u,196670u,3499u,3501u,327737u,57u,3502u,61u,3499u,196855u,
3504u,0u,262394u,3502u,3503u,3520u,131320u,3503u,262205u,6u,3505u,248u,262205u,6u,3506u,3471u,
327808u,6u,3507u,3505u,3506u,393281u,584u,3508u,578u,548u,3507u,262205u,10u,3509u,3508u,262205u,
10u,3510u,3395u,327866u,57u,3511u,3509u,3510u,196855u,3513u,0u,262394u,3511u,3512u,3513u,131320u,
3512u,262205u,6u,3514u,248u,262205u,6u,3515u,3471u,327808u,6u,3516u,3514u,3515u,393281u,584u,
3517u,578u,548u,3516u,262205u,10u,3518u,3517u,196670u,3395u,3518u,262205u,6u,3519u,3471u,196670u,
3406u,3519u,131321u,3513u,131320u,3513u,131321u,3504u,131320u,3520u,262205u,6u,3521u,248u,262205u,6u,
3522u,3471u,327808u,6u,3523u,3521u,3522u,393281u,584u,3524u,578u,548u,3523u,262205u,10u,3525u,
3524u,262205u,10u,3526u,3395u,327809u,10u,3527u,3526u,3525u,196670u,3395u,3527u,131321u,3504u,131320u,
3504u,131321u,3421u,131320u,3421u,262205u,6u,3528u,3417u,327808u,6u,3529u,3528u,551u,196670u,3417u,
3529u,131321u,3418u,131320u,3420u,131321u,3411u,131320u,3411u,262205u,6u,3530u,3407u,327808u,6u,3531u,
3530u,551u,196670u,3407u,3531u,131321u,3408u,131320u,3410u,262205u,6u,3533u,249u,196670u,3532u,3533u,
327737u,57u,3534u,65u,3532u,262312u,57u,3535u,3534u,196855u,3537u,0u,262394u,3535u,3536u,3537u,
131320u,3536u,262205u,6u,3538u,250u,262205u,6u,3539u,3361u,327808u,6u,3540u,3538u,3539u,327745u,
58u,3541u,252u,2754u,262205u,57u,3542u,3541u,196855u,3545u,0u,262394u,3542u,3544u,3547u,131320u,
3544u,262205u,10u,3546u,3395u,196670u,3543u,3546u,131321u,3545u,131320u,3547u,262205u,6u,3549u,3405u,
196670u,3548u,3549u,327737u,57u,3550u,65u,3548u,196855u,3553u,0u,262394u,3550u,3552u,3558u,131320u,
3552u,262205u,10u,3554u,3395u,262205u,6u,3555u,3405u,262255u,10u,3556u,3555u,327816u,10u,3557u,
3554u,3556u,196670u,3551u,3557u,131321u,3553u,131320u,3558u,196670u,3551u,630u,131321u,3553u,131320u,3553u,
262205u,10u,3559u,3551u,196670u,3543u,3559u,131321u,3545u,131320u,3545u,262205u,10u,3560u,3543u,393281u,
584u,3561u,578u,548u,3540u,196670u,3561u,3560u,65789u,131320u,3537u,327745u,58u,3564u,252u,2754u,
262205u,57u,3565u,3564u,196670u,3563u,3565u,327737u,57u,3566u,61u,3563u,196855u,3568u,0u,262394u,
3566u,3567u,3568u,131320u,3567u,262205u,6u,3569u,3406u,327855u,57u,3570u,3569u,548u,196855u,3572u,
0u,262394u,3570u,3571u,3572u,131320u,3571u,262205u,6u,3573u,251u,262205u,6u,3574u,3406u,327808u,
6u,3575u,3573u,3574u,262205u,6u,3576u,249u,262205u,6u,3577u,3361u,327808u,6u,3578u,3576u,
3577u,196670u,3579u,3575u,393281u,584u,3581u,578u,548u,3578u,262205u,10u,3582u,3581u,196670u,3580u,
3582u,393273u,2u,3583u,55u,3579u,3580u,131321u,3572u,131320u,3572u,65789u,131320u,3568u,262205u,6u,
3586u,3405u,196670u,3585u,3586u,327737u,57u,3587u,65u,3585u,262312u,57u,3588u,3587u,196855u,3590u,
0u,262394u,3588u,3589u,3590u,131320u,3589u,65789u,131320u,3590u,262205u,6u,3593u,249u,262205u,6u,
3594u,3361u,327808u,6u,3595u,3593u,3594u,393281u,584u,3596u,578u,548u,3595u,262205u,10u,3597u,
3596u,262205u,6u,3598u,3405u,262255u,10u,3599u,3598u,327816u,10u,3600u,3597u,3599u,196670u,3592u,
3600u,196670u,3601u,548u,131321u,3602u,131320u,3602u,262390u,3604u,3605u,0u,131321u,3606u,131320u,3606u,
262205u,6u,3607u,3601u,327745u,9u,3608u,252u,2404u,262205u,6u,3609u,3608u,327857u,57u,3610u,
3607u,3609u,262394u,3610u,3603u,3604u,131320u,3603u,196670u,3611u,548u,131321u,3612u,131320u,3612u,262390u,
3614u,3615u,0u,131321u,3616u,131320u,3616u,262205u,6u,3617u,3611u,327745u,9u,3618u,252u,3118u,
262205u,6u,3619u,3618u,327857u,57u,3620u,3617u,3619u,262394u,3620u,3613u,3614u,131320u,3613u,262205u,
6u,3622u,3382u,327745u,9u,3623u,252u,2510u,262205u,6u,3624u,3623u,327812u,6u,3625u,3622u,
3624u,262205u,6u,3626u,3601u,327808u,6u,3627u,3625u,3626u,327745u,9u,3628u,252u,2750u,262205u,
6u,3629u,3628u,327810u,6u,3630u,3627u,3629u,196670u,3621u,3630u,262205u,6u,3632u,3390u,327745u,
9u,3633u,252u,2624u,262205u,6u,3634u,3633u,327812u,6u,3635u,3632u,3634u,262205u,6u,3636u,
3611u,327808u,6u,3637u,3635u,3636u,327745u,9u,3638u,252u,2409u,262205u,6u,3639u,3638u,327810u,
6u,3640u,3637u,3639u,196670u,3631u,3640u,262205u,6u,3641u,3621u,327855u,57u,3642u,3641u,548u,
196855u,3644u,0u,262394u,3642u,3643u,3644u,131320u,3643u,262205u,6u,3645u,3621u,327745u,9u,3646u,
252u,551u,262205u,6u,3647u,3646u,327857u,57u,3648u,3645u,3647u,131321u,3644u,131320u,3644u,458997u,
57u,3649u,3642u,3613u,3648u,3643u,262205u,6u,3650u,3631u,327855u,57u,3651u,3650u,548u,327847u,
57u,3652u,3649u,3651u,196855u,3654u,0u,262394u,3652u,3653u,3654u,131320u,3653u,262205u,6u,3655u,
3631u,327745u,9u,3656u,252u,2490u,262205u,6u,3657u,3656u,327857u,57u,3658u,3655u,3657u,131321u,
3654u,131320u,3654u,458997u,57u,3659u,3652u,3644u,3658u,3653u,196855u,3661u,0u,262394u,3659u,3660u,
3661u,131320u,3660u,262205u,6u,3662u,251u,327745u,58u,3664u,252u,3472u,262205u,57u,3665u,3664u,
196670u,3663u,3665u,327737u,57u,3666u,61u,3663u,196855u,3669u,0u,262394u,3666u,3668u,3676u,131320u,
3668u,262205u,6u,3670u,3621u,327745u,9u,3671u,252u,548u,262205u,6u,3672u,3671u,327812u,6u,
3673u,3670u,3672u,262205u,6u,3674u,3373u,327808u,6u,3675u,3673u,3674u,196670u,3667u,3675u,131321u,
3669u,131320u,3676u,262205u,6u,3677u,3373u,327745u,9u,3678u,252u,551u,262205u,6u,3679u,3678u,
327812u,6u,3680u,3677u,3679u,262205u,6u,3681u,3621u,327808u,6u,3682u,3680u,3681u,327745u,9u,
3683u,252u,2490u,262205u,6u,3684u,3683u,327812u,6u,3685u,3682u,3684u,262205u,6u,3686u,3631u,
327808u,6u,3687u,3685u,3686u,196670u,3667u,3687u,131321u,3669u,131320u,3669u,262205u,6u,3688u,3667u,
327808u,6u,3689u,3662u,3688u,196670u,3690u,3689u,262205u,10u,3692u,3592u,196670u,3691u,3692u,393273u,
2u,3693u,55u,3690u,3691u,131321u,3661u,131320u,3661u,131321u,3615u,131320u,3615u,262205u,6u,3694u,
3611u,327808u,6u,3695u,3694u,551u,196670u,3611u,3695u,131321u,3612u,131320u,3614u,131321u,3605u,131320u,
3605u,262205u,6u,3696u,3601u,327808u,6u,3697u,3696u,551u,196670u,3601u,3697u,131321u,3602u,131320u,
3604u,65789u,65592u,327734u,2u,264u,0u,256u,196663u,9u,257u,196663u,9u,258u,196663u,9u,
259u,196663u,9u,260u,196663u,9u,261u,196663u,9u,262u,196663u,58u,263u,131320u,265u,262203u,
9u,3698u,7u,262203u,9u,3710u,7u,262203u,9u,3712u,7u,262203u,9u,3713u,7u,262203u,
58u,3750u,7u,262205u,6u,3699u,549u,262205u,6u,3700u,550u,327812u,6u,3701u,3699u,3700u,
262205u,6u,3702u,547u,327808u,6u,3703u,3701u,3702u,196670u,3698u,3703u,262205u,6u,3704u,3698u,
262205u,6u,3705u,262u,327855u,57u,3706u,3704u,3705u,196855u,3708u,0u,262394u,3706u,3707u,3708u,
131320u,3707u,65789u,131320u,3708u,262205u,6u,3711u,3698u,196670u,3710u,3711u,196670u,3712u,548u,262205u,
6u,3714u,261u,327810u,6u,3715u,3714u,551u,196670u,3713u,3715u,131321u,3716u,131320u,3716u,262390u,
3718u,3719u,0u,131321u,3720u,131320u,3720u,262205u,6u,3721u,3713u,327855u,57u,3722u,3721u,548u,
262394u,3722u,3717u,3718u,131320u,3717u,262205u,6u,3723u,3710u,262205u,6u,3724u,259u,262205u,6u,
3725u,3713u,327808u,6u,3726u,3724u,3725u,393281u,584u,3727u,578u,548u,3726u,262205u,10u,3728u,
3727u,262268u,6u,3729u,3728u,327819u,6u,3730u,3723u,3729u,262205u,6u,3731u,260u,262205u,6u,
3732u,3713u,327808u,6u,3733u,3731u,3732u,393281u,584u,3734u,578u,548u,3733u,262205u,10u,3735u,
3734u,262268u,6u,3736u,3735u,327812u,6u,3737u,3730u,3736u,262205u,6u,3738u,3712u,327808u,6u,
3739u,3738u,3737u,196670u,3712u,3739u,262205u,6u,3740u,259u,262205u,6u,3741u,3713u,327808u,6u,
3742u,3740u,3741u,393281u,584u,3743u,578u,548u,3742u,262205u,10u,3744u,3743u,262268u,6u,3745u,
3744u,262205u,6u,3746u,3710u,327815u,6u,3747u,3746u,3745u,196670u,3710u,3747u,131321u,3719u,131320u,
3719u,262205u,6u,3748u,3713u,327810u,6u,3749u,3748u,551u,196670u,3713u,3749u,131321u,3716u,131320u,
3718u,262205u,57u,3751u,263u,196670u,3750u,3751u,327737u,57u,3752u,61u,3750u,196855u,3754u,0u,
262394u,3752u,3753u,3764u,131320u,3753u,262205u,6u,3755u,258u,262205u,6u,3756u,3712u,327808u,6u,
3757u,3755u,3756u,262205u,6u,3758u,257u,262205u,6u,3759u,3698u,327808u,6u,3760u,3758u,3759u,
393281u,584u,3761u,578u,548u,3760u,262205u,10u,3762u,3761u,393281u,584u,3763u,578u,548u,3757u,
196670u,3763u,3762u,131321u,3754u,131320u,3764u,262205u,6u,3765u,258u,262205u,6u,3766u,3698u,327808u,
6u,3767u,3765u,3766u,262205u,6u,3768u,257u,262205u,6u,3769u,3712u,327808u,6u,3770u,3768u,
3769u,393281u,584u,3771u,578u,548u,3770u,262205u,10u,3772u,3771u,393281u,584u,3773u,578u,548u,
3767u,196670u,3773u,3772u,131321u,3754u,131320u,3754u,65789u,65592u,327734u,2u,280u,0u,266u,196663u,
9u,267u,196663u,9u,268u,196663u,9u,269u,196663u,9u,270u,196663u,9u,271u,196663u,9u,
272u,196663u,9u,273u,196663u,9u,274u,196663u,9u,275u,196663u,9u,276u,196663u,9u,277u,
196663u,9u,278u,196663u,58u,279u,131320u,281u,262203u,9u,3774u,7u,262203u,9u,3786u,7u,
262203u,9u,3792u,7u,262203u,9u,3798u,7u,262203u,9u,3802u,7u,262203u,9u,3803u,7u,
262203u,9u,3808u,7u,262203u,9u,3838u,7u,262203u,9u,3842u,7u,262203u,58u,3920u,7u,
262203u,9u,3935u,7u,262203u,11u,3936u,7u,262203u,11u,3946u,7u,262205u,6u,3775u,549u,
262205u,6u,3776u,550u,327812u,6u,3777u,3775u,3776u,262205u,6u,3778u,547u,327808u,6u,3779u,
3777u,3778u,196670u,3774u,3779u,262205u,6u,3780u,3774u,262205u,6u,3781u,269u,327855u,57u,3782u,
3780u,3781u,196855u,3784u,0u,262394u,3782u,3783u,3784u,131320u,3783u,65789u,131320u,3784u,262205u,6u,
3787u,3774u,262205u,6u,3788u,273u,262205u,6u,3789u,274u,327812u,6u,3790u,3788u,3789u,327815u,
6u,3791u,3787u,3790u,196670u,3786u,3791u,262205u,6u,3793u,3774u,262205u,6u,3794u,274u,327815u,
6u,3795u,3793u,3794u,262205u,6u,3796u,273u,327819u,6u,3797u,3795u,3796u,196670u,3792u,3797u,
262205u,6u,3799u,3774u,262205u,6u,3800u,274u,327819u,6u,3801u,3799u,3800u,196670u,3798u,3801u,
196670u,3802u,3173u,262205u,6u,3804u,277u,196670u,3803u,3804u,327737u,57u,3805u,65u,3803u,196855u,
3807u,0u,262394u,3805u,3806u,3837u,131320u,3806u,262205u,6u,3809u,3786u,262205u,6u,3810u,277u,
327812u,6u,3811u,3809u,3810u,262205u,6u,3812u,277u,327812u,6u,3813u,3811u,3812u,262205u,6u,
3814u,3792u,262205u,6u,3815u,277u,327819u,6u,3816u,3814u,3815u,262205u,6u,3817u,277u,327812u,
6u,3818u,3816u,3817u,327808u,6u,3819u,3813u,3818u,262205u,6u,3820u,3798u,262205u,6u,3821u,
277u,327819u,6u,3822u,3820u,3821u,327808u,6u,3823u,3819u,3822u,196670u,3808u,3823u,262205u,6u,
3824u,3808u,262205u,6u,3825u,271u,327812u,6u,3826u,3824u,3825u,262205u,6u,3827u,3792u,262205u,
6u,3828u,277u,327815u,6u,3829u,3827u,3828u,327808u,6u,3830u,3826u,3829u,262205u,6u,3831u,
272u,327812u,6u,3832u,3830u,3831u,262205u,6u,3833u,3798u,262205u,6u,3834u,277u,327815u,6u,
3835u,3833u,3834u,327808u,6u,3836u,3832u,3835u,196670u,3802u,3836u,131321u,3807u,131320u,3837u,262205u,
6u,3839u,3792u,262205u,6u,3840u,275u,327810u,6u,3841u,3839u,3840u,196670u,3838u,3841u,262205u,
6u,3843u,3798u,262205u,6u,3844u,276u,327810u,6u,3845u,3843u,3844u,196670u,3842u,3845u,262205u,
6u,3846u,278u,327850u,57u,3847u,3846u,2490u,196855u,3849u,0u,262394u,3847u,3848u,3849u,131320u,
3848u,262205u,6u,3850u,3838u,327857u,57u,3851u,3850u,548u,196855u,3853u,0u,262394u,3851u,3852u,
3853u,131320u,3852u,262205u,6u,3854u,3838u,262270u,6u,3855u,3854u,196670u,3838u,3855u,131321u,3853u,
131320u,3853u,262205u,6u,3856u,3838u,262205u,6u,3857u,271u,327855u,57u,3858u,3856u,3857u,196855u,
3860u,0u,262394u,3858u,3859u,3860u,131320u,3859u,262205u,6u,3861u,271u,327812u,6u,3862u,2490u,
3861u,262205u,6u,3863u,3838u,327810u,6u,3864u,3862u,3863u,327810u,6u,3865u,3864u,2490u,196670u,
3838u,3865u,131321u,3860u,131320u,3860u,262205u,6u,3866u,3842u,327857u,57u,3867u,3866u,548u,196855u,
3869u,0u,262394u,3867u,3868u,3869u,131320u,3868u,262205u,6u,3870u,3842u,262270u,6u,3871u,3870u,
196670u,3842u,3871u,131321u,3869u,131320u,3869u,262205u,6u,3872u,3842u,262205u,6u,3873u,272u,327855u,
57u,3874u,3872u,3873u,196855u,3876u,0u,262394u,3874u,3875u,3876u,131320u,3875u,262205u,6u,3877u,
272u,327812u,6u,3878u,2490u,3877u,262205u,6u,3879u,3842u,327810u,6u,3880u,3878u,3879u,327810u,
6u,3881u,3880u,2490u,196670u,3842u,3881u,131321u,3876u,131320u,3876u,131321u,3849u,131320u,3849u,262205u,
6u,3882u,278u,327851u,57u,3883u,3882u,551u,196855u,3885u,0u,262394u,3883u,3884u,3885u,131320u,
3884u,262205u,6u,3886u,3838u,262205u,6u,3887u,271u,327810u,6u,3888u,3887u,551u,458764u,6u,
3889u,1u,39u,3886u,3888u,458764u,6u,3890u,1u,42u,548u,3889u,196670u,3838u,3890u,262205u,
6u,3891u,3842u,262205u,6u,3892u,272u,327810u,6u,3893u,3892u,551u,458764u,6u,3894u,1u,
39u,3891u,3893u,458764u,6u,3895u,1u,42u,548u,3894u,196670u,3842u,3895u,131321u,3885u,131320u,
3885u,262205u,6u,3896u,3838u,327855u,57u,3897u,3896u,548u,262205u,6u,3898u,3838u,262205u,6u,
3899u,271u,327857u,57u,3900u,3898u,3899u,327847u,57u,3901u,3897u,3900u,262205u,6u,3902u,3842u,
327855u,57u,3903u,3902u,548u,327847u,57u,3904u,3901u,3903u,262205u,6u,3905u,3842u,262205u,6u,
3906u,272u,327857u,57u,3907u,3905u,3906u,327847u,57u,3908u,3904u,3907u,196855u,3910u,0u,262394u,
3908u,3909u,3910u,131320u,3909u,262205u,6u,3911u,3786u,262205u,6u,3912u,271u,327812u,6u,3913u,
3911u,3912u,262205u,6u,3914u,3838u,327808u,6u,3915u,3913u,3914u,262205u,6u,3916u,272u,327812u,
6u,3917u,3915u,3916u,262205u,6u,3918u,3842u,327808u,6u,3919u,3917u,3918u,196670u,3802u,3919u,
131321u,3910u,131320u,3910u,131321u,3807u,131320u,3807u,262205u,57u,3921u,279u,196670u,3920u,3921u,327737u,
57u,3922u,61u,3920u,196855u,3924u,0u,262394u,3922u,3923u,3940u,131320u,3923u,262205u,6u,3925u,
3802u,327855u,57u,3926u,3925u,548u,196855u,3928u,0u,262394u,3926u,3927u,3928u,131320u,3927u,262205u,
6u,3929u,268u,262205u,6u,3930u,3802u,327808u,6u,3931u,3929u,3930u,262205u,6u,3932u,267u,
262205u,6u,3933u,3774u,327808u,6u,3934u,3932u,3933u,196670u,3935u,3931u,393281u,584u,3937u,578u,
548u,3934u,262205u,10u,3938u,3937u,196670u,3936u,3938u,393273u,2u,3939u,55u,3935u,3936u,131321u,
3928u,131320u,3928u,131321u,3924u,131320u,3940u,262205u,6u,3941u,268u,262205u,6u,3942u,3774u,327808u,
6u,3943u,3941u,3942u,262205u,6u,3944u,3802u,327855u,57u,3945u,3944u,548u,196855u,3948u,0u,
262394u,3945u,3947u,3954u,131320u,3947u,262205u,6u,3949u,267u,262205u,6u,3950u,3802u,327808u,6u,
3951u,3949u,3950u,393281u,584u,3952u,578u,548u,3951u,262205u,10u,3953u,3952u,196670u,3946u,3953u,
131321u,3948u,131320u,3954u,196670u,3946u,630u,131321u,3948u,131320u,3948u,262205u,10u,3955u,3946u,393281u,
584u,3956u,578u,548u,3943u,196670u,3956u,3955u,131321u,3924u,131320u,3924u,65789u,65592u,327734u,2u,
293u,0u,284u,196663u,9u,285u,196663u,9u,286u,196663u,9u,287u,196663u,9u,288u,196663u,
9u,289u,196663u,9u,290u,196663u,283u,291u,196663u,9u,292u,131320u,294u,262203u,9u,3957u,
7u,262203u,9u,3969u,7u,262203u,9u,3977u,7u,262203u,9u,3985u,7u,262203u,11u,3990u,
7u,262203u,11u,3993u,7u,262203u,11u,4006u,7u,262203u,9u,4007u,7u,262203u,11u,4010u,
7u,262203u,58u,4024u,7u,262203u,9u,4037u,7u,262203u,11u,4038u,7u,262203u,9u,4041u,
7u,262203u,9u,4047u,7u,262203u,9u,4051u,7u,262203u,9u,4060u,7u,262203u,9u,4070u,
7u,262203u,9u,4080u,7u,262203u,9u,4090u,7u,262203u,9u,4100u,7u,262203u,9u,4104u,
7u,262203u,9u,4193u,7u,262203u,9u,4205u,7u,262203u,9u,4210u,7u,262203u,9u,4234u,
7u,262203u,9u,4249u,7u,262203u,11u,4250u,7u,262203u,9u,4262u,7u,262203u,11u,4263u,
7u,262203u,9u,4285u,7u,262205u,6u,3958u,549u,262205u,6u,3959u,550u,327812u,6u,3960u,
3958u,3959u,262205u,6u,3961u,547u,327808u,6u,3962u,3960u,3961u,196670u,3957u,3962u,262205u,6u,
3963u,3957u,262205u,6u,3964u,292u,327855u,57u,3965u,3963u,3964u,196855u,3967u,0u,262394u,3965u,
3966u,3967u,131320u,3966u,65789u,131320u,3967u,262205u,6u,3970u,3957u,327745u,9u,3971u,291u,2395u,
262205u,6u,3972u,3971u,327745u,9u,3973u,291u,3377u,262205u,6u,3974u,3973u,327812u,6u,3975u,
3972u,3974u,327815u,6u,3976u,3970u,3975u,196670u,3969u,3976u,262205u,6u,3978u,3957u,327745u,9u,
3979u,291u,3377u,262205u,6u,3980u,3979u,327815u,6u,3981u,3978u,3980u,327745u,9u,3982u,291u,
2395u,262205u,6u,3983u,3982u,327819u,6u,3984u,3981u,3983u,196670u,3977u,3984u,262205u,6u,3986u,
3957u,327745u,9u,3987u,291u,3377u,262205u,6u,3988u,3987u,327819u,6u,3989u,3986u,3988u,196670u,
3985u,3989u,327745u,58u,3991u,291u,3472u,262205u,57u,3992u,3991u,196855u,3995u,0u,262394u,3992u,
3994u,4004u,131320u,3994u,262205u,6u,3996u,286u,327745u,9u,3997u,291u,2754u,262205u,6u,3998u,
3997u,262205u,6u,3999u,3969u,327808u,6u,4000u,3998u,3999u,327808u,6u,4001u,3996u,4000u,393281u,
584u,4002u,578u,548u,4001u,262205u,10u,4003u,4002u,196670u,3993u,4003u,131321u,3995u,131320u,4004u,
196670u,3993u,630u,131321u,3995u,131320u,3995u,262205u,10u,4005u,3993u,196670u,3990u,4005u,262205u,6u,
4008u,287u,196670u,4007u,4008u,327737u,57u,4009u,65u,4007u,196855u,4012u,0u,262394u,4009u,4011u,
4018u,131320u,4011u,262205u,6u,4013u,287u,262205u,6u,4014u,3957u,327808u,6u,4015u,4013u,4014u,
393281u,584u,4016u,578u,548u,4015u,262205u,10u,4017u,4016u,196670u,4010u,4017u,131321u,4012u,131320u,
4018u,196670u,4010u,630u,131321u,4012u,131320u,4012u,262205u,10u,4019u,4010u,196670u,4006u,4019u,262205u,
6u,4020u,287u,327851u,57u,4021u,4020u,548u,196855u,4023u,0u,262394u,4021u,4022u,4023u,131320u,
4022u,327745u,58u,4025u,291u,3472u,262205u,57u,4026u,4025u,196670u,4024u,4026u,327737u,57u,4027u,
61u,4024u,131321u,4023u,131320u,4023u,458997u,57u,4028u,4021u,4012u,4027u,4022u,196855u,4030u,0u,
262394u,4028u,4029u,4030u,131320u,4029u,262205u,6u,4031u,290u,327745u,9u,4032u,291u,2754u,262205u,
6u,4033u,4032u,327808u,6u,4034u,4031u,4033u,262205u,6u,4035u,3969u,327808u,6u,4036u,4034u,
4035u,196670u,4037u,4036u,262205u,10u,4039u,4006u,196670u,4038u,4039u,393273u,2u,4040u,55u,4037u,
4038u,131321u,4030u,131320u,4030u,327745u,9u,4042u,291u,548u,262205u,6u,4043u,4042u,327850u,57u,
4044u,4043u,2490u,262205u,6u,4045u,3969u,393385u,6u,4046u,4044u,4045u,548u,196670u,4041u,4046u,
327745u,9u,4048u,291u,548u,262205u,6u,4049u,4048u,327850u,57u,4050u,4049u,2490u,196855u,4053u,
0u,262394u,4050u,4052u,4056u,131320u,4052u,262205u,6u,4054u,3969u,327808u,6u,4055u,4054u,551u,
196670u,4051u,4055u,131321u,4053u,131320u,4056u,327745u,9u,4057u,291u,551u,262205u,6u,4058u,4057u,
196670u,4051u,4058u,131321u,4053u,131320u,4053u,262205u,6u,4059u,4051u,196670u,4047u,4059u,262205u,6u,
4061u,4041u,196670u,4060u,4061u,131321u,4062u,131320u,4062u,262390u,4064u,4065u,0u,131321u,4066u,131320u,
4066u,262205u,6u,4067u,4060u,262205u,6u,4068u,4047u,327857u,57u,4069u,4067u,4068u,262394u,4069u,
4063u,4064u,131320u,4063u,196670u,4070u,548u,131321u,4071u,131320u,4071u,262390u,4073u,4074u,0u,131321u,
4075u,131320u,4075u,262205u,6u,4076u,4070u,327745u,9u,4077u,291u,2510u,262205u,6u,4078u,4077u,
327857u,57u,4079u,4076u,4078u,262394u,4079u,4072u,4073u,131320u,4072u,196670u,4080u,548u,131321u,4081u,
131320u,4081u,262390u,4083u,4084u,0u,131321u,4085u,131320u,4085u,262205u,6u,4086u,4080u,327745u,9u,
4087u,291u,2624u,262205u,6u,4088u,4087u,327857u,57u,4089u,4086u,4088u,262394u,4089u,4082u,4083u,
131320u,4082u,262205u,6u,4091u,3977u,327745u,9u,4092u,291u,2750u,262205u,6u,4093u,4092u,327812u,
6u,4094u,4091u,4093u,262205u,6u,4095u,4070u,327808u,6u,4096u,4094u,4095u,327745u,9u,4097u,
291u,2409u,262205u,6u,4098u,4097u,327810u,6u,4099u,4096u,4098u,196670u,4090u,4099u,327745u,9u,
4101u,291u,548u,262205u,6u,4102u,4101u,327850u,57u,4103u,4102u,551u,196855u,4106u,0u,262394u,
4103u,4105u,4107u,131320u,4105u,196670u,4104u,548u,131321u,4106u,131320u,4107u,262205u,6u,4108u,3985u,
327745u,9u,4109u,291u,2750u,262205u,6u,4110u,4109u,327812u,6u,4111u,4108u,4110u,262205u,6u,
4112u,4080u,327808u,6u,4113u,4111u,4112u,327745u,9u,4114u,291u,2409u,262205u,6u,4115u,4114u,
327810u,6u,4116u,4113u,4115u,196670u,4104u,4116u,131321u,4106u,131320u,4106u,262205u,6u,4117u,4104u,
196670u,4100u,4117u,327745u,9u,4118u,291u,548u,262205u,6u,4119u,4118u,327850u,57u,4120u,4119u,
2404u,196855u,4122u,0u,262394u,4120u,4121u,4122u,131320u,4121u,262205u,6u,4123u,3977u,327745u,9u,
4124u,291u,2409u,262205u,6u,4125u,4124u,327808u,6u,4126u,4123u,4125u,262205u,6u,4127u,4070u,
327810u,6u,4128u,4126u,4127u,196670u,4090u,4128u,262205u,6u,4129u,3985u,327745u,9u,4130u,291u,
2409u,262205u,6u,4131u,4130u,327808u,6u,4132u,4129u,4131u,262205u,6u,4133u,4080u,327810u,6u,
4134u,4132u,4133u,196670u,4100u,4134u,262205u,6u,4135u,4090u,327857u,57u,4136u,4135u,548u,262205u,
6u,4137u,4100u,327857u,57u,4138u,4137u,548u,327846u,57u,4139u,4136u,4138u,262312u,57u,4140u,
4139u,196855u,4142u,0u,262394u,4140u,4141u,4142u,131320u,4141u,262205u,6u,4143u,4090u,327745u,9u,
4144u,291u,2750u,262205u,6u,4145u,4144u,327819u,6u,4146u,4143u,4145u,327851u,57u,4147u,4146u,
548u,131321u,4142u,131320u,4142u,458997u,57u,4148u,4139u,4121u,4147u,4141u,262312u,57u,4149u,4148u,
196855u,4151u,0u,262394u,4149u,4150u,4151u,131320u,4150u,262205u,6u,4152u,4100u,327745u,9u,4153u,
291u,2750u,262205u,6u,4154u,4153u,327819u,6u,4155u,4152u,4154u,327851u,57u,4156u,4155u,548u,
131321u,4151u,131320u,4151u,458997u,57u,4157u,4148u,4142u,4156u,4150u,196855u,4159u,0u,262394u,4157u,
4158u,4159u,131320u,4158u,131321u,4084u,131320u,4159u,327745u,9u,4161u,291u,2750u,262205u,6u,4162u,
4161u,262205u,6u,4163u,4090u,327815u,6u,4164u,4163u,4162u,196670u,4090u,4164u,327745u,9u,4165u,
291u,2750u,262205u,6u,4166u,4165u,262205u,6u,4167u,4100u,327815u,6u,4168u,4167u,4166u,196670u,
4100u,4168u,131321u,4122u,131320u,4122u,262205u,6u,4169u,4090u,327857u,57u,4170u,4169u,548u,262312u,
57u,4171u,4170u,196855u,4173u,0u,262394u,4171u,4172u,4173u,131320u,4172u,262205u,6u,4174u,4090u,
327745u,9u,4175u,291u,2404u,262205u,6u,4176u,4175u,327855u,57u,4177u,4174u,4176u,131321u,4173u,
131320u,4173u,458997u,57u,4178u,4170u,4122u,4177u,4172u,262205u,6u,4179u,4100u,327857u,57u,4180u,
4179u,548u,327846u,57u,4181u,4178u,4180u,262312u,57u,4182u,4181u,196855u,4184u,0u,262394u,4182u,
4183u,4184u,131320u,4183u,262205u,6u,4185u,4100u,327745u,9u,4186u,291u,3118u,262205u,6u,4187u,
4186u,327855u,57u,4188u,4185u,4187u,131321u,4184u,131320u,4184u,458997u,57u,4189u,4181u,4173u,4188u,
4183u,196855u,4191u,0u,262394u,4189u,4190u,4191u,131320u,4190u,131321u,4084u,131320u,4191u,262205u,6u,
4194u,4060u,327745u,9u,4195u,291u,2404u,262205u,6u,4196u,4195u,327812u,6u,4197u,4194u,4196u,
262205u,6u,4198u,4090u,327808u,6u,4199u,4197u,4198u,327745u,9u,4200u,291u,3118u,262205u,6u,
4201u,4200u,327812u,6u,4202u,4199u,4201u,262205u,6u,4203u,4100u,327808u,6u,4204u,4202u,4203u,
196670u,4193u,4204u,262205u,6u,4206u,3969u,327745u,9u,4207u,291u,548u,262205u,6u,4208u,4207u,
327850u,57u,4209u,4208u,2490u,196855u,4212u,0u,262394u,4209u,4211u,4213u,131320u,4211u,196670u,4210u,
551u,131321u,4212u,131320u,4213u,327745u,9u,4214u,291u,551u,262205u,6u,4215u,4214u,196670u,4210u,
4215u,131321u,4212u,131320u,4212u,262205u,6u,4216u,4210u,327812u,6u,4217u,4206u,4216u,327745u,9u,
4218u,291u,548u,262205u,6u,4219u,4218u,327850u,57u,4220u,4219u,2490u,262205u,6u,4221u,4060u,
393385u,6u,4222u,4220u,548u,4221u,327808u,6u,4223u,4217u,4222u,327745u,9u,4224u,291u,2510u,
262205u,6u,4225u,4224u,327812u,6u,4226u,4223u,4225u,262205u,6u,4227u,4070u,327808u,6u,4228u,
4226u,4227u,327745u,9u,4229u,291u,2624u,262205u,6u,4230u,4229u,327812u,6u,4231u,4228u,4230u,
262205u,6u,4232u,4080u,327808u,6u,4233u,4231u,4232u,196670u,4205u,4233u,262205u,6u,4235u,287u,
196670u,4234u,4235u,327737u,57u,4236u,65u,4234u,196855u,4238u,0u,262394u,4236u,4237u,4265u,131320u,
4237u,262205u,6u,4239u,289u,262205u,6u,4240u,4193u,327808u,6u,4241u,4239u,4240u,262205u,10u,
4242u,4006u,262205u,6u,4243u,286u,262205u,6u,4244u,4205u,327808u,6u,4245u,4243u,4244u,393281u,
584u,4246u,578u,548u,4245u,262205u,10u,4247u,4246u,327813u,10u,4248u,4242u,4247u,196670u,4249u,
4241u,196670u,4250u,4248u,393273u,2u,4251u,55u,4249u,4250u,262205u,6u,4252u,290u,262205u,6u,
4253u,4205u,327808u,6u,4254u,4252u,4253u,262205u,10u,4255u,4006u,262205u,6u,4256u,285u,262205u,
6u,4257u,4193u,327808u,6u,4258u,4256u,4257u,393281u,584u,4259u,578u,548u,4258u,262205u,10u,
4260u,4259u,327813u,10u,4261u,4255u,4260u,196670u,4262u,4254u,196670u,4263u,4261u,393273u,2u,4264u,
55u,4262u,4263u,131321u,4238u,131320u,4265u,262205u,6u,4266u,285u,262205u,6u,4267u,4193u,327808u,
6u,4268u,4266u,4267u,393281u,584u,4269u,578u,548u,4268u,262205u,10u,4270u,4269u,262205u,6u,
4271u,286u,262205u,6u,4272u,4205u,327808u,6u,4273u,4271u,4272u,393281u,584u,4274u,578u,548u,
4273u,262205u,10u,4275u,4274u,327813u,10u,4276u,4270u,4275u,262205u,10u,4277u,3990u,327809u,10u,
4278u,4277u,4276u,196670u,3990u,4278u,131321u,4238u,131320u,4238u,131321u,4084u,131320u,4084u,262205u,6u,
4279u,4080u,327808u,6u,4280u,4279u,551u,196670u,4080u,4280u,131321u,4081u,131320u,4083u,131321u,4074u,
131320u,4074u,262205u,6u,4281u,4070u,327808u,6u,4282u,4281u,551u,196670u,4070u,4282u,131321u,4071u,
131320u,4073u,131321u,4065u,131320u,4065u,262205u,6u,4283u,4060u,327808u,6u,4284u,4283u,551u,196670u,
4060u,4284u,131321u,4062u,131320u,4064u,262205u,6u,4286u,287u,196670u,4285u,4286u,327737u,57u,4287u,
65u,4285u,262312u,57u,4288u,4287u,196855u,4290u,0u,262394u,4288u,4289u,4290u,131320u,4289u,262205u,
6u,4291u,288u,262205u,6u,4292u,3957u,327808u,6u,4293u,4291u,4292u,262205u,10u,4294u,3990u,
393281u,584u,4295u,578u,548u,4293u,196670u,4295u,4294u,131321u,4290u,131320u,4290u,65789u,65592u,327734u,
2u,304u,0u,295u,196663u,9u,296u,196663u,9u,297u,196663u,9u,298u,196663u,9u,299u,
196663u,11u,300u,196663u,58u,301u,196663u,58u,302u,196663u,58u,303u,131320u,305u,262203u,9u,
4296u,7u,262203u,58u,4308u,7u,262203u,11u,4336u,7u,262203u,11u,4338u,7u,262203u,58u,
4339u,7u,262203u,11u,4342u,7u,262203u,11u,4354u,7u,262203u,58u,4361u,7u,262203u,11u,
4372u,7u,262203u,11u,4376u,7u,262203u,58u,4393u,7u,262203u,11u,4396u,7u,262203u,11u,
4403u,7u,262203u,11u,4407u,7u,262203u,11u,4429u,7u,262203u,11u,4433u,7u,262205u,6u,
4297u,549u,262205u,6u,4298u,550u,327812u,6u,4299u,4297u,4298u,262205u,6u,4300u,547u,327808u,
6u,4301u,4299u,4300u,196670u,4296u,4301u,262205u,6u,4302u,4296u,262205u,6u,4303u,299u,327855u,
57u,4304u,4302u,4303u,196855u,4306u,0u,262394u,4304u,4305u,4306u,131320u,4305u,65789u,131320u,4306u,
262205u,57u,4309u,303u,196670u,4308u,4309u,327737u,57u,4310u,61u,4308u,262312u,57u,4311u,4310u,
262205u,10u,4312u,300u,327860u,57u,4313u,4312u,630u,327846u,57u,4314u,4311u,4313u,196855u,4316u,
0u,262394u,4314u,4315u,4316u,131320u,4315u,262205u,6u,4317u,298u,262205u,6u,4318u,4296u,327808u,
6u,4319u,4317u,4318u,262205u,6u,4320u,296u,262205u,6u,4321u,4296u,327808u,6u,4322u,4320u,
4321u,393281u,584u,4323u,578u,548u,4322u,262205u,10u,4324u,4323u,393281u,584u,4325u,578u,548u,
4319u,196670u,4325u,4324u,65789u,131320u,4316u,262205u,10u,4327u,300u,327860u,57u,4328u,4327u,701u,
196855u,4330u,0u,262394u,4328u,4329u,4330u,131320u,4329u,262205u,6u,4331u,298u,262205u,6u,4332u,
4296u,327808u,6u,4333u,4331u,4332u,393281u,584u,4334u,578u,548u,4333u,196670u,4334u,630u,65789u,
131320u,4330u,196670u,4336u,4337u,262205u,57u,4340u,301u,196670u,4339u,4340u,327737u,57u,4341u,61u,
4339u,196855u,4344u,0u,262394u,4341u,4343u,4356u,131320u,4343u,262205u,10u,4345u,300u,327811u,10u,
4346u,701u,4345u,262205u,10u,4347u,300u,262205u,10u,4348u,4336u,327813u,10u,4349u,4347u,4348u,
262205u,10u,4350u,4336u,327813u,10u,4351u,4349u,4350u,327809u,10u,4352u,701u,4351u,327813u,10u,
4353u,4346u,4352u,196670u,4354u,4353u,327737u,10u,4355u,76u,4354u,196670u,4342u,4355u,131321u,4344u,
131320u,4356u,262205u,10u,4357u,300u,327811u,10u,4358u,701u,4357u,327816u,10u,4359u,701u,4358u,
196670u,4342u,4359u,131321u,4344u,131320u,4344u,262205u,10u,4360u,4342u,196670u,4338u,4360u,262205u,57u,
4362u,302u,196670u,4361u,4362u,327737u,57u,4363u,61u,4361u,196855u,4365u,0u,262394u,4363u,4364u,
4389u,131320u,4364u,262205u,6u,4366u,298u,262205u,6u,4367u,4296u,327808u,6u,4368u,4366u,4367u,
262205u,6u,4369u,297u,262205u,6u,4370u,4296u,327808u,6u,4371u,4369u,4370u,393281u,584u,4373u,
578u,548u,4371u,262205u,10u,4374u,4373u,196670u,4372u,4374u,327737u,57u,4375u,69u,4372u,196855u,
4378u,0u,262394u,4375u,4377u,4386u,131320u,4377u,262205u,6u,4379u,296u,262205u,6u,4380u,4296u,
327808u,6u,4381u,4379u,4380u,393281u,584u,4382u,578u,548u,4381u,262205u,10u,4383u,4382u,262205u,
10u,4384u,4338u,327813u,10u,4385u,4383u,4384u,196670u,4376u,4385u,131321u,4378u,131320u,4386u,196670u,
4376u,630u,131321u,4378u,131320u,4378u,262205u,10u,4387u,4376u,393281u,584u,4388u,578u,548u,4368u,
196670u,4388u,4387u,131321u,4365u,131320u,4389u,262205u,6u,4390u,298u,262205u,6u,4391u,4296u,327808u,
6u,4392u,4390u,4391u,262205u,57u,4394u,301u,196670u,4393u,4394u,327737u,57u,4395u,61u,4393u,
196855u,4398u,0u,262394u,4395u,4397u,4425u,131320u,4397u,262205u,10u,4399u,4338u,262205u,6u,4400u,
297u,262205u,6u,4401u,4296u,327808u,6u,4402u,4400u,4401u,393281u,584u,4404u,578u,548u,4402u,
262205u,10u,4405u,4404u,196670u,4403u,4405u,327737u,57u,4406u,69u,4403u,196855u,4409u,0u,262394u,
4406u,4408u,4415u,131320u,4408u,262205u,6u,4410u,296u,262205u,6u,4411u,4296u,327808u,6u,4412u,
4410u,4411u,393281u,584u,4413u,578u,548u,4412u,262205u,10u,4414u,4413u,196670u,4407u,4414u,131321u,
4409u,131320u,4415u,262205u,10u,4416u,4336u,196670u,4407u,4416u,131321u,4409u,131320u,4409u,262205u,10u,
4417u,4407u,327813u,10u,4418u,4399u,4417u,262205u,10u,4419u,4338u,262205u,10u,4420u,4336u,327813u,
10u,4421u,4419u,4420u,262205u,10u,4422u,300u,327813u,10u,4423u,4421u,4422u,327811u,10u,4424u,
4418u,4423u,196670u,4396u,4424u,131321u,4398u,131320u,4425u,262205u,6u,4426u,297u,262205u,6u,4427u,
4296u,327808u,6u,4428u,4426u,4427u,393281u,584u,4430u,578u,548u,4428u,262205u,10u,4431u,4430u,
196670u,4429u,4431u,327737u,57u,4432u,69u,4429u,196855u,4435u,0u,262394u,4432u,4434u,4443u,131320u,
4434u,262205u,6u,4436u,296u,262205u,6u,4437u,4296u,327808u,6u,4438u,4436u,4437u,393281u,584u,
4439u,578u,548u,4438u,262205u,10u,4440u,4439u,262205u,10u,4441u,4338u,327813u,10u,4442u,4440u,
4441u,196670u,4433u,4442u,131321u,4435u,131320u,4443u,196670u,4433u,630u,131321u,4435u,131320u,4435u,262205u,
10u,4444u,4433u,196670u,4396u,4444u,131321u,4398u,131320u,4398u,262205u,10u,4445u,4396u,393281u,584u,
4446u,578u,548u,4392u,196670u,4446u,4445u,131321u,4365u,131320u,4365u,65789u,65592u,327734u,2u,315u,
0u,306u,196663u,9u,307u,196663u,9u,308u,196663u,9u,309u,196663u,9u,310u,196663u,9u,
311u,196663u,9u,312u,196663u,9u,313u,196663u,58u,314u,131320u,316u,262203u,9u,4447u,7u,
262203u,11u,4459u,7u,262203u,9u,4471u,7u,262203u,58u,4480u,7u,262203u,11u,4483u,7u,
262203u,11u,4543u,7u,262203u,11u,4547u,7u,262205u,6u,4448u,549u,262205u,6u,4449u,550u,
327812u,6u,4450u,4448u,4449u,262205u,6u,4451u,547u,327808u,6u,4452u,4450u,4451u,196670u,4447u,
4452u,262205u,6u,4453u,4447u,262205u,6u,4454u,313u,327855u,57u,4455u,4453u,4454u,196855u,4457u,
0u,262394u,4455u,4456u,4457u,131320u,4456u,65789u,131320u,4457u,262205u,6u,4462u,308u,262205u,6u,
4463u,4447u,327808u,6u,4464u,4462u,4463u,393281u,584u,4465u,578u,548u,4464u,262205u,10u,4466u,
4465u,458764u,10u,4467u,1u,37u,4461u,4466u,458764u,10u,4468u,1u,40u,4460u,4467u,327813u,
10u,4469u,687u,4468u,393228u,10u,4470u,1u,27u,4469u,196670u,4459u,4470u,262205u,6u,4472u,
310u,196670u,4471u,4472u,327737u,57u,4473u,65u,4471u,262312u,57u,4474u,4473u,196855u,4476u,0u,
262394u,4474u,4475u,4476u,131320u,4475u,262205u,6u,4477u,311u,262205u,6u,4478u,4447u,327808u,6u,
4479u,4477u,4478u,262205u,57u,4481u,314u,196670u,4480u,4481u,327737u,57u,4482u,61u,4480u,196855u,
4485u,0u,262394u,4482u,4484u,4499u,131320u,4484u,262205u,6u,4486u,307u,262205u,6u,4487u,4447u,
327808u,6u,4488u,4486u,4487u,393281u,584u,4489u,578u,548u,4488u,262205u,10u,4490u,4489u,262205u,
10u,4491u,4459u,262205u,6u,4492u,309u,262205u,6u,4493u,4447u,327808u,6u,4494u,4492u,4493u,
393281u,584u,4495u,578u,548u,4494u,262205u,10u,4496u,4495u,327813u,10u,4497u,4491u,4496u,327809u,
10u,4498u,4490u,4497u,196670u,4483u,4498u,131321u,4485u,131320u,4499u,262205u,6u,4500u,307u,262205u,
6u,4501u,4447u,327808u,6u,4502u,4500u,4501u,393281u,584u,4503u,578u,548u,4502u,262205u,10u,
4504u,4503u,196670u,4483u,4504u,131321u,4485u,131320u,4485u,262205u,10u,4505u,4483u,393281u,584u,4506u,
578u,548u,4479u,196670u,4506u,4505u,65789u,131320u,4476u,262205u,6u,4508u,311u,262205u,6u,4509u,
4447u,327808u,6u,4510u,4508u,4509u,262205u,6u,4511u,310u,262205u,6u,4512u,4447u,327808u,6u,
4513u,4511u,4512u,393281u,584u,4514u,578u,548u,4513u,262205u,10u,4515u,4514u,393281u,584u,4516u,
578u,548u,4510u,196670u,4516u,4515u,262205u,6u,4517u,312u,262205u,6u,4518u,4447u,327808u,6u,
4519u,4517u,4518u,393281u,584u,4520u,578u,548u,4519u,196670u,4520u,630u,262205u,6u,4521u,309u,
327851u,57u,4522u,4521u,548u,196855u,4524u,0u,262394u,4522u,4523u,4524u,131320u,4523u,262205u,6u,
4525u,308u,262205u,6u,4526u,4447u,327808u,6u,4527u,4525u,4526u,393281u,584u,4528u,578u,548u,
4527u,262205u,10u,4529u,4528u,327870u,57u,4530u,4529u,4460u,131321u,4524u,131320u,4524u,458997u,57u,
4531u,4522u,4476u,4530u,4523u,196855u,4533u,0u,262394u,4531u,4532u,4533u,131320u,4532u,262205u,6u,
4534u,308u,262205u,6u,4535u,4447u,327808u,6u,4536u,4534u,4535u,393281u,584u,4537u,578u,548u,
4536u,262205u,10u,4538u,4537u,327868u,57u,4539u,4538u,4461u,131321u,4533u,131320u,4533u,458997u,57u,
4540u,4531u,4524u,4539u,4532u,196855u,4542u,0u,262394u,4540u,4541u,4542u,131320u,4541u,262205u,10u,
4544u,4459u,327866u,57u,4546u,4544u,4545u,196855u,4549u,0u,262394u,4546u,4548u,4552u,131320u,4548u,
262205u,10u,4550u,4459u,327816u,10u,4551u,701u,4550u,196670u,4547u,4551u,131321u,4549u,131320u,4552u,
196670u,4547u,630u,131321u,4549u,131320u,4549u,262205u,10u,4553u,4547u,196670u,4543u,4553u,262205u,6u,
4554u,312u,262205u,6u,4555u,4447u,327808u,6u,4556u,4554u,4555u,262205u,6u,4557u,310u,262205u,
6u,4558u,4447u,327808u,6u,4559u,4557u,4558u,393281u,584u,4560u,578u,548u,4559u,262205u,10u,
4561u,4560u,327813u,10u,4562u,4561u,687u,262205u,10u,4563u,4459u,327813u,10u,4564u,4562u,4563u,
262205u,6u,4565u,309u,262205u,6u,4566u,4447u,327808u,6u,4567u,4565u,4566u,393281u,584u,4568u,
578u,548u,4567u,262205u,10u,4569u,4568u,262205u,6u,4570u,307u,262205u,6u,4571u,4447u,327808u,
6u,4572u,4570u,4571u,393281u,584u,4573u,578u,548u,4572u,262205u,10u,4574u,4573u,327811u,10u,
4575u,4569u,4574u,262205u,10u,4576u,4543u,327813u,10u,4577u,4575u,4576u,327813u,10u,4578u,4564u,
4577u,393281u,584u,4579u,578u,548u,4556u,196670u,4579u,4578u,131321u,4542u,131320u,4542u,65789u,65592u,
327734u,2u,329u,0u,319u,196663u,9u,320u,196663u,9u,321u,196663u,9u,322u,196663u,9u,
323u,196663u,9u,324u,196663u,9u,325u,196663u,318u,326u,196663u,11u,327u,196663u,58u,328u,
131320u,330u,262203u,9u,4580u,7u,262203u,9u,4581u,7u,262203u,9u,4583u,7u,262203u,11u,
4589u,7u,262203u,58u,4590u,7u,262203u,9u,4591u,7u,262203u,58u,4597u,7u,262203u,58u,
4604u,7u,262203u,58u,4610u,7u,262203u,9u,4615u,7u,262203u,11u,4650u,7u,262203u,58u,
4651u,7u,262203u,11u,4654u,7u,262203u,11u,4657u,7u,262203u,9u,4659u,7u,262203u,11u,
4672u,7u,262203u,58u,4673u,7u,262203u,9u,4678u,7u,262203u,11u,4688u,7u,262203u,11u,
4719u,7u,262203u,58u,4720u,7u,262203u,11u,4723u,7u,262203u,11u,4726u,7u,262203u,9u,
4728u,7u,262203u,11u,4744u,7u,262203u,11u,4748u,7u,262203u,11u,4750u,7u,262203u,11u,
4753u,7u,262203u,11u,4763u,7u,262203u,11u,4766u,7u,262203u,9u,4779u,7u,262203u,9u,
4785u,7u,262203u,9u,4795u,7u,262203u,11u,4833u,7u,262203u,11u,4834u,7u,262203u,9u,
4835u,7u,262203u,9u,4845u,7u,262203u,11u,4889u,7u,262203u,9u,4891u,7u,262203u,11u,
4894u,7u,262203u,9u,4896u,7u,262203u,58u,4903u,7u,262203u,58u,4915u,7u,262203u,9u,
4929u,7u,262203u,9u,4939u,7u,262203u,11u,4957u,7u,196670u,4580u,548u,262205u,6u,4582u,
549u,196670u,4581u,4582u,327745u,9u,4584u,326u,2490u,262205u,6u,4585u,4584u,327745u,9u,4586u,
326u,551u,262205u,6u,4587u,4586u,327812u,6u,4588u,4585u,4587u,196670u,4583u,4588u,196670u,4589u,
630u,262205u,6u,4592u,323u,196670u,4591u,4592u,327737u,57u,4593u,65u,4591u,262312u,57u,4594u,
4593u,196855u,4596u,0u,262394u,4594u,4595u,4596u,131320u,4595u,262205u,57u,4598u,328u,196670u,4597u,
4598u,327737u,57u,4599u,61u,4597u,131321u,4596u,131320u,4596u,458997u,57u,4600u,4593u,330u,4599u,
4595u,262312u,57u,4601u,4600u,196855u,4603u,0u,262394u,4601u,4602u,4603u,131320u,4602u,327745u,58u,
4605u,326u,2510u,262205u,57u,4606u,4605u,196670u,4604u,4606u,327737u,57u,4607u,61u,4604u,262312u,
57u,4608u,4607u,131321u,4603u,131320u,4603u,458997u,57u,4609u,4600u,4596u,4608u,4602u,196670u,4590u,
4609u,262205u,57u,4611u,4590u,196670u,4610u,4611u,327737u,57u,4612u,61u,4610u,196855u,4614u,0u,
262394u,4612u,4613u,4614u,131320u,4613u,262205u,6u,4616u,547u,196670u,4615u,4616u,131321u,4617u,131320u,
4617u,262390u,4619u,4620u,0u,131321u,4621u,131320u,4621u,262205u,6u,4622u,4615u,262205u,6u,4623u,
4583u,327857u,57u,4624u,4622u,4623u,262394u,4624u,4618u,4619u,131320u,4618u,262205u,6u,4625u,320u,
262205u,6u,4626u,4615u,327745u,9u,4627u,326u,551u,262205u,6u,4628u,4627u,327815u,6u,4629u,
4626u,4628u,327745u,9u,4630u,326u,548u,262205u,6u,4631u,4630u,327812u,6u,4632u,4629u,4631u,
262205u,6u,4633u,4581u,327808u,6u,4634u,4632u,4633u,327745u,9u,4635u,326u,551u,262205u,6u,
4636u,4635u,327812u,6u,4637u,4634u,4636u,262205u,6u,4638u,4615u,327745u,9u,4639u,326u,551u,
262205u,6u,4640u,4639u,327819u,6u,4641u,4638u,4640u,327808u,6u,4642u,4637u,4641u,327808u,6u,
4643u,4625u,4642u,393281u,584u,4644u,578u,548u,4643u,262205u,10u,4645u,4644u,262205u,10u,4646u,
4589u,327809u,10u,4647u,4646u,4645u,196670u,4589u,4647u,131321u,4620u,131320u,4620u,262205u,6u,4648u,
4615u,327808u,6u,4649u,4648u,551u,196670u,4615u,4649u,131321u,4617u,131320u,4619u,131321u,4614u,131320u,
4614u,262205u,57u,4652u,4590u,196670u,4651u,4652u,327737u,57u,4653u,61u,4651u,196855u,4656u,0u,
262394u,4653u,4655u,4665u,131320u,4655u,262205u,10u,4658u,4589u,196670u,4657u,4658u,262205u,6u,4660u,
4580u,196670u,4659u,4660u,393273u,10u,4661u,46u,4657u,4659u,262205u,6u,4662u,4583u,262255u,10u,
4663u,4662u,327816u,10u,4664u,4661u,4663u,196670u,4654u,4664u,131321u,4656u,131320u,4665u,262205u,6u,
4666u,322u,262205u,6u,4667u,4581u,327808u,6u,4668u,4666u,4667u,393281u,584u,4669u,578u,548u,
4668u,262205u,10u,4670u,4669u,196670u,4654u,4670u,131321u,4656u,131320u,4656u,262205u,10u,4671u,4654u,
196670u,4650u,4671u,196670u,4672u,630u,262205u,57u,4674u,4590u,196670u,4673u,4674u,327737u,57u,4675u,
61u,4673u,196855u,4677u,0u,262394u,4675u,4676u,4677u,131320u,4676u,262205u,6u,4679u,547u,196670u,
4678u,4679u,131321u,4680u,131320u,4680u,262390u,4682u,4683u,0u,131321u,4684u,131320u,4684u,262205u,6u,
4685u,4678u,262205u,6u,4686u,4583u,327857u,57u,4687u,4685u,4686u,262394u,4687u,4681u,4682u,131320u,
4681u,262205u,6u,4689u,320u,262205u,6u,4690u,4678u,327745u,9u,4691u,326u,551u,262205u,6u,
4692u,4691u,327815u,6u,4693u,4690u,4692u,327745u,9u,4694u,326u,548u,262205u,6u,4695u,4694u,
327812u,6u,4696u,4693u,4695u,262205u,6u,4697u,4581u,327808u,6u,4698u,4696u,4697u,327745u,9u,
4699u,326u,551u,262205u,6u,4700u,4699u,327812u,6u,4701u,4698u,4700u,262205u,6u,4702u,4678u,
327745u,9u,4703u,326u,551u,262205u,6u,4704u,4703u,327819u,6u,4705u,4702u,4704u,327808u,6u,
4706u,4701u,4705u,327808u,6u,4707u,4689u,4706u,393281u,584u,4708u,578u,548u,4707u,262205u,10u,
4709u,4708u,262205u,10u,4710u,4650u,327811u,10u,4711u,4709u,4710u,196670u,4688u,4711u,262205u,10u,
4712u,4688u,262205u,10u,4713u,4688u,327813u,10u,4714u,4712u,4713u,262205u,10u,4715u,4672u,327809u,
10u,4716u,4715u,4714u,196670u,4672u,4716u,131321u,4683u,131320u,4683u,262205u,6u,4717u,4678u,327808u,
6u,4718u,4717u,551u,196670u,4678u,4718u,131321u,4680u,131320u,4682u,131321u,4677u,131320u,4677u,262205u,
57u,4721u,4590u,196670u,4720u,4721u,327737u,57u,4722u,61u,4720u,196855u,4725u,0u,262394u,4722u,
4724u,4734u,131320u,4724u,262205u,10u,4727u,4672u,196670u,4726u,4727u,262205u,6u,4729u,4580u,196670u,
4728u,4729u,393273u,10u,4730u,46u,4726u,4728u,262205u,6u,4731u,4583u,262255u,10u,4732u,4731u,
327816u,10u,4733u,4730u,4732u,196670u,4723u,4733u,131321u,4725u,131320u,4734u,262205u,6u,4735u,322u,
327745u,9u,4736u,326u,548u,262205u,6u,4737u,4736u,262205u,6u,4738u,4581u,327808u,6u,4739u,
4737u,4738u,327808u,6u,4740u,4735u,4739u,393281u,584u,4741u,578u,548u,4740u,262205u,10u,4742u,
4741u,196670u,4723u,4742u,131321u,4725u,131320u,4725u,262205u,10u,4743u,4723u,196670u,4719u,4743u,262205u,
10u,4745u,4719u,262205u,10u,4746u,327u,327809u,10u,4747u,4745u,4746u,196670u,4748u,4747u,327737u,
10u,4749u,76u,4748u,196670u,4744u,4749u,327745u,58u,4751u,326u,2404u,262205u,57u,4752u,4751u,
196855u,4755u,0u,262394u,4752u,4754u,4761u,131320u,4754u,262205u,6u,4756u,321u,262205u,6u,4757u,
4581u,327808u,6u,4758u,4756u,4757u,393281u,584u,4759u,578u,548u,4758u,262205u,10u,4760u,4759u,
196670u,4753u,4760u,131321u,4755u,131320u,4761u,196670u,4753u,701u,131321u,4755u,131320u,4755u,262205u,10u,
4762u,4753u,196670u,4750u,4762u,327745u,58u,4764u,326u,3118u,262205u,57u,4765u,4764u,196855u,4768u,
0u,262394u,4765u,4767u,4777u,131320u,4767u,262205u,6u,4769u,321u,327745u,9u,4770u,326u,548u,
262205u,6u,4771u,4770u,262205u,6u,4772u,4581u,327808u,6u,4773u,4771u,4772u,327808u,6u,4774u,
4769u,4773u,393281u,584u,4775u,578u,548u,4774u,262205u,10u,4776u,4775u,196670u,4766u,4776u,131321u,
4768u,131320u,4777u,196670u,4766u,630u,131321u,4768u,131320u,4768u,262205u,10u,4778u,4766u,196670u,4763u,
4778u,262205u,6u,4780u,323u,196670u,4779u,4780u,327737u,57u,4781u,65u,4779u,262312u,57u,4782u,
4781u,196855u,4784u,0u,262394u,4782u,4783u,4784u,131320u,4783u,262205u,6u,4786u,547u,196670u,4785u,
4786u,131321u,4787u,131320u,4787u,262390u,4789u,4790u,0u,131321u,4791u,131320u,4791u,262205u,6u,4792u,
4785u,262205u,6u,4793u,4583u,327857u,57u,4794u,4792u,4793u,262394u,4794u,4788u,4789u,131320u,4788u,
262205u,6u,4796u,4785u,327745u,9u,4797u,326u,551u,262205u,6u,4798u,4797u,327815u,6u,4799u,
4796u,4798u,327745u,9u,4800u,326u,548u,262205u,6u,4801u,4800u,327812u,6u,4802u,4799u,4801u,
262205u,6u,4803u,4581u,327808u,6u,4804u,4802u,4803u,327745u,9u,4805u,326u,551u,262205u,6u,
4806u,4805u,327812u,6u,4807u,4804u,4806u,262205u,6u,4808u,4785u,327745u,9u,4809u,326u,551u,
262205u,6u,4810u,4809u,327819u,6u,4811u,4808u,4810u,327808u,6u,4812u,4807u,4811u,196670u,4795u,
4812u,262205u,6u,4813u,324u,262205u,6u,4814u,4795u,327808u,6u,4815u,4813u,4814u,262205u,6u,
4816u,320u,262205u,6u,4817u,4795u,327808u,6u,4818u,4816u,4817u,393281u,584u,4819u,578u,548u,
4818u,262205u,10u,4820u,4819u,262205u,10u,4821u,4650u,327811u,10u,4822u,4820u,4821u,262205u,10u,
4823u,4744u,327813u,10u,4824u,4822u,4823u,262205u,10u,4825u,4750u,327813u,10u,4826u,4824u,4825u,
262205u,10u,4827u,4763u,327809u,10u,4828u,4826u,4827u,393281u,584u,4829u,578u,548u,4815u,196670u,
4829u,4828u,131321u,4790u,131320u,4790u,262205u,6u,4830u,4785u,327808u,6u,4831u,4830u,551u,196670u,
4785u,4831u,131321u,4787u,131320u,4789u,65789u,131320u,4784u,196670u,4833u,630u,196670u,4834u,630u,262205u,
6u,4836u,547u,196670u,4835u,4836u,131321u,4837u,131320u,4837u,262390u,4839u,4840u,0u,131321u,4841u,
131320u,4841u,262205u,6u,4842u,4835u,262205u,6u,4843u,4583u,327857u,57u,4844u,4842u,4843u,262394u,
4844u,4838u,4839u,131320u,4838u,262205u,6u,4846u,4835u,327745u,9u,4847u,326u,551u,262205u,6u,
4848u,4847u,327815u,6u,4849u,4846u,4848u,327745u,9u,4850u,326u,548u,262205u,6u,4851u,4850u,
327812u,6u,4852u,4849u,4851u,262205u,6u,4853u,4581u,327808u,6u,4854u,4852u,4853u,327745u,9u,
4855u,326u,551u,262205u,6u,4856u,4855u,327812u,6u,4857u,4854u,4856u,262205u,6u,4858u,4835u,
327745u,9u,4859u,326u,551u,262205u,6u,4860u,4859u,327819u,6u,4861u,4858u,4860u,327808u,6u,
4862u,4857u,4861u,196670u,4845u,4862u,262205u,6u,4863u,323u,262205u,6u,4864u,4845u,327808u,6u,
4865u,4863u,4864u,393281u,584u,4866u,578u,548u,4865u,262205u,10u,4867u,4866u,262205u,10u,4868u,
4833u,327809u,10u,4869u,4868u,4867u,196670u,4833u,4869u,262205u,6u,4870u,323u,262205u,6u,4871u,
4845u,327808u,6u,4872u,4870u,4871u,393281u,584u,4873u,578u,548u,4872u,262205u,10u,4874u,4873u,
262205u,6u,4875u,320u,262205u,6u,4876u,4845u,327808u,6u,4877u,4875u,4876u,393281u,584u,4878u,
578u,548u,4877u,262205u,10u,4879u,4878u,262205u,10u,4880u,4650u,327811u,10u,4881u,4879u,4880u,
327813u,10u,4882u,4874u,4881u,262205u,10u,4883u,4744u,327813u,10u,4884u,4882u,4883u,262205u,10u,
4885u,4834u,327809u,10u,4886u,4885u,4884u,196670u,4834u,4886u,131321u,4840u,131320u,4840u,262205u,6u,
4887u,4835u,327808u,6u,4888u,4887u,551u,196670u,4835u,4888u,131321u,4837u,131320u,4839u,262205u,10u,
4890u,4833u,196670u,4889u,4890u,262205u,6u,4892u,4580u,196670u,4891u,4892u,393273u,10u,4893u,46u,
4889u,4891u,196670u,4833u,4893u,262205u,10u,4895u,4834u,196670u,4894u,4895u,262205u,6u,4897u,4580u,
196670u,4896u,4897u,393273u,10u,4898u,46u,4894u,4896u,196670u,4834u,4898u,262205u,6u,4899u,547u,
327850u,57u,4900u,4899u,548u,196855u,4902u,0u,262394u,4900u,4901u,4902u,131320u,4901u,327745u,58u,
4904u,326u,2404u,262205u,57u,4905u,4904u,196670u,4903u,4905u,327737u,57u,4906u,61u,4903u,131321u,
4902u,131320u,4902u,458997u,57u,4907u,4900u,4839u,4906u,4901u,196855u,4909u,0u,262394u,4907u,4908u,
4909u,131320u,4908u,262205u,6u,4910u,325u,262205u,6u,4911u,4581u,327808u,6u,4912u,4910u,4911u,
262205u,10u,4913u,4834u,393281u,584u,4914u,578u,548u,4912u,196670u,4914u,4913u,327745u,58u,4916u,
326u,3118u,262205u,57u,4917u,4916u,196670u,4915u,4917u,327737u,57u,4918u,61u,4915u,196855u,4920u,
0u,262394u,4918u,4919u,4920u,131320u,4919u,262205u,6u,4921u,325u,327745u,9u,4922u,326u,548u,
262205u,6u,4923u,4922u,262205u,6u,4924u,4581u,327808u,6u,4925u,4923u,4924u,327808u,6u,4926u,
4921u,4925u,262205u,10u,4927u,4833u,393281u,584u,4928u,578u,548u,4926u,196670u,4928u,4927u,131321u,
4920u,131320u,4920u,131321u,4909u,131320u,4909u,262205u,6u,4930u,547u,196670u,4929u,4930u,131321u,4931u,
131320u,4931u,262390u,4933u,4934u,0u,131321u,4935u,131320u,4935u,262205u,6u,4936u,4929u,262205u,6u,
4937u,4583u,327857u,57u,4938u,4936u,4937u,262394u,4938u,4932u,4933u,131320u,4932u,262205u,6u,4940u,
4929u,327745u,9u,4941u,326u,551u,262205u,6u,4942u,4941u,327815u,6u,4943u,4940u,4942u,327745u,
9u,4944u,326u,548u,262205u,6u,4945u,4944u,327812u,6u,4946u,4943u,4945u,262205u,6u,4947u,
4581u,327808u,6u,4948u,4946u,4947u,327745u,9u,4949u,326u,551u,262205u,6u,4950u,4949u,327812u,
6u,4951u,4948u,4950u,262205u,6u,4952u,4929u,327745u,9u,4953u,326u,551u,262205u,6u,4954u,
4953u,327819u,6u,4955u,4952u,4954u,327808u,6u,4956u,4951u,4955u,196670u,4939u,4956u,262205u,6u,
4958u,320u,262205u,6u,4959u,4939u,327808u,6u,4960u,4958u,4959u,393281u,584u,4961u,578u,548u,
4960u,262205u,10u,4962u,4961u,262205u,10u,4963u,4650u,327811u,10u,4964u,4962u,4963u,262205u,10u,
4965u,4744u,327813u,10u,4966u,4964u,4965u,196670u,4957u,4966u,262205u,6u,4967u,324u,262205u,6u,
4968u,4939u,327808u,6u,4969u,4967u,4968u,262205u,10u,4970u,4744u,262205u,10u,4971u,4750u,327813u,
10u,4972u,4970u,4971u,262205u,6u,4973u,323u,262205u,6u,4974u,4939u,327808u,6u,4975u,4973u,
4974u,393281u,584u,4976u,578u,548u,4975u,262205u,10u,4977u,4976u,262205u,10u,4978u,4833u,262205u,
6u,4979u,4583u,262255u,10u,4980u,4979u,327816u,10u,4981u,4978u,4980u,327811u,10u,4982u,4977u,
4981u,262205u,10u,4983u,4957u,262205u,10u,4984u,4834u,327813u,10u,4985u,4983u,4984u,262205u,6u,
4986u,4583u,262255u,10u,4987u,4986u,327816u,10u,4988u,4985u,4987u,327811u,10u,4989u,4982u,4988u,
327813u,10u,4990u,4972u,4989u,393281u,584u,4991u,578u,548u,4969u,196670u,4991u,4990u,131321u,4934u,
131320u,4934u,262205u,6u,4992u,4929u,327808u,6u,4993u,4992u,551u,196670u,4929u,4993u,131321u,4931u,
131320u,4933u,65789u,65592u,327734u,2u,336u,0u,108u,196663u,9u,331u,196663u,9u,332u,196663u,
9u,333u,196663u,9u,334u,196663u,9u,335u,131320u,337u,262203u,9u,4994u,7u,262205u,6u,
4995u,549u,262205u,6u,4996u,550u,327812u,6u,4997u,4995u,4996u,262205u,6u,4998u,547u,327808u,
6u,4999u,4997u,4998u,196670u,4994u,4999u,262205u,6u,5000u,4994u,262205u,6u,5001u,333u,327857u,
57u,5002u,5000u,5001u,262205u,6u,5003u,335u,327855u,57u,5004u,5003u,548u,327847u,57u,5005u,
5002u,5004u,196855u,5007u,0u,262394u,5005u,5006u,5007u,131320u,5006u,262205u,6u,5008u,331u,262205u,
6u,5009u,4994u,327808u,6u,5010u,5008u,5009u,262205u,6u,5011u,332u,262205u,6u,5012u,335u,
262205u,6u,5013u,4994u,262205u,6u,5014u,334u,327819u,6u,5015u,5013u,5014u,327808u,6u,5016u,
5012u,5015u,327808u,6u,5017u,5011u,5016u,393281u,584u,5018u,578u,548u,5017u,262205u,10u,5019u,
5018u,393281u,584u,5020u,578u,548u,5010u,262205u,10u,5021u,5020u,327809u,10u,5022u,5021u,5019u,
393281u,584u,5023u,578u,548u,5010u,196670u,5023u,5022u,131321u,5007u,131320u,5007u,65789u,65592u,327734u,
2u,348u,0u,340u,196663u,9u,341u,196663u,9u,342u,196663u,9u,343u,196663u,339u,344u,
196663u,9u,345u,196663u,58u,346u,196663u,9u,347u,131320u,349u,262203u,9u,5024u,7u,262203u,
9u,5036u,7u,262203u,9u,5041u,7u,262203u,9u,5046u,7u,262203u,9u,5047u,7u,262203u,
9u,5053u,7u,262203u,11u,5062u,7u,262203u,58u,5063u,7u,262203u,58u,5069u,7u,262203u,
9u,5096u,7u,262203u,9u,5105u,7u,262203u,58u,5106u,7u,262203u,9u,5111u,7u,262203u,
9u,5128u,7u,262205u,6u,5025u,549u,262205u,6u,5026u,550u,327812u,6u,5027u,5025u,5026u,
262205u,6u,5028u,547u,327808u,6u,5029u,5027u,5028u,196670u,5024u,5029u,262205u,6u,5030u,5024u,
262205u,6u,5031u,347u,327855u,57u,5032u,5030u,5031u,196855u,5034u,0u,262394u,5032u,5033u,5034u,
131320u,5033u,65789u,131320u,5034u,262205u,6u,5037u,5024u,327745u,9u,5038u,344u,548u,262205u,6u,
5039u,5038u,327815u,6u,5040u,5037u,5039u,196670u,5036u,5040u,262205u,6u,5042u,5024u,327745u,9u,
5043u,344u,548u,262205u,6u,5044u,5043u,327819u,6u,5045u,5042u,5044u,196670u,5041u,5045u,262205u,
6u,5048u,345u,196670u,5047u,5048u,327737u,57u,5049u,65u,5047u,327745u,58u,5050u,344u,2510u,
262205u,57u,5051u,5050u,327847u,57u,5052u,5049u,5051u,196855u,5055u,0u,262394u,5052u,5054u,5058u,
131320u,5054u,327745u,9u,5056u,344u,551u,262205u,6u,5057u,5056u,196670u,5053u,5057u,131321u,5055u,
131320u,5058u,327745u,9u,5059u,344u,548u,262205u,6u,5060u,5059u,196670u,5053u,5060u,131321u,5055u,
131320u,5055u,262205u,6u,5061u,5053u,196670u,5046u,5061u,196670u,5062u,630u,262205u,57u,5064u,346u,
196670u,5063u,5064u,327737u,57u,5065u,61u,5063u,262312u,57u,5066u,5065u,196855u,5068u,0u,262394u,
5066u,5067u,5068u,131320u,5067u,327745u,58u,5070u,344u,2510u,262205u,57u,5071u,5070u,196670u,5069u,
5071u,327737u,57u,5072u,61u,5069u,262312u,57u,5073u,5072u,131321u,5068u,131320u,5068u,458997u,57u,
5074u,5066u,5055u,5073u,5067u,196855u,5076u,0u,262394u,5074u,5075u,5076u,131320u,5075u,327745u,9u,
5077u,344u,2750u,262205u,6u,5078u,5077u,327855u,57u,5079u,5078u,548u,131321u,5076u,131320u,5076u,
458997u,57u,5080u,5074u,5068u,5079u,5075u,196855u,5082u,0u,262394u,5080u,5081u,5082u,131320u,5081u,
262205u,6u,5083u,342u,327745u,9u,5084u,344u,2750u,262205u,6u,5085u,5084u,262205u,6u,5086u,
345u,327745u,9u,5087u,344u,548u,262205u,6u,5088u,5087u,327812u,6u,5089u,5086u,5088u,327808u,
6u,5090u,5085u,5089u,262205u,6u,5091u,5041u,327808u,6u,5092u,5090u,5091u,327808u,6u,5093u,
5083u,5092u,393281u,584u,5094u,578u,548u,5093u,262205u,10u,5095u,5094u,196670u,5062u,5095u,131321u,
5082u,131320u,5082u,196670u,5096u,548u,131321u,5097u,131320u,5097u,262390u,5099u,5100u,0u,131321u,5101u,
131320u,5101u,262205u,6u,5102u,5096u,262205u,6u,5103u,5046u,327857u,57u,5104u,5102u,5103u,262394u,
5104u,5098u,5099u,131320u,5098u,327745u,58u,5107u,344u,2510u,262205u,57u,5108u,5107u,196670u,5106u,
5108u,327737u,57u,5109u,61u,5106u,262312u,57u,5110u,5109u,196855u,5113u,0u,262394u,5110u,5112u,
5125u,131320u,5112u,262205u,6u,5114u,345u,327745u,9u,5115u,344u,548u,262205u,6u,5116u,5115u,
327812u,6u,5117u,5114u,5116u,262205u,6u,5118u,5041u,327808u,6u,5119u,5117u,5118u,327745u,9u,
5120u,344u,548u,262205u,6u,5121u,5120u,327812u,6u,5122u,5119u,5121u,262205u,6u,5123u,5096u,
327808u,6u,5124u,5122u,5123u,196670u,5111u,5124u,131321u,5113u,131320u,5125u,262205u,6u,5126u,345u,
327851u,57u,5127u,5126u,548u,196855u,5130u,0u,262394u,5127u,5129u,5150u,131320u,5129u,327745u,9u,
5131u,344u,548u,262205u,6u,5132u,5131u,327745u,9u,5133u,344u,548u,262205u,6u,5134u,5133u,
327812u,6u,5135u,5132u,5134u,262205u,6u,5136u,5096u,327812u,6u,5137u,5136u,2490u,327745u,9u,
5138u,344u,548u,262205u,6u,5139u,5138u,327812u,6u,5140u,5137u,5139u,327808u,6u,5141u,5135u,
5140u,262205u,6u,5142u,345u,327810u,6u,5143u,5142u,551u,327745u,9u,5144u,344u,548u,262205u,
6u,5145u,5144u,327812u,6u,5146u,5143u,5145u,327808u,6u,5147u,5141u,5146u,262205u,6u,5148u,
5041u,327808u,6u,5149u,5147u,5148u,196670u,5128u,5149u,131321u,5130u,131320u,5150u,262205u,6u,5151u,
5096u,327745u,9u,5152u,344u,548u,262205u,6u,5153u,5152u,327812u,6u,5154u,5151u,5153u,262205u,
6u,5155u,5041u,327808u,6u,5156u,5154u,5155u,196670u,5128u,5156u,131321u,5130u,131320u,5130u,262205u,
6u,5157u,5128u,196670u,5111u,5157u,131321u,5113u,131320u,5113u,262205u,6u,5158u,5111u,196670u,5105u,
5158u,262205u,6u,5159u,341u,262205u,6u,5160u,5036u,262205u,6u,5161u,5046u,327812u,6u,5162u,
5160u,5161u,262205u,6u,5163u,5096u,327808u,6u,5164u,5162u,5163u,327808u,6u,5165u,5159u,5164u,
393281u,584u,5166u,578u,548u,5165u,262205u,10u,5167u,5166u,262205u,6u,5168u,342u,262205u,6u,
5169u,5105u,327808u,6u,5170u,5168u,5169u,393281u,584u,5171u,578u,548u,5170u,262205u,10u,5172u,
5171u,327813u,10u,5173u,5167u,5172u,262205u,10u,5174u,5062u,327809u,10u,5175u,5174u,5173u,196670u,
5062u,5175u,131321u,5100u,131320u,5100u,262205u,6u,5176u,5096u,327808u,6u,5177u,5176u,551u,196670u,
5096u,5177u,131321u,5097u,131320u,5099u,262205u,6u,5178u,343u,262205u,6u,5179u,5024u,327808u,6u,
5180u,5178u,5179u,262205u,10u,5181u,5062u,393281u,584u,5182u,578u,548u,5180u,196670u,5182u,5181u,
65789u,65592u,327734u,2u,355u,0u,350u,196663u,9u,351u,196663u,9u,352u,196663u,9u,353u,
196663u,339u,354u,131320u,356u,262203u,9u,5183u,7u,262203u,9u,5184u,7u,262203u,9u,5189u,
7u,262203u,9u,5194u,7u,262203u,9u,5200u,7u,262203u,11u,5210u,7u,262203u,11u,5213u,
7u,262203u,11u,5215u,7u,262203u,9u,5217u,7u,262203u,11u,5228u,7u,262203u,58u,5229u,
7u,262203u,9u,5240u,7u,262203u,11u,5297u,7u,262203u,9u,5299u,7u,262203u,11u,5302u,
7u,262203u,9u,5303u,7u,262203u,11u,5314u,7u,262203u,11u,5337u,7u,262203u,11u,5338u,
7u,262203u,9u,5340u,7u,262203u,9u,5346u,7u,196670u,5183u,548u,262205u,6u,5185u,549u,
327745u,9u,5186u,354u,2404u,262205u,6u,5187u,5186u,327815u,6u,5188u,5185u,5187u,196670u,5184u,
5188u,262205u,6u,5190u,549u,327745u,9u,5191u,354u,2404u,262205u,6u,5192u,5191u,327819u,6u,
5193u,5190u,5192u,196670u,5189u,5193u,327745u,9u,5195u,354u,548u,262205u,6u,5196u,5195u,327745u,
9u,5197u,354u,2490u,262205u,6u,5198u,5197u,327815u,6u,5199u,5196u,5198u,196670u,5194u,5199u,
262205u,6u,5201u,5184u,327745u,9u,5202u,354u,2404u,262205u,6u,5203u,5202u,327812u,6u,5204u,
5201u,5203u,262205u,6u,5205u,5189u,327808u,6u,5206u,5204u,5205u,327745u,9u,5207u,354u,3118u,
262205u,6u,5208u,5207u,327812u,6u,5209u,5206u,5208u,196670u,5200u,5209u,262205u,6u,5211u,5194u,
262255u,10u,5212u,5211u,196670u,5213u,5212u,327737u,10u,5214u,76u,5213u,196670u,5210u,5214u,196670u,
5215u,5216u,262205u,6u,5218u,547u,196670u,5217u,5218u,131321u,5219u,131320u,5219u,262390u,5221u,5222u,
0u,131321u,5223u,131320u,5223u,262205u,6u,5224u,5217u,327745u,9u,5225u,354u,3118u,262205u,6u,
5226u,5225u,327857u,57u,5227u,5224u,5226u,262394u,5227u,5220u,5221u,131320u,5220u,196670u,5228u,1554u,
327745u,58u,5230u,354u,2624u,262205u,57u,5231u,5230u,196670u,5229u,5231u,327737u,57u,5232u,61u,
5229u,262312u,57u,5233u,5232u,262205u,6u,5234u,5217u,262205u,6u,5235u,5189u,327859u,57u,5236u,
5234u,5235u,327846u,57u,5237u,5233u,5236u,196855u,5239u,0u,262394u,5237u,5238u,5239u,131320u,5238u,
196670u,5228u,630u,196670u,5240u,548u,131321u,5241u,131320u,5241u,262390u,5243u,5244u,0u,131321u,5245u,
131320u,5245u,262205u,6u,5246u,5240u,262205u,6u,5247u,5194u,327857u,57u,5248u,5246u,5247u,262394u,
5248u,5242u,5243u,131320u,5242u,262205u,6u,5249u,351u,262205u,6u,5250u,5189u,327745u,9u,5251u,
354u,548u,262205u,6u,5252u,5251u,327812u,6u,5253u,5250u,5252u,262205u,6u,5254u,5184u,262205u,
6u,5255u,5194u,327812u,6u,5256u,5254u,5255u,327808u,6u,5257u,5253u,5256u,262205u,6u,5258u,
5240u,327808u,6u,5259u,5257u,5258u,327808u,6u,5260u,5249u,5259u,393281u,584u,5261u,578u,548u,
5260u,262205u,10u,5262u,5261u,262205u,6u,5263u,352u,262205u,6u,5264u,5217u,327745u,9u,5265u,
354u,548u,262205u,6u,5266u,5265u,327812u,6u,5267u,5264u,5266u,262205u,6u,5268u,5184u,262205u,
6u,5269u,5194u,327812u,6u,5270u,5268u,5269u,327808u,6u,5271u,5267u,5270u,262205u,6u,5272u,
5240u,327808u,6u,5273u,5271u,5272u,327808u,6u,5274u,5263u,5273u,393281u,584u,5275u,578u,548u,
5274u,262205u,10u,5276u,5275u,327813u,10u,5277u,5262u,5276u,262205u,10u,5278u,5228u,327809u,10u,
5279u,5278u,5277u,196670u,5228u,5279u,131321u,5244u,131320u,5244u,262205u,6u,5280u,5240u,327808u,6u,
5281u,5280u,551u,196670u,5240u,5281u,131321u,5241u,131320u,5243u,262205u,10u,5282u,5210u,262205u,10u,
5283u,5228u,327813u,10u,5284u,5283u,5282u,196670u,5228u,5284u,131321u,5239u,131320u,5239u,262205u,6u,
5285u,353u,262205u,6u,5286u,5200u,262205u,6u,5287u,5217u,327808u,6u,5288u,5286u,5287u,327808u,
6u,5289u,5285u,5288u,262205u,10u,5290u,5228u,393281u,584u,5291u,578u,548u,5289u,196670u,5291u,
5290u,262205u,10u,5292u,5215u,262205u,10u,5293u,5228u,458764u,10u,5294u,1u,40u,5292u,5293u,
196670u,5215u,5294u,131321u,5222u,131320u,5222u,262205u,6u,5295u,5217u,327808u,6u,5296u,5295u,551u,
196670u,5217u,5296u,131321u,5219u,131320u,5221u,262205u,10u,5298u,5215u,196670u,5297u,5298u,262205u,6u,
5300u,5183u,196670u,5299u,5300u,393273u,10u,5301u,50u,5297u,5299u,196670u,5215u,5301u,196670u,5302u,
630u,262205u,6u,5304u,547u,196670u,5303u,5304u,131321u,5305u,131320u,5305u,262390u,5307u,5308u,0u,
131321u,5309u,131320u,5309u,262205u,6u,5310u,5303u,327745u,9u,5311u,354u,3118u,262205u,6u,5312u,
5311u,327857u,57u,5313u,5310u,5312u,262394u,5313u,5306u,5307u,131320u,5306u,262205u,6u,5315u,353u,
262205u,6u,5316u,5200u,262205u,6u,5317u,5303u,327808u,6u,5318u,5316u,5317u,327808u,6u,5319u,
5315u,5318u,393281u,584u,5320u,578u,548u,5319u,262205u,10u,5321u,5320u,262205u,10u,5322u,5215u,
327811u,10u,5323u,5321u,5322u,393228u,10u,5324u,1u,27u,5323u,196670u,5314u,5324u,262205u,6u,
5325u,353u,262205u,6u,5326u,5200u,262205u,6u,5327u,5303u,327808u,6u,5328u,5326u,5327u,327808u,
6u,5329u,5325u,5328u,262205u,10u,5330u,5314u,393281u,584u,5331u,578u,548u,5329u,196670u,5331u,
5330u,262205u,10u,5332u,5314u,262205u,10u,5333u,5302u,327809u,10u,5334u,5333u,5332u,196670u,5302u,
5334u,131321u,5308u,131320u,5308u,262205u,6u,5335u,5303u,327808u,6u,5336u,5335u,551u,196670u,5303u,
5336u,131321u,5305u,131320u,5307u,262205u,10u,5339u,5302u,196670u,5338u,5339u,262205u,6u,5341u,5183u,
196670u,5340u,5341u,393273u,10u,5342u,46u,5338u,5340u,327809u,10u,5344u,5342u,5343u,327816u,10u,
5345u,701u,5344u,196670u,5337u,5345u,262205u,6u,5347u,547u,196670u,5346u,5347u,131321u,5348u,131320u,
5348u,262390u,5350u,5351u,0u,131321u,5352u,131320u,5352u,262205u,6u,5353u,5346u,327745u,9u,5354u,
354u,3118u,262205u,6u,5355u,5354u,327857u,57u,5356u,5353u,5355u,262394u,5356u,5349u,5350u,131320u,
5349u,262205u,6u,5357u,353u,262205u,6u,5358u,5200u,262205u,6u,5359u,5346u,327808u,6u,5360u,
5358u,5359u,327808u,6u,5361u,5357u,5360u,262205u,10u,5362u,5337u,393281u,584u,5363u,578u,548u,
5361u,262205u,10u,5364u,5363u,327813u,10u,5365u,5364u,5362u,393281u,584u,5366u,578u,548u,5361u,
196670u,5366u,5365u,131321u,5351u,131320u,5351u,262205u,6u,5367u,5346u,327808u,6u,5368u,5367u,551u,
196670u,5346u,5368u,131321u,5348u,131320u,5350u,65789u,65592u,327734u,2u,363u,0u,357u,196663u,9u,
358u,196663u,9u,359u,196663u,9u,360u,196663u,339u,361u,196663u,9u,362u,131320u,364u,262203u,
9u,5369u,7u,262203u,9u,5381u,7u,262203u,9u,5386u,7u,262203u,9u,5391u,7u,262203u,
11u,5399u,7u,262203u,9u,5400u,7u,262205u,6u,5370u,549u,262205u,6u,5371u,550u,327812u,
6u,5372u,5370u,5371u,262205u,6u,5373u,547u,327808u,6u,5374u,5372u,5373u,196670u,5369u,5374u,
262205u,6u,5375u,5369u,262205u,6u,5376u,362u,327855u,57u,5377u,5375u,5376u,196855u,5379u,0u,
262394u,5377u,5378u,5379u,131320u,5378u,65789u,131320u,5379u,262205u,6u,5382u,5369u,327745u,9u,5383u,
361u,548u,262205u,6u,5384u,5383u,327815u,6u,5385u,5382u,5384u,196670u,5381u,5385u,262205u,6u,
5387u,5369u,327745u,9u,5388u,361u,548u,262205u,6u,5389u,5388u,327819u,6u,5390u,5387u,5389u,
196670u,5386u,5390u,262205u,6u,5392u,5386u,327745u,9u,5393u,361u,548u,262205u,6u,5394u,5393u,
327745u,9u,5395u,361u,2490u,262205u,6u,5396u,5395u,327815u,6u,5397u,5394u,5396u,327815u,6u,
5398u,5392u,5397u,196670u,5391u,5398u,196670u,5399u,630u,196670u,5400u,548u,131321u,5401u,131320u,5401u,
262390u,5403u,5404u,0u,131321u,5405u,131320u,5405u,262205u,6u,5406u,5400u,327745u,9u,5407u,361u,
3118u,262205u,6u,5408u,5407u,327857u,57u,5409u,5406u,5408u,262394u,5409u,5402u,5403u,131320u,5402u,
262205u,6u,5410u,358u,262205u,6u,5411u,5391u,327745u,9u,5412u,361u,2404u,262205u,6u,5413u,
5412u,327812u,6u,5414u,5411u,5413u,262205u,6u,5415u,5381u,327808u,6u,5416u,5414u,5415u,327745u,
9u,5417u,361u,3118u,262205u,6u,5418u,5417u,327812u,6u,5419u,5416u,5418u,262205u,6u,5420u,
5400u,327808u,6u,5421u,5419u,5420u,327808u,6u,5422u,5410u,5421u,393281u,584u,5423u,578u,548u,
5422u,262205u,10u,5424u,5423u,262205u,6u,5425u,359u,262205u,6u,5426u,5400u,327745u,9u,5427u,
361u,548u,262205u,6u,5428u,5427u,327812u,6u,5429u,5426u,5428u,262205u,6u,5430u,5386u,327808u,
6u,5431u,5429u,5430u,327808u,6u,5432u,5425u,5431u,393281u,584u,5433u,578u,548u,5432u,262205u,
10u,5434u,5433u,327813u,10u,5435u,5424u,5434u,262205u,10u,5436u,5399u,327809u,10u,5437u,5436u,
5435u,196670u,5399u,5437u,131321u,5404u,131320u,5404u,262205u,6u,5438u,5400u,327808u,6u,5439u,5438u,
551u,196670u,5400u,5439u,131321u,5401u,131320u,5403u,262205u,6u,5440u,360u,262205u,6u,5441u,5369u,
327808u,6u,5442u,5440u,5441u,262205u,10u,5443u,5399u,393281u,584u,5444u,578u,548u,5442u,196670u,
5444u,5443u,65789u,65592u,327734u,2u,372u,0u,365u,196663u,9u,366u,196663u,9u,367u,196663u,
9u,368u,196663u,9u,369u,196663u,339u,370u,196663u,9u,371u,131320u,373u,262203u,9u,5445u,
7u,262203u,9u,5457u,7u,262203u,9u,5462u,7u,262203u,9u,5467u,7u,262203u,11u,5476u,
7u,262203u,9u,5477u,7u,262203u,9u,5497u,7u,262203u,9u,5507u,7u,262203u,11u,5510u,
7u,262205u,6u,5446u,549u,262205u,6u,5447u,550u,327812u,6u,5448u,5446u,5447u,262205u,6u,
5449u,547u,327808u,6u,5450u,5448u,5449u,196670u,5445u,5450u,262205u,6u,5451u,5445u,262205u,6u,
5452u,371u,327855u,57u,5453u,5451u,5452u,196855u,5455u,0u,262394u,5453u,5454u,5455u,131320u,5454u,
65789u,131320u,5455u,262205u,6u,5458u,5445u,327745u,9u,5459u,370u,548u,262205u,6u,5460u,5459u,
327815u,6u,5461u,5458u,5460u,196670u,5457u,5461u,262205u,6u,5463u,5445u,327745u,9u,5464u,370u,
548u,262205u,6u,5465u,5464u,327819u,6u,5466u,5463u,5465u,196670u,5462u,5466u,327745u,9u,5468u,
370u,2395u,262205u,6u,5469u,5468u,327745u,9u,5470u,370u,548u,262205u,6u,5471u,5470u,327745u,
9u,5472u,370u,548u,262205u,6u,5473u,5472u,327812u,6u,5474u,5471u,5473u,327810u,6u,5475u,
5469u,5474u,196670u,5467u,5475u,196670u,5476u,630u,262205u,6u,5478u,368u,196670u,5477u,5478u,327737u,
57u,5479u,65u,5477u,262312u,57u,5480u,5479u,196855u,5482u,0u,262394u,5480u,5481u,5482u,131320u,
5481u,327745u,9u,5483u,370u,2409u,262205u,6u,5484u,5483u,327855u,57u,5485u,5484u,548u,131321u,
5482u,131320u,5482u,458997u,57u,5486u,5480u,5455u,5485u,5481u,196855u,5488u,0u,262394u,5486u,5487u,
5488u,131320u,5487u,262205u,6u,5489u,367u,327745u,9u,5490u,370u,2409u,262205u,6u,5491u,5490u,
262205u,6u,5492u,5462u,327808u,6u,5493u,5491u,5492u,327808u,6u,5494u,5489u,5493u,393281u,584u,
5495u,578u,548u,5494u,262205u,10u,5496u,5495u,196670u,5476u,5496u,131321u,5488u,131320u,5488u,196670u,
5497u,548u,131321u,5498u,131320u,5498u,262390u,5500u,5501u,0u,131321u,5502u,131320u,5502u,262205u,6u,
5503u,5497u,327745u,9u,5504u,370u,548u,262205u,6u,5505u,5504u,327857u,57u,5506u,5503u,5505u,
262394u,5506u,5499u,5500u,131320u,5499u,262205u,6u,5508u,368u,196670u,5507u,5508u,327737u,57u,5509u,
65u,5507u,196855u,5512u,0u,262394u,5509u,5511u,5536u,131320u,5511u,262205u,6u,5513u,368u,262205u,
6u,5514u,5457u,327745u,9u,5515u,370u,548u,262205u,6u,5516u,5515u,327812u,6u,5517u,5514u,
5516u,262205u,6u,5518u,5497u,327808u,6u,5519u,5517u,5518u,327808u,6u,5520u,5513u,5519u,393281u,
584u,5521u,578u,548u,5520u,262205u,10u,5522u,5521u,262205u,6u,5523u,367u,262205u,6u,5524u,
5467u,262205u,6u,5525u,5462u,327745u,9u,5526u,370u,548u,262205u,6u,5527u,5526u,327812u,6u,
5528u,5525u,5527u,327808u,6u,5529u,5524u,5528u,262205u,6u,5530u,5497u,327808u,6u,5531u,5529u,
5530u,327808u,6u,5532u,5523u,5531u,393281u,584u,5533u,578u,548u,5532u,262205u,10u,5534u,5533u,
327813u,10u,5535u,5522u,5534u,196670u,5510u,5535u,131321u,5512u,131320u,5536u,262205u,6u,5537u,366u,
262205u,6u,5538u,5457u,327745u,9u,5539u,370u,548u,262205u,6u,5540u,5539u,327812u,6u,5541u,
5538u,5540u,262205u,6u,5542u,5497u,327808u,6u,5543u,5541u,5542u,327808u,6u,5544u,5537u,5543u,
393281u,584u,5545u,578u,548u,5544u,262205u,10u,5546u,5545u,262205u,6u,5547u,367u,262205u,6u,
5548u,5467u,262205u,6u,5549u,5497u,327745u,9u,5550u,370u,548u,262205u,6u,5551u,5550u,327812u,
6u,5552u,5549u,5551u,327808u,6u,5553u,5548u,5552u,262205u,6u,5554u,5462u,327808u,6u,5555u,
5553u,5554u,327808u,6u,5556u,5547u,5555u,393281u,584u,5557u,578u,548u,5556u,262205u,10u,5558u,
5557u,327813u,10u,5559u,5546u,5558u,196670u,5510u,5559u,131321u,5512u,131320u,5512u,262205u,10u,5560u,
5510u,262205u,10u,5561u,5476u,327809u,10u,5562u,5561u,5560u,196670u,5476u,5562u,131321u,5501u,131320u,
5501u,262205u,6u,5563u,5497u,327808u,6u,5564u,5563u,551u,196670u,5497u,5564u,131321u,5498u,131320u,
5500u,262205u,6u,5565u,369u,262205u,6u,5566u,5445u,327808u,6u,5567u,5565u,5566u,262205u,10u,
5568u,5476u,393281u,584u,5569u,578u,548u,5567u,196670u,5569u,5568u,65789u,65592u,327734u,2u,380u,
0u,374u,196663u,9u,375u,196663u,9u,376u,196663u,9u,377u,196663u,9u,378u,196663u,339u,
379u,131320u,381u,262203u,9u,5570u,7u,262203u,9u,5571u,7u,262203u,9u,5576u,7u,262203u,
9u,5581u,7u,262203u,9u,5587u,7u,262203u,11u,5597u,7u,262203u,9u,5598u,7u,262203u,
11u,5609u,7u,262203u,9u,5610u,7u,262203u,11u,5672u,7u,262203u,9u,5674u,7u,262203u,
9u,5677u,7u,262203u,11u,5699u,7u,196670u,5570u,548u,262205u,6u,5572u,549u,327745u,9u,
5573u,379u,2404u,262205u,6u,5574u,5573u,327815u,6u,5575u,5572u,5574u,196670u,5571u,5575u,262205u,
6u,5577u,549u,327745u,9u,5578u,379u,2404u,262205u,6u,5579u,5578u,327819u,6u,5580u,5577u,
5579u,196670u,5576u,5580u,327745u,9u,5582u,379u,548u,262205u,6u,5583u,5582u,327745u,9u,5584u,
379u,2490u,262205u,6u,5585u,5584u,327815u,6u,5586u,5583u,5585u,196670u,5581u,5586u,262205u,6u,
5588u,5571u,327745u,9u,5589u,379u,2404u,262205u,6u,5590u,5589u,327812u,6u,5591u,5588u,5590u,
262205u,6u,5592u,5576u,327808u,6u,5593u,5591u,5592u,327745u,9u,5594u,379u,3118u,262205u,6u,
5595u,5594u,327812u,6u,5596u,5593u,5595u,196670u,5587u,5596u,196670u,5597u,630u,262205u,6u,5599u,
547u,196670u,5598u,5599u,131321u,5600u,131320u,5600u,262390u,5602u,5603u,0u,131321u,5604u,131320u,5604u,
262205u,6u,5605u,5598u,327745u,9u,5606u,379u,3118u,262205u,6u,5607u,5606u,327857u,57u,5608u,
5605u,5607u,262394u,5608u,5601u,5602u,131320u,5601u,196670u,5609u,630u,196670u,5610u,548u,131321u,5611u,
131320u,5611u,262390u,5613u,5614u,0u,131321u,5615u,131320u,5615u,262205u,6u,5616u,5610u,262205u,6u,
5617u,5581u,327857u,57u,5618u,5616u,5617u,262394u,5618u,5612u,5613u,131320u,5612u,262205u,6u,5619u,
377u,262205u,6u,5620u,5576u,327745u,9u,5621u,379u,548u,262205u,6u,5622u,5621u,327812u,6u,
5623u,5620u,5622u,262205u,6u,5624u,5571u,262205u,6u,5625u,5581u,327812u,6u,5626u,5624u,5625u,
327808u,6u,5627u,5623u,5626u,262205u,6u,5628u,5610u,327808u,6u,5629u,5627u,5628u,327808u,6u,
5630u,5619u,5629u,393281u,584u,5631u,578u,548u,5630u,262205u,10u,5632u,5631u,262205u,6u,5633u,
376u,262205u,6u,5634u,5598u,327745u,9u,5635u,379u,548u,262205u,6u,5636u,5635u,327812u,6u,
5637u,5634u,5636u,262205u,6u,5638u,5571u,262205u,6u,5639u,5581u,327812u,6u,5640u,5638u,5639u,
327808u,6u,5641u,5637u,5640u,262205u,6u,5642u,5610u,327808u,6u,5643u,5641u,5642u,327808u,6u,
5644u,5633u,5643u,393281u,584u,5645u,578u,548u,5644u,262205u,10u,5646u,5645u,327813u,10u,5647u,
5632u,5646u,262205u,10u,5648u,5609u,327809u,10u,5649u,5648u,5647u,196670u,5609u,5649u,131321u,5614u,
131320u,5614u,262205u,6u,5650u,5610u,327808u,6u,5651u,5650u,551u,196670u,5610u,5651u,131321u,5611u,
131320u,5613u,262205u,6u,5652u,378u,262205u,6u,5653u,5587u,262205u,6u,5654u,5598u,327808u,6u,
5655u,5653u,5654u,327808u,6u,5656u,5652u,5655u,262205u,10u,5657u,5609u,393281u,584u,5658u,578u,
548u,5656u,196670u,5658u,5657u,262205u,10u,5659u,5609u,262205u,6u,5660u,375u,262205u,6u,5661u,
5587u,262205u,6u,5662u,5598u,327808u,6u,5663u,5661u,5662u,327808u,6u,5664u,5660u,5663u,393281u,
584u,5665u,578u,548u,5664u,262205u,10u,5666u,5665u,327813u,10u,5667u,5659u,5666u,262205u,10u,
5668u,5597u,327809u,10u,5669u,5668u,5667u,196670u,5597u,5669u,131321u,5603u,131320u,5603u,262205u,6u,
5670u,5598u,327808u,6u,5671u,5670u,551u,196670u,5598u,5671u,131321u,5600u,131320u,5602u,262205u,10u,
5673u,5597u,196670u,5672u,5673u,262205u,6u,5675u,5570u,196670u,5674u,5675u,393273u,10u,5676u,46u,
5672u,5674u,196670u,5597u,5676u,262205u,6u,5678u,547u,196670u,5677u,5678u,131321u,5679u,131320u,5679u,
262390u,5681u,5682u,0u,131321u,5683u,131320u,5683u,262205u,6u,5684u,5677u,327745u,9u,5685u,379u,
3118u,262205u,6u,5686u,5685u,327857u,57u,5687u,5684u,5686u,262394u,5687u,5680u,5681u,131320u,5680u,
262205u,6u,5688u,378u,262205u,6u,5689u,5587u,262205u,6u,5690u,5677u,327808u,6u,5691u,5689u,
5690u,327808u,6u,5692u,5688u,5691u,327745u,58u,5693u,379u,2624u,262205u,57u,5694u,5693u,262205u,
6u,5695u,5677u,262205u,6u,5696u,5576u,327853u,57u,5697u,5695u,5696u,327847u,57u,5698u,5694u,
5697u,196855u,5701u,0u,262394u,5698u,5700u,5702u,131320u,5700u,196670u,5699u,630u,131321u,5701u,131320u,
5702u,262205u,6u,5703u,378u,262205u,6u,5704u,5587u,262205u,6u,5705u,5677u,327808u,6u,5706u,
5704u,5705u,327808u,6u,5707u,5703u,5706u,393281u,584u,5708u,578u,548u,5707u,262205u,10u,5709u,
5708u,262205u,10u,5710u,5597u,327811u,10u,5711u,5709u,5710u,262205u,6u,5712u,375u,262205u,6u,
5713u,5587u,262205u,6u,5714u,5677u,327808u,6u,5715u,5713u,5714u,327808u,6u,5716u,5712u,5715u,
393281u,584u,5717u,578u,548u,5716u,262205u,10u,5718u,5717u,327813u,10u,5719u,5711u,5718u,196670u,
5699u,5719u,131321u,5701u,131320u,5701u,262205u,10u,5720u,5699u,393281u,584u,5721u,578u,548u,5692u,
196670u,5721u,5720u,131321u,5682u,131320u,5682u,262205u,6u,5722u,5677u,327808u,6u,5723u,5722u,551u,
196670u,5677u,5723u,131321u,5679u,131320u,5681u,65789u,65592u,327734u,2u,392u,0u,382u,196663u,9u,
383u,196663u,9u,384u,196663u,9u,385u,196663u,9u,386u,196663u,9u,387u,196663u,9u,388u,
196663u,339u,389u,196663u,9u,390u,196663u,9u,391u,131320u,393u,262203u,9u,5724u,7u,262203u,
9u,5736u,7u,262203u,9u,5741u,7u,262203u,9u,5746u,7u,262203u,11u,5754u,7u,262203u,
9u,5759u,7u,262203u,9u,5804u,7u,262203u,9u,5845u,7u,262203u,11u,5890u,7u,262203u,
11u,5902u,7u,262205u,6u,5725u,549u,262205u,6u,5726u,550u,327812u,6u,5727u,5725u,5726u,
262205u,6u,5728u,547u,327808u,6u,5729u,5727u,5728u,196670u,5724u,5729u,262205u,6u,5730u,5724u,
262205u,6u,5731u,391u,327855u,57u,5732u,5730u,5731u,196855u,5734u,0u,262394u,5732u,5733u,5734u,
131320u,5733u,65789u,131320u,5734u,262205u,6u,5737u,5724u,327745u,9u,5738u,389u,548u,262205u,6u,
5739u,5738u,327815u,6u,5740u,5737u,5739u,196670u,5736u,5740u,262205u,6u,5742u,5724u,327745u,9u,
5743u,389u,548u,262205u,6u,5744u,5743u,327819u,6u,5745u,5742u,5744u,196670u,5741u,5745u,262205u,
6u,5747u,5741u,327745u,9u,5748u,389u,548u,262205u,6u,5749u,5748u,327745u,9u,5750u,389u,
2490u,262205u,6u,5751u,5750u,327815u,6u,5752u,5749u,5751u,327815u,6u,5753u,5747u,5752u,196670u,
5746u,5753u,196670u,5754u,630u,262205u,6u,5755u,390u,327850u,57u,5756u,5755u,548u,196855u,5758u,
0u,262394u,5756u,5757u,5799u,131320u,5757u,196670u,5759u,548u,131321u,5760u,131320u,5760u,262390u,5762u,
5763u,0u,131321u,5764u,131320u,5764u,262205u,6u,5765u,5759u,327745u,9u,5766u,389u,3118u,262205u,
6u,5767u,5766u,327857u,57u,5768u,5765u,5767u,262394u,5768u,5761u,5762u,131320u,5761u,262205u,6u,
5769u,387u,262205u,6u,5770u,5746u,327745u,9u,5771u,389u,2404u,262205u,6u,5772u,5771u,327812u,
6u,5773u,5770u,5772u,262205u,6u,5774u,5736u,327808u,6u,5775u,5773u,5774u,327745u,9u,5776u,
389u,3118u,262205u,6u,5777u,5776u,327812u,6u,5778u,5775u,5777u,262205u,6u,5779u,5759u,327808u,
6u,5780u,5778u,5779u,327808u,6u,5781u,5769u,5780u,393281u,584u,5782u,578u,548u,5781u,262205u,
10u,5783u,5782u,262205u,6u,5784u,384u,262205u,6u,5785u,5759u,327745u,9u,5786u,389u,548u,
262205u,6u,5787u,5786u,327812u,6u,5788u,5785u,5787u,262205u,6u,5789u,5741u,327808u,6u,5790u,
5788u,5789u,327808u,6u,5791u,5784u,5790u,393281u,584u,5792u,578u,548u,5791u,262205u,10u,5793u,
5792u,327813u,10u,5794u,5783u,5793u,262205u,10u,5795u,5754u,327809u,10u,5796u,5795u,5794u,196670u,
5754u,5796u,131321u,5763u,131320u,5763u,262205u,6u,5797u,5759u,327808u,6u,5798u,5797u,551u,196670u,
5759u,5798u,131321u,5760u,131320u,5762u,131321u,5758u,131320u,5799u,262205u,6u,5800u,390u,327850u,57u,
5801u,5800u,551u,196855u,5803u,0u,262394u,5801u,5802u,5844u,131320u,5802u,196670u,5804u,548u,131321u,
5805u,131320u,5805u,262390u,5807u,5808u,0u,131321u,5809u,131320u,5809u,262205u,6u,5810u,5804u,327745u,
9u,5811u,389u,2404u,262205u,6u,5812u,5811u,327857u,57u,5813u,5810u,5812u,262394u,5813u,5806u,
5807u,131320u,5806u,262205u,6u,5814u,387u,262205u,6u,5815u,5746u,327745u,9u,5816u,389u,2404u,
262205u,6u,5817u,5816u,327812u,6u,5818u,5815u,5817u,262205u,6u,5819u,5804u,327808u,6u,5820u,
5818u,5819u,327745u,9u,5821u,389u,3118u,262205u,6u,5822u,5821u,327812u,6u,5823u,5820u,5822u,
262205u,6u,5824u,5736u,327808u,6u,5825u,5823u,5824u,327808u,6u,5826u,5814u,5825u,393281u,584u,
5827u,578u,548u,5826u,262205u,10u,5828u,5827u,262205u,6u,5829u,383u,262205u,6u,5830u,5804u,
327745u,9u,5831u,389u,548u,262205u,6u,5832u,5831u,327812u,6u,5833u,5830u,5832u,262205u,6u,
5834u,5741u,327808u,6u,5835u,5833u,5834u,327808u,6u,5836u,5829u,5835u,393281u,584u,5837u,578u,
548u,5836u,262205u,10u,5838u,5837u,327813u,10u,5839u,5828u,5838u,262205u,10u,5840u,5754u,327809u,
10u,5841u,5840u,5839u,196670u,5754u,5841u,131321u,5808u,131320u,5808u,262205u,6u,5842u,5804u,327808u,
6u,5843u,5842u,551u,196670u,5804u,5843u,131321u,5805u,131320u,5807u,131321u,5803u,131320u,5844u,196670u,
5845u,548u,131321u,5846u,131320u,5846u,262390u,5848u,5849u,0u,131321u,5850u,131320u,5850u,262205u,6u,
5851u,5845u,327745u,9u,5852u,389u,2404u,262205u,6u,5853u,5852u,327857u,57u,5854u,5851u,5853u,
262394u,5854u,5847u,5848u,131320u,5847u,262205u,6u,5855u,386u,262205u,6u,5856u,5746u,327745u,9u,
5857u,389u,2404u,262205u,6u,5858u,5857u,327812u,6u,5859u,5856u,5858u,262205u,6u,5860u,5845u,
327808u,6u,5861u,5859u,5860u,327745u,9u,5862u,389u,3118u,262205u,6u,5863u,5862u,327812u,6u,
5864u,5861u,5863u,262205u,6u,5865u,5736u,327808u,6u,5866u,5864u,5865u,327808u,6u,5867u,5855u,
5866u,393281u,584u,5868u,578u,548u,5867u,262205u,10u,5869u,5868u,262205u,6u,5870u,385u,262205u,
6u,5871u,5845u,327745u,9u,5872u,389u,548u,262205u,6u,5873u,5872u,327812u,6u,5874u,5871u,
5873u,262205u,6u,5875u,5741u,327808u,6u,5876u,5874u,5875u,327808u,6u,5877u,5870u,5876u,393281u,
584u,5878u,578u,548u,5877u,262205u,10u,5879u,5878u,327813u,10u,5880u,5869u,5879u,262205u,10u,
5881u,5754u,327809u,10u,5882u,5881u,5880u,196670u,5754u,5882u,131321u,5849u,131320u,5849u,262205u,6u,
5883u,5845u,327808u,6u,5884u,5883u,551u,196670u,5845u,5884u,131321u,5846u,131320u,5848u,131321u,5803u,
131320u,5803u,131321u,5758u,131320u,5758u,262205u,6u,5885u,388u,262205u,6u,5886u,5724u,327808u,6u,
5887u,5885u,5886u,262205u,6u,5888u,390u,327850u,57u,5889u,5888u,2490u,196855u,5892u,0u,262394u,
5889u,5891u,5894u,131320u,5891u,262205u,10u,5893u,5754u,196670u,5890u,5893u,131321u,5892u,131320u,5894u,
262205u,10u,5895u,5754u,327745u,9u,5896u,389u,548u,262205u,6u,5897u,5896u,327745u,9u,5898u,
389u,2490u,262205u,6u,5899u,5898u,327815u,6u,5900u,5897u,5899u,262255u,10u,5901u,5900u,196670u,
5902u,5901u,327737u,10u,5903u,76u,5902u,327813u,10u,5904u,5895u,5903u,196670u,5890u,5904u,131321u,
5892u,131320u,5892u,262205u,10u,5905u,5890u,393281u,584u,5906u,578u,548u,5887u,196670u,5906u,5905u,
65789u,65592u,327734u,2u,405u,0u,394u,196663u,9u,395u,196663u,9u,396u,196663u,9u,397u,
196663u,9u,398u,196663u,9u,399u,196663u,9u,400u,196663u,9u,401u,196663u,9u,402u,196663u,
339u,403u,196663u,9u,404u,131320u,406u,262203u,9u,5907u,7u,262203u,11u,5919u,7u,262203u,
9u,5920u,7u,262203u,9u,5934u,7u,262203u,9u,5941u,7u,262203u,9u,5948u,7u,262203u,
58u,5984u,7u,262203u,9u,5991u,7u,262203u,9u,5999u,7u,262203u,9u,6007u,7u,262203u,
9u,6012u,7u,262203u,9u,6015u,7u,262203u,9u,6026u,7u,262203u,9u,6071u,7u,262203u,
9u,6076u,7u,262203u,9u,6081u,7u,262203u,9u,6117u,7u,262203u,9u,6125u,7u,262203u,
9u,6131u,7u,262203u,9u,6137u,7u,262203u,9u,6142u,7u,262203u,9u,6150u,7u,262205u,
6u,5908u,549u,262205u,6u,5909u,550u,327812u,6u,5910u,5908u,5909u,262205u,6u,5911u,547u,
327808u,6u,5912u,5910u,5911u,196670u,5907u,5912u,262205u,6u,5913u,5907u,262205u,6u,5914u,404u,
327855u,57u,5915u,5913u,5914u,196855u,5917u,0u,262394u,5915u,5916u,5917u,131320u,5916u,65789u,131320u,
5917u,196670u,5919u,630u,327745u,9u,5921u,403u,2395u,262205u,6u,5922u,5921u,327745u,9u,5923u,
403u,548u,262205u,6u,5924u,5923u,327745u,9u,5925u,403u,548u,262205u,6u,5926u,5925u,327812u,
6u,5927u,5924u,5926u,327810u,6u,5928u,5922u,5927u,196670u,5920u,5928u,262205u,6u,5929u,5907u,
262205u,6u,5930u,5920u,327855u,57u,5931u,5929u,5930u,196855u,5933u,0u,262394u,5931u,5932u,5983u,
131320u,5932u,262205u,6u,5935u,5907u,262205u,6u,5936u,5920u,327810u,6u,5937u,5935u,5936u,327745u,
9u,5938u,403u,548u,262205u,6u,5939u,5938u,327815u,6u,5940u,5937u,5939u,196670u,5934u,5940u,
262205u,6u,5942u,5907u,262205u,6u,5943u,5920u,327810u,6u,5944u,5942u,5943u,327745u,9u,5945u,
403u,548u,262205u,6u,5946u,5945u,327819u,6u,5947u,5944u,5946u,196670u,5941u,5947u,196670u,5948u,
548u,131321u,5949u,131320u,5949u,262390u,5951u,5952u,0u,131321u,5953u,131320u,5953u,262205u,6u,5954u,
5948u,327745u,9u,5955u,403u,2404u,262205u,6u,5956u,5955u,327857u,57u,5957u,5954u,5956u,262394u,
5957u,5950u,5951u,131320u,5950u,262205u,6u,5958u,397u,262205u,6u,5959u,5948u,327745u,9u,5960u,
403u,548u,262205u,6u,5961u,5960u,327812u,6u,5962u,5959u,5961u,262205u,6u,5963u,5934u,327808u,
6u,5964u,5962u,5963u,327808u,6u,5965u,5958u,5964u,393281u,584u,5966u,578u,548u,5965u,262205u,
10u,5967u,5966u,262205u,6u,5968u,398u,262205u,6u,5969u,5948u,327745u,9u,5970u,403u,548u,
262205u,6u,5971u,5970u,327812u,6u,5972u,5969u,5971u,262205u,6u,5973u,5941u,327808u,6u,5974u,
5972u,5973u,327808u,6u,5975u,5968u,5974u,393281u,584u,5976u,578u,548u,5975u,262205u,10u,5977u,
5976u,327813u,10u,5978u,5967u,5977u,262205u,10u,5979u,5919u,327809u,10u,5980u,5979u,5978u,196670u,
5919u,5980u,131321u,5952u,131320u,5952u,262205u,6u,5981u,5948u,327808u,6u,5982u,5981u,551u,196670u,
5948u,5982u,131321u,5949u,131320u,5951u,131321u,5933u,131320u,5983u,327745u,58u,5985u,403u,2510u,262205u,
57u,5986u,5985u,196670u,5984u,5986u,327737u,57u,5987u,61u,5984u,262312u,57u,5988u,5987u,196855u,
5990u,0u,262394u,5988u,5989u,6061u,131320u,5989u,262205u,6u,5992u,5907u,327745u,9u,5993u,403u,
548u,262205u,6u,5994u,5993u,327815u,6u,5995u,5992u,5994u,327745u,9u,5996u,403u,548u,262205u,
6u,5997u,5996u,327815u,6u,5998u,5995u,5997u,196670u,5991u,5998u,262205u,6u,6000u,5907u,327745u,
9u,6001u,403u,548u,262205u,6u,6002u,6001u,327815u,6u,6003u,6000u,6002u,327745u,9u,6004u,
403u,548u,262205u,6u,6005u,6004u,327819u,6u,6006u,6003u,6005u,196670u,5999u,6006u,262205u,6u,
6008u,5907u,327745u,9u,6009u,403u,548u,262205u,6u,6010u,6009u,327819u,6u,6011u,6008u,6010u,
196670u,6007u,6011u,262205u,6u,6013u,5991u,327850u,57u,6014u,6013u,548u,196855u,6017u,0u,262394u,
6014u,6016u,6019u,131320u,6016u,262205u,6u,6018u,399u,196670u,6015u,6018u,131321u,6017u,131320u,6019u,
262205u,6u,6020u,5991u,327850u,57u,6021u,6020u,551u,262205u,6u,6022u,400u,262205u,6u,6023u,
401u,393385u,6u,6024u,6021u,6022u,6023u,196670u,6015u,6024u,131321u,6017u,131320u,6017u,262205u,6u,
6025u,6015u,196670u,6012u,6025u,196670u,6026u,548u,131321u,6027u,131320u,6027u,262390u,6029u,6030u,0u,
131321u,6031u,131320u,6031u,262205u,6u,6032u,6026u,327745u,9u,6033u,403u,2404u,262205u,6u,6034u,
6033u,327857u,57u,6035u,6032u,6034u,262394u,6035u,6028u,6029u,131320u,6028u,262205u,6u,6036u,395u,
262205u,6u,6037u,6026u,327745u,9u,6038u,403u,548u,262205u,6u,6039u,6038u,327812u,6u,6040u,
6037u,6039u,262205u,6u,6041u,6007u,327808u,6u,6042u,6040u,6041u,327808u,6u,6043u,6036u,6042u,
393281u,584u,6044u,578u,548u,6043u,262205u,10u,6045u,6044u,262205u,6u,6046u,6012u,262205u,6u,
6047u,6026u,327745u,9u,6048u,403u,548u,262205u,6u,6049u,6048u,327812u,6u,6050u,6047u,6049u,
262205u,6u,6051u,5999u,327808u,6u,6052u,6050u,6051u,327808u,6u,6053u,6046u,6052u,393281u,584u,
6054u,578u,548u,6053u,262205u,10u,6055u,6054u,327813u,10u,6056u,6045u,6055u,262205u,10u,6057u,
5919u,327809u,10u,6058u,6057u,6056u,196670u,5919u,6058u,131321u,6030u,131320u,6030u,262205u,6u,6059u,
6026u,327808u,6u,6060u,6059u,551u,196670u,6026u,6060u,131321u,6027u,131320u,6029u,131321u,5990u,131320u,
6061u,262205u,6u,6062u,5907u,327745u,9u,6063u,403u,548u,262205u,6u,6064u,6063u,327745u,9u,
6065u,403u,548u,262205u,6u,6066u,6065u,327812u,6u,6067u,6064u,6066u,327857u,57u,6068u,6062u,
6067u,196855u,6070u,0u,262394u,6068u,6069u,6116u,131320u,6069u,262205u,6u,6072u,5907u,327745u,9u,
6073u,403u,548u,262205u,6u,6074u,6073u,327815u,6u,6075u,6072u,6074u,196670u,6071u,6075u,262205u,
6u,6077u,5907u,327745u,9u,6078u,403u,548u,262205u,6u,6079u,6078u,327819u,6u,6080u,6077u,
6079u,196670u,6076u,6080u,196670u,6081u,548u,131321u,6082u,131320u,6082u,262390u,6084u,6085u,0u,131321u,
6086u,131320u,6086u,262205u,6u,6087u,6081u,327745u,9u,6088u,403u,2404u,262205u,6u,6089u,6088u,
327857u,57u,6090u,6087u,6089u,262394u,6090u,6083u,6084u,131320u,6083u,262205u,6u,6091u,395u,262205u,
6u,6092u,6081u,327745u,9u,6093u,403u,548u,262205u,6u,6094u,6093u,327812u,6u,6095u,6092u,
6094u,262205u,6u,6096u,6071u,327808u,6u,6097u,6095u,6096u,327808u,6u,6098u,6091u,6097u,393281u,
584u,6099u,578u,548u,6098u,262205u,10u,6100u,6099u,262205u,6u,6101u,399u,262205u,6u,6102u,
6081u,327745u,9u,6103u,403u,548u,262205u,6u,6104u,6103u,327812u,6u,6105u,6102u,6104u,262205u,
6u,6106u,6076u,327808u,6u,6107u,6105u,6106u,327808u,6u,6108u,6101u,6107u,393281u,584u,6109u,
578u,548u,6108u,262205u,10u,6110u,6109u,327813u,10u,6111u,6100u,6110u,262205u,10u,6112u,5919u,
327809u,10u,6113u,6112u,6111u,196670u,5919u,6113u,131321u,6085u,131320u,6085u,262205u,6u,6114u,6081u,
327808u,6u,6115u,6114u,551u,196670u,6081u,6115u,131321u,6082u,131320u,6084u,131321u,6070u,131320u,6116u,
262205u,6u,6118u,5907u,327745u,9u,6119u,403u,548u,262205u,6u,6120u,6119u,327745u,9u,6121u,
403u,548u,262205u,6u,6122u,6121u,327812u,6u,6123u,6120u,6122u,327810u,6u,6124u,6118u,6123u,
196670u,6117u,6124u,262205u,6u,6126u,6117u,327745u,9u,6127u,403u,548u,262205u,6u,6128u,6127u,
327812u,6u,6129u,2490u,6128u,327815u,6u,6130u,6126u,6129u,196670u,6125u,6130u,262205u,6u,6132u,
6117u,327745u,9u,6133u,403u,548u,262205u,6u,6134u,6133u,327812u,6u,6135u,2490u,6134u,327819u,
6u,6136u,6132u,6135u,196670u,6131u,6136u,262205u,6u,6138u,6131u,327745u,9u,6139u,403u,548u,
262205u,6u,6140u,6139u,327819u,6u,6141u,6138u,6140u,196670u,6137u,6141u,262205u,6u,6143u,6131u,
327745u,9u,6144u,403u,548u,262205u,6u,6145u,6144u,327857u,57u,6146u,6143u,6145u,262205u,6u,
6147u,400u,262205u,6u,6148u,401u,393385u,6u,6149u,6146u,6147u,6148u,196670u,6142u,6149u,196670u,
6150u,548u,131321u,6151u,131320u,6151u,262390u,6153u,6154u,0u,131321u,6155u,131320u,6155u,262205u,6u,
6156u,6150u,327745u,9u,6157u,403u,3118u,262205u,6u,6158u,6157u,327857u,57u,6159u,6156u,6158u,
262394u,6159u,6152u,6153u,131320u,6152u,262205u,6u,6160u,396u,262205u,6u,6161u,6150u,327745u,9u,
6162u,403u,551u,262205u,6u,6163u,6162u,327812u,6u,6164u,6161u,6163u,262205u,6u,6165u,6125u,
327808u,6u,6166u,6164u,6165u,327808u,6u,6167u,6160u,6166u,393281u,584u,6168u,578u,548u,6167u,
262205u,10u,6169u,6168u,262205u,6u,6170u,6142u,262205u,6u,6171u,6150u,327745u,9u,6172u,403u,
548u,262205u,6u,6173u,6172u,327812u,6u,6174u,6171u,6173u,262205u,6u,6175u,6137u,327808u,6u,
6176u,6174u,6175u,327808u,6u,6177u,6170u,6176u,393281u,584u,6178u,578u,548u,6177u,262205u,10u,
6179u,6178u,327813u,10u,6180u,6169u,6179u,262205u,10u,6181u,5919u,327809u,10u,6182u,6181u,6180u,
196670u,5919u,6182u,131321u,6154u,131320u,6154u,262205u,6u,6183u,6150u,327808u,6u,6184u,6183u,551u,
196670u,6150u,6184u,131321u,6151u,131320u,6153u,131321u,6070u,131320u,6070u,131321u,5990u,131320u,5990u,131321u,
5933u,131320u,5933u,262205u,6u,6185u,402u,262205u,6u,6186u,5907u,327808u,6u,6187u,6185u,6186u,
262205u,10u,6188u,5919u,393281u,584u,6189u,578u,548u,6187u,196670u,6189u,6188u,65789u,65592u,327734u,
2u,416u,0u,407u,196663u,9u,408u,196663u,9u,409u,196663u,9u,410u,196663u,9u,411u,
196663u,9u,412u,196663u,339u,413u,196663u,58u,414u,196663u,9u,415u,131320u,417u,262203u,9u,
6190u,7u,262203u,9u,6202u,7u,262203u,58u,6203u,7u,262203u,9u,6206u,7u,262203u,9u,
6215u,7u,262203u,9u,6219u,7u,262203u,11u,6223u,7u,262203u,58u,6224u,7u,262203u,9u,
6231u,7u,262203u,58u,6318u,7u,262203u,9u,6324u,7u,262203u,9u,6360u,7u,262205u,6u,
6191u,549u,262205u,6u,6192u,550u,327812u,6u,6193u,6191u,6192u,262205u,6u,6194u,547u,327808u,
6u,6195u,6193u,6194u,196670u,6190u,6195u,262205u,6u,6196u,6190u,262205u,6u,6197u,415u,327855u,
57u,6198u,6196u,6197u,196855u,6200u,0u,262394u,6198u,6199u,6200u,131320u,6199u,65789u,131320u,6200u,
262205u,57u,6204u,414u,196670u,6203u,6204u,327737u,57u,6205u,61u,6203u,196855u,6208u,0u,262394u,
6205u,6207u,6211u,131320u,6207u,327745u,9u,6209u,413u,551u,262205u,6u,6210u,6209u,196670u,6206u,
6210u,131321u,6208u,131320u,6211u,327745u,9u,6212u,413u,548u,262205u,6u,6213u,6212u,196670u,6206u,
6213u,131321u,6208u,131320u,6208u,262205u,6u,6214u,6206u,196670u,6202u,6214u,262205u,6u,6216u,6190u,
262205u,6u,6217u,6202u,327815u,6u,6218u,6216u,6217u,196670u,6215u,6218u,262205u,6u,6220u,6190u,
262205u,6u,6221u,6202u,327819u,6u,6222u,6220u,6221u,196670u,6219u,6222u,196670u,6223u,630u,327745u,
58u,6225u,413u,2510u,262205u,57u,6226u,6225u,196670u,6224u,6226u,327737u,57u,6227u,61u,6224u,
262312u,57u,6228u,6227u,196855u,6230u,0u,262394u,6228u,6229u,6317u,131320u,6229u,196670u,6231u,548u,
131321u,6232u,131320u,6232u,262390u,6234u,6235u,0u,131321u,6236u,131320u,6236u,262205u,6u,6237u,6231u,
327745u,9u,6238u,413u,548u,262205u,6u,6239u,6238u,327857u,57u,6240u,6237u,6239u,262394u,6240u,
6233u,6234u,131320u,6233u,262205u,6u,6241u,409u,262205u,6u,6242u,6215u,327745u,9u,6243u,413u,
548u,262205u,6u,6244u,6243u,327812u,6u,6245u,6242u,6244u,262205u,6u,6246u,6231u,327808u,6u,
6247u,6245u,6246u,327808u,6u,6248u,6241u,6247u,393281u,584u,6249u,578u,548u,6248u,262205u,10u,
6250u,6249u,262205u,6u,6251u,408u,262205u,6u,6252u,6231u,327745u,9u,6253u,413u,548u,262205u,
6u,6254u,6253u,327812u,6u,6255u,6252u,6254u,262205u,6u,6256u,6219u,327808u,6u,6257u,6255u,
6256u,327808u,6u,6258u,6251u,6257u,393281u,584u,6259u,578u,548u,6258u,262205u,10u,6260u,6259u,
327813u,10u,6261u,6250u,6260u,262205u,6u,6262u,410u,262205u,6u,6263u,6215u,327745u,9u,6264u,
413u,548u,262205u,6u,6265u,6264u,327812u,6u,6266u,6263u,6265u,262205u,6u,6267u,6231u,327808u,
6u,6268u,6266u,6267u,327808u,6u,6269u,6262u,6268u,393281u,584u,6270u,578u,548u,6269u,262205u,
10u,6271u,6270u,262205u,6u,6272u,408u,327745u,9u,6273u,413u,548u,262205u,6u,6274u,6273u,
262205u,6u,6275u,6231u,327808u,6u,6276u,6274u,6275u,327745u,9u,6277u,413u,548u,262205u,6u,
6278u,6277u,327812u,6u,6279u,6276u,6278u,262205u,6u,6280u,6219u,327808u,6u,6281u,6279u,6280u,
327808u,6u,6282u,6272u,6281u,393281u,584u,6283u,578u,548u,6282u,262205u,10u,6284u,6283u,327813u,
10u,6285u,6271u,6284u,327809u,10u,6286u,6261u,6285u,262205u,6u,6287u,411u,262205u,6u,6288u,
6215u,327745u,9u,6289u,413u,548u,262205u,6u,6290u,6289u,327812u,6u,6291u,6288u,6290u,262205u,
6u,6292u,6231u,327808u,6u,6293u,6291u,6292u,327808u,6u,6294u,6287u,6293u,393281u,584u,6295u,
578u,548u,6294u,262205u,10u,6296u,6295u,262205u,6u,6297u,408u,327745u,9u,6298u,413u,548u,
262205u,6u,6299u,6298u,327812u,6u,6300u,2490u,6299u,262205u,6u,6301u,6231u,327808u,6u,6302u,
6300u,6301u,327745u,9u,6303u,413u,548u,262205u,6u,6304u,6303u,327812u,6u,6305u,6302u,6304u,
262205u,6u,6306u,6219u,327808u,6u,6307u,6305u,6306u,327808u,6u,6308u,6297u,6307u,393281u,584u,
6309u,578u,548u,6308u,262205u,10u,6310u,6309u,327813u,10u,6311u,6296u,6310u,327809u,10u,6312u,
6286u,6311u,262205u,10u,6313u,6223u,327809u,10u,6314u,6313u,6312u,196670u,6223u,6314u,131321u,6235u,
131320u,6235u,262205u,6u,6315u,6231u,327808u,6u,6316u,6315u,551u,196670u,6231u,6316u,131321u,6232u,
131320u,6234u,131321u,6230u,131320u,6317u,262205u,57u,6319u,414u,196670u,6318u,6319u,327737u,57u,6320u,
61u,6318u,262312u,57u,6321u,6320u,196855u,6323u,0u,262394u,6321u,6322u,6359u,131320u,6322u,196670u,
6324u,548u,131321u,6325u,131320u,6325u,262390u,6327u,6328u,0u,131321u,6329u,131320u,6329u,262205u,6u,
6330u,6324u,327745u,9u,6331u,413u,548u,262205u,6u,6332u,6331u,327857u,57u,6333u,6330u,6332u,
262394u,6333u,6326u,6327u,131320u,6326u,262205u,6u,6334u,409u,262205u,6u,6335u,6215u,327745u,9u,
6336u,413u,548u,262205u,6u,6337u,6336u,327812u,6u,6338u,6335u,6337u,262205u,6u,6339u,6324u,
327808u,6u,6340u,6338u,6339u,327808u,6u,6341u,6334u,6340u,393281u,584u,6342u,578u,548u,6341u,
262205u,10u,6343u,6342u,262205u,6u,6344u,408u,262205u,6u,6345u,6219u,327745u,9u,6346u,413u,
548u,262205u,6u,6347u,6346u,327812u,6u,6348u,6345u,6347u,262205u,6u,6349u,6324u,327808u,6u,
6350u,6348u,6349u,327808u,6u,6351u,6344u,6350u,393281u,584u,6352u,578u,548u,6351u,262205u,10u,
6353u,6352u,327813u,10u,6354u,6343u,6353u,262205u,10u,6355u,6223u,327809u,10u,6356u,6355u,6354u,
196670u,6223u,6356u,131321u,6328u,131320u,6328u,262205u,6u,6357u,6324u,327808u,6u,6358u,6357u,551u,
196670u,6324u,6358u,131321u,6325u,131320u,6327u,131321u,6323u,131320u,6359u,196670u,6360u,548u,131321u,6361u,
131320u,6361u,262390u,6363u,6364u,0u,131321u,6365u,131320u,6365u,262205u,6u,6366u,6360u,327745u,9u,
6367u,413u,548u,262205u,6u,6368u,6367u,327857u,57u,6369u,6366u,6368u,262394u,6369u,6362u,6363u,
131320u,6362u,262205u,6u,6370u,410u,262205u,6u,6371u,6215u,327745u,9u,6372u,413u,548u,262205u,
6u,6373u,6372u,327812u,6u,6374u,6371u,6373u,262205u,6u,6375u,6360u,327808u,6u,6376u,6374u,
6375u,327808u,6u,6377u,6370u,6376u,393281u,584u,6378u,578u,548u,6377u,262205u,10u,6379u,6378u,
262205u,6u,6380u,408u,327745u,9u,6381u,413u,548u,262205u,6u,6382u,6381u,327745u,9u,6383u,
413u,548u,262205u,6u,6384u,6383u,327812u,6u,6385u,6382u,6384u,262205u,6u,6386u,6219u,327812u,
6u,6387u,6386u,2490u,327745u,9u,6388u,413u,548u,262205u,6u,6389u,6388u,327812u,6u,6390u,
6387u,6389u,327808u,6u,6391u,6385u,6390u,262205u,6u,6392u,6360u,327808u,6u,6393u,6391u,6392u,
327808u,6u,6394u,6380u,6393u,393281u,584u,6395u,578u,548u,6394u,262205u,10u,6396u,6395u,327813u,
10u,6397u,6379u,6396u,262205u,6u,6398u,411u,262205u,6u,6399u,6215u,327745u,9u,6400u,413u,
548u,262205u,6u,6401u,6400u,327812u,6u,6402u,6399u,6401u,262205u,6u,6403u,6360u,327808u,6u,
6404u,6402u,6403u,327808u,6u,6405u,6398u,6404u,393281u,584u,6406u,578u,548u,6405u,262205u,10u,
6407u,6406u,262205u,6u,6408u,408u,327745u,9u,6409u,413u,548u,262205u,6u,6410u,6409u,327745u,
9u,6411u,413u,548u,262205u,6u,6412u,6411u,327812u,6u,6413u,6410u,6412u,262205u,6u,6414u,
6219u,327812u,6u,6415u,6414u,2490u,327745u,9u,6416u,413u,548u,262205u,6u,6417u,6416u,327812u,
6u,6418u,6415u,6417u,327808u,6u,6419u,6413u,6418u,327745u,9u,6420u,413u,548u,262205u,6u,
6421u,6420u,327808u,6u,6422u,6419u,6421u,262205u,6u,6423u,6360u,327808u,6u,6424u,6422u,6423u,
327808u,6u,6425u,6408u,6424u,393281u,584u,6426u,578u,548u,6425u,262205u,10u,6427u,6426u,327813u,
10u,6428u,6407u,6427u,327809u,10u,6429u,6397u,6428u,262205u,10u,6430u,6223u,327809u,10u,6431u,
6430u,6429u,196670u,6223u,6431u,131321u,6364u,131320u,6364u,262205u,6u,6432u,6360u,327808u,6u,6433u,
6432u,551u,196670u,6360u,6433u,131321u,6361u,131320u,6363u,131321u,6323u,131320u,6323u,131321u,6230u,131320u,
6230u,262205u,6u,6434u,412u,262205u,6u,6435u,6190u,327808u,6u,6436u,6434u,6435u,262205u,10u,
6437u,6223u,393281u,584u,6438u,578u,548u,6436u,196670u,6438u,6437u,65789u,65592u,327734u,10u,419u,
0u,71u,196663u,11u,418u,131320u,420u,262205u,10u,6439u,418u,262271u,10u,6440u,6439u,393228u,
10u,6441u,1u,27u,6440u,327809u,10u,6442u,701u,6441u,327816u,10u,6443u,701u,6442u,131326u,
6443u,65592u,327734u,57u,428u,0u,423u,196663u,19u,424u,196663u,19u,425u,196663u,422u,426u,
196663u,8u,427u,131320u,429u,262203u,9u,6446u,7u,262203u,9u,6449u,7u,262203u,9u,6452u,
7u,262203u,9u,6476u,7u,262203u,9u,6479u,7u,262203u,58u,6492u,7u,262203u,9u,6495u,
7u,262203u,9u,6499u,7u,262203u,9u,6503u,7u,262203u,58u,6504u,7u,262203u,9u,6507u,
7u,262203u,9u,6513u,7u,262203u,58u,6514u,7u,262203u,9u,6517u,7u,262203u,9u,6523u,
7u,262203u,9u,6538u,7u,262203u,9u,6540u,7u,262203u,9u,6544u,7u,262203u,58u,6545u,
7u,262203u,9u,6548u,7u,262203u,9u,6556u,7u,262203u,58u,6557u,7u,262203u,9u,6560u,
7u,262203u,19u,6568u,7u,262203u,8u,6570u,7u,262203u,9u,6572u,7u,262203u,11u,6573u,
7u,262203u,19u,6579u,7u,262203u,9u,6581u,7u,262203u,11u,6582u,7u,262203u,9u,6585u,
7u,262203u,9u,6594u,7u,262203u,9u,6601u,7u,262203u,9u,6608u,7u,262203u,9u,6617u,
7u,262203u,9u,6623u,7u,262203u,11u,6629u,7u,262203u,9u,6630u,7u,262203u,9u,6654u,
7u,262203u,9u,6679u,7u,262203u,9u,6691u,7u,262203u,9u,6714u,7u,262203u,9u,6716u,
7u,262203u,9u,6717u,7u,262203u,58u,6731u,7u,262203u,9u,6734u,7u,262203u,9u,6739u,
7u,262203u,9u,6744u,7u,262203u,58u,6745u,7u,262203u,9u,6748u,7u,262203u,9u,6755u,
7u,262203u,58u,6756u,7u,262203u,9u,6759u,7u,262203u,9u,6766u,7u,262203u,9u,6781u,
7u,262203u,9u,6783u,7u,262203u,9u,6787u,7u,262203u,58u,6788u,7u,262203u,9u,6791u,
7u,262203u,9u,6799u,7u,262203u,58u,6800u,7u,262203u,9u,6803u,7u,262203u,19u,6811u,
7u,262203u,8u,6813u,7u,262203u,9u,6815u,7u,262203u,11u,6816u,7u,262203u,19u,6819u,
7u,262203u,8u,6821u,7u,262203u,9u,6823u,7u,262203u,11u,6824u,7u,262203u,19u,6827u,
7u,262203u,8u,6829u,7u,262203u,9u,6831u,7u,262203u,11u,6832u,7u,262203u,19u,6835u,
7u,262203u,8u,6837u,7u,262203u,9u,6839u,7u,262203u,11u,6840u,7u,262203u,19u,6846u,
7u,262203u,9u,6848u,7u,262203u,11u,6849u,7u,262203u,9u,6852u,7u,262203u,9u,6861u,
7u,262203u,9u,6868u,7u,262203u,9u,6877u,7u,262203u,9u,6883u,7u,262203u,11u,6889u,
7u,262203u,9u,6890u,7u,262203u,9u,6914u,7u,262203u,9u,6939u,7u,262203u,9u,6951u,
7u,262203u,11u,6967u,7u,262203u,9u,6971u,7u,262203u,9u,6979u,7u,262203u,11u,6987u,
7u,262203u,9u,6988u,7u,262203u,9u,7012u,7u,262203u,9u,7037u,7u,262203u,9u,7051u,
7u,262203u,11u,7069u,7u,262203u,9u,7073u,7u,262203u,9u,7082u,7u,262203u,11u,7091u,
7u,262203u,9u,7092u,7u,262203u,9u,7116u,7u,262203u,9u,7148u,7u,262203u,9u,7163u,
7u,262203u,9u,7187u,7u,262203u,9u,7194u,7u,262203u,11u,7203u,7u,262203u,11u,7210u,
7u,262203u,11u,7217u,7u,262203u,9u,7241u,7u,262203u,9u,7243u,7u,262203u,9u,7244u,
7u,262203u,58u,7251u,7u,262203u,9u,7254u,7u,262203u,9u,7259u,7u,262203u,9u,7264u,
7u,262203u,58u,7265u,7u,262203u,9u,7268u,7u,262203u,9u,7275u,7u,262203u,58u,7276u,
7u,262203u,9u,7279u,7u,262203u,9u,7286u,7u,262203u,9u,7301u,7u,262203u,9u,7303u,
7u,262203u,9u,7307u,7u,262203u,58u,7308u,7u,262203u,9u,7311u,7u,262203u,9u,7319u,
7u,262203u,58u,7320u,7u,262203u,9u,7323u,7u,262203u,19u,7331u,7u,262203u,8u,7333u,
7u,262203u,9u,7335u,7u,262203u,11u,7336u,7u,262203u,19u,7339u,7u,262203u,8u,7341u,
7u,262203u,9u,7343u,7u,262203u,11u,7344u,7u,262203u,19u,7350u,7u,262203u,9u,7352u,
7u,262203u,11u,7353u,7u,262203u,9u,7356u,7u,262203u,9u,7365u,7u,262203u,9u,7372u,
7u,262203u,9u,7379u,7u,262203u,9u,7388u,7u,262203u,9u,7394u,7u,262203u,9u,7402u,
7u,262203u,9u,7411u,7u,262203u,9u,7420u,7u,262203u,9u,7426u,7u,262203u,9u,7434u,
7u,262203u,9u,7443u,7u,262203u,11u,7452u,7u,262203u,9u,7453u,7u,262203u,9u,7455u,
7u,262203u,9u,7457u,7u,262203u,9u,7460u,7u,262203u,9u,7462u,7u,262203u,9u,7465u,
7u,262203u,11u,7469u,7u,262203u,9u,7470u,7u,262203u,9u,7472u,7u,262203u,9u,7474u,
7u,262203u,9u,7477u,7u,262203u,9u,7479u,7u,262203u,9u,7482u,7u,262203u,11u,7486u,
7u,262203u,9u,7487u,7u,262203u,9u,7489u,7u,262203u,9u,7491u,7u,262203u,9u,7494u,
7u,262203u,9u,7496u,7u,262203u,9u,7499u,7u,262203u,11u,7503u,7u,262203u,9u,7504u,
7u,262203u,9u,7506u,7u,262203u,9u,7508u,7u,262203u,9u,7511u,7u,262203u,9u,7513u,
7u,262203u,9u,7516u,7u,262203u,9u,7520u,7u,262203u,9u,7561u,7u,262203u,11u,7602u,
7u,262203u,11u,7603u,7u,262203u,11u,7606u,7u,262203u,11u,7607u,7u,262203u,11u,7610u,
7u,262203u,11u,7613u,7u,262203u,11u,7614u,7u,262203u,11u,7617u,7u,262203u,9u,7649u,
7u,262203u,9u,7651u,7u,262203u,9u,7652u,7u,327745u,9u,6447u,426u,551u,262205u,6u,
6448u,6447u,196670u,6446u,6448u,327745u,9u,6450u,426u,2490u,262205u,6u,6451u,6450u,196670u,6449u,
6451u,327745u,9u,6453u,426u,2404u,262205u,6u,6454u,6453u,196670u,6452u,6454u,262205u,6u,6455u,
6446u,327859u,57u,6456u,6455u,548u,262205u,6u,6457u,6449u,327859u,57u,6458u,6457u,548u,327846u,
57u,6459u,6456u,6458u,262205u,6u,6460u,6452u,327859u,57u,6461u,6460u,548u,327846u,57u,6462u,
6459u,6461u,196855u,6464u,0u,262394u,6462u,6463u,6464u,131320u,6463u,131326u,6465u,131320u,6464u,327745u,
9u,6467u,424u,551u,262205u,6u,6468u,6467u,262205u,6u,6469u,6446u,262205u,6u,6470u,6449u,
327812u,6u,6471u,6469u,6470u,327851u,57u,6472u,6468u,6471u,196855u,6474u,0u,262394u,6472u,6473u,
6474u,131320u,6473u,131326u,6465u,131320u,6474u,327745u,9u,6477u,426u,2510u,262205u,6u,6478u,6477u,
196670u,6476u,6478u,262205u,6u,6480u,6476u,196670u,6479u,6480u,327737u,57u,6481u,65u,6479u,262312u,
57u,6482u,6481u,196855u,6484u,0u,262394u,6482u,6483u,6484u,131320u,6483u,131326u,6465u,131320u,6484u,
327745u,9u,6486u,426u,548u,262205u,6u,6487u,6486u,327850u,57u,6489u,6487u,6488u,196855u,6491u,
0u,262394u,6489u,6490u,6491u,131320u,6490u,327745u,58u,6493u,426u,3118u,262205u,57u,6494u,6493u,
196670u,6492u,6494u,262205u,6u,6496u,6452u,262205u,6u,6497u,6449u,327812u,6u,6498u,6496u,6497u,
196670u,6495u,6498u,262205u,6u,6500u,6452u,262205u,6u,6501u,6452u,327812u,6u,6502u,6500u,6501u,
196670u,6499u,6502u,262205u,57u,6505u,6492u,196670u,6504u,6505u,327737u,57u,6506u,61u,6504u,196855u,
6509u,0u,262394u,6506u,6508u,6511u,131320u,6508u,262205u,6u,6510u,6452u,196670u,6507u,6510u,131321u,
6509u,131320u,6511u,196670u,6507u,548u,131321u,6509u,131320u,6509u,262205u,6u,6512u,6507u,196670u,6503u,
6512u,262205u,57u,6515u,6492u,196670u,6514u,6515u,327737u,57u,6516u,61u,6514u,196855u,6519u,0u,
262394u,6516u,6518u,6521u,131320u,6518u,262205u,6u,6520u,6452u,196670u,6517u,6520u,131321u,6519u,131320u,
6521u,196670u,6517u,548u,131321u,6519u,131320u,6519u,262205u,6u,6522u,6517u,196670u,6513u,6522u,262205u,
6u,6524u,6495u,262205u,6u,6525u,6499u,327808u,6u,6526u,6524u,6525u,262205u,6u,6527u,6503u,
327808u,6u,6528u,6526u,6527u,262205u,6u,6529u,6513u,327808u,6u,6530u,6528u,6529u,196670u,6523u,
6530u,327745u,9u,6531u,426u,2624u,262205u,6u,6532u,6531u,262205u,6u,6533u,6523u,327857u,57u,
6534u,6532u,6533u,196855u,6536u,0u,262394u,6534u,6535u,6536u,131320u,6535u,131326u,6465u,131320u,6536u,
262205u,6u,6539u,6476u,196670u,6538u,6539u,262205u,6u,6541u,6538u,262205u,6u,6542u,6495u,327808u,
6u,6543u,6541u,6542u,196670u,6540u,6543u,262205u,57u,6546u,6492u,196670u,6545u,6546u,327737u,57u,
6547u,61u,6545u,196855u,6550u,0u,262394u,6547u,6549u,6554u,131320u,6549u,262205u,6u,6551u,6540u,
262205u,6u,6552u,6499u,327808u,6u,6553u,6551u,6552u,196670u,6548u,6553u,131321u,6550u,131320u,6554u,
196670u,6548u,548u,131321u,6550u,131320u,6550u,262205u,6u,6555u,6548u,196670u,6544u,6555u,262205u,57u,
6558u,6492u,196670u,6557u,6558u,327737u,57u,6559u,61u,6557u,196855u,6562u,0u,262394u,6559u,6561u,
6566u,131320u,6561u,262205u,6u,6563u,6544u,262205u,6u,6564u,6503u,327808u,6u,6565u,6563u,6564u,
196670u,6560u,6565u,131321u,6562u,131320u,6566u,196670u,6560u,548u,131321u,6562u,131320u,6562u,262205u,6u,
6567u,6560u,196670u,6556u,6567u,262205u,6u,6569u,6452u,262205u,7u,6571u,427u,196670u,6570u,6571u,
196670u,6572u,6569u,196670u,6573u,630u,458809u,12u,6574u,17u,6570u,6572u,6573u,262205u,7u,6575u,
6570u,196670u,427u,6575u,196670u,6568u,6574u,262205u,6u,6576u,6446u,262205u,6u,6577u,6452u,327812u,
6u,6578u,6576u,6577u,262205u,12u,6580u,425u,196670u,6579u,6580u,196670u,6581u,6578u,196670u,6582u,
630u,458809u,2u,6583u,24u,6579u,6581u,6582u,262205u,12u,6584u,6579u,196670u,425u,6584u,196670u,
6585u,548u,131321u,6586u,131320u,6586u,262390u,6588u,6589u,0u,131321u,6590u,131320u,6590u,262205u,6u,
6591u,6585u,262205u,6u,6592u,6446u,327857u,57u,6593u,6591u,6592u,262394u,6593u,6587u,6588u,131320u,
6587u,327745u,9u,6595u,424u,548u,262205u,6u,6596u,6595u,262205u,6u,6597u,6585u,262205u,6u,
6598u,6449u,327812u,6u,6599u,6597u,6598u,327808u,6u,6600u,6596u,6599u,196670u,6594u,6600u,327745u,
9u,6602u,425u,548u,262205u,6u,6603u,6602u,262205u,6u,6604u,6585u,262205u,6u,6605u,6452u,
327812u,6u,6606u,6604u,6605u,327808u,6u,6607u,6603u,6606u,196670u,6601u,6607u,196670u,6608u,548u,
131321u,6609u,131320u,6609u,262390u,6611u,6612u,0u,131321u,6613u,131320u,6613u,262205u,6u,6614u,6608u,
262205u,6u,6615u,6452u,327857u,57u,6616u,6614u,6615u,262394u,6616u,6610u,6611u,131320u,6610u,262205u,
6u,6618u,6538u,262205u,6u,6619u,6608u,262205u,6u,6620u,6449u,327812u,6u,6621u,6619u,6620u,
327808u,6u,6622u,6618u,6621u,196670u,6617u,6622u,262205u,6u,6624u,6540u,262205u,6u,6625u,6608u,
262205u,6u,6626u,6452u,327812u,6u,6627u,6625u,6626u,327808u,6u,6628u,6624u,6627u,196670u,6623u,
6628u,196670u,6629u,630u,196670u,6630u,548u,131321u,6631u,131320u,6631u,262390u,6633u,6634u,0u,131321u,
6635u,131320u,6635u,262205u,6u,6636u,6630u,262205u,6u,6637u,6449u,327857u,57u,6638u,6636u,6637u,
262394u,6638u,6632u,6633u,131320u,6632u,262205u,6u,6639u,6617u,262205u,6u,6640u,6630u,327808u,6u,
6641u,6639u,6640u,393281u,584u,6642u,578u,548u,6641u,262205u,10u,6643u,6642u,262205u,6u,6644u,
6594u,262205u,6u,6645u,6630u,327808u,6u,6646u,6644u,6645u,393281u,584u,6647u,578u,548u,6646u,
262205u,10u,6648u,6647u,327813u,10u,6649u,6643u,6648u,262205u,10u,6650u,6629u,327809u,10u,6651u,
6650u,6649u,196670u,6629u,6651u,131321u,6634u,131320u,6634u,262205u,6u,6652u,6630u,327808u,6u,6653u,
6652u,551u,196670u,6630u,6653u,131321u,6631u,131320u,6633u,196670u,6654u,548u,131321u,6655u,131320u,6655u,
262390u,6657u,6658u,0u,131321u,6659u,131320u,6659u,262205u,6u,6660u,6654u,262205u,6u,6661u,6452u,
327857u,57u,6662u,6660u,6661u,262394u,6662u,6656u,6657u,131320u,6656u,262205u,6u,6663u,6623u,262205u,
6u,6664u,6654u,327808u,6u,6665u,6663u,6664u,393281u,584u,6666u,578u,548u,6665u,262205u,10u,
6667u,6666u,327745u,9u,6668u,6568u,548u,262205u,6u,6669u,6668u,262205u,6u,6670u,6654u,327808u,
6u,6671u,6669u,6670u,393281u,584u,6672u,578u,548u,6671u,262205u,10u,6673u,6672u,327813u,10u,
6674u,6667u,6673u,262205u,10u,6675u,6629u,327809u,10u,6676u,6675u,6674u,196670u,6629u,6676u,131321u,
6658u,131320u,6658u,262205u,6u,6677u,6654u,327808u,6u,6678u,6677u,551u,196670u,6654u,6678u,131321u,
6655u,131320u,6657u,262205u,6u,6680u,6544u,196670u,6679u,6680u,327737u,57u,6681u,65u,6679u,196855u,
6683u,0u,262394u,6681u,6682u,6683u,131320u,6682u,262205u,6u,6684u,6544u,262205u,6u,6685u,6608u,
327808u,6u,6686u,6684u,6685u,393281u,584u,6687u,578u,548u,6686u,262205u,10u,6688u,6687u,262205u,
10u,6689u,6629u,327809u,10u,6690u,6689u,6688u,196670u,6629u,6690u,131321u,6683u,131320u,6683u,262205u,
6u,6692u,6556u,196670u,6691u,6692u,327737u,57u,6693u,65u,6691u,196855u,6695u,0u,262394u,6693u,
6694u,6695u,131320u,6694u,262205u,6u,6696u,6556u,262205u,6u,6697u,6608u,327808u,6u,6698u,6696u,
6697u,393281u,584u,6699u,578u,548u,6698u,262205u,10u,6700u,6699u,262205u,10u,6701u,6629u,327809u,
10u,6702u,6701u,6700u,196670u,6629u,6702u,131321u,6695u,131320u,6695u,262205u,6u,6703u,6601u,262205u,
6u,6704u,6608u,327808u,6u,6705u,6703u,6704u,262205u,10u,6706u,6629u,393228u,10u,6707u,1u,
21u,6706u,393281u,584u,6708u,578u,548u,6705u,196670u,6708u,6707u,131321u,6612u,131320u,6612u,262205u,
6u,6709u,6608u,327808u,6u,6710u,6709u,551u,196670u,6608u,6710u,131321u,6609u,131320u,6611u,262205u,
6u,6711u,6601u,262205u,6u,6712u,6452u,327808u,6u,6713u,6711u,6712u,262205u,6u,6715u,6601u,
196670u,6714u,6715u,196670u,6716u,6713u,327745u,9u,6718u,6568u,548u,262205u,6u,6719u,6718u,196670u,
6717u,6719u,458809u,2u,6720u,35u,6714u,6716u,6717u,131321u,6589u,131320u,6589u,262205u,6u,6721u,
6585u,327808u,6u,6722u,6721u,551u,196670u,6585u,6722u,131321u,6586u,131320u,6588u,131326u,6723u,131320u,
6491u,327745u,9u,6725u,426u,548u,262205u,6u,6726u,6725u,327850u,57u,6728u,6726u,6727u,196855u,
6730u,0u,262394u,6728u,6729u,6730u,131320u,6729u,327745u,58u,6732u,426u,3118u,262205u,57u,6733u,
6732u,196670u,6731u,6733u,262205u,6u,6735u,6452u,327812u,6u,6736u,2404u,6735u,262205u,6u,6737u,
6449u,327812u,6u,6738u,6736u,6737u,196670u,6734u,6738u,262205u,6u,6740u,6452u,327812u,6u,6741u,
2404u,6740u,262205u,6u,6742u,6452u,327812u,6u,6743u,6741u,6742u,196670u,6739u,6743u,262205u,57u,
6746u,6731u,196670u,6745u,6746u,327737u,57u,6747u,61u,6745u,196855u,6750u,0u,262394u,6747u,6749u,
6753u,131320u,6749u,262205u,6u,6751u,6452u,327812u,6u,6752u,2404u,6751u,196670u,6748u,6752u,131321u,
6750u,131320u,6753u,196670u,6748u,548u,131321u,6750u,131320u,6750u,262205u,6u,6754u,6748u,196670u,6744u,
6754u,262205u,57u,6757u,6731u,196670u,6756u,6757u,327737u,57u,6758u,61u,6756u,196855u,6761u,0u,
262394u,6758u,6760u,6764u,131320u,6760u,262205u,6u,6762u,6452u,327812u,6u,6763u,2404u,6762u,196670u,
6759u,6763u,131321u,6761u,131320u,6764u,196670u,6759u,548u,131321u,6761u,131320u,6761u,262205u,6u,6765u,
6759u,196670u,6755u,6765u,262205u,6u,6767u,6734u,262205u,6u,6768u,6739u,327808u,6u,6769u,6767u,
6768u,262205u,6u,6770u,6744u,327808u,6u,6771u,6769u,6770u,262205u,6u,6772u,6755u,327808u,6u,
6773u,6771u,6772u,196670u,6766u,6773u,327745u,9u,6774u,426u,2624u,262205u,6u,6775u,6774u,262205u,
6u,6776u,6766u,327857u,57u,6777u,6775u,6776u,196855u,6779u,0u,262394u,6777u,6778u,6779u,131320u,
6778u,131326u,6465u,131320u,6779u,262205u,6u,6782u,6476u,196670u,6781u,6782u,262205u,6u,6784u,6781u,
262205u,6u,6785u,6734u,327808u,6u,6786u,6784u,6785u,196670u,6783u,6786u,262205u,57u,6789u,6731u,
196670u,6788u,6789u,327737u,57u,6790u,61u,6788u,196855u,6793u,0u,262394u,6790u,6792u,6797u,131320u,
6792u,262205u,6u,6794u,6783u,262205u,6u,6795u,6739u,327808u,6u,6796u,6794u,6795u,196670u,6791u,
6796u,131321u,6793u,131320u,6797u,196670u,6791u,548u,131321u,6793u,131320u,6793u,262205u,6u,6798u,6791u,
196670u,6787u,6798u,262205u,57u,6801u,6731u,196670u,6800u,6801u,327737u,57u,6802u,61u,6800u,196855u,
6805u,0u,262394u,6802u,6804u,6809u,131320u,6804u,262205u,6u,6806u,6787u,262205u,6u,6807u,6744u,
327808u,6u,6808u,6806u,6807u,196670u,6803u,6808u,131321u,6805u,131320u,6809u,196670u,6803u,548u,131321u,
6805u,131320u,6805u,262205u,6u,6810u,6803u,196670u,6799u,6810u,262205u,6u,6812u,6452u,262205u,7u,
6814u,427u,196670u,6813u,6814u,196670u,6815u,6812u,196670u,6816u,630u,458809u,12u,6817u,17u,6813u,
6815u,6816u,262205u,7u,6818u,6813u,196670u,427u,6818u,196670u,6811u,6817u,262205u,6u,6820u,6452u,
262205u,7u,6822u,427u,196670u,6821u,6822u,196670u,6823u,6820u,196670u,6824u,630u,458809u,12u,6825u,
17u,6821u,6823u,6824u,262205u,7u,6826u,6821u,196670u,427u,6826u,196670u,6819u,6825u,262205u,6u,
6828u,6452u,262205u,7u,6830u,427u,196670u,6829u,6830u,196670u,6831u,6828u,196670u,6832u,630u,458809u,
12u,6833u,17u,6829u,6831u,6832u,262205u,7u,6834u,6829u,196670u,427u,6834u,196670u,6827u,6833u,
262205u,6u,6836u,6452u,262205u,7u,6838u,427u,196670u,6837u,6838u,196670u,6839u,6836u,196670u,6840u,
630u,458809u,12u,6841u,17u,6837u,6839u,6840u,262205u,7u,6842u,6837u,196670u,427u,6842u,196670u,
6835u,6841u,262205u,6u,6843u,6446u,262205u,6u,6844u,6452u,327812u,6u,6845u,6843u,6844u,262205u,
12u,6847u,425u,196670u,6846u,6847u,196670u,6848u,6845u,196670u,6849u,630u,458809u,2u,6850u,24u,
6846u,6848u,6849u,262205u,12u,6851u,6846u,196670u,425u,6851u,196670u,6852u,548u,131321u,6853u,131320u,
6853u,262390u,6855u,6856u,0u,131321u,6857u,131320u,6857u,262205u,6u,6858u,6852u,262205u,6u,6859u,
6446u,327857u,57u,6860u,6858u,6859u,262394u,6860u,6854u,6855u,131320u,6854u,327745u,9u,6862u,424u,
548u,262205u,6u,6863u,6862u,262205u,6u,6864u,6852u,262205u,6u,6865u,6449u,327812u,6u,6866u,
6864u,6865u,327808u,6u,6867u,6863u,6866u,196670u,6861u,6867u,196670u,6868u,548u,131321u,6869u,131320u,
6869u,262390u,6871u,6872u,0u,131321u,6873u,131320u,6873u,262205u,6u,6874u,6868u,262205u,6u,6875u,
6452u,327857u,57u,6876u,6874u,6875u,262394u,6876u,6870u,6871u,131320u,6870u,262205u,6u,6878u,6781u,
262205u,6u,6879u,6868u,262205u,6u,6880u,6449u,327812u,6u,6881u,6879u,6880u,327808u,6u,6882u,
6878u,6881u,196670u,6877u,6882u,262205u,6u,6884u,6783u,262205u,6u,6885u,6868u,262205u,6u,6886u,
6452u,327812u,6u,6887u,6885u,6886u,327808u,6u,6888u,6884u,6887u,196670u,6883u,6888u,196670u,6889u,
630u,196670u,6890u,548u,131321u,6891u,131320u,6891u,262390u,6893u,6894u,0u,131321u,6895u,131320u,6895u,
262205u,6u,6896u,6890u,262205u,6u,6897u,6449u,327857u,57u,6898u,6896u,6897u,262394u,6898u,6892u,
6893u,131320u,6892u,262205u,6u,6899u,6877u,262205u,6u,6900u,6890u,327808u,6u,6901u,6899u,6900u,
393281u,584u,6902u,578u,548u,6901u,262205u,10u,6903u,6902u,262205u,6u,6904u,6861u,262205u,6u,
6905u,6890u,327808u,6u,6906u,6904u,6905u,393281u,584u,6907u,578u,548u,6906u,262205u,10u,6908u,
6907u,327813u,10u,6909u,6903u,6908u,262205u,10u,6910u,6889u,327809u,10u,6911u,6910u,6909u,196670u,
6889u,6911u,131321u,6894u,131320u,6894u,262205u,6u,6912u,6890u,327808u,6u,6913u,6912u,551u,196670u,
6890u,6913u,131321u,6891u,131320u,6893u,196670u,6914u,548u,131321u,6915u,131320u,6915u,262390u,6917u,6918u,
0u,131321u,6919u,131320u,6919u,262205u,6u,6920u,6914u,262205u,6u,6921u,6452u,327857u,57u,6922u,
6920u,6921u,262394u,6922u,6916u,6917u,131320u,6916u,262205u,6u,6923u,6883u,262205u,6u,6924u,6914u,
327808u,6u,6925u,6923u,6924u,393281u,584u,6926u,578u,548u,6925u,262205u,10u,6927u,6926u,327745u,
9u,6928u,6811u,548u,262205u,6u,6929u,6928u,262205u,6u,6930u,6914u,327808u,6u,6931u,6929u,
6930u,393281u,584u,6932u,578u,548u,6931u,262205u,10u,6933u,6932u,327813u,10u,6934u,6927u,6933u,
262205u,10u,6935u,6889u,327809u,10u,6936u,6935u,6934u,196670u,6889u,6936u,131321u,6918u,131320u,6918u,
262205u,6u,6937u,6914u,327808u,6u,6938u,6937u,551u,196670u,6914u,6938u,131321u,6915u,131320u,6917u,
262205u,6u,6940u,6787u,196670u,6939u,6940u,327737u,57u,6941u,65u,6939u,196855u,6943u,0u,262394u,
6941u,6942u,6943u,131320u,6942u,262205u,6u,6944u,6787u,262205u,6u,6945u,6868u,327808u,6u,6946u,
6944u,6945u,393281u,584u,6947u,578u,548u,6946u,262205u,10u,6948u,6947u,262205u,10u,6949u,6889u,
327809u,10u,6950u,6949u,6948u,196670u,6889u,6950u,131321u,6943u,131320u,6943u,262205u,6u,6952u,6799u,
196670u,6951u,6952u,327737u,57u,6953u,65u,6951u,196855u,6955u,0u,262394u,6953u,6954u,6955u,131320u,
6954u,262205u,6u,6956u,6799u,262205u,6u,6957u,6868u,327808u,6u,6958u,6956u,6957u,393281u,584u,
6959u,578u,548u,6958u,262205u,10u,6960u,6959u,262205u,10u,6961u,6889u,327809u,10u,6962u,6961u,
6960u,196670u,6889u,6962u,131321u,6955u,131320u,6955u,327745u,9u,6963u,6819u,548u,262205u,6u,6964u,
6963u,262205u,6u,6965u,6868u,327808u,6u,6966u,6964u,6965u,262205u,10u,6968u,6889u,196670u,6967u,
6968u,327737u,10u,6969u,419u,6967u,393281u,584u,6970u,578u,548u,6966u,196670u,6970u,6969u,262205u,
6u,6972u,6781u,262205u,6u,6973u,6452u,262205u,6u,6974u,6868u,327808u,6u,6975u,6973u,6974u,
262205u,6u,6976u,6449u,327812u,6u,6977u,6975u,6976u,327808u,6u,6978u,6972u,6977u,196670u,6971u,
6978u,262205u,6u,6980u,6783u,262205u,6u,6981u,6452u,262205u,6u,6982u,6868u,327808u,6u,6983u,
6981u,6982u,262205u,6u,6984u,6452u,327812u,6u,6985u,6983u,6984u,327808u,6u,6986u,6980u,6985u,
196670u,6979u,6986u,196670u,6987u,630u,196670u,6988u,548u,131321u,6989u,131320u,6989u,262390u,6991u,6992u,
0u,131321u,6993u,131320u,6993u,262205u,6u,6994u,6988u,262205u,6u,6995u,6449u,327857u,57u,6996u,
6994u,6995u,262394u,6996u,6990u,6991u,131320u,6990u,262205u,6u,6997u,6971u,262205u,6u,6998u,6988u,
327808u,6u,6999u,6997u,6998u,393281u,584u,7000u,578u,548u,6999u,262205u,10u,7001u,7000u,262205u,
6u,7002u,6861u,262205u,6u,7003u,6988u,327808u,6u,7004u,7002u,7003u,393281u,584u,7005u,578u,
548u,7004u,262205u,10u,7006u,7005u,327813u,10u,7007u,7001u,7006u,262205u,10u,7008u,6987u,327809u,
10u,7009u,7008u,7007u,196670u,6987u,7009u,131321u,6992u,131320u,6992u,262205u,6u,7010u,6988u,327808u,
6u,7011u,7010u,551u,196670u,6988u,7011u,131321u,6989u,131320u,6991u,196670u,7012u,548u,131321u,7013u,
131320u,7013u,262390u,7015u,7016u,0u,131321u,7017u,131320u,7017u,262205u,6u,7018u,7012u,262205u,6u,
7019u,6452u,327857u,57u,7020u,7018u,7019u,262394u,7020u,7014u,7015u,131320u,7014u,262205u,6u,7021u,
6979u,262205u,6u,7022u,7012u,327808u,6u,7023u,7021u,7022u,393281u,584u,7024u,578u,548u,7023u,
262205u,10u,7025u,7024u,327745u,9u,7026u,6811u,548u,262205u,6u,7027u,7026u,262205u,6u,7028u,
7012u,327808u,6u,7029u,7027u,7028u,393281u,584u,7030u,578u,548u,7029u,262205u,10u,7031u,7030u,
327813u,10u,7032u,7025u,7031u,262205u,10u,7033u,6987u,327809u,10u,7034u,7033u,7032u,196670u,6987u,
7034u,131321u,7016u,131320u,7016u,262205u,6u,7035u,7012u,327808u,6u,7036u,7035u,551u,196670u,7012u,
7036u,131321u,7013u,131320u,7015u,262205u,6u,7038u,6787u,196670u,7037u,7038u,327737u,57u,7039u,65u,
7037u,196855u,7041u,0u,262394u,7039u,7040u,7041u,131320u,7040u,262205u,6u,7042u,6787u,262205u,6u,
7043u,6452u,262205u,6u,7044u,6868u,327808u,6u,7045u,7043u,7044u,327808u,6u,7046u,7042u,7045u,
393281u,584u,7047u,578u,548u,7046u,262205u,10u,7048u,7047u,262205u,10u,7049u,6987u,327809u,10u,
7050u,7049u,7048u,196670u,6987u,7050u,131321u,7041u,131320u,7041u,262205u,6u,7052u,6799u,196670u,7051u,
7052u,327737u,57u,7053u,65u,7051u,196855u,7055u,0u,262394u,7053u,7054u,7055u,131320u,7054u,262205u,
6u,7056u,6799u,262205u,6u,7057u,6452u,262205u,6u,7058u,6868u,327808u,6u,7059u,7057u,7058u,
327808u,6u,7060u,7056u,7059u,393281u,584u,7061u,578u,548u,7060u,262205u,10u,7062u,7061u,262205u,
10u,7063u,6987u,327809u,10u,7064u,7063u,7062u,196670u,6987u,7064u,131321u,7055u,131320u,7055u,327745u,
9u,7065u,6827u,548u,262205u,6u,7066u,7065u,262205u,6u,7067u,6868u,327808u,6u,7068u,7066u,
7067u,262205u,10u,7070u,6987u,196670u,7069u,7070u,327737u,10u,7071u,419u,7069u,393281u,584u,7072u,
578u,548u,7068u,196670u,7072u,7071u,262205u,6u,7074u,6781u,262205u,6u,7075u,6452u,327812u,6u,
7076u,2490u,7075u,262205u,6u,7077u,6868u,327808u,6u,7078u,7076u,7077u,262205u,6u,7079u,6449u,
327812u,6u,7080u,7078u,7079u,327808u,6u,7081u,7074u,7080u,196670u,7073u,7081u,262205u,6u,7083u,
6783u,262205u,6u,7084u,6452u,327812u,6u,7085u,2490u,7084u,262205u,6u,7086u,6868u,327808u,6u,
7087u,7085u,7086u,262205u,6u,7088u,6452u,327812u,6u,7089u,7087u,7088u,327808u,6u,7090u,7083u,
7089u,196670u,7082u,7090u,196670u,7091u,630u,196670u,7092u,548u,131321u,7093u,131320u,7093u,262390u,7095u,
7096u,0u,131321u,7097u,131320u,7097u,262205u,6u,7098u,7092u,262205u,6u,7099u,6449u,327857u,57u,
7100u,7098u,7099u,262394u,7100u,7094u,7095u,131320u,7094u,262205u,6u,7101u,7073u,262205u,6u,7102u,
7092u,327808u,6u,7103u,7101u,7102u,393281u,584u,7104u,578u,548u,7103u,262205u,10u,7105u,7104u,
262205u,6u,7106u,6861u,262205u,6u,7107u,7092u,327808u,6u,7108u,7106u,7107u,393281u,584u,7109u,
578u,548u,7108u,262205u,10u,7110u,7109u,327813u,10u,7111u,7105u,7110u,262205u,10u,7112u,7091u,
327809u,10u,7113u,7112u,7111u,196670u,7091u,7113u,131321u,7096u,131320u,7096u,262205u,6u,7114u,7092u,
327808u,6u,7115u,7114u,551u,196670u,7092u,7115u,131321u,7093u,131320u,7095u,196670u,7116u,548u,131321u,
7117u,131320u,7117u,262390u,7119u,7120u,0u,131321u,7121u,131320u,7121u,262205u,6u,7122u,7116u,262205u,
6u,7123u,6452u,327857u,57u,7124u,7122u,7123u,262394u,7124u,7118u,7119u,131320u,7118u,262205u,6u,
7125u,7082u,262205u,6u,7126u,7116u,327808u,6u,7127u,7125u,7126u,393281u,584u,7128u,578u,548u,
7127u,262205u,10u,7129u,7128u,327745u,9u,7130u,6819u,548u,262205u,6u,7131u,7130u,262205u,6u,
7132u,7116u,327808u,6u,7133u,7131u,7132u,393281u,584u,7134u,578u,548u,7133u,262205u,10u,7135u,
7134u,327745u,9u,7136u,6811u,548u,262205u,6u,7137u,7136u,262205u,6u,7138u,7116u,327808u,6u,
7139u,7137u,7138u,393281u,584u,7140u,578u,548u,7139u,262205u,10u,7141u,7140u,327813u,10u,7142u,
7135u,7141u,327813u,10u,7143u,7129u,7142u,262205u,10u,7144u,7091u,327809u,10u,7145u,7144u,7143u,
196670u,7091u,7145u,131321u,7120u,131320u,7120u,262205u,6u,7146u,7116u,327808u,6u,7147u,7146u,551u,
196670u,7116u,7147u,131321u,7117u,131320u,7119u,262205u,6u,7149u,6787u,196670u,7148u,7149u,327737u,57u,
7150u,65u,7148u,196855u,7152u,0u,262394u,7150u,7151u,7152u,131320u,7151u,262205u,6u,7153u,6787u,
262205u,6u,7154u,6452u,327812u,6u,7155u,2490u,7154u,262205u,6u,7156u,6868u,327808u,6u,7157u,
7155u,7156u,327808u,6u,7158u,7153u,7157u,393281u,584u,7159u,578u,548u,7158u,262205u,10u,7160u,
7159u,262205u,10u,7161u,7091u,327809u,10u,7162u,7161u,7160u,196670u,7091u,7162u,131321u,7152u,131320u,
7152u,262205u,6u,7164u,6799u,196670u,7163u,7164u,327737u,57u,7165u,65u,7163u,196855u,7167u,0u,
262394u,7165u,7166u,7167u,131320u,7166u,262205u,6u,7168u,6799u,262205u,6u,7169u,6452u,327812u,6u,
7170u,2490u,7169u,262205u,6u,7171u,6868u,327808u,6u,7172u,7170u,7171u,327808u,6u,7173u,7168u,
7172u,393281u,584u,7174u,578u,548u,7173u,262205u,10u,7175u,7174u,262205u,10u,7176u,7091u,327809u,
10u,7177u,7176u,7175u,196670u,7091u,7177u,131321u,7167u,131320u,7167u,327745u,9u,7178u,6835u,548u,
262205u,6u,7179u,7178u,262205u,6u,7180u,6868u,327808u,6u,7181u,7179u,7180u,262205u,10u,7182u,
7091u,393228u,10u,7183u,1u,21u,7182u,393281u,584u,7184u,578u,548u,7181u,196670u,7184u,7183u,
131321u,6872u,131320u,6872u,262205u,6u,7185u,6868u,327808u,6u,7186u,7185u,551u,196670u,6868u,7186u,
131321u,6869u,131320u,6871u,327745u,9u,7188u,425u,548u,262205u,6u,7189u,7188u,262205u,6u,7190u,
6852u,262205u,6u,7191u,6452u,327812u,6u,7192u,7190u,7191u,327808u,6u,7193u,7189u,7192u,196670u,
7187u,7193u,196670u,7194u,548u,131321u,7195u,131320u,7195u,262390u,7197u,7198u,0u,131321u,7199u,131320u,
7199u,262205u,6u,7200u,7194u,262205u,6u,7201u,6452u,327857u,57u,7202u,7200u,7201u,262394u,7202u,
7196u,7197u,131320u,7196u,327745u,9u,7204u,6827u,548u,262205u,6u,7205u,7204u,262205u,6u,7206u,
7194u,327808u,6u,7207u,7205u,7206u,393281u,584u,7208u,578u,548u,7207u,262205u,10u,7209u,7208u,
196670u,7203u,7209u,327745u,9u,7211u,6835u,548u,262205u,6u,7212u,7211u,262205u,6u,7213u,7194u,
327808u,6u,7214u,7212u,7213u,393281u,584u,7215u,578u,548u,7214u,262205u,10u,7216u,7215u,196670u,
7210u,7216u,327745u,9u,7218u,6811u,548u,262205u,6u,7219u,7218u,262205u,6u,7220u,7194u,327808u,
6u,7221u,7219u,7220u,393281u,584u,7222u,578u,548u,7221u,262205u,10u,7223u,7222u,196670u,7217u,
7223u,262205u,6u,7224u,7187u,262205u,6u,7225u,7194u,327808u,6u,7226u,7224u,7225u,262205u,10u,
7227u,7203u,327811u,10u,7228u,701u,7227u,262205u,10u,7229u,7210u,327813u,10u,7230u,7228u,7229u,
262205u,10u,7231u,7203u,262205u,10u,7232u,7217u,327813u,10u,7233u,7231u,7232u,327809u,10u,7234u,
7230u,7233u,393281u,584u,7235u,578u,548u,7226u,196670u,7235u,7234u,131321u,7198u,131320u,7198u,262205u,
6u,7236u,7194u,327808u,6u,7237u,7236u,551u,196670u,7194u,7237u,131321u,7195u,131320u,7197u,262205u,
6u,7238u,7187u,262205u,6u,7239u,6452u,327808u,6u,7240u,7238u,7239u,262205u,6u,7242u,7187u,
196670u,7241u,7242u,196670u,7243u,7240u,327745u,9u,7245u,6811u,548u,262205u,6u,7246u,7245u,196670u,
7244u,7246u,458809u,2u,7247u,35u,7241u,7243u,7244u,131321u,6856u,131320u,6856u,262205u,6u,7248u,
6852u,327808u,6u,7249u,7248u,551u,196670u,6852u,7249u,131321u,6853u,131320u,6855u,131326u,6723u,131320u,
6730u,327745u,58u,7252u,426u,3118u,262205u,57u,7253u,7252u,196670u,7251u,7253u,262205u,6u,7255u,
6452u,327812u,6u,7256u,3118u,7255u,262205u,6u,7257u,6449u,327812u,6u,7258u,7256u,7257u,196670u,
7254u,7258u,262205u,6u,7260u,6452u,327812u,6u,7261u,3118u,7260u,262205u,6u,7262u,6452u,327812u,
6u,7263u,7261u,7262u,196670u,7259u,7263u,262205u,57u,7266u,7251u,196670u,7265u,7266u,327737u,57u,
7267u,61u,7265u,196855u,7270u,0u,262394u,7267u,7269u,7273u,131320u,7269u,262205u,6u,7271u,6452u,
327812u,6u,7272u,3118u,7271u,196670u,7268u,7272u,131321u,7270u,131320u,7273u,196670u,7268u,548u,131321u,
7270u,131320u,7270u,262205u,6u,7274u,7268u,196670u,7264u,7274u,262205u,57u,7277u,7251u,196670u,7276u,
7277u,327737u,57u,7278u,61u,7276u,196855u,7281u,0u,262394u,7278u,7280u,7284u,131320u,7280u,262205u,
6u,7282u,6452u,327812u,6u,7283u,3118u,7282u,196670u,7279u,7283u,131321u,7281u,131320u,7284u,196670u,
7279u,548u,131321u,7281u,131320u,7281u,262205u,6u,7285u,7279u,196670u,7275u,7285u,262205u,6u,7287u,
7254u,262205u,6u,7288u,7259u,327808u,6u,7289u,7287u,7288u,262205u,6u,7290u,7264u,327808u,6u,
7291u,7289u,7290u,262205u,6u,7292u,7275u,327808u,6u,7293u,7291u,7292u,196670u,7286u,7293u,327745u,
9u,7294u,426u,2624u,262205u,6u,7295u,7294u,262205u,6u,7296u,7286u,327857u,57u,7297u,7295u,
7296u,196855u,7299u,0u,262394u,7297u,7298u,7299u,131320u,7298u,131326u,6465u,131320u,7299u,262205u,6u,
7302u,6476u,196670u,7301u,7302u,262205u,6u,7304u,7301u,262205u,6u,7305u,7254u,327808u,6u,7306u,
7304u,7305u,196670u,7303u,7306u,262205u,57u,7309u,7251u,196670u,7308u,7309u,327737u,57u,7310u,61u,
7308u,196855u,7313u,0u,262394u,7310u,7312u,7317u,131320u,7312u,262205u,6u,7314u,7303u,262205u,6u,
7315u,7259u,327808u,6u,7316u,7314u,7315u,196670u,7311u,7316u,131321u,7313u,131320u,7317u,196670u,7311u,
548u,131321u,7313u,131320u,7313u,262205u,6u,7318u,7311u,196670u,7307u,7318u,262205u,57u,7321u,7251u,
196670u,7320u,7321u,327737u,57u,7322u,61u,7320u,196855u,7325u,0u,262394u,7322u,7324u,7329u,131320u,
7324u,262205u,6u,7326u,7307u,262205u,6u,7327u,7264u,327808u,6u,7328u,7326u,7327u,196670u,7323u,
7328u,131321u,7325u,131320u,7329u,196670u,7323u,548u,131321u,7325u,131320u,7325u,262205u,6u,7330u,7323u,
196670u,7319u,7330u,262205u,6u,7332u,6452u,262205u,7u,7334u,427u,196670u,7333u,7334u,196670u,7335u,
7332u,196670u,7336u,630u,458809u,12u,7337u,17u,7333u,7335u,7336u,262205u,7u,7338u,7333u,196670u,
427u,7338u,196670u,7331u,7337u,262205u,6u,7340u,6452u,262205u,7u,7342u,427u,196670u,7341u,7342u,
196670u,7343u,7340u,196670u,7344u,630u,458809u,12u,7345u,17u,7341u,7343u,7344u,262205u,7u,7346u,
7341u,196670u,427u,7346u,196670u,7339u,7345u,262205u,6u,7347u,6446u,262205u,6u,7348u,6452u,327812u,
6u,7349u,7347u,7348u,262205u,12u,7351u,425u,196670u,7350u,7351u,196670u,7352u,7349u,196670u,7353u,
630u,458809u,2u,7354u,24u,7350u,7352u,7353u,262205u,12u,7355u,7350u,196670u,425u,7355u,196670u,
7356u,548u,131321u,7357u,131320u,7357u,262390u,7359u,7360u,0u,131321u,7361u,131320u,7361u,262205u,6u,
7362u,7356u,262205u,6u,7363u,6446u,327857u,57u,7364u,7362u,7363u,262394u,7364u,7358u,7359u,131320u,
7358u,327745u,9u,7366u,424u,548u,262205u,6u,7367u,7366u,262205u,6u,7368u,7356u,262205u,6u,
7369u,6449u,327812u,6u,7370u,7368u,7369u,327808u,6u,7371u,7367u,7370u,196670u,7365u,7371u,327745u,
9u,7373u,425u,548u,262205u,6u,7374u,7373u,262205u,6u,7375u,7356u,262205u,6u,7376u,6452u,
327812u,6u,7377u,7375u,7376u,327808u,6u,7378u,7374u,7377u,196670u,7372u,7378u,196670u,7379u,548u,
131321u,7380u,131320u,7380u,262390u,7382u,7383u,0u,131321u,7384u,131320u,7384u,262205u,6u,7385u,7379u,
262205u,6u,7386u,6452u,327857u,57u,7387u,7385u,7386u,262394u,7387u,7381u,7382u,131320u,7381u,262205u,
6u,7389u,7301u,262205u,6u,7390u,7379u,262205u,6u,7391u,6449u,327812u,6u,7392u,7390u,7391u,
327808u,6u,7393u,7389u,7392u,196670u,7388u,7393u,262205u,6u,7395u,7301u,262205u,6u,7396u,6452u,
262205u,6u,7397u,7379u,327808u,6u,7398u,7396u,7397u,262205u,6u,7399u,6449u,327812u,6u,7400u,
7398u,7399u,327808u,6u,7401u,7395u,7400u,196670u,7394u,7401u,262205u,6u,7403u,7301u,262205u,6u,
7404u,6452u,327812u,6u,7405u,2490u,7404u,262205u,6u,7406u,7379u,327808u,6u,7407u,7405u,7406u,
262205u,6u,7408u,6449u,327812u,6u,7409u,7407u,7408u,327808u,6u,7410u,7403u,7409u,196670u,7402u,
7410u,262205u,6u,7412u,7301u,262205u,6u,7413u,6452u,327812u,6u,7414u,2404u,7413u,262205u,6u,
7415u,7379u,327808u,6u,7416u,7414u,7415u,262205u,6u,7417u,6449u,327812u,6u,7418u,7416u,7417u,
327808u,6u,7419u,7412u,7418u,196670u,7411u,7419u,262205u,6u,7421u,7303u,262205u,6u,7422u,7379u,
262205u,6u,7423u,6452u,327812u,6u,7424u,7422u,7423u,327808u,6u,7425u,7421u,7424u,196670u,7420u,
7425u,262205u,6u,7427u,7303u,262205u,6u,7428u,6452u,262205u,6u,7429u,7379u,327808u,6u,7430u,
7428u,7429u,262205u,6u,7431u,6452u,327812u,6u,7432u,7430u,7431u,327808u,6u,7433u,7427u,7432u,
196670u,7426u,7433u,262205u,6u,7435u,7303u,262205u,6u,7436u,6452u,327812u,6u,7437u,2490u,7436u,
262205u,6u,7438u,7379u,327808u,6u,7439u,7437u,7438u,262205u,6u,7440u,6452u,327812u,6u,7441u,
7439u,7440u,327808u,6u,7442u,7435u,7441u,196670u,7434u,7442u,262205u,6u,7444u,7303u,262205u,6u,
7445u,6452u,327812u,6u,7446u,2404u,7445u,262205u,6u,7447u,7379u,327808u,6u,7448u,7446u,7447u,
262205u,6u,7449u,6452u,327812u,6u,7450u,7448u,7449u,327808u,6u,7451u,7444u,7450u,196670u,7443u,
7451u,262205u,6u,7454u,7388u,196670u,7453u,7454u,262205u,6u,7456u,7365u,196670u,7455u,7456u,262205u,
6u,7458u,6449u,196670u,7457u,7458u,458809u,10u,7459u,41u,7453u,7455u,7457u,262205u,6u,7461u,
7420u,196670u,7460u,7461u,327745u,9u,7463u,7331u,548u,262205u,6u,7464u,7463u,196670u,7462u,7464u,
262205u,6u,7466u,6452u,196670u,7465u,7466u,458809u,10u,7467u,41u,7460u,7462u,7465u,327809u,10u,
7468u,7459u,7467u,196670u,7452u,7468u,262205u,6u,7471u,7394u,196670u,7470u,7471u,262205u,6u,7473u,
7365u,196670u,7472u,7473u,262205u,6u,7475u,6449u,196670u,7474u,7475u,458809u,10u,7476u,41u,7470u,
7472u,7474u,262205u,6u,7478u,7426u,196670u,7477u,7478u,327745u,9u,7480u,7331u,548u,262205u,6u,
7481u,7480u,196670u,7479u,7481u,262205u,6u,7483u,6452u,196670u,7482u,7483u,458809u,10u,7484u,41u,
7477u,7479u,7482u,327809u,10u,7485u,7476u,7484u,196670u,7469u,7485u,262205u,6u,7488u,7402u,196670u,
7487u,7488u,262205u,6u,7490u,7365u,196670u,7489u,7490u,262205u,6u,7492u,6449u,196670u,7491u,7492u,
458809u,10u,7493u,41u,7487u,7489u,7491u,262205u,6u,7495u,7434u,196670u,7494u,7495u,327745u,9u,
7497u,7331u,548u,262205u,6u,7498u,7497u,196670u,7496u,7498u,262205u,6u,7500u,6452u,196670u,7499u,
7500u,458809u,10u,7501u,41u,7494u,7496u,7499u,327809u,10u,7502u,7493u,7501u,196670u,7486u,7502u,
262205u,6u,7505u,7411u,196670u,7504u,7505u,262205u,6u,7507u,7365u,196670u,7506u,7507u,262205u,6u,
7509u,6449u,196670u,7508u,7509u,458809u,10u,7510u,41u,7504u,7506u,7508u,262205u,6u,7512u,7443u,
196670u,7511u,7512u,327745u,9u,7514u,7331u,548u,262205u,6u,7515u,7514u,196670u,7513u,7515u,262205u,
6u,7517u,6452u,196670u,7516u,7517u,458809u,10u,7518u,41u,7511u,7513u,7516u,327809u,10u,7519u,
7510u,7518u,196670u,7503u,7519u,262205u,6u,7521u,7307u,196670u,7520u,7521u,327737u,57u,7522u,65u,
7520u,196855u,7524u,0u,262394u,7522u,7523u,7524u,131320u,7523u,262205u,6u,7525u,7307u,262205u,6u,
7526u,7379u,327808u,6u,7527u,7525u,7526u,393281u,584u,7528u,578u,548u,7527u,262205u,10u,7529u,
7528u,262205u,10u,7530u,7452u,327809u,10u,7531u,7530u,7529u,196670u,7452u,7531u,262205u,6u,7532u,
7307u,262205u,6u,7533u,6452u,262205u,6u,7534u,7379u,327808u,6u,7535u,7533u,7534u,327808u,6u,
7536u,7532u,7535u,393281u,584u,7537u,578u,548u,7536u,262205u,10u,7538u,7537u,262205u,10u,7539u,
7469u,327809u,10u,7540u,7539u,7538u,196670u,7469u,7540u,262205u,6u,7541u,7307u,262205u,6u,7542u,
6452u,327812u,6u,7543u,2490u,7542u,262205u,6u,7544u,7379u,327808u,6u,7545u,7543u,7544u,327808u,
6u,7546u,7541u,7545u,393281u,584u,7547u,578u,548u,7546u,262205u,10u,7548u,7547u,262205u,10u,
7549u,7486u,327809u,10u,7550u,7549u,7548u,196670u,7486u,7550u,262205u,6u,7551u,7307u,262205u,6u,
7552u,6452u,327812u,6u,7553u,2404u,7552u,262205u,6u,7554u,7379u,327808u,6u,7555u,7553u,7554u,
327808u,6u,7556u,7551u,7555u,393281u,584u,7557u,578u,548u,7556u,262205u,10u,7558u,7557u,262205u,
10u,7559u,7503u,327809u,10u,7560u,7559u,7558u,196670u,7503u,7560u,131321u,7524u,131320u,7524u,262205u,
6u,7562u,7319u,196670u,7561u,7562u,327737u,57u,7563u,65u,7561u,196855u,7565u,0u,262394u,7563u,
7564u,7565u,131320u,7564u,262205u,6u,7566u,7319u,262205u,6u,7567u,7379u,327808u,6u,7568u,7566u,
7567u,393281u,584u,7569u,578u,548u,7568u,262205u,10u,7570u,7569u,262205u,10u,7571u,7452u,327809u,
10u,7572u,7571u,7570u,196670u,7452u,7572u,262205u,6u,7573u,7319u,262205u,6u,7574u,6452u,262205u,
6u,7575u,7379u,327808u,6u,7576u,7574u,7575u,327808u,6u,7577u,7573u,7576u,393281u,584u,7578u,
578u,548u,7577u,262205u,10u,7579u,7578u,262205u,10u,7580u,7469u,327809u,10u,7581u,7580u,7579u,
196670u,7469u,7581u,262205u,6u,7582u,7319u,262205u,6u,7583u,6452u,327812u,6u,7584u,2490u,7583u,
262205u,6u,7585u,7379u,327808u,6u,7586u,7584u,7585u,327808u,6u,7587u,7582u,7586u,393281u,584u,
7588u,578u,548u,7587u,262205u,10u,7589u,7588u,262205u,10u,7590u,7486u,327809u,10u,7591u,7590u,
7589u,196670u,7486u,7591u,262205u,6u,7592u,7319u,262205u,6u,7593u,6452u,327812u,6u,7594u,2404u,
7593u,262205u,6u,7595u,7379u,327808u,6u,7596u,7594u,7595u,327808u,6u,7597u,7592u,7596u,393281u,
584u,7598u,578u,548u,7597u,262205u,10u,7599u,7598u,262205u,10u,7600u,7503u,327809u,10u,7601u,
7600u,7599u,196670u,7503u,7601u,131321u,7565u,131320u,7565u,262205u,10u,7604u,7452u,196670u,7603u,7604u,
327737u,10u,7605u,419u,7603u,196670u,7602u,7605u,262205u,10u,7608u,7469u,196670u,7607u,7608u,327737u,
10u,7609u,419u,7607u,196670u,7606u,7609u,262205u,10u,7611u,7486u,393228u,10u,7612u,1u,21u,
7611u,196670u,7610u,7612u,262205u,10u,7615u,7503u,196670u,7614u,7615u,327737u,10u,7616u,419u,7614u,
196670u,7613u,7616u,262205u,10u,7618u,7606u,327745u,9u,7619u,7339u,548u,262205u,6u,7620u,7619u,
262205u,6u,7621u,7379u,327808u,6u,7622u,7620u,7621u,393281u,584u,7623u,578u,548u,7622u,262205u,
10u,7624u,7623u,327813u,10u,7625u,7618u,7624u,262205u,10u,7626u,7602u,262205u,10u,7627u,7610u,
327813u,10u,7628u,7626u,7627u,327809u,10u,7629u,7625u,7628u,196670u,7617u,7629u,327745u,9u,7630u,
7339u,548u,262205u,6u,7631u,7630u,262205u,6u,7632u,7379u,327808u,6u,7633u,7631u,7632u,262205u,
10u,7634u,7617u,393281u,584u,7635u,578u,548u,7633u,196670u,7635u,7634u,262205u,6u,7636u,7372u,
262205u,6u,7637u,7379u,327808u,6u,7638u,7636u,7637u,262205u,10u,7639u,7613u,262205u,10u,7640u,
7617u,393228u,10u,7641u,1u,21u,7640u,327813u,10u,7642u,7639u,7641u,393281u,584u,7643u,578u,
548u,7638u,196670u,7643u,7642u,131321u,7383u,131320u,7383u,262205u,6u,7644u,7379u,327808u,6u,7645u,
7644u,551u,196670u,7379u,7645u,131321u,7380u,131320u,7382u,262205u,6u,7646u,7372u,262205u,6u,7647u,
6452u,327808u,6u,7648u,7646u,7647u,262205u,6u,7650u,7372u,196670u,7649u,7650u,196670u,7651u,7648u,
327745u,9u,7653u,7331u,548u,262205u,6u,7654u,7653u,196670u,7652u,7654u,458809u,2u,7655u,35u,
7649u,7651u,7652u,131321u,7360u,131320u,7360u,262205u,6u,7656u,7356u,327808u,6u,7657u,7656u,551u,
196670u,7356u,7657u,131321u,7357u,131320u,7359u,131326u,6723u,65592u,327734u,57u,436u,0u,430u,196663u,
19u,431u,196663u,19u,432u,196663u,19u,433u,196663u,422u,434u,196663u,8u,435u,131320u,437u,
262203u,19u,7667u,7u,262203u,19u,7669u,7u,262203u,9u,7671u,7u,262203u,9u,7674u,7u,
262203u,9u,7677u,7u,262203u,58u,7709u,7u,262203u,9u,7712u,7u,262203u,9u,7715u,7u,
262203u,9u,7722u,7u,262203u,9u,7727u,7u,262203u,9u,7732u,7u,262203u,58u,7733u,7u,
262203u,9u,7736u,7u,262203u,9u,7743u,7u,262203u,58u,7744u,7u,262203u,9u,7747u,7u,
262203u,9u,7754u,7u,262203u,9u,7769u,7u,262203u,9u,7771u,7u,262203u,9u,7775u,7u,
262203u,58u,7776u,7u,262203u,9u,7779u,7u,262203u,9u,7787u,7u,262203u,58u,7788u,7u,
262203u,9u,7791u,7u,262203u,19u,7799u,7u,262203u,8u,7803u,7u,262203u,9u,7805u,7u,
262203u,11u,7806u,7u,262203u,19u,7809u,7u,262203u,8u,7813u,7u,262203u,9u,7815u,7u,
262203u,11u,7816u,7u,262203u,19u,7819u,7u,262203u,8u,7823u,7u,262203u,9u,7825u,7u,
262203u,11u,7826u,7u,262203u,19u,7829u,7u,262203u,8u,7833u,7u,262203u,9u,7835u,7u,
262203u,11u,7836u,7u,262203u,19u,7839u,7u,262203u,8u,7843u,7u,262203u,9u,7845u,7u,
262203u,11u,7846u,7u,262203u,19u,7849u,7u,262203u,8u,7853u,7u,262203u,9u,7855u,7u,
262203u,11u,7856u,7u,262203u,19u,7859u,7u,262203u,8u,7861u,7u,262203u,9u,7863u,7u,
262203u,11u,7864u,7u,262203u,19u,7867u,7u,262203u,8u,7869u,7u,262203u,9u,7871u,7u,
262203u,11u,7872u,7u,262203u,9u,7875u,7u,262203u,9u,7884u,7u,262203u,9u,7891u,7u,
262203u,9u,7900u,7u,262203u,9u,7906u,7u,262203u,9u,7914u,7u,262203u,9u,7923u,7u,
262203u,9u,7932u,7u,262203u,9u,7938u,7u,262203u,9u,7946u,7u,262203u,9u,7955u,7u,
262203u,11u,7964u,7u,262203u,9u,7965u,7u,262203u,9u,7967u,7u,262203u,9u,7969u,7u,
262203u,9u,7972u,7u,262203u,9u,7974u,7u,262203u,9u,7977u,7u,262203u,11u,7981u,7u,
262203u,9u,7982u,7u,262203u,9u,7984u,7u,262203u,9u,7986u,7u,262203u,9u,7989u,7u,
262203u,9u,7991u,7u,262203u,9u,7994u,7u,262203u,11u,7998u,7u,262203u,9u,7999u,7u,
262203u,9u,8001u,7u,262203u,9u,8003u,7u,262203u,9u,8006u,7u,262203u,9u,8008u,7u,
262203u,9u,8011u,7u,262203u,11u,8015u,7u,262203u,9u,8016u,7u,262203u,9u,8018u,7u,
262203u,9u,8020u,7u,262203u,9u,8023u,7u,262203u,9u,8025u,7u,262203u,9u,8028u,7u,
262203u,9u,8032u,7u,262203u,9u,8073u,7u,262203u,11u,8114u,7u,262203u,11u,8115u,7u,
262203u,11u,8118u,7u,262203u,11u,8119u,7u,262203u,11u,8122u,7u,262203u,11u,8125u,7u,
262203u,11u,8126u,7u,262203u,11u,8129u,7u,262203u,11u,8142u,7u,262203u,19u,8230u,7u,
262203u,9u,8233u,7u,262203u,11u,8236u,7u,262203u,19u,8240u,7u,262203u,9u,8242u,7u,
262203u,11u,8245u,7u,262203u,19u,8248u,7u,262203u,8u,8250u,7u,262203u,9u,8252u,7u,
262203u,11u,8253u,7u,262203u,19u,8256u,7u,262203u,8u,8258u,7u,262203u,9u,8260u,7u,
262203u,11u,8261u,7u,262203u,9u,8264u,7u,262203u,9u,8274u,7u,262203u,9u,8281u,7u,
262203u,9u,8284u,7u,262203u,9u,8296u,7u,262203u,9u,8299u,7u,262203u,19u,8311u,7u,
262203u,8u,8313u,7u,262203u,9u,8315u,7u,262203u,11u,8316u,7u,262203u,19u,8319u,7u,
262203u,8u,8321u,7u,262203u,9u,8323u,7u,262203u,11u,8324u,7u,262203u,9u,8327u,7u,
262203u,9u,8336u,7u,262203u,11u,8342u,7u,262203u,11u,8349u,7u,262203u,11u,8356u,7u,
262203u,11u,8363u,7u,262203u,11u,8370u,7u,262203u,11u,8377u,7u,262203u,11u,8380u,7u,
262203u,11u,8383u,7u,262203u,11u,8393u,7u,262203u,11u,8396u,7u,262203u,11u,8406u,7u,
262203u,11u,8420u,7u,262203u,11u,8424u,7u,262203u,11u,8440u,7u,262203u,11u,8444u,7u,
262203u,11u,8448u,7u,262203u,11u,8460u,7u,262203u,11u,8467u,7u,262203u,11u,8474u,7u,
262203u,11u,8481u,7u,262203u,9u,8488u,7u,262203u,9u,8490u,7u,262203u,9u,8494u,7u,
262203u,9u,8499u,7u,262203u,9u,8504u,7u,262203u,11u,8513u,7u,262203u,9u,8640u,7u,
262203u,11u,8649u,7u,262203u,11u,8652u,7u,262203u,58u,8787u,7u,262203u,9u,8792u,7u,
262203u,19u,8894u,7u,262203u,19u,8896u,7u,262203u,19u,8901u,7u,262203u,19u,8903u,7u,
262203u,19u,8911u,7u,262203u,19u,8913u,7u,262203u,9u,8915u,7u,262203u,9u,8918u,7u,
262203u,9u,8921u,7u,262203u,58u,8953u,7u,262203u,9u,8956u,7u,262203u,9u,8959u,7u,
262203u,9u,8966u,7u,262203u,9u,8971u,7u,262203u,9u,8976u,7u,262203u,58u,8977u,7u,
262203u,9u,8980u,7u,262203u,9u,8987u,7u,262203u,58u,8988u,7u,262203u,9u,8991u,7u,
262203u,9u,8998u,7u,262203u,9u,9013u,7u,262203u,9u,9015u,7u,262203u,9u,9019u,7u,
262203u,58u,9020u,7u,262203u,9u,9023u,7u,262203u,9u,9031u,7u,262203u,58u,9032u,7u,
262203u,9u,9035u,7u,262203u,19u,9043u,7u,262203u,8u,9047u,7u,262203u,9u,9049u,7u,
262203u,11u,9050u,7u,262203u,19u,9053u,7u,262203u,8u,9057u,7u,262203u,9u,9059u,7u,
262203u,11u,9060u,7u,262203u,19u,9063u,7u,262203u,8u,9067u,7u,262203u,9u,9069u,7u,
262203u,11u,9070u,7u,262203u,19u,9073u,7u,262203u,8u,9077u,7u,262203u,9u,9079u,7u,
262203u,11u,9080u,7u,262203u,19u,9083u,7u,262203u,8u,9085u,7u,262203u,9u,9087u,7u,
262203u,11u,9088u,7u,262203u,9u,9091u,7u,262203u,9u,9100u,7u,262203u,9u,9107u,7u,
262203u,9u,9116u,7u,262203u,9u,9122u,7u,262203u,11u,9128u,7u,262203u,9u,9129u,7u,
262203u,9u,9153u,7u,262203u,9u,9178u,7u,262203u,9u,9190u,7u,262203u,11u,9202u,7u,
262203u,11u,9203u,7u,262203u,9u,9206u,7u,262203u,9u,9214u,7u,262203u,11u,9222u,7u,
262203u,9u,9223u,7u,262203u,9u,9247u,7u,262203u,9u,9272u,7u,262203u,9u,9286u,7u,
262203u,11u,9300u,7u,262203u,11u,9301u,7u,262203u,9u,9304u,7u,262203u,9u,9313u,7u,
262203u,11u,9322u,7u,262203u,9u,9323u,7u,262203u,9u,9347u,7u,262203u,9u,9374u,7u,
262203u,9u,9389u,7u,262203u,11u,9404u,7u,262203u,11u,9407u,7u,262203u,9u,9421u,7u,
262203u,9u,9453u,7u,262203u,19u,9488u,7u,262203u,9u,9491u,7u,262203u,11u,9494u,7u,
262203u,19u,9498u,7u,262203u,9u,9500u,7u,262203u,11u,9503u,7u,262203u,19u,9506u,7u,
262203u,8u,9508u,7u,262203u,9u,9510u,7u,262203u,11u,9511u,7u,262203u,9u,9514u,7u,
262203u,9u,9524u,7u,262203u,9u,9531u,7u,262203u,9u,9534u,7u,262203u,19u,9546u,7u,
262203u,8u,9548u,7u,262203u,9u,9550u,7u,262203u,11u,9551u,7u,262203u,19u,9554u,7u,
262203u,8u,9556u,7u,262203u,9u,9558u,7u,262203u,11u,9559u,7u,262203u,19u,9562u,7u,
262203u,8u,9564u,7u,262203u,9u,9566u,7u,262203u,11u,9567u,7u,262203u,19u,9570u,7u,
262203u,8u,9572u,7u,262203u,9u,9574u,7u,262203u,11u,9575u,7u,262203u,9u,9578u,7u,
262203u,9u,9587u,7u,262203u,11u,9593u,7u,262203u,11u,9600u,7u,262203u,11u,9607u,7u,
262203u,11u,9614u,7u,262203u,11u,9617u,7u,262203u,11u,9627u,7u,262203u,11u,9641u,7u,
262203u,11u,9646u,7u,262203u,19u,9687u,7u,262203u,8u,9689u,7u,262203u,9u,9691u,7u,
262203u,11u,9692u,7u,262203u,9u,9695u,7u,262203u,9u,9704u,7u,262203u,9u,9709u,7u,
262203u,11u,9718u,7u,262203u,9u,9774u,7u,262203u,11u,9783u,7u,262203u,11u,9786u,7u,
262203u,11u,9796u,7u,262203u,58u,9857u,7u,262203u,9u,9862u,7u,262203u,9u,9902u,7u,
262203u,11u,9911u,7u,262203u,11u,9914u,7u,262203u,11u,9924u,7u,262203u,11u,9935u,7u,
262203u,9u,9973u,7u,262203u,9u,9982u,7u,262203u,9u,9984u,7u,262203u,9u,9988u,7u,
262203u,11u,9997u,7u,262203u,9u,10090u,7u,262203u,11u,10099u,7u,262203u,11u,10102u,7u,
262203u,58u,10199u,7u,262203u,9u,10204u,7u,262203u,19u,10278u,7u,262203u,19u,10280u,7u,
262203u,19u,10288u,7u,262203u,19u,10290u,7u,262203u,9u,10292u,7u,262203u,9u,10295u,7u,
262203u,9u,10298u,7u,262203u,58u,10330u,7u,262203u,9u,10333u,7u,262203u,9u,10336u,7u,
262203u,9u,10343u,7u,262203u,9u,10347u,7u,262203u,9u,10351u,7u,262203u,58u,10352u,7u,
262203u,9u,10355u,7u,262203u,9u,10361u,7u,262203u,58u,10362u,7u,262203u,9u,10365u,7u,
262203u,9u,10371u,7u,262203u,9u,10386u,7u,262203u,9u,10388u,7u,262203u,9u,10392u,7u,
262203u,58u,10393u,7u,262203u,9u,10396u,7u,262203u,9u,10404u,7u,262203u,58u,10405u,7u,
262203u,9u,10408u,7u,262203u,19u,10416u,7u,262203u,8u,10420u,7u,262203u,9u,10422u,7u,
262203u,11u,10423u,7u,262203u,19u,10426u,7u,262203u,8u,10428u,7u,262203u,9u,10430u,7u,
262203u,11u,10431u,7u,262203u,9u,10434u,7u,262203u,9u,10443u,7u,262203u,9u,10450u,7u,
262203u,9u,10457u,7u,262203u,11u,10466u,7u,262203u,9u,10467u,7u,262203u,9u,10473u,7u,
262203u,9u,10479u,7u,262203u,9u,10503u,7u,262203u,9u,10528u,7u,262203u,9u,10540u,7u,
262203u,9u,10560u,7u,262203u,19u,10590u,7u,262203u,9u,10593u,7u,262203u,11u,10596u,7u,
262203u,19u,10600u,7u,262203u,9u,10602u,7u,262203u,11u,10605u,7u,262203u,19u,10608u,7u,
262203u,8u,10610u,7u,262203u,9u,10612u,7u,262203u,11u,10613u,7u,262203u,9u,10616u,7u,
262203u,9u,10626u,7u,262203u,9u,10633u,7u,262203u,9u,10636u,7u,262203u,19u,10648u,7u,
262203u,8u,10650u,7u,262203u,9u,10652u,7u,262203u,11u,10653u,7u,262203u,9u,10656u,7u,
262203u,11u,10665u,7u,262203u,11u,10676u,7u,262203u,11u,10694u,7u,262203u,9u,10701u,7u,
262203u,9u,10707u,7u,262203u,9u,10713u,7u,262203u,9u,10762u,7u,262203u,11u,10771u,7u,
262203u,11u,10774u,7u,262203u,58u,10818u,7u,262203u,19u,10853u,7u,262203u,19u,10855u,7u,
327745u,9u,7660u,434u,548u,262205u,6u,7661u,7660u,196855u,7666u,0u,590075u,7661u,7665u,63u,
7662u,64u,7663u,65u,7664u,131320u,7665u,131326u,6465u,131320u,7662u,262205u,12u,7668u,431u,196670u,
7667u,7668u,262205u,12u,7670u,432u,196670u,7669u,7670u,327745u,9u,7672u,434u,551u,262205u,6u,
7673u,7672u,196670u,7671u,7673u,327745u,9u,7675u,434u,2490u,262205u,6u,7676u,7675u,196670u,7674u,
7676u,327745u,9u,7678u,434u,2404u,262205u,6u,7679u,7678u,196670u,7677u,7679u,262205u,6u,7680u,
7671u,327859u,57u,7681u,7680u,548u,262205u,6u,7682u,7674u,327859u,57u,7683u,7682u,548u,327846u,
57u,7684u,7681u,7683u,262205u,6u,7685u,7677u,327859u,57u,7686u,7685u,548u,327846u,57u,7687u,
7684u,7686u,196855u,7689u,0u,262394u,7687u,7688u,7689u,131320u,7688u,131326u,6465u,131320u,7689u,327745u,
9u,7691u,7667u,551u,262205u,6u,7692u,7691u,262205u,6u,7693u,7671u,262205u,6u,7694u,7674u,
327812u,6u,7695u,7693u,7694u,327851u,57u,7696u,7692u,7695u,196855u,7698u,0u,262394u,7696u,7697u,
7698u,131320u,7697u,131326u,6465u,131320u,7698u,327745u,9u,7700u,7669u,551u,262205u,6u,7701u,7700u,
262205u,6u,7702u,7671u,262205u,6u,7703u,7677u,327812u,6u,7704u,7702u,7703u,327851u,57u,7705u,
7701u,7704u,196855u,7707u,0u,262394u,7705u,7706u,7707u,131320u,7706u,131326u,6465u,131320u,7707u,327745u,
58u,7710u,434u,3118u,262205u,57u,7711u,7710u,196670u,7709u,7711u,327745u,9u,7713u,434u,2510u,
262205u,6u,7714u,7713u,196670u,7712u,7714u,262205u,6u,7716u,7712u,196670u,7715u,7716u,327737u,57u,
7717u,65u,7715u,262312u,57u,7718u,7717u,196855u,7720u,0u,262394u,7718u,7719u,7720u,131320u,7719u,
131326u,6465u,131320u,7720u,262205u,6u,7723u,7677u,327812u,6u,7724u,3118u,7723u,262205u,6u,7725u,
7674u,327812u,6u,7726u,7724u,7725u,196670u,7722u,7726u,262205u,6u,7728u,7677u,327812u,6u,7729u,
3118u,7728u,262205u,6u,7730u,7677u,327812u,6u,7731u,7729u,7730u,196670u,7727u,7731u,262205u,57u,
7734u,7709u,196670u,7733u,7734u,327737u,57u,7735u,61u,7733u,196855u,7738u,0u,262394u,7735u,7737u,
7741u,131320u,7737u,262205u,6u,7739u,7677u,327812u,6u,7740u,3118u,7739u,196670u,7736u,7740u,131321u,
7738u,131320u,7741u,196670u,7736u,548u,131321u,7738u,131320u,7738u,262205u,6u,7742u,7736u,196670u,7732u,
7742u,262205u,57u,7745u,7709u,196670u,7744u,7745u,327737u,57u,7746u,61u,7744u,196855u,7749u,0u,
262394u,7746u,7748u,7752u,131320u,7748u,262205u,6u,7750u,7677u,327812u,6u,7751u,3118u,7750u,196670u,
7747u,7751u,131321u,7749u,131320u,7752u,196670u,7747u,548u,131321u,7749u,131320u,7749u,262205u,6u,7753u,
7747u,196670u,7743u,7753u,262205u,6u,7755u,7722u,262205u,6u,7756u,7727u,327808u,6u,7757u,7755u,
7756u,262205u,6u,7758u,7732u,327808u,6u,7759u,7757u,7758u,262205u,6u,7760u,7743u,327808u,6u,
7761u,7759u,7760u,196670u,7754u,7761u,327745u,9u,7762u,434u,2624u,262205u,6u,7763u,7762u,262205u,
6u,7764u,7754u,327857u,57u,7765u,7763u,7764u,196855u,7767u,0u,262394u,7765u,7766u,7767u,131320u,
7766u,131326u,6465u,131320u,7767u,262205u,6u,7770u,7712u,196670u,7769u,7770u,262205u,6u,7772u,7769u,
262205u,6u,7773u,7722u,327808u,6u,7774u,7772u,7773u,196670u,7771u,7774u,262205u,57u,7777u,7709u,
196670u,7776u,7777u,327737u,57u,7778u,61u,7776u,196855u,7781u,0u,262394u,7778u,7780u,7785u,131320u,
7780u,262205u,6u,7782u,7771u,262205u,6u,7783u,7727u,327808u,6u,7784u,7782u,7783u,196670u,7779u,
7784u,131321u,7781u,131320u,7785u,196670u,7779u,548u,131321u,7781u,131320u,7781u,262205u,6u,7786u,7779u,
196670u,7775u,7786u,262205u,57u,7789u,7709u,196670u,7788u,7789u,327737u,57u,7790u,61u,7788u,196855u,
7793u,0u,262394u,7790u,7792u,7797u,131320u,7792u,262205u,6u,7794u,7775u,262205u,6u,7795u,7732u,
327808u,6u,7796u,7794u,7795u,196670u,7791u,7796u,131321u,7793u,131320u,7797u,196670u,7791u,548u,131321u,
7793u,131320u,7793u,262205u,6u,7798u,7791u,196670u,7787u,7798u,262205u,6u,7800u,7671u,262205u,6u,
7801u,7677u,327812u,6u,7802u,7800u,7801u,262205u,7u,7804u,435u,196670u,7803u,7804u,196670u,7805u,
7802u,196670u,7806u,630u,458809u,12u,7807u,17u,7803u,7805u,7806u,262205u,7u,7808u,7803u,196670u,
435u,7808u,196670u,7799u,7807u,262205u,6u,7810u,7671u,262205u,6u,7811u,7677u,327812u,6u,7812u,
7810u,7811u,262205u,7u,7814u,435u,196670u,7813u,7814u,196670u,7815u,7812u,196670u,7816u,630u,458809u,
12u,7817u,17u,7813u,7815u,7816u,262205u,7u,7818u,7813u,196670u,435u,7818u,196670u,7809u,7817u,
262205u,6u,7820u,7671u,262205u,6u,7821u,7677u,327812u,6u,7822u,7820u,7821u,262205u,7u,7824u,
435u,196670u,7823u,7824u,196670u,7825u,7822u,196670u,7826u,630u,458809u,12u,7827u,17u,7823u,7825u,
7826u,262205u,7u,7828u,7823u,196670u,435u,7828u,196670u,7819u,7827u,262205u,6u,7830u,7671u,262205u,
6u,7831u,7677u,327812u,6u,7832u,7830u,7831u,262205u,7u,7834u,435u,196670u,7833u,7834u,196670u,
7835u,7832u,196670u,7836u,630u,458809u,12u,7837u,17u,7833u,7835u,7836u,262205u,7u,7838u,7833u,
196670u,435u,7838u,196670u,7829u,7837u,262205u,6u,7840u,7671u,262205u,6u,7841u,7677u,327812u,6u,
7842u,7840u,7841u,262205u,7u,7844u,435u,196670u,7843u,7844u,196670u,7845u,7842u,196670u,7846u,630u,
458809u,12u,7847u,17u,7843u,7845u,7846u,262205u,7u,7848u,7843u,196670u,435u,7848u,196670u,7839u,
7847u,262205u,6u,7850u,7671u,262205u,6u,7851u,7677u,327812u,6u,7852u,7850u,7851u,262205u,7u,
7854u,435u,196670u,7853u,7854u,196670u,7855u,7852u,196670u,7856u,630u,458809u,12u,7857u,17u,7853u,
7855u,7856u,262205u,7u,7858u,7853u,196670u,435u,7858u,196670u,7849u,7857u,262205u,6u,7860u,7677u,
262205u,7u,7862u,435u,196670u,7861u,7862u,196670u,7863u,7860u,196670u,7864u,630u,458809u,12u,7865u,
17u,7861u,7863u,7864u,262205u,7u,7866u,7861u,196670u,435u,7866u,196670u,7859u,7865u,262205u,6u,
7868u,7677u,262205u,7u,7870u,435u,196670u,7869u,7870u,196670u,7871u,7868u,196670u,7872u,630u,458809u,
12u,7873u,17u,7869u,7871u,7872u,262205u,7u,7874u,7869u,196670u,435u,7874u,196670u,7867u,7873u,
196670u,7875u,548u,131321u,7876u,131320u,7876u,262390u,7878u,7879u,0u,131321u,7880u,131320u,7880u,262205u,
6u,7881u,7875u,262205u,6u,7882u,7671u,327857u,57u,7883u,7881u,7882u,262394u,7883u,7877u,7878u,
131320u,7877u,327745u,9u,7885u,7667u,548u,262205u,6u,7886u,7885u,262205u,6u,7887u,7875u,262205u,
6u,7888u,7674u,327812u,6u,7889u,7887u,7888u,327808u,6u,7890u,7886u,7889u,196670u,7884u,7890u,
196670u,7891u,548u,131321u,7892u,131320u,7892u,262390u,7894u,7895u,0u,131321u,7896u,131320u,7896u,262205u,
6u,7897u,7891u,262205u,6u,7898u,7677u,327857u,57u,7899u,7897u,7898u,262394u,7899u,7893u,7894u,
131320u,7893u,262205u,6u,7901u,7769u,262205u,6u,7902u,7891u,262205u,6u,7903u,7674u,327812u,6u,
7904u,7902u,7903u,327808u,6u,7905u,7901u,7904u,196670u,7900u,7905u,262205u,6u,7907u,7769u,262205u,
6u,7908u,7677u,262205u,6u,7909u,7891u,327808u,6u,7910u,7908u,7909u,262205u,6u,7911u,7674u,
327812u,6u,7912u,7910u,7911u,327808u,6u,7913u,7907u,7912u,196670u,7906u,7913u,262205u,6u,7915u,
7769u,262205u,6u,7916u,7677u,327812u,6u,7917u,2490u,7916u,262205u,6u,7918u,7891u,327808u,6u,
7919u,7917u,7918u,262205u,6u,7920u,7674u,327812u,6u,7921u,7919u,7920u,327808u,6u,7922u,7915u,
7921u,196670u,7914u,7922u,262205u,6u,7924u,7769u,262205u,6u,7925u,7677u,327812u,6u,7926u,2404u,
7925u,262205u,6u,7927u,7891u,327808u,6u,7928u,7926u,7927u,262205u,6u,7929u,7674u,327812u,6u,
7930u,7928u,7929u,327808u,6u,7931u,7924u,7930u,196670u,7923u,7931u,262205u,6u,7933u,7771u,262205u,
6u,7934u,7891u,262205u,6u,7935u,7677u,327812u,6u,7936u,7934u,7935u,327808u,6u,7937u,7933u,
7936u,196670u,7932u,7937u,262205u,6u,7939u,7771u,262205u,6u,7940u,7677u,262205u,6u,7941u,7891u,
327808u,6u,7942u,7940u,7941u,262205u,6u,7943u,7677u,327812u,6u,7944u,7942u,7943u,327808u,6u,
7945u,7939u,7944u,196670u,7938u,7945u,262205u,6u,7947u,7771u,262205u,6u,7948u,7677u,327812u,6u,
7949u,2490u,7948u,262205u,6u,7950u,7891u,327808u,6u,7951u,7949u,7950u,262205u,6u,7952u,7677u,
327812u,6u,7953u,7951u,7952u,327808u,6u,7954u,7947u,7953u,196670u,7946u,7954u,262205u,6u,7956u,
7771u,262205u,6u,7957u,7677u,327812u,6u,7958u,2404u,7957u,262205u,6u,7959u,7891u,327808u,6u,
7960u,7958u,7959u,262205u,6u,7961u,7677u,327812u,6u,7962u,7960u,7961u,327808u,6u,7963u,7956u,
7962u,196670u,7955u,7963u,262205u,6u,7966u,7900u,196670u,7965u,7966u,262205u,6u,7968u,7884u,196670u,
7967u,7968u,262205u,6u,7970u,7674u,196670u,7969u,7970u,458809u,10u,7971u,41u,7965u,7967u,7969u,
262205u,6u,7973u,7932u,196670u,7972u,7973u,327745u,9u,7975u,7859u,548u,262205u,6u,7976u,7975u,
196670u,7974u,7976u,262205u,6u,7978u,7677u,196670u,7977u,7978u,458809u,10u,7979u,41u,7972u,7974u,
7977u,327809u,10u,7980u,7971u,7979u,196670u,7964u,7980u,262205u,6u,7983u,7906u,196670u,7982u,7983u,
262205u,6u,7985u,7884u,196670u,7984u,7985u,262205u,6u,7987u,7674u,196670u,7986u,7987u,458809u,10u,
7988u,41u,7982u,7984u,7986u,262205u,6u,7990u,7938u,196670u,7989u,7990u,327745u,9u,7992u,7859u,
548u,262205u,6u,7993u,7992u,196670u,7991u,7993u,262205u,6u,7995u,7677u,196670u,7994u,7995u,458809u,
10u,7996u,41u,7989u,7991u,7994u,327809u,10u,7997u,7988u,7996u,196670u,7981u,7997u,262205u,6u,
8000u,7914u,196670u,7999u,8000u,262205u,6u,8002u,7884u,196670u,8001u,8002u,262205u,6u,8004u,7674u,
196670u,8003u,8004u,458809u,10u,8005u,41u,7999u,8001u,8003u,262205u,6u,8007u,7946u,196670u,8006u,
8007u,327745u,9u,8009u,7859u,548u,262205u,6u,8010u,8009u,196670u,8008u,8010u,262205u,6u,8012u,
7677u,196670u,8011u,8012u,458809u,10u,8013u,41u,8006u,8008u,8011u,327809u,10u,8014u,8005u,8013u,
196670u,7998u,8014u,262205u,6u,8017u,7923u,196670u,8016u,8017u,262205u,6u,8019u,7884u,196670u,8018u,
8019u,262205u,6u,8021u,7674u,196670u,8020u,8021u,458809u,10u,8022u,41u,8016u,8018u,8020u,262205u,
6u,8024u,7955u,196670u,8023u,8024u,327745u,9u,8026u,7859u,548u,262205u,6u,8027u,8026u,196670u,
8025u,8027u,262205u,6u,8029u,7677u,196670u,8028u,8029u,458809u,10u,8030u,41u,8023u,8025u,8028u,
327809u,10u,8031u,8022u,8030u,196670u,8015u,8031u,262205u,6u,8033u,7775u,196670u,8032u,8033u,327737u,
57u,8034u,65u,8032u,196855u,8036u,0u,262394u,8034u,8035u,8036u,131320u,8035u,262205u,6u,8037u,
7775u,262205u,6u,8038u,7891u,327808u,6u,8039u,8037u,8038u,393281u,584u,8040u,578u,548u,8039u,
262205u,10u,8041u,8040u,262205u,10u,8042u,7964u,327809u,10u,8043u,8042u,8041u,196670u,7964u,8043u,
262205u,6u,8044u,7775u,262205u,6u,8045u,7677u,262205u,6u,8046u,7891u,327808u,6u,8047u,8045u,
8046u,327808u,6u,8048u,8044u,8047u,393281u,584u,8049u,578u,548u,8048u,262205u,10u,8050u,8049u,
262205u,10u,8051u,7981u,327809u,10u,8052u,8051u,8050u,196670u,7981u,8052u,262205u,6u,8053u,7775u,
262205u,6u,8054u,7677u,327812u,6u,8055u,2490u,8054u,262205u,6u,8056u,7891u,327808u,6u,8057u,
8055u,8056u,327808u,6u,8058u,8053u,8057u,393281u,584u,8059u,578u,548u,8058u,262205u,10u,8060u,
8059u,262205u,10u,8061u,7998u,327809u,10u,8062u,8061u,8060u,196670u,7998u,8062u,262205u,6u,8063u,
7775u,262205u,6u,8064u,7677u,327812u,6u,8065u,2404u,8064u,262205u,6u,8066u,7891u,327808u,6u,
8067u,8065u,8066u,327808u,6u,8068u,8063u,8067u,393281u,584u,8069u,578u,548u,8068u,262205u,10u,
8070u,8069u,262205u,10u,8071u,8015u,327809u,10u,8072u,8071u,8070u,196670u,8015u,8072u,131321u,8036u,
131320u,8036u,262205u,6u,8074u,7787u,196670u,8073u,8074u,327737u,57u,8075u,65u,8073u,196855u,8077u,
0u,262394u,8075u,8076u,8077u,131320u,8076u,262205u,6u,8078u,7787u,262205u,6u,8079u,7891u,327808u,
6u,8080u,8078u,8079u,393281u,584u,8081u,578u,548u,8080u,262205u,10u,8082u,8081u,262205u,10u,
8083u,7964u,327809u,10u,8084u,8083u,8082u,196670u,7964u,8084u,262205u,6u,8085u,7787u,262205u,6u,
8086u,7677u,262205u,6u,8087u,7891u,327808u,6u,8088u,8086u,8087u,327808u,6u,8089u,8085u,8088u,
393281u,584u,8090u,578u,548u,8089u,262205u,10u,8091u,8090u,262205u,10u,8092u,7981u,327809u,10u,
8093u,8092u,8091u,196670u,7981u,8093u,262205u,6u,8094u,7787u,262205u,6u,8095u,7677u,327812u,6u,
8096u,2490u,8095u,262205u,6u,8097u,7891u,327808u,6u,8098u,8096u,8097u,327808u,6u,8099u,8094u,
8098u,393281u,584u,8100u,578u,548u,8099u,262205u,10u,8101u,8100u,262205u,10u,8102u,7998u,327809u,
10u,8103u,8102u,8101u,196670u,7998u,8103u,262205u,6u,8104u,7787u,262205u,6u,8105u,7677u,327812u,
6u,8106u,2404u,8105u,262205u,6u,8107u,7891u,327808u,6u,8108u,8106u,8107u,327808u,6u,8109u,
8104u,8108u,393281u,584u,8110u,578u,548u,8109u,262205u,10u,8111u,8110u,262205u,10u,8112u,8015u,
327809u,10u,8113u,8112u,8111u,196670u,8015u,8113u,131321u,8077u,131320u,8077u,262205u,10u,8116u,7964u,
196670u,8115u,8116u,327737u,10u,8117u,419u,8115u,196670u,8114u,8117u,262205u,10u,8120u,7981u,196670u,
8119u,8120u,327737u,10u,8121u,419u,8119u,196670u,8118u,8121u,262205u,10u,8123u,7998u,393228u,10u,
8124u,1u,21u,8123u,196670u,8122u,8124u,262205u,10u,8127u,8015u,196670u,8126u,8127u,327737u,10u,
8128u,419u,8126u,196670u,8125u,8128u,262205u,10u,8130u,8118u,327745u,9u,8131u,7867u,548u,262205u,
6u,8132u,8131u,262205u,6u,8133u,7891u,327808u,6u,8134u,8132u,8133u,393281u,584u,8135u,578u,
548u,8134u,262205u,10u,8136u,8135u,327813u,10u,8137u,8130u,8136u,262205u,10u,8138u,8114u,262205u,
10u,8139u,8122u,327813u,10u,8140u,8138u,8139u,327809u,10u,8141u,8137u,8140u,196670u,8129u,8141u,
262205u,10u,8143u,8125u,262205u,10u,8144u,8129u,393228u,10u,8145u,1u,21u,8144u,327813u,10u,
8146u,8143u,8145u,196670u,8142u,8146u,327745u,9u,8147u,7799u,548u,262205u,6u,8148u,8147u,262205u,
6u,8149u,7875u,262205u,6u,8150u,7677u,327812u,6u,8151u,8149u,8150u,262205u,6u,8152u,7891u,
327808u,6u,8153u,8151u,8152u,327808u,6u,8154u,8148u,8153u,262205u,10u,8155u,8114u,393281u,584u,
8156u,578u,548u,8154u,196670u,8156u,8155u,327745u,9u,8157u,7809u,548u,262205u,6u,8158u,8157u,
262205u,6u,8159u,7875u,262205u,6u,8160u,7677u,327812u,6u,8161u,8159u,8160u,262205u,6u,8162u,
7891u,327808u,6u,8163u,8161u,8162u,327808u,6u,8164u,8158u,8163u,262205u,10u,8165u,8118u,393281u,
584u,8166u,578u,548u,8164u,196670u,8166u,8165u,327745u,9u,8167u,7819u,548u,262205u,6u,8168u,
8167u,262205u,6u,8169u,7875u,262205u,6u,8170u,7677u,327812u,6u,8171u,8169u,8170u,262205u,6u,
8172u,7891u,327808u,6u,8173u,8171u,8172u,327808u,6u,8174u,8168u,8173u,262205u,10u,8175u,8122u,
393281u,584u,8176u,578u,548u,8174u,196670u,8176u,8175u,327745u,9u,8177u,7829u,548u,262205u,6u,
8178u,8177u,262205u,6u,8179u,7875u,262205u,6u,8180u,7677u,327812u,6u,8181u,8179u,8180u,262205u,
6u,8182u,7891u,327808u,6u,8183u,8181u,8182u,327808u,6u,8184u,8178u,8183u,262205u,10u,8185u,
8125u,393281u,584u,8186u,578u,548u,8184u,196670u,8186u,8185u,327745u,9u,8187u,7839u,548u,262205u,
6u,8188u,8187u,262205u,6u,8189u,7875u,262205u,6u,8190u,7677u,327812u,6u,8191u,8189u,8190u,
262205u,6u,8192u,7891u,327808u,6u,8193u,8191u,8192u,327808u,6u,8194u,8188u,8193u,262205u,10u,
8195u,8129u,393281u,584u,8196u,578u,548u,8194u,196670u,8196u,8195u,327745u,9u,8197u,7849u,548u,
262205u,6u,8198u,8197u,262205u,6u,8199u,7875u,262205u,6u,8200u,7677u,327812u,6u,8201u,8199u,
8200u,262205u,6u,8202u,7891u,327808u,6u,8203u,8201u,8202u,327808u,6u,8204u,8198u,8203u,262205u,
10u,8205u,8142u,393281u,584u,8206u,578u,548u,8204u,196670u,8206u,8205u,327745u,9u,8207u,7867u,
548u,262205u,6u,8208u,8207u,262205u,6u,8209u,7891u,327808u,6u,8210u,8208u,8209u,262205u,10u,
8211u,8129u,393281u,584u,8212u,578u,548u,8210u,196670u,8212u,8211u,327745u,9u,8213u,7859u,548u,
262205u,6u,8214u,8213u,262205u,6u,8215u,7891u,327808u,6u,8216u,8214u,8215u,262205u,10u,8217u,
8142u,393281u,584u,8218u,578u,548u,8216u,196670u,8218u,8217u,131321u,7895u,131320u,7895u,262205u,6u,
8219u,7891u,327808u,6u,8220u,8219u,551u,196670u,7891u,8220u,131321u,7892u,131320u,7894u,131321u,7879u,
131320u,7879u,262205u,6u,8221u,7875u,327808u,6u,8222u,8221u,551u,196670u,7875u,8222u,131321u,7876u,
131320u,7878u,393281u,9u,8223u,434u,2750u,551u,262205u,6u,8224u,8223u,327745u,9u,8225u,434u,
2624u,262205u,6u,8226u,8225u,327851u,57u,8227u,8224u,8226u,196855u,8229u,0u,262394u,8227u,8228u,
8229u,131320u,8228u,327745u,19u,8231u,434u,2750u,262205u,12u,8232u,8231u,196670u,8230u,8232u,327745u,
9u,8234u,434u,2624u,262205u,6u,8235u,8234u,196670u,8233u,8235u,196670u,8236u,630u,458809u,2u,
8237u,24u,8230u,8233u,8236u,262205u,12u,8238u,8230u,327745u,19u,8239u,434u,2750u,196670u,8239u,
8238u,131321u,8229u,131320u,8229u,262205u,12u,8241u,433u,196670u,8240u,8241u,327745u,9u,8243u,7667u,
551u,262205u,6u,8244u,8243u,196670u,8242u,8244u,196670u,8245u,630u,458809u,2u,8246u,24u,8240u,
8242u,8245u,262205u,12u,8247u,8240u,196670u,433u,8247u,262205u,6u,8249u,7677u,262205u,7u,8251u,
435u,196670u,8250u,8251u,196670u,8252u,8249u,196670u,8253u,630u,458809u,12u,8254u,17u,8250u,8252u,
8253u,262205u,7u,8255u,8250u,196670u,435u,8255u,196670u,8248u,8254u,262205u,6u,8257u,7677u,262205u,
7u,8259u,435u,196670u,8258u,8259u,196670u,8260u,8257u,196670u,8261u,630u,458809u,12u,8262u,17u,
8258u,8260u,8261u,262205u,7u,8263u,8258u,196670u,435u,8263u,196670u,8256u,8262u,262205u,6u,8265u,
7671u,327810u,6u,8266u,8265u,551u,196670u,8264u,8266u,131321u,8267u,131320u,8267u,262390u,8269u,8270u,
0u,131321u,8271u,131320u,8271u,262205u,6u,8272u,8264u,327855u,57u,8273u,8272u,548u,262394u,8273u,
8268u,8269u,131320u,8268u,327745u,9u,8275u,7667u,548u,262205u,6u,8276u,8275u,262205u,6u,8277u,
8264u,262205u,6u,8278u,7674u,327812u,6u,8279u,8277u,8278u,327808u,6u,8280u,8276u,8279u,196670u,
8274u,8280u,262205u,6u,8282u,8264u,327853u,57u,8283u,8282u,548u,196855u,8286u,0u,262394u,8283u,
8285u,8294u,131320u,8285u,327745u,9u,8287u,7849u,548u,262205u,6u,8288u,8287u,262205u,6u,8289u,
8264u,327810u,6u,8290u,8289u,551u,262205u,6u,8291u,7677u,327812u,6u,8292u,8290u,8291u,327808u,
6u,8293u,8288u,8292u,196670u,8284u,8293u,131321u,8286u,131320u,8294u,196670u,8284u,548u,131321u,8286u,
131320u,8286u,262205u,6u,8295u,8284u,196670u,8281u,8295u,262205u,6u,8297u,8264u,327853u,57u,8298u,
8297u,548u,196855u,8301u,0u,262394u,8298u,8300u,8309u,131320u,8300u,327745u,9u,8302u,7839u,548u,
262205u,6u,8303u,8302u,262205u,6u,8304u,8264u,327810u,6u,8305u,8304u,551u,262205u,6u,8306u,
7677u,327812u,6u,8307u,8305u,8306u,327808u,6u,8308u,8303u,8307u,196670u,8299u,8308u,131321u,8301u,
131320u,8309u,196670u,8299u,548u,131321u,8301u,131320u,8301u,262205u,6u,8310u,8299u,196670u,8296u,8310u,
262205u,6u,8312u,7677u,262205u,7u,8314u,435u,196670u,8313u,8314u,196670u,8315u,8312u,196670u,8316u,
630u,458809u,12u,8317u,17u,8313u,8315u,8316u,262205u,7u,8318u,8313u,196670u,435u,8318u,196670u,
8311u,8317u,262205u,6u,8320u,7677u,262205u,7u,8322u,435u,196670u,8321u,8322u,196670u,8323u,8320u,
196670u,8324u,630u,458809u,12u,8325u,17u,8321u,8323u,8324u,262205u,7u,8326u,8321u,196670u,435u,
8326u,196670u,8319u,8325u,196670u,8327u,548u,131321u,8328u,131320u,8328u,262390u,8330u,8331u,0u,131321u,
8332u,131320u,8332u,262205u,6u,8333u,8327u,262205u,6u,8334u,7677u,327857u,57u,8335u,8333u,8334u,
262394u,8335u,8329u,8330u,131320u,8329u,262205u,6u,8337u,8264u,262205u,6u,8338u,7677u,327812u,6u,
8339u,8337u,8338u,262205u,6u,8340u,8327u,327808u,6u,8341u,8339u,8340u,196670u,8336u,8341u,327745u,
9u,8343u,7799u,548u,262205u,6u,8344u,8343u,262205u,6u,8345u,8336u,327808u,6u,8346u,8344u,
8345u,393281u,584u,8347u,578u,548u,8346u,262205u,10u,8348u,8347u,196670u,8342u,8348u,327745u,9u,
8350u,7809u,548u,262205u,6u,8351u,8350u,262205u,6u,8352u,8336u,327808u,6u,8353u,8351u,8352u,
393281u,584u,8354u,578u,548u,8353u,262205u,10u,8355u,8354u,196670u,8349u,8355u,327745u,9u,8357u,
7819u,548u,262205u,6u,8358u,8357u,262205u,6u,8359u,8336u,327808u,6u,8360u,8358u,8359u,393281u,
584u,8361u,578u,548u,8360u,262205u,10u,8362u,8361u,196670u,8356u,8362u,327745u,9u,8364u,7829u,
548u,262205u,6u,8365u,8364u,262205u,6u,8366u,8336u,327808u,6u,8367u,8365u,8366u,393281u,584u,
8368u,578u,548u,8367u,262205u,10u,8369u,8368u,196670u,8363u,8369u,327745u,9u,8371u,7839u,548u,
262205u,6u,8372u,8371u,262205u,6u,8373u,8336u,327808u,6u,8374u,8372u,8373u,393281u,584u,8375u,
578u,548u,8374u,262205u,10u,8376u,8375u,196670u,8370u,8376u,262205u,10u,8378u,8370u,393228u,10u,
8379u,1u,21u,8378u,196670u,8377u,8379u,262205u,6u,8381u,8281u,327851u,57u,8382u,8381u,548u,
196855u,8385u,0u,262394u,8382u,8384u,8391u,131320u,8384u,262205u,6u,8386u,8281u,262205u,6u,8387u,
8327u,327808u,6u,8388u,8386u,8387u,393281u,584u,8389u,578u,548u,8388u,262205u,10u,8390u,8389u,
196670u,8383u,8390u,131321u,8385u,131320u,8391u,196670u,8383u,630u,131321u,8385u,131320u,8385u,262205u,10u,
8392u,8383u,196670u,8380u,8392u,262205u,6u,8394u,8296u,327851u,57u,8395u,8394u,548u,196855u,8398u,
0u,262394u,8395u,8397u,8404u,131320u,8397u,262205u,6u,8399u,8296u,262205u,6u,8400u,8327u,327808u,
6u,8401u,8399u,8400u,393281u,584u,8402u,578u,548u,8401u,262205u,10u,8403u,8402u,196670u,8396u,
8403u,131321u,8398u,131320u,8404u,196670u,8396u,630u,131321u,8398u,131320u,8398u,262205u,10u,8405u,8396u,
196670u,8393u,8405u,327745u,9u,8407u,7669u,548u,262205u,6u,8408u,8407u,262205u,6u,8409u,8336u,
327808u,6u,8410u,8408u,8409u,393281u,584u,8411u,578u,548u,8410u,262205u,10u,8412u,8411u,327745u,
9u,8413u,8248u,548u,262205u,6u,8414u,8413u,262205u,6u,8415u,8327u,327808u,6u,8416u,8414u,
8415u,393281u,584u,8417u,578u,548u,8416u,262205u,10u,8418u,8417u,327809u,10u,8419u,8412u,8418u,
196670u,8406u,8419u,262205u,10u,8421u,8406u,262205u,10u,8422u,8377u,327813u,10u,8423u,8421u,8422u,
196670u,8420u,8423u,327745u,9u,8425u,8256u,548u,262205u,6u,8426u,8425u,262205u,6u,8427u,8327u,
327808u,6u,8428u,8426u,8427u,393281u,584u,8429u,578u,548u,8428u,262205u,10u,8430u,8429u,262205u,
10u,8431u,8406u,262205u,10u,8432u,8363u,327813u,10u,8433u,8431u,8432u,262205u,10u,8434u,8377u,
262205u,10u,8435u,8377u,327813u,10u,8436u,8434u,8435u,327811u,10u,8437u,701u,8436u,327813u,10u,
8438u,8433u,8437u,327809u,10u,8439u,8430u,8438u,196670u,8424u,8439u,262205u,10u,8441u,8424u,262205u,
10u,8442u,8356u,327813u,10u,8443u,8441u,8442u,196670u,8440u,8443u,262205u,10u,8445u,8424u,262205u,
10u,8446u,8342u,327813u,10u,8447u,8445u,8446u,196670u,8444u,8447u,262205u,10u,8449u,8424u,262205u,
10u,8450u,8393u,327813u,10u,8451u,8449u,8450u,196670u,8448u,8451u,327745u,9u,8452u,8319u,548u,
262205u,6u,8453u,8452u,262205u,6u,8454u,8327u,327808u,6u,8455u,8453u,8454u,262205u,10u,8456u,
8424u,262205u,10u,8457u,8349u,327813u,10u,8458u,8456u,8457u,393281u,584u,8459u,578u,548u,8455u,
196670u,8459u,8458u,262205u,10u,8461u,8440u,262205u,10u,8462u,8342u,327813u,10u,8463u,8461u,8462u,
262205u,10u,8464u,8342u,327811u,10u,8465u,701u,8464u,327813u,10u,8466u,8463u,8465u,196670u,8460u,
8466u,262205u,10u,8468u,8448u,262205u,10u,8469u,8349u,327813u,10u,8470u,8468u,8469u,262205u,10u,
8471u,8349u,327811u,10u,8472u,701u,8471u,327813u,10u,8473u,8470u,8472u,196670u,8467u,8473u,262205u,
10u,8475u,8444u,262205u,10u,8476u,8356u,262205u,10u,8477u,8356u,327813u,10u,8478u,8476u,8477u,
327811u,10u,8479u,701u,8478u,327813u,10u,8480u,8475u,8479u,196670u,8474u,8480u,262205u,10u,8482u,
8420u,262205u,10u,8483u,8363u,327813u,10u,8484u,8482u,8483u,262205u,10u,8485u,8363u,327811u,10u,
8486u,701u,8485u,327813u,10u,8487u,8484u,8486u,196670u,8481u,8487u,262205u,6u,8489u,8327u,196670u,
8488u,8489u,262205u,6u,8491u,7677u,262205u,6u,8492u,8327u,327808u,6u,8493u,8491u,8492u,196670u,
8490u,8493u,262205u,6u,8495u,7677u,327812u,6u,8496u,2490u,8495u,262205u,6u,8497u,8327u,327808u,
6u,8498u,8496u,8497u,196670u,8494u,8498u,262205u,6u,8500u,7677u,327812u,6u,8501u,2404u,8500u,
262205u,6u,8502u,8327u,327808u,6u,8503u,8501u,8502u,196670u,8499u,8503u,196670u,8504u,548u,131321u,
8505u,131320u,8505u,262390u,8507u,8508u,0u,131321u,8509u,131320u,8509u,262205u,6u,8510u,8504u,262205u,
6u,8511u,7674u,327857u,57u,8512u,8510u,8511u,262394u,8512u,8506u,8507u,131320u,8506u,262205u,6u,
8514u,8274u,262205u,6u,8515u,8504u,327808u,6u,8516u,8514u,8515u,393281u,584u,8517u,578u,548u,
8516u,262205u,10u,8518u,8517u,196670u,8513u,8518u,393281u,9u,8519u,434u,2750u,548u,262205u,6u,
8520u,8519u,262205u,6u,8521u,8488u,262205u,6u,8522u,7674u,327812u,6u,8523u,8521u,8522u,262205u,
6u,8524u,8504u,327808u,6u,8525u,8523u,8524u,327808u,6u,8526u,8520u,8525u,262205u,10u,8527u,
8460u,262205u,10u,8528u,8513u,327813u,10u,8529u,8527u,8528u,393281u,584u,8530u,578u,548u,8526u,
262205u,10u,8531u,8530u,327809u,10u,8532u,8531u,8529u,393281u,584u,8533u,578u,548u,8526u,196670u,
8533u,8532u,393281u,9u,8534u,434u,2750u,548u,262205u,6u,8535u,8534u,262205u,6u,8536u,8490u,
262205u,6u,8537u,7674u,327812u,6u,8538u,8536u,8537u,262205u,6u,8539u,8504u,327808u,6u,8540u,
8538u,8539u,327808u,6u,8541u,8535u,8540u,262205u,10u,8542u,8467u,262205u,10u,8543u,8513u,327813u,
10u,8544u,8542u,8543u,393281u,584u,8545u,578u,548u,8541u,262205u,10u,8546u,8545u,327809u,10u,
8547u,8546u,8544u,393281u,584u,8548u,578u,548u,8541u,196670u,8548u,8547u,393281u,9u,8549u,434u,
2750u,548u,262205u,6u,8550u,8549u,262205u,6u,8551u,8494u,262205u,6u,8552u,7674u,327812u,6u,
8553u,8551u,8552u,262205u,6u,8554u,8504u,327808u,6u,8555u,8553u,8554u,327808u,6u,8556u,8550u,
8555u,262205u,10u,8557u,8474u,262205u,10u,8558u,8513u,327813u,10u,8559u,8557u,8558u,393281u,584u,
8560u,578u,548u,8556u,262205u,10u,8561u,8560u,327809u,10u,8562u,8561u,8559u,393281u,584u,8563u,
578u,548u,8556u,196670u,8563u,8562u,393281u,9u,8564u,434u,2750u,548u,262205u,6u,8565u,8564u,
262205u,6u,8566u,8499u,262205u,6u,8567u,7674u,327812u,6u,8568u,8566u,8567u,262205u,6u,8569u,
8504u,327808u,6u,8570u,8568u,8569u,327808u,6u,8571u,8565u,8570u,262205u,10u,8572u,8481u,262205u,
10u,8573u,8513u,327813u,10u,8574u,8572u,8573u,393281u,584u,8575u,578u,548u,8571u,262205u,10u,
8576u,8575u,327809u,10u,8577u,8576u,8574u,393281u,584u,8578u,578u,548u,8571u,196670u,8578u,8577u,
327745u,9u,8579u,433u,548u,262205u,6u,8580u,8579u,262205u,6u,8581u,8264u,262205u,6u,8582u,
7674u,327812u,6u,8583u,8581u,8582u,262205u,6u,8584u,8504u,327808u,6u,8585u,8583u,8584u,327808u,
6u,8586u,8580u,8585u,262205u,10u,8587u,8460u,262205u,6u,8588u,7769u,262205u,6u,8589u,8488u,
262205u,6u,8590u,7674u,327812u,6u,8591u,8589u,8590u,262205u,6u,8592u,8504u,327808u,6u,8593u,
8591u,8592u,327808u,6u,8594u,8588u,8593u,393281u,584u,8595u,578u,548u,8594u,262205u,10u,8596u,
8595u,327813u,10u,8597u,8587u,8596u,262205u,10u,8598u,8467u,262205u,6u,8599u,7769u,262205u,6u,
8600u,8490u,262205u,6u,8601u,7674u,327812u,6u,8602u,8600u,8601u,262205u,6u,8603u,8504u,327808u,
6u,8604u,8602u,8603u,327808u,6u,8605u,8599u,8604u,393281u,584u,8606u,578u,548u,8605u,262205u,
10u,8607u,8606u,327813u,10u,8608u,8598u,8607u,327809u,10u,8609u,8597u,8608u,262205u,10u,8610u,
8474u,262205u,6u,8611u,7769u,262205u,6u,8612u,8494u,262205u,6u,8613u,7674u,327812u,6u,8614u,
8612u,8613u,262205u,6u,8615u,8504u,327808u,6u,8616u,8614u,8615u,327808u,6u,8617u,8611u,8616u,
393281u,584u,8618u,578u,548u,8617u,262205u,10u,8619u,8618u,327813u,10u,8620u,8610u,8619u,327809u,
10u,8621u,8609u,8620u,262205u,10u,8622u,8481u,262205u,6u,8623u,7769u,262205u,6u,8624u,8499u,
262205u,6u,8625u,7674u,327812u,6u,8626u,8624u,8625u,262205u,6u,8627u,8504u,327808u,6u,8628u,
8626u,8627u,327808u,6u,8629u,8623u,8628u,393281u,584u,8630u,578u,548u,8629u,262205u,10u,8631u,
8630u,327813u,10u,8632u,8622u,8631u,327809u,10u,8633u,8621u,8632u,393281u,584u,8634u,578u,548u,
8586u,262205u,10u,8635u,8634u,327809u,10u,8636u,8635u,8633u,393281u,584u,8637u,578u,548u,8586u,
196670u,8637u,8636u,131321u,8508u,131320u,8508u,262205u,6u,8638u,8504u,327808u,6u,8639u,8638u,551u,
196670u,8504u,8639u,131321u,8505u,131320u,8507u,196670u,8640u,548u,131321u,8641u,131320u,8641u,262390u,8643u,
8644u,0u,131321u,8645u,131320u,8645u,262205u,6u,8646u,8640u,262205u,6u,8647u,7677u,327857u,57u,
8648u,8646u,8647u,262394u,8648u,8642u,8643u,131320u,8642u,262205u,6u,8650u,8281u,327851u,57u,8651u,
8650u,548u,196855u,8654u,0u,262394u,8651u,8653u,8660u,131320u,8653u,262205u,6u,8655u,8281u,262205u,
6u,8656u,8640u,327808u,6u,8657u,8655u,8656u,393281u,584u,8658u,578u,548u,8657u,262205u,10u,
8659u,8658u,196670u,8652u,8659u,131321u,8654u,131320u,8660u,196670u,8652u,630u,131321u,8654u,131320u,8654u,
262205u,10u,8661u,8652u,196670u,8649u,8661u,393281u,9u,8662u,434u,2750u,548u,262205u,6u,8663u,
8662u,262205u,6u,8664u,7722u,262205u,6u,8665u,8488u,262205u,6u,8666u,7677u,327812u,6u,8667u,
8665u,8666u,327808u,6u,8668u,8664u,8667u,262205u,6u,8669u,8640u,327808u,6u,8670u,8668u,8669u,
327808u,6u,8671u,8663u,8670u,262205u,10u,8672u,8460u,262205u,10u,8673u,8649u,327813u,10u,8674u,
8672u,8673u,393281u,584u,8675u,578u,548u,8671u,262205u,10u,8676u,8675u,327809u,10u,8677u,8676u,
8674u,393281u,584u,8678u,578u,548u,8671u,196670u,8678u,8677u,393281u,9u,8679u,434u,2750u,548u,
262205u,6u,8680u,8679u,262205u,6u,8681u,7722u,262205u,6u,8682u,8490u,262205u,6u,8683u,7677u,
327812u,6u,8684u,8682u,8683u,327808u,6u,8685u,8681u,8684u,262205u,6u,8686u,8640u,327808u,6u,
8687u,8685u,8686u,327808u,6u,8688u,8680u,8687u,262205u,10u,8689u,8467u,262205u,10u,8690u,8649u,
327813u,10u,8691u,8689u,8690u,393281u,584u,8692u,578u,548u,8688u,262205u,10u,8693u,8692u,327809u,
10u,8694u,8693u,8691u,393281u,584u,8695u,578u,548u,8688u,196670u,8695u,8694u,393281u,9u,8696u,
434u,2750u,548u,262205u,6u,8697u,8696u,262205u,6u,8698u,7722u,262205u,6u,8699u,8494u,262205u,
6u,8700u,7677u,327812u,6u,8701u,8699u,8700u,327808u,6u,8702u,8698u,8701u,262205u,6u,8703u,
8640u,327808u,6u,8704u,8702u,8703u,327808u,6u,8705u,8697u,8704u,262205u,10u,8706u,8474u,262205u,
10u,8707u,8649u,327813u,10u,8708u,8706u,8707u,393281u,584u,8709u,578u,548u,8705u,262205u,10u,
8710u,8709u,327809u,10u,8711u,8710u,8708u,393281u,584u,8712u,578u,548u,8705u,196670u,8712u,8711u,
393281u,9u,8713u,434u,2750u,548u,262205u,6u,8714u,8713u,262205u,6u,8715u,7722u,262205u,6u,
8716u,8499u,262205u,6u,8717u,7677u,327812u,6u,8718u,8716u,8717u,327808u,6u,8719u,8715u,8718u,
262205u,6u,8720u,8640u,327808u,6u,8721u,8719u,8720u,327808u,6u,8722u,8714u,8721u,262205u,10u,
8723u,8481u,262205u,10u,8724u,8649u,327813u,10u,8725u,8723u,8724u,393281u,584u,8726u,578u,548u,
8722u,262205u,10u,8727u,8726u,327809u,10u,8728u,8727u,8725u,393281u,584u,8729u,578u,548u,8722u,
196670u,8729u,8728u,327745u,9u,8730u,8311u,548u,262205u,6u,8731u,8730u,262205u,6u,8732u,8640u,
327808u,6u,8733u,8731u,8732u,262205u,10u,8734u,8460u,262205u,6u,8735u,7771u,262205u,6u,8736u,
8488u,262205u,6u,8737u,7677u,327812u,6u,8738u,8736u,8737u,262205u,6u,8739u,8640u,327808u,6u,
8740u,8738u,8739u,327808u,6u,8741u,8735u,8740u,393281u,584u,8742u,578u,548u,8741u,262205u,10u,
8743u,8742u,327813u,10u,8744u,8734u,8743u,262205u,10u,8745u,8467u,262205u,6u,8746u,7771u,262205u,
6u,8747u,8490u,262205u,6u,8748u,7677u,327812u,6u,8749u,8747u,8748u,262205u,6u,8750u,8640u,
327808u,6u,8751u,8749u,8750u,327808u,6u,8752u,8746u,8751u,393281u,584u,8753u,578u,548u,8752u,
262205u,10u,8754u,8753u,327813u,10u,8755u,8745u,8754u,327809u,10u,8756u,8744u,8755u,262205u,10u,
8757u,8474u,262205u,6u,8758u,7771u,262205u,6u,8759u,8494u,262205u,6u,8760u,7677u,327812u,6u,
8761u,8759u,8760u,262205u,6u,8762u,8640u,327808u,6u,8763u,8761u,8762u,327808u,6u,8764u,8758u,
8763u,393281u,584u,8765u,578u,548u,8764u,262205u,10u,8766u,8765u,327813u,10u,8767u,8757u,8766u,
327809u,10u,8768u,8756u,8767u,262205u,10u,8769u,8481u,262205u,6u,8770u,7771u,262205u,6u,8771u,
8499u,262205u,6u,8772u,7677u,327812u,6u,8773u,8771u,8772u,262205u,6u,8774u,8640u,327808u,6u,
8775u,8773u,8774u,327808u,6u,8776u,8770u,8775u,393281u,584u,8777u,578u,548u,8776u,262205u,10u,
8778u,8777u,327813u,10u,8779u,8769u,8778u,327809u,10u,8780u,8768u,8779u,393281u,584u,8781u,578u,
548u,8733u,262205u,10u,8782u,8781u,327809u,10u,8783u,8782u,8780u,393281u,584u,8784u,578u,548u,
8733u,196670u,8784u,8783u,131321u,8644u,131320u,8644u,262205u,6u,8785u,8640u,327808u,6u,8786u,8785u,
551u,196670u,8640u,8786u,131321u,8641u,131320u,8643u,262205u,57u,8788u,7709u,196670u,8787u,8788u,327737u,
57u,8789u,61u,8787u,196855u,8791u,0u,262394u,8789u,8790u,8791u,131320u,8790u,262205u,6u,8793u,
7722u,262205u,6u,8794u,7727u,327808u,6u,8795u,8793u,8794u,196670u,8792u,8795u,393281u,9u,8796u,
434u,2750u,548u,262205u,6u,8797u,8796u,262205u,6u,8798u,8792u,262205u,6u,8799u,8488u,327808u,
6u,8800u,8798u,8799u,327808u,6u,8801u,8797u,8800u,262205u,10u,8802u,8460u,393281u,584u,8803u,
578u,548u,8801u,262205u,10u,8804u,8803u,327809u,10u,8805u,8804u,8802u,393281u,584u,8806u,578u,
548u,8801u,196670u,8806u,8805u,393281u,9u,8807u,434u,2750u,548u,262205u,6u,8808u,8807u,262205u,
6u,8809u,8792u,262205u,6u,8810u,8490u,327808u,6u,8811u,8809u,8810u,327808u,6u,8812u,8808u,
8811u,262205u,10u,8813u,8467u,393281u,584u,8814u,578u,548u,8812u,262205u,10u,8815u,8814u,327809u,
10u,8816u,8815u,8813u,393281u,584u,8817u,578u,548u,8812u,196670u,8817u,8816u,393281u,9u,8818u,
434u,2750u,548u,262205u,6u,8819u,8818u,262205u,6u,8820u,8792u,262205u,6u,8821u,8494u,327808u,
6u,8822u,8820u,8821u,327808u,6u,8823u,8819u,8822u,262205u,10u,8824u,8474u,393281u,584u,8825u,
578u,548u,8823u,262205u,10u,8826u,8825u,327809u,10u,8827u,8826u,8824u,393281u,584u,8828u,578u,
548u,8823u,196670u,8828u,8827u,393281u,9u,8829u,434u,2750u,548u,262205u,6u,8830u,8829u,262205u,
6u,8831u,8792u,262205u,6u,8832u,8499u,327808u,6u,8833u,8831u,8832u,327808u,6u,8834u,8830u,
8833u,262205u,10u,8835u,8481u,393281u,584u,8836u,578u,548u,8834u,262205u,10u,8837u,8836u,327809u,
10u,8838u,8837u,8835u,393281u,584u,8839u,578u,548u,8834u,196670u,8839u,8838u,393281u,9u,8840u,
434u,2750u,548u,262205u,6u,8841u,8840u,262205u,6u,8842u,8792u,262205u,6u,8843u,7732u,327808u,
6u,8844u,8842u,8843u,262205u,6u,8845u,8488u,327808u,6u,8846u,8844u,8845u,327808u,6u,8847u,
8841u,8846u,262205u,10u,8848u,8460u,393281u,584u,8849u,578u,548u,8847u,262205u,10u,8850u,8849u,
327809u,10u,8851u,8850u,8848u,393281u,584u,8852u,578u,548u,8847u,196670u,8852u,8851u,393281u,9u,
8853u,434u,2750u,548u,262205u,6u,8854u,8853u,262205u,6u,8855u,8792u,262205u,6u,8856u,7732u,
327808u,6u,8857u,8855u,8856u,262205u,6u,8858u,8490u,327808u,6u,8859u,8857u,8858u,327808u,6u,
8860u,8854u,8859u,262205u,10u,8861u,8467u,393281u,584u,8862u,578u,548u,8860u,262205u,10u,8863u,
8862u,327809u,10u,8864u,8863u,8861u,393281u,584u,8865u,578u,548u,8860u,196670u,8865u,8864u,393281u,
9u,8866u,434u,2750u,548u,262205u,6u,8867u,8866u,262205u,6u,8868u,8792u,262205u,6u,8869u,
7732u,327808u,6u,8870u,8868u,8869u,262205u,6u,8871u,8494u,327808u,6u,8872u,8870u,8871u,327808u,
6u,8873u,8867u,8872u,262205u,10u,8874u,8474u,393281u,584u,8875u,578u,548u,8873u,262205u,10u,
8876u,8875u,327809u,10u,8877u,8876u,8874u,393281u,584u,8878u,578u,548u,8873u,196670u,8878u,8877u,
393281u,9u,8879u,434u,2750u,548u,262205u,6u,8880u,8879u,262205u,6u,8881u,8792u,262205u,6u,
8882u,7732u,327808u,6u,8883u,8881u,8882u,262205u,6u,8884u,8499u,327808u,6u,8885u,8883u,8884u,
327808u,6u,8886u,8880u,8885u,262205u,10u,8887u,8481u,393281u,584u,8888u,578u,548u,8886u,262205u,
10u,8889u,8888u,327809u,10u,8890u,8889u,8887u,393281u,584u,8891u,578u,548u,8886u,196670u,8891u,
8890u,131321u,8791u,131320u,8791u,131321u,8331u,131320u,8331u,262205u,6u,8892u,8327u,327808u,6u,8893u,
8892u,551u,196670u,8327u,8893u,131321u,8328u,131320u,8330u,262205u,12u,8895u,8248u,196670u,8894u,8895u,
262205u,12u,8897u,8311u,196670u,8896u,8897u,393273u,2u,8898u,29u,8894u,8896u,262205u,12u,8899u,
8894u,196670u,8248u,8899u,262205u,12u,8900u,8896u,196670u,8311u,8900u,262205u,12u,8902u,8256u,196670u,
8901u,8902u,262205u,12u,8904u,8319u,196670u,8903u,8904u,393273u,2u,8905u,29u,8901u,8903u,262205u,
12u,8906u,8901u,196670u,8256u,8906u,262205u,12u,8907u,8903u,196670u,8319u,8907u,131321u,8270u,131320u,
8270u,262205u,6u,8908u,8264u,327810u,6u,8909u,8908u,551u,196670u,8264u,8909u,131321u,8267u,131320u,
8269u,131326u,6723u,131320u,7663u,262205u,12u,8912u,431u,196670u,8911u,8912u,262205u,12u,8914u,432u,
196670u,8913u,8914u,327745u,9u,8916u,434u,551u,262205u,6u,8917u,8916u,196670u,8915u,8917u,327745u,
9u,8919u,434u,2490u,262205u,6u,8920u,8919u,196670u,8918u,8920u,327745u,9u,8922u,434u,2404u,
262205u,6u,8923u,8922u,196670u,8921u,8923u,262205u,6u,8924u,8915u,327859u,57u,8925u,8924u,548u,
262205u,6u,8926u,8918u,327859u,57u,8927u,8926u,548u,327846u,57u,8928u,8925u,8927u,262205u,6u,
8929u,8921u,327859u,57u,8930u,8929u,548u,327846u,57u,8931u,8928u,8930u,196855u,8933u,0u,262394u,
8931u,8932u,8933u,131320u,8932u,131326u,6465u,131320u,8933u,327745u,9u,8935u,8911u,551u,262205u,6u,
8936u,8935u,262205u,6u,8937u,8915u,262205u,6u,8938u,8918u,327812u,6u,8939u,8937u,8938u,327851u,
57u,8940u,8936u,8939u,196855u,8942u,0u,262394u,8940u,8941u,8942u,131320u,8941u,131326u,6465u,131320u,
8942u,327745u,9u,8944u,8913u,551u,262205u,6u,8945u,8944u,262205u,6u,8946u,8915u,262205u,6u,
8947u,8921u,327812u,6u,8948u,8946u,8947u,327851u,57u,8949u,8945u,8948u,196855u,8951u,0u,262394u,
8949u,8950u,8951u,131320u,8950u,131326u,6465u,131320u,8951u,327745u,58u,8954u,434u,3118u,262205u,57u,
8955u,8954u,196670u,8953u,8955u,327745u,9u,8957u,434u,2510u,262205u,6u,8958u,8957u,196670u,8956u,
8958u,262205u,6u,8960u,8956u,196670u,8959u,8960u,327737u,57u,8961u,65u,8959u,262312u,57u,8962u,
8961u,196855u,8964u,0u,262394u,8962u,8963u,8964u,131320u,8963u,131326u,6465u,131320u,8964u,262205u,6u,
8967u,8921u,327812u,6u,8968u,2404u,8967u,262205u,6u,8969u,8918u,327812u,6u,8970u,8968u,8969u,
196670u,8966u,8970u,262205u,6u,8972u,8921u,327812u,6u,8973u,2404u,8972u,262205u,6u,8974u,8921u,
327812u,6u,8975u,8973u,8974u,196670u,8971u,8975u,262205u,57u,8978u,8953u,196670u,8977u,8978u,327737u,
57u,8979u,61u,8977u,196855u,8982u,0u,262394u,8979u,8981u,8985u,131320u,8981u,262205u,6u,8983u,
8921u,327812u,6u,8984u,2404u,8983u,196670u,8980u,8984u,131321u,8982u,131320u,8985u,196670u,8980u,548u,
131321u,8982u,131320u,8982u,262205u,6u,8986u,8980u,196670u,8976u,8986u,262205u,57u,8989u,8953u,196670u,
8988u,8989u,327737u,57u,8990u,61u,8988u,196855u,8993u,0u,262394u,8990u,8992u,8996u,131320u,8992u,
262205u,6u,8994u,8921u,327812u,6u,8995u,2404u,8994u,196670u,8991u,8995u,131321u,8993u,131320u,8996u,
196670u,8991u,548u,131321u,8993u,131320u,8993u,262205u,6u,8997u,8991u,196670u,8987u,8997u,262205u,6u,
8999u,8966u,262205u,6u,9000u,8971u,327808u,6u,9001u,8999u,9000u,262205u,6u,9002u,8976u,327808u,
6u,9003u,9001u,9002u,262205u,6u,9004u,8987u,327808u,6u,9005u,9003u,9004u,196670u,8998u,9005u,
327745u,9u,9006u,434u,2624u,262205u,6u,9007u,9006u,262205u,6u,9008u,8998u,327857u,57u,9009u,
9007u,9008u,196855u,9011u,0u,262394u,9009u,9010u,9011u,131320u,9010u,131326u,6465u,131320u,9011u,262205u,
6u,9014u,8956u,196670u,9013u,9014u,262205u,6u,9016u,9013u,262205u,6u,9017u,8966u,327808u,6u,
9018u,9016u,9017u,196670u,9015u,9018u,262205u,57u,9021u,8953u,196670u,9020u,9021u,327737u,57u,9022u,
61u,9020u,196855u,9025u,0u,262394u,9022u,9024u,9029u,131320u,9024u,262205u,6u,9026u,9015u,262205u,
6u,9027u,8971u,327808u,6u,9028u,9026u,9027u,196670u,9023u,9028u,131321u,9025u,131320u,9029u,196670u,
9023u,548u,131321u,9025u,131320u,9025u,262205u,6u,9030u,9023u,196670u,9019u,9030u,262205u,57u,9033u,
8953u,196670u,9032u,9033u,327737u,57u,9034u,61u,9032u,196855u,9037u,0u,262394u,9034u,9036u,9041u,
131320u,9036u,262205u,6u,9038u,9019u,262205u,6u,9039u,8976u,327808u,6u,9040u,9038u,9039u,196670u,
9035u,9040u,131321u,9037u,131320u,9041u,196670u,9035u,548u,131321u,9037u,131320u,9037u,262205u,6u,9042u,
9035u,196670u,9031u,9042u,262205u,6u,9044u,8915u,262205u,6u,9045u,8921u,327812u,6u,9046u,9044u,
9045u,262205u,7u,9048u,435u,196670u,9047u,9048u,196670u,9049u,9046u,196670u,9050u,630u,458809u,12u,
9051u,17u,9047u,9049u,9050u,262205u,7u,9052u,9047u,196670u,435u,9052u,196670u,9043u,9051u,262205u,
6u,9054u,8915u,262205u,6u,9055u,8921u,327812u,6u,9056u,9054u,9055u,262205u,7u,9058u,435u,
196670u,9057u,9058u,196670u,9059u,9056u,196670u,9060u,630u,458809u,12u,9061u,17u,9057u,9059u,9060u,
262205u,7u,9062u,9057u,196670u,435u,9062u,196670u,9053u,9061u,262205u,6u,9064u,8915u,262205u,6u,
9065u,8921u,327812u,6u,9066u,9064u,9065u,262205u,7u,9068u,435u,196670u,9067u,9068u,196670u,9069u,
9066u,196670u,9070u,630u,458809u,12u,9071u,17u,9067u,9069u,9070u,262205u,7u,9072u,9067u,196670u,
435u,9072u,196670u,9063u,9071u,262205u,6u,9074u,8915u,262205u,6u,9075u,8921u,327812u,6u,9076u,
9074u,9075u,262205u,7u,9078u,435u,196670u,9077u,9078u,196670u,9079u,9076u,196670u,9080u,630u,458809u,
12u,9081u,17u,9077u,9079u,9080u,262205u,7u,9082u,9077u,196670u,435u,9082u,196670u,9073u,9081u,
262205u,6u,9084u,8921u,262205u,7u,9086u,435u,196670u,9085u,9086u,196670u,9087u,9084u,196670u,9088u,
630u,458809u,12u,9089u,17u,9085u,9087u,9088u,262205u,7u,9090u,9085u,196670u,435u,9090u,196670u,
9083u,9089u,196670u,9091u,548u,131321u,9092u,131320u,9092u,262390u,9094u,9095u,0u,131321u,9096u,131320u,
9096u,262205u,6u,9097u,9091u,262205u,6u,9098u,8915u,327857u,57u,9099u,9097u,9098u,262394u,9099u,
9093u,9094u,131320u,9093u,327745u,9u,9101u,8911u,548u,262205u,6u,9102u,9101u,262205u,6u,9103u,
9091u,262205u,6u,9104u,8918u,327812u,6u,9105u,9103u,9104u,327808u,6u,9106u,9102u,9105u,196670u,
9100u,9106u,196670u,9107u,548u,131321u,9108u,131320u,9108u,262390u,9110u,9111u,0u,131321u,9112u,131320u,
9112u,262205u,6u,9113u,9107u,262205u,6u,9114u,8921u,327857u,57u,9115u,9113u,9114u,262394u,9115u,
9109u,9110u,131320u,9109u,262205u,6u,9117u,9013u,262205u,6u,9118u,9107u,262205u,6u,9119u,8918u,
327812u,6u,9120u,9118u,9119u,327808u,6u,9121u,9117u,9120u,196670u,9116u,9121u,262205u,6u,9123u,
9015u,262205u,6u,9124u,9107u,262205u,6u,9125u,8921u,327812u,6u,9126u,9124u,9125u,327808u,6u,
9127u,9123u,9126u,196670u,9122u,9127u,196670u,9128u,630u,196670u,9129u,548u,131321u,9130u,131320u,9130u,
262390u,9132u,9133u,0u,131321u,9134u,131320u,9134u,262205u,6u,9135u,9129u,262205u,6u,9136u,8918u,
327857u,57u,9137u,9135u,9136u,262394u,9137u,9131u,9132u,131320u,9131u,262205u,6u,9138u,9116u,262205u,
6u,9139u,9129u,327808u,6u,9140u,9138u,9139u,393281u,584u,9141u,578u,548u,9140u,262205u,10u,
9142u,9141u,262205u,6u,9143u,9100u,262205u,6u,9144u,9129u,327808u,6u,9145u,9143u,9144u,393281u,
584u,9146u,578u,548u,9145u,262205u,10u,9147u,9146u,327813u,10u,9148u,9142u,9147u,262205u,10u,
9149u,9128u,327809u,10u,9150u,9149u,9148u,196670u,9128u,9150u,131321u,9133u,131320u,9133u,262205u,6u,
9151u,9129u,327808u,6u,9152u,9151u,551u,196670u,9129u,9152u,131321u,9130u,131320u,9132u,196670u,9153u,
548u,131321u,9154u,131320u,9154u,262390u,9156u,9157u,0u,131321u,9158u,131320u,9158u,262205u,6u,9159u,
9153u,262205u,6u,9160u,8921u,327857u,57u,9161u,9159u,9160u,262394u,9161u,9155u,9156u,131320u,9155u,
262205u,6u,9162u,9122u,262205u,6u,9163u,9153u,327808u,6u,9164u,9162u,9163u,393281u,584u,9165u,
578u,548u,9164u,262205u,10u,9166u,9165u,327745u,9u,9167u,9083u,548u,262205u,6u,9168u,9167u,
262205u,6u,9169u,9153u,327808u,6u,9170u,9168u,9169u,393281u,584u,9171u,578u,548u,9170u,262205u,
10u,9172u,9171u,327813u,10u,9173u,9166u,9172u,262205u,10u,9174u,9128u,327809u,10u,9175u,9174u,
9173u,196670u,9128u,9175u,131321u,9157u,131320u,9157u,262205u,6u,9176u,9153u,327808u,6u,9177u,9176u,
551u,196670u,9153u,9177u,131321u,9154u,131320u,9156u,262205u,6u,9179u,9019u,196670u,9178u,9179u,327737u,
57u,9180u,65u,9178u,196855u,9182u,0u,262394u,9180u,9181u,9182u,131320u,9181u,262205u,6u,9183u,
9019u,262205u,6u,9184u,9107u,327808u,6u,9185u,9183u,9184u,393281u,584u,9186u,578u,548u,9185u,
262205u,10u,9187u,9186u,262205u,10u,9188u,9128u,327809u,10u,9189u,9188u,9187u,196670u,9128u,9189u,
131321u,9182u,131320u,9182u,262205u,6u,9191u,9031u,196670u,9190u,9191u,327737u,57u,9192u,65u,9190u,
196855u,9194u,0u,262394u,9192u,9193u,9194u,131320u,9193u,262205u,6u,9195u,9031u,262205u,6u,9196u,
9107u,327808u,6u,9197u,9195u,9196u,393281u,584u,9198u,578u,548u,9197u,262205u,10u,9199u,9198u,
262205u,10u,9200u,9128u,327809u,10u,9201u,9200u,9199u,196670u,9128u,9201u,131321u,9194u,131320u,9194u,
262205u,10u,9204u,9128u,196670u,9203u,9204u,327737u,10u,9205u,419u,9203u,196670u,9202u,9205u,262205u,
6u,9207u,9013u,262205u,6u,9208u,8921u,262205u,6u,9209u,9107u,327808u,6u,9210u,9208u,9209u,
262205u,6u,9211u,8918u,327812u,6u,9212u,9210u,9211u,327808u,6u,9213u,9207u,9212u,196670u,9206u,
9213u,262205u,6u,9215u,9015u,262205u,6u,9216u,8921u,262205u,6u,9217u,9107u,327808u,6u,9218u,
9216u,9217u,262205u,6u,9219u,8921u,327812u,6u,9220u,9218u,9219u,327808u,6u,9221u,9215u,9220u,
196670u,9214u,9221u,196670u,9222u,630u,196670u,9223u,548u,131321u,9224u,131320u,9224u,262390u,9226u,9227u,
0u,131321u,9228u,131320u,9228u,262205u,6u,9229u,9223u,262205u,6u,9230u,8918u,327857u,57u,9231u,
9229u,9230u,262394u,9231u,9225u,9226u,131320u,9225u,262205u,6u,9232u,9206u,262205u,6u,9233u,9223u,
327808u,6u,9234u,9232u,9233u,393281u,584u,9235u,578u,548u,9234u,262205u,10u,9236u,9235u,262205u,
6u,9237u,9100u,262205u,6u,9238u,9223u,327808u,6u,9239u,9237u,9238u,393281u,584u,9240u,578u,
548u,9239u,262205u,10u,9241u,9240u,327813u,10u,9242u,9236u,9241u,262205u,10u,9243u,9222u,327809u,
10u,9244u,9243u,9242u,196670u,9222u,9244u,131321u,9227u,131320u,9227u,262205u,6u,9245u,9223u,327808u,
6u,9246u,9245u,551u,196670u,9223u,9246u,131321u,9224u,131320u,9226u,196670u,9247u,548u,131321u,9248u,
131320u,9248u,262390u,9250u,9251u,0u,131321u,9252u,131320u,9252u,262205u,6u,9253u,9247u,262205u,6u,
9254u,8921u,327857u,57u,9255u,9253u,9254u,262394u,9255u,9249u,9250u,131320u,9249u,262205u,6u,9256u,
9214u,262205u,6u,9257u,9247u,327808u,6u,9258u,9256u,9257u,393281u,584u,9259u,578u,548u,9258u,
262205u,10u,9260u,9259u,327745u,9u,9261u,9083u,548u,262205u,6u,9262u,9261u,262205u,6u,9263u,
9247u,327808u,6u,9264u,9262u,9263u,393281u,584u,9265u,578u,548u,9264u,262205u,10u,9266u,9265u,
327813u,10u,9267u,9260u,9266u,262205u,10u,9268u,9222u,327809u,10u,9269u,9268u,9267u,196670u,9222u,
9269u,131321u,9251u,131320u,9251u,262205u,6u,9270u,9247u,327808u,6u,9271u,9270u,551u,196670u,9247u,
9271u,131321u,9248u,131320u,9250u,262205u,6u,9273u,9019u,196670u,9272u,9273u,327737u,57u,9274u,65u,
9272u,196855u,9276u,0u,262394u,9274u,9275u,9276u,131320u,9275u,262205u,6u,9277u,9019u,262205u,6u,
9278u,8921u,262205u,6u,9279u,9107u,327808u,6u,9280u,9278u,9279u,327808u,6u,9281u,9277u,9280u,
393281u,584u,9282u,578u,548u,9281u,262205u,10u,9283u,9282u,262205u,10u,9284u,9222u,327809u,10u,
9285u,9284u,9283u,196670u,9222u,9285u,131321u,9276u,131320u,9276u,262205u,6u,9287u,9031u,196670u,9286u,
9287u,327737u,57u,9288u,65u,9286u,196855u,9290u,0u,262394u,9288u,9289u,9290u,131320u,9289u,262205u,
6u,9291u,9031u,262205u,6u,9292u,8921u,262205u,6u,9293u,9107u,327808u,6u,9294u,9292u,9293u,
327808u,6u,9295u,9291u,9294u,393281u,584u,9296u,578u,548u,9295u,262205u,10u,9297u,9296u,262205u,
10u,9298u,9222u,327809u,10u,9299u,9298u,9297u,196670u,9222u,9299u,131321u,9290u,131320u,9290u,262205u,
10u,9302u,9222u,196670u,9301u,9302u,327737u,10u,9303u,419u,9301u,196670u,9300u,9303u,262205u,6u,
9305u,9013u,262205u,6u,9306u,8921u,327812u,6u,9307u,2490u,9306u,262205u,6u,9308u,9107u,327808u,
6u,9309u,9307u,9308u,262205u,6u,9310u,8918u,327812u,6u,9311u,9309u,9310u,327808u,6u,9312u,
9305u,9311u,196670u,9304u,9312u,262205u,6u,9314u,9015u,262205u,6u,9315u,8921u,327812u,6u,9316u,
2490u,9315u,262205u,6u,9317u,9107u,327808u,6u,9318u,9316u,9317u,262205u,6u,9319u,8921u,327812u,
6u,9320u,9318u,9319u,327808u,6u,9321u,9314u,9320u,196670u,9313u,9321u,196670u,9322u,630u,196670u,
9323u,548u,131321u,9324u,131320u,9324u,262390u,9326u,9327u,0u,131321u,9328u,131320u,9328u,262205u,6u,
9329u,9323u,262205u,6u,9330u,8918u,327857u,57u,9331u,9329u,9330u,262394u,9331u,9325u,9326u,131320u,
9325u,262205u,6u,9332u,9304u,262205u,6u,9333u,9323u,327808u,6u,9334u,9332u,9333u,393281u,584u,
9335u,578u,548u,9334u,262205u,10u,9336u,9335u,262205u,6u,9337u,9100u,262205u,6u,9338u,9323u,
327808u,6u,9339u,9337u,9338u,393281u,584u,9340u,578u,548u,9339u,262205u,10u,9341u,9340u,327813u,
10u,9342u,9336u,9341u,262205u,10u,9343u,9322u,327809u,10u,9344u,9343u,9342u,196670u,9322u,9344u,
131321u,9327u,131320u,9327u,262205u,6u,9345u,9323u,327808u,6u,9346u,9345u,551u,196670u,9323u,9346u,
131321u,9324u,131320u,9326u,196670u,9347u,548u,131321u,9348u,131320u,9348u,262390u,9350u,9351u,0u,131321u,
9352u,131320u,9352u,262205u,6u,9353u,9347u,262205u,6u,9354u,8921u,327857u,57u,9355u,9353u,9354u,
262394u,9355u,9349u,9350u,131320u,9349u,262205u,6u,9356u,9313u,262205u,6u,9357u,9347u,327808u,6u,
9358u,9356u,9357u,393281u,584u,9359u,578u,548u,9358u,262205u,10u,9360u,9359u,262205u,10u,9361u,
9202u,327745u,9u,9362u,9083u,548u,262205u,6u,9363u,9362u,262205u,6u,9364u,9347u,327808u,6u,
9365u,9363u,9364u,393281u,584u,9366u,578u,548u,9365u,262205u,10u,9367u,9366u,327813u,10u,9368u,
9361u,9367u,327813u,10u,9369u,9360u,9368u,262205u,10u,9370u,9322u,327809u,10u,9371u,9370u,9369u,
196670u,9322u,9371u,131321u,9351u,131320u,9351u,262205u,6u,9372u,9347u,327808u,6u,9373u,9372u,551u,
196670u,9347u,9373u,131321u,9348u,131320u,9350u,262205u,6u,9375u,9019u,196670u,9374u,9375u,327737u,57u,
9376u,65u,9374u,196855u,9378u,0u,262394u,9376u,9377u,9378u,131320u,9377u,262205u,6u,9379u,9019u,
262205u,6u,9380u,8921u,327812u,6u,9381u,2490u,9380u,262205u,6u,9382u,9107u,327808u,6u,9383u,
9381u,9382u,327808u,6u,9384u,9379u,9383u,393281u,584u,9385u,578u,548u,9384u,262205u,10u,9386u,
9385u,262205u,10u,9387u,9322u,327809u,10u,9388u,9387u,9386u,196670u,9322u,9388u,131321u,9378u,131320u,
9378u,262205u,6u,9390u,9031u,196670u,9389u,9390u,327737u,57u,9391u,65u,9389u,196855u,9393u,0u,
262394u,9391u,9392u,9393u,131320u,9392u,262205u,6u,9394u,9031u,262205u,6u,9395u,8921u,327812u,6u,
9396u,2490u,9395u,262205u,6u,9397u,9107u,327808u,6u,9398u,9396u,9397u,327808u,6u,9399u,9394u,
9398u,393281u,584u,9400u,578u,548u,9399u,262205u,10u,9401u,9400u,262205u,10u,9402u,9322u,327809u,
10u,9403u,9402u,9401u,196670u,9322u,9403u,131321u,9393u,131320u,9393u,262205u,10u,9405u,9322u,393228u,
10u,9406u,1u,21u,9405u,196670u,9404u,9406u,262205u,10u,9408u,9300u,327811u,10u,9409u,701u,
9408u,262205u,10u,9410u,9404u,327813u,10u,9411u,9409u,9410u,262205u,10u,9412u,9300u,327745u,9u,
9413u,9083u,548u,262205u,6u,9414u,9413u,262205u,6u,9415u,9107u,327808u,6u,9416u,9414u,9415u,
393281u,584u,9417u,578u,548u,9416u,262205u,10u,9418u,9417u,327813u,10u,9419u,9412u,9418u,327809u,
10u,9420u,9411u,9419u,196670u,9407u,9420u,262205u,6u,9422u,9091u,262205u,6u,9423u,8921u,327812u,
6u,9424u,9422u,9423u,262205u,6u,9425u,9107u,327808u,6u,9426u,9424u,9425u,196670u,9421u,9426u,
327745u,9u,9427u,9053u,548u,262205u,6u,9428u,9427u,262205u,6u,9429u,9421u,327808u,6u,9430u,
9428u,9429u,262205u,10u,9431u,9202u,393281u,584u,9432u,578u,548u,9430u,196670u,9432u,9431u,327745u,
9u,9433u,9063u,548u,262205u,6u,9434u,9433u,262205u,6u,9435u,9421u,327808u,6u,9436u,9434u,
9435u,262205u,10u,9437u,9300u,393281u,584u,9438u,578u,548u,9436u,196670u,9438u,9437u,327745u,9u,
9439u,9073u,548u,262205u,6u,9440u,9439u,262205u,6u,9441u,9421u,327808u,6u,9442u,9440u,9441u,
262205u,10u,9443u,9404u,393281u,584u,9444u,578u,548u,9442u,196670u,9444u,9443u,327745u,9u,9445u,
9043u,548u,262205u,6u,9446u,9445u,262205u,6u,9447u,9421u,327808u,6u,9448u,9446u,9447u,262205u,
10u,9449u,9407u,393281u,584u,9450u,578u,548u,9448u,196670u,9450u,9449u,131321u,9111u,131320u,9111u,
262205u,6u,9451u,9107u,327808u,6u,9452u,9451u,551u,196670u,9107u,9452u,131321u,9108u,131320u,9110u,
196670u,9453u,548u,131321u,9454u,131320u,9454u,262390u,9456u,9457u,0u,131321u,9458u,131320u,9458u,262205u,
6u,9459u,9453u,262205u,6u,9460u,8921u,327857u,57u,9461u,9459u,9460u,262394u,9461u,9455u,9456u,
131320u,9455u,327745u,9u,9462u,9083u,548u,262205u,6u,9463u,9462u,262205u,6u,9464u,9453u,327808u,
6u,9465u,9463u,9464u,327745u,9u,9466u,9043u,548u,262205u,6u,9467u,9466u,262205u,6u,9468u,
9091u,262205u,6u,9469u,8921u,327812u,6u,9470u,9468u,9469u,262205u,6u,9471u,9453u,327808u,6u,
9472u,9470u,9471u,327808u,6u,9473u,9467u,9472u,393281u,584u,9474u,578u,548u,9473u,262205u,10u,
9475u,9474u,393281u,584u,9476u,578u,548u,9465u,196670u,9476u,9475u,131321u,9457u,131320u,9457u,262205u,
6u,9477u,9453u,327808u,6u,9478u,9477u,551u,196670u,9453u,9478u,131321u,9454u,131320u,9456u,131321u,
9095u,131320u,9095u,262205u,6u,9479u,9091u,327808u,6u,9480u,9479u,551u,196670u,9091u,9480u,131321u,
9092u,131320u,9094u,393281u,9u,9481u,434u,2750u,551u,262205u,6u,9482u,9481u,327745u,9u,9483u,
434u,2624u,262205u,6u,9484u,9483u,327851u,57u,9485u,9482u,9484u,196855u,9487u,0u,262394u,9485u,
9486u,9487u,131320u,9486u,327745u,19u,9489u,434u,2750u,262205u,12u,9490u,9489u,196670u,9488u,9490u,
327745u,9u,9492u,434u,2624u,262205u,6u,9493u,9492u,196670u,9491u,9493u,196670u,9494u,630u,458809u,
2u,9495u,24u,9488u,9491u,9494u,262205u,12u,9496u,9488u,327745u,19u,9497u,434u,2750u,196670u,
9497u,9496u,131321u,9487u,131320u,9487u,262205u,12u,9499u,433u,196670u,9498u,9499u,327745u,9u,9501u,
8911u,551u,262205u,6u,9502u,9501u,196670u,9500u,9502u,196670u,9503u,630u,458809u,2u,9504u,24u,
9498u,9500u,9503u,262205u,12u,9505u,9498u,196670u,433u,9505u,262205u,6u,9507u,8921u,262205u,7u,
9509u,435u,196670u,9508u,9509u,196670u,9510u,9507u,196670u,9511u,630u,458809u,12u,9512u,17u,9508u,
9510u,9511u,262205u,7u,9513u,9508u,196670u,435u,9513u,196670u,9506u,9512u,262205u,6u,9515u,8915u,
327810u,6u,9516u,9515u,551u,196670u,9514u,9516u,131321u,9517u,131320u,9517u,262390u,9519u,9520u,0u,
131321u,9521u,131320u,9521u,262205u,6u,9522u,9514u,327855u,57u,9523u,9522u,548u,262394u,9523u,9518u,
9519u,131320u,9518u,327745u,9u,9525u,8911u,548u,262205u,6u,9526u,9525u,262205u,6u,9527u,9514u,
262205u,6u,9528u,8918u,327812u,6u,9529u,9527u,9528u,327808u,6u,9530u,9526u,9529u,196670u,9524u,
9530u,262205u,6u,9532u,9514u,327853u,57u,9533u,9532u,548u,196855u,9536u,0u,262394u,9533u,9535u,
9544u,131320u,9535u,327745u,9u,9537u,9043u,548u,262205u,6u,9538u,9537u,262205u,6u,9539u,9514u,
327810u,6u,9540u,9539u,551u,262205u,6u,9541u,8921u,327812u,6u,9542u,9540u,9541u,327808u,6u,
9543u,9538u,9542u,196670u,9534u,9543u,131321u,9536u,131320u,9544u,196670u,9534u,548u,131321u,9536u,131320u,
9536u,262205u,6u,9545u,9534u,196670u,9531u,9545u,262205u,6u,9547u,8921u,262205u,7u,9549u,435u,
196670u,9548u,9549u,196670u,9550u,9547u,196670u,9551u,630u,458809u,12u,9552u,17u,9548u,9550u,9551u,
262205u,7u,9553u,9548u,196670u,435u,9553u,196670u,9546u,9552u,262205u,6u,9555u,8921u,262205u,7u,
9557u,435u,196670u,9556u,9557u,196670u,9558u,9555u,196670u,9559u,630u,458809u,12u,9560u,17u,9556u,
9558u,9559u,262205u,7u,9561u,9556u,196670u,435u,9561u,196670u,9554u,9560u,262205u,6u,9563u,8921u,
262205u,7u,9565u,435u,196670u,9564u,9565u,196670u,9566u,9563u,196670u,9567u,630u,458809u,12u,9568u,
17u,9564u,9566u,9567u,262205u,7u,9569u,9564u,196670u,435u,9569u,196670u,9562u,9568u,262205u,6u,
9571u,8921u,262205u,7u,9573u,435u,196670u,9572u,9573u,196670u,9574u,9571u,196670u,9575u,630u,458809u,
12u,9576u,17u,9572u,9574u,9575u,262205u,7u,9577u,9572u,196670u,435u,9577u,196670u,9570u,9576u,
196670u,9578u,548u,131321u,9579u,131320u,9579u,262390u,9581u,9582u,0u,131321u,9583u,131320u,9583u,262205u,
6u,9584u,9578u,262205u,6u,9585u,8921u,327857u,57u,9586u,9584u,9585u,262394u,9586u,9580u,9581u,
131320u,9580u,262205u,6u,9588u,9514u,262205u,6u,9589u,8921u,327812u,6u,9590u,9588u,9589u,262205u,
6u,9591u,9578u,327808u,6u,9592u,9590u,9591u,196670u,9587u,9592u,327745u,9u,9594u,9053u,548u,
262205u,6u,9595u,9594u,262205u,6u,9596u,9587u,327808u,6u,9597u,9595u,9596u,393281u,584u,9598u,
578u,548u,9597u,262205u,10u,9599u,9598u,196670u,9593u,9599u,327745u,9u,9601u,9063u,548u,262205u,
6u,9602u,9601u,262205u,6u,9603u,9587u,327808u,6u,9604u,9602u,9603u,393281u,584u,9605u,578u,
548u,9604u,262205u,10u,9606u,9605u,196670u,9600u,9606u,327745u,9u,9608u,9073u,548u,262205u,6u,
9609u,9608u,262205u,6u,9610u,9587u,327808u,6u,9611u,9609u,9610u,393281u,584u,9612u,578u,548u,
9611u,262205u,10u,9613u,9612u,196670u,9607u,9613u,262205u,6u,9615u,9531u,327851u,57u,9616u,9615u,
548u,196855u,9619u,0u,262394u,9616u,9618u,9625u,131320u,9618u,262205u,6u,9620u,9531u,262205u,6u,
9621u,9578u,327808u,6u,9622u,9620u,9621u,393281u,584u,9623u,578u,548u,9622u,262205u,10u,9624u,
9623u,196670u,9617u,9624u,131321u,9619u,131320u,9625u,196670u,9617u,630u,131321u,9619u,131320u,9619u,262205u,
10u,9626u,9617u,196670u,9614u,9626u,327745u,9u,9628u,8913u,548u,262205u,6u,9629u,9628u,262205u,
6u,9630u,9587u,327808u,6u,9631u,9629u,9630u,393281u,584u,9632u,578u,548u,9631u,262205u,10u,
9633u,9632u,327745u,9u,9634u,9506u,548u,262205u,6u,9635u,9634u,262205u,6u,9636u,9578u,327808u,
6u,9637u,9635u,9636u,393281u,584u,9638u,578u,548u,9637u,262205u,10u,9639u,9638u,327809u,10u,
9640u,9633u,9639u,196670u,9627u,9640u,262205u,10u,9642u,9627u,262205u,10u,9643u,9600u,327811u,10u,
9644u,701u,9643u,327813u,10u,9645u,9642u,9644u,196670u,9641u,9645u,262205u,10u,9647u,9627u,262205u,
10u,9648u,9614u,262205u,10u,9649u,9607u,327811u,10u,9650u,9648u,9649u,327813u,10u,9651u,9647u,
9650u,196670u,9646u,9651u,327745u,9u,9652u,9546u,548u,262205u,6u,9653u,9652u,262205u,6u,9654u,
9578u,327808u,6u,9655u,9653u,9654u,262205u,10u,9656u,9627u,262205u,10u,9657u,9600u,327813u,10u,
9658u,9656u,9657u,393281u,584u,9659u,578u,548u,9655u,262205u,10u,9660u,9659u,327809u,10u,9661u,
9660u,9658u,393281u,584u,9662u,578u,548u,9655u,196670u,9662u,9661u,327745u,9u,9663u,9570u,548u,
262205u,6u,9664u,9663u,262205u,6u,9665u,9578u,327808u,6u,9666u,9664u,9665u,262205u,10u,9667u,
9641u,262205u,10u,9668u,9607u,262205u,10u,9669u,9607u,327813u,10u,9670u,9668u,9669u,327811u,10u,
9671u,701u,9670u,327813u,10u,9672u,9667u,9671u,393281u,584u,9673u,578u,548u,9666u,196670u,9673u,
9672u,327745u,9u,9674u,9562u,548u,262205u,6u,9675u,9674u,262205u,6u,9676u,9578u,327808u,6u,
9677u,9675u,9676u,262205u,10u,9678u,9646u,262205u,10u,9679u,9600u,327813u,10u,9680u,9678u,9679u,
262205u,10u,9681u,9600u,327811u,10u,9682u,701u,9681u,327813u,10u,9683u,9680u,9682u,393281u,584u,
9684u,578u,548u,9677u,196670u,9684u,9683u,131321u,9582u,131320u,9582u,262205u,6u,9685u,9578u,327808u,
6u,9686u,9685u,551u,196670u,9578u,9686u,131321u,9579u,131320u,9581u,262205u,6u,9688u,8921u,262205u,
7u,9690u,435u,196670u,9689u,9690u,196670u,9691u,9688u,196670u,9692u,630u,458809u,12u,9693u,17u,
9689u,9691u,9692u,262205u,7u,9694u,9689u,196670u,435u,9694u,196670u,9687u,9693u,196670u,9695u,548u,
131321u,9696u,131320u,9696u,262390u,9698u,9699u,0u,131321u,9700u,131320u,9700u,262205u,6u,9701u,9695u,
262205u,6u,9702u,8921u,327857u,57u,9703u,9701u,9702u,262394u,9703u,9697u,9698u,131320u,9697u,262205u,
6u,9705u,8921u,327812u,6u,9706u,2490u,9705u,262205u,6u,9707u,9695u,327808u,6u,9708u,9706u,
9707u,196670u,9704u,9708u,196670u,9709u,548u,131321u,9710u,131320u,9710u,262390u,9712u,9713u,0u,131321u,
9714u,131320u,9714u,262205u,6u,9715u,9709u,262205u,6u,9716u,8918u,327857u,57u,9717u,9715u,9716u,
262394u,9717u,9711u,9712u,131320u,9711u,262205u,6u,9719u,9524u,262205u,6u,9720u,9709u,327808u,6u,
9721u,9719u,9720u,393281u,584u,9722u,578u,548u,9721u,262205u,10u,9723u,9722u,196670u,9718u,9723u,
393281u,9u,9724u,434u,2750u,548u,262205u,6u,9725u,9724u,262205u,6u,9726u,9704u,262205u,6u,
9727u,8918u,327812u,6u,9728u,9726u,9727u,262205u,6u,9729u,9709u,327808u,6u,9730u,9728u,9729u,
327808u,6u,9731u,9725u,9730u,327745u,9u,9732u,9570u,548u,262205u,6u,9733u,9732u,262205u,6u,
9734u,9695u,327808u,6u,9735u,9733u,9734u,393281u,584u,9736u,578u,548u,9735u,262205u,10u,9737u,
9736u,262205u,10u,9738u,9718u,327813u,10u,9739u,9737u,9738u,393281u,584u,9740u,578u,548u,9731u,
262205u,10u,9741u,9740u,327809u,10u,9742u,9741u,9739u,393281u,584u,9743u,578u,548u,9731u,196670u,
9743u,9742u,327745u,9u,9744u,433u,548u,262205u,6u,9745u,9744u,262205u,6u,9746u,9514u,262205u,
6u,9747u,8918u,327812u,6u,9748u,9746u,9747u,262205u,6u,9749u,9709u,327808u,6u,9750u,9748u,
9749u,327808u,6u,9751u,9745u,9750u,327745u,9u,9752u,9570u,548u,262205u,6u,9753u,9752u,262205u,
6u,9754u,9695u,327808u,6u,9755u,9753u,9754u,393281u,584u,9756u,578u,548u,9755u,262205u,10u,
9757u,9756u,262205u,6u,9758u,9013u,262205u,6u,9759u,9704u,262205u,6u,9760u,8918u,327812u,6u,
9761u,9759u,9760u,262205u,6u,9762u,9709u,327808u,6u,9763u,9761u,9762u,327808u,6u,9764u,9758u,
9763u,393281u,584u,9765u,578u,548u,9764u,262205u,10u,9766u,9765u,327813u,10u,9767u,9757u,9766u,
393281u,584u,9768u,578u,548u,9751u,262205u,10u,9769u,9768u,327809u,10u,9770u,9769u,9767u,393281u,
584u,9771u,578u,548u,9751u,196670u,9771u,9770u,131321u,9713u,131320u,9713u,262205u,6u,9772u,9709u,
327808u,6u,9773u,9772u,551u,196670u,9709u,9773u,131321u,9710u,131320u,9712u,196670u,9774u,548u,131321u,
9775u,131320u,9775u,262390u,9777u,9778u,0u,131321u,9779u,131320u,9779u,262205u,6u,9780u,9774u,262205u,
6u,9781u,8921u,327857u,57u,9782u,9780u,9781u,262394u,9782u,9776u,9777u,131320u,9776u,262205u,6u,
9784u,9531u,327851u,57u,9785u,9784u,548u,196855u,9788u,0u,262394u,9785u,9787u,9794u,131320u,9787u,
262205u,6u,9789u,9531u,262205u,6u,9790u,9774u,327808u,6u,9791u,9789u,9790u,393281u,584u,9792u,
578u,548u,9791u,262205u,10u,9793u,9792u,196670u,9786u,9793u,131321u,9788u,131320u,9794u,196670u,9786u,
630u,131321u,9788u,131320u,9788u,262205u,10u,9795u,9786u,196670u,9783u,9795u,327745u,9u,9797u,9053u,
548u,262205u,6u,9798u,9797u,262205u,6u,9799u,9514u,262205u,6u,9800u,8921u,327812u,6u,9801u,
9799u,9800u,262205u,6u,9802u,9774u,327808u,6u,9803u,9801u,9802u,327808u,6u,9804u,9798u,9803u,
393281u,584u,9805u,578u,548u,9804u,262205u,10u,9806u,9805u,196670u,9796u,9806u,393281u,9u,9807u,
434u,2750u,548u,262205u,6u,9808u,9807u,262205u,6u,9809u,8966u,262205u,6u,9810u,9704u,262205u,
6u,9811u,8921u,327812u,6u,9812u,9810u,9811u,327808u,6u,9813u,9809u,9812u,262205u,6u,9814u,
9774u,327808u,6u,9815u,9813u,9814u,327808u,6u,9816u,9808u,9815u,327745u,9u,9817u,9570u,548u,
262205u,6u,9818u,9817u,262205u,6u,9819u,9695u,327808u,6u,9820u,9818u,9819u,393281u,584u,9821u,
578u,548u,9820u,262205u,10u,9822u,9821u,262205u,10u,9823u,9796u,262205u,10u,9824u,9783u,327813u,
10u,9825u,9823u,9824u,327813u,10u,9826u,9822u,9825u,393281u,584u,9827u,578u,548u,9816u,262205u,
10u,9828u,9827u,327809u,10u,9829u,9828u,9826u,393281u,584u,9830u,578u,548u,9816u,196670u,9830u,
9829u,327745u,9u,9831u,9687u,548u,262205u,6u,9832u,9831u,262205u,6u,9833u,9774u,327808u,6u,
9834u,9832u,9833u,327745u,9u,9835u,9570u,548u,262205u,6u,9836u,9835u,262205u,6u,9837u,9695u,
327808u,6u,9838u,9836u,9837u,393281u,584u,9839u,578u,548u,9838u,262205u,10u,9840u,9839u,262205u,
6u,9841u,9015u,262205u,6u,9842u,9704u,262205u,6u,9843u,8921u,327812u,6u,9844u,9842u,9843u,
262205u,6u,9845u,9774u,327808u,6u,9846u,9844u,9845u,327808u,6u,9847u,9841u,9846u,393281u,584u,
9848u,578u,548u,9847u,262205u,10u,9849u,9848u,327813u,10u,9850u,9840u,9849u,393281u,584u,9851u,
578u,548u,9834u,262205u,10u,9852u,9851u,327809u,10u,9853u,9852u,9850u,393281u,584u,9854u,578u,
548u,9834u,196670u,9854u,9853u,131321u,9778u,131320u,9778u,262205u,6u,9855u,9774u,327808u,6u,9856u,
9855u,551u,196670u,9774u,9856u,131321u,9775u,131320u,9777u,262205u,57u,9858u,8953u,196670u,9857u,9858u,
327737u,57u,9859u,61u,9857u,196855u,9861u,0u,262394u,9859u,9860u,9861u,131320u,9860u,262205u,6u,
9863u,8966u,262205u,6u,9864u,8971u,327808u,6u,9865u,9863u,9864u,196670u,9862u,9865u,393281u,9u,
9866u,434u,2750u,548u,262205u,6u,9867u,9866u,262205u,6u,9868u,9862u,262205u,6u,9869u,9704u,
327808u,6u,9870u,9868u,9869u,327808u,6u,9871u,9867u,9870u,327745u,9u,9872u,9570u,548u,262205u,
6u,9873u,9872u,262205u,6u,9874u,9695u,327808u,6u,9875u,9873u,9874u,393281u,584u,9876u,578u,
548u,9875u,262205u,10u,9877u,9876u,393281u,584u,9878u,578u,548u,9871u,262205u,10u,9879u,9878u,
327809u,10u,9880u,9879u,9877u,393281u,584u,9881u,578u,548u,9871u,196670u,9881u,9880u,393281u,9u,
9882u,434u,2750u,548u,262205u,6u,9883u,9882u,262205u,6u,9884u,9862u,262205u,6u,9885u,8976u,
327808u,6u,9886u,9884u,9885u,262205u,6u,9887u,9704u,327808u,6u,9888u,9886u,9887u,327808u,6u,
9889u,9883u,9888u,327745u,9u,9890u,9570u,548u,262205u,6u,9891u,9890u,262205u,6u,9892u,9695u,
327808u,6u,9893u,9891u,9892u,393281u,584u,9894u,578u,548u,9893u,262205u,10u,9895u,9894u,393281u,
584u,9896u,578u,548u,9889u,262205u,10u,9897u,9896u,327809u,10u,9898u,9897u,9895u,393281u,584u,
9899u,578u,548u,9889u,196670u,9899u,9898u,131321u,9861u,131320u,9861u,131321u,9699u,131320u,9699u,262205u,
6u,9900u,9695u,327808u,6u,9901u,9900u,551u,196670u,9695u,9901u,131321u,9696u,131320u,9698u,196670u,
9902u,548u,131321u,9903u,131320u,9903u,262390u,9905u,9906u,0u,131321u,9907u,131320u,9907u,262205u,6u,
9908u,9902u,262205u,6u,9909u,8921u,327857u,57u,9910u,9908u,9909u,262394u,9910u,9904u,9905u,131320u,
9904u,262205u,6u,9912u,9531u,327851u,57u,9913u,9912u,548u,196855u,9916u,0u,262394u,9913u,9915u,
9922u,131320u,9915u,262205u,6u,9917u,9531u,262205u,6u,9918u,9902u,327808u,6u,9919u,9917u,9918u,
393281u,584u,9920u,578u,548u,9919u,262205u,10u,9921u,9920u,196670u,9914u,9921u,131321u,9916u,131320u,
9922u,196670u,9914u,630u,131321u,9916u,131320u,9916u,262205u,10u,9923u,9914u,196670u,9911u,9923u,327745u,
9u,9925u,9053u,548u,262205u,6u,9926u,9925u,262205u,6u,9927u,9514u,262205u,6u,9928u,8921u,
327812u,6u,9929u,9927u,9928u,262205u,6u,9930u,9902u,327808u,6u,9931u,9929u,9930u,327808u,6u,
9932u,9926u,9931u,393281u,584u,9933u,578u,548u,9932u,262205u,10u,9934u,9933u,196670u,9924u,9934u,
327745u,9u,9936u,9687u,548u,262205u,6u,9937u,9936u,262205u,6u,9938u,9902u,327808u,6u,9939u,
9937u,9938u,393281u,584u,9940u,578u,548u,9939u,262205u,10u,9941u,9940u,262205u,10u,9942u,9911u,
327813u,10u,9943u,9941u,9942u,196670u,9935u,9943u,327745u,9u,9944u,9546u,548u,262205u,6u,9945u,
9944u,262205u,6u,9946u,9902u,327808u,6u,9947u,9945u,9946u,327745u,9u,9948u,9687u,548u,262205u,
6u,9949u,9948u,262205u,6u,9950u,9902u,327808u,6u,9951u,9949u,9950u,393281u,584u,9952u,578u,
548u,9951u,262205u,10u,9953u,9952u,262205u,10u,9954u,9924u,327813u,10u,9955u,9953u,9954u,393281u,
584u,9956u,578u,548u,9947u,262205u,10u,9957u,9956u,327809u,10u,9958u,9957u,9955u,393281u,584u,
9959u,578u,548u,9947u,196670u,9959u,9958u,327745u,9u,9960u,9554u,548u,262205u,6u,9961u,9960u,
262205u,6u,9962u,9902u,327808u,6u,9963u,9961u,9962u,262205u,10u,9964u,9935u,262205u,10u,9965u,
9924u,327813u,10u,9966u,9964u,9965u,262205u,10u,9967u,9924u,327811u,10u,9968u,701u,9967u,327813u,
10u,9969u,9966u,9968u,393281u,584u,9970u,578u,548u,9963u,196670u,9970u,9969u,131321u,9906u,131320u,
9906u,262205u,6u,9971u,9902u,327808u,6u,9972u,9971u,551u,196670u,9902u,9972u,131321u,9903u,131320u,
9905u,196670u,9973u,548u,131321u,9974u,131320u,9974u,262390u,9976u,9977u,0u,131321u,9978u,131320u,9978u,
262205u,6u,9979u,9973u,262205u,6u,9980u,8921u,327857u,57u,9981u,9979u,9980u,262394u,9981u,9975u,
9976u,131320u,9975u,262205u,6u,9983u,9973u,196670u,9982u,9983u,262205u,6u,9985u,8921u,262205u,6u,
9986u,9973u,327808u,6u,9987u,9985u,9986u,196670u,9984u,9987u,196670u,9988u,548u,131321u,9989u,131320u,
9989u,262390u,9991u,9992u,0u,131321u,9993u,131320u,9993u,262205u,6u,9994u,9988u,262205u,6u,9995u,
8918u,327857u,57u,9996u,9994u,9995u,262394u,9996u,9990u,9991u,131320u,9990u,262205u,6u,9998u,9524u,
262205u,6u,9999u,9988u,327808u,6u,10000u,9998u,9999u,393281u,584u,10001u,578u,548u,10000u,262205u,
10u,10002u,10001u,196670u,9997u,10002u,393281u,9u,10003u,434u,2750u,548u,262205u,6u,10004u,10003u,
262205u,6u,10005u,9982u,262205u,6u,10006u,8918u,327812u,6u,10007u,10005u,10006u,262205u,6u,10008u,
9988u,327808u,6u,10009u,10007u,10008u,327808u,6u,10010u,10004u,10009u,327745u,9u,10011u,9554u,548u,
262205u,6u,10012u,10011u,262205u,6u,10013u,9973u,327808u,6u,10014u,10012u,10013u,393281u,584u,10015u,
578u,548u,10014u,262205u,10u,10016u,10015u,262205u,10u,10017u,9997u,327813u,10u,10018u,10016u,10017u,
393281u,584u,10019u,578u,548u,10010u,262205u,10u,10020u,10019u,327809u,10u,10021u,10020u,10018u,393281u,
584u,10022u,578u,548u,10010u,196670u,10022u,10021u,393281u,9u,10023u,434u,2750u,548u,262205u,6u,
10024u,10023u,262205u,6u,10025u,9984u,262205u,6u,10026u,8918u,327812u,6u,10027u,10025u,10026u,262205u,
6u,10028u,9988u,327808u,6u,10029u,10027u,10028u,327808u,6u,10030u,10024u,10029u,327745u,9u,10031u,
9562u,548u,262205u,6u,10032u,10031u,262205u,6u,10033u,9973u,327808u,6u,10034u,10032u,10033u,393281u,
584u,10035u,578u,548u,10034u,262205u,10u,10036u,10035u,262205u,10u,10037u,9997u,327813u,10u,10038u,
10036u,10037u,393281u,584u,10039u,578u,548u,10030u,262205u,10u,10040u,10039u,327809u,10u,10041u,10040u,
10038u,393281u,584u,10042u,578u,548u,10030u,196670u,10042u,10041u,327745u,9u,10043u,433u,548u,262205u,
6u,10044u,10043u,262205u,6u,10045u,9514u,262205u,6u,10046u,8918u,327812u,6u,10047u,10045u,10046u,
262205u,6u,10048u,9988u,327808u,6u,10049u,10047u,10048u,327808u,6u,10050u,10044u,10049u,327745u,9u,
10051u,9554u,548u,262205u,6u,10052u,10051u,262205u,6u,10053u,9973u,327808u,6u,10054u,10052u,10053u,
393281u,584u,10055u,578u,548u,10054u,262205u,10u,10056u,10055u,262205u,6u,10057u,9013u,262205u,6u,
10058u,9982u,262205u,6u,10059u,8918u,327812u,6u,10060u,10058u,10059u,262205u,6u,10061u,9988u,327808u,
6u,10062u,10060u,10061u,327808u,6u,10063u,10057u,10062u,393281u,584u,10064u,578u,548u,10063u,262205u,
10u,10065u,10064u,327813u,10u,10066u,10056u,10065u,327745u,9u,10067u,9562u,548u,262205u,6u,10068u,
10067u,262205u,6u,10069u,9973u,327808u,6u,10070u,10068u,10069u,393281u,584u,10071u,578u,548u,10070u,
262205u,10u,10072u,10071u,262205u,6u,10073u,9013u,262205u,6u,10074u,9984u,262205u,6u,10075u,8918u,
327812u,6u,10076u,10074u,10075u,262205u,6u,10077u,9988u,327808u,6u,10078u,10076u,10077u,327808u,6u,
10079u,10073u,10078u,393281u,584u,10080u,578u,548u,10079u,262205u,10u,10081u,10080u,327813u,10u,10082u,
10072u,10081u,327809u,10u,10083u,10066u,10082u,393281u,584u,10084u,578u,548u,10050u,262205u,10u,10085u,
10084u,327809u,10u,10086u,10085u,10083u,393281u,584u,10087u,578u,548u,10050u,196670u,10087u,10086u,131321u,
9992u,131320u,9992u,262205u,6u,10088u,9988u,327808u,6u,10089u,10088u,551u,196670u,9988u,10089u,131321u,
9989u,131320u,9991u,196670u,10090u,548u,131321u,10091u,131320u,10091u,262390u,10093u,10094u,0u,131321u,10095u,
131320u,10095u,262205u,6u,10096u,10090u,262205u,6u,10097u,8921u,327857u,57u,10098u,10096u,10097u,262394u,
10098u,10092u,10093u,131320u,10092u,262205u,6u,10100u,9531u,327851u,57u,10101u,10100u,548u,196855u,10104u,
0u,262394u,10101u,10103u,10110u,131320u,10103u,262205u,6u,10105u,9531u,262205u,6u,10106u,10090u,327808u,
6u,10107u,10105u,10106u,393281u,584u,10108u,578u,548u,10107u,262205u,10u,10109u,10108u,196670u,10102u,
10109u,131321u,10104u,131320u,10110u,196670u,10102u,630u,131321u,10104u,131320u,10104u,262205u,10u,10111u,10102u,
196670u,10099u,10111u,393281u,9u,10112u,434u,2750u,548u,262205u,6u,10113u,10112u,262205u,6u,10114u,
8966u,262205u,6u,10115u,9982u,262205u,6u,10116u,8921u,327812u,6u,10117u,10115u,10116u,327808u,6u,
10118u,10114u,10117u,262205u,6u,10119u,10090u,327808u,6u,10120u,10118u,10119u,327808u,6u,10121u,10113u,
10120u,327745u,9u,10122u,9554u,548u,262205u,6u,10123u,10122u,262205u,6u,10124u,9973u,327808u,6u,
10125u,10123u,10124u,393281u,584u,10126u,578u,548u,10125u,262205u,10u,10127u,10126u,262205u,10u,10128u,
10099u,327813u,10u,10129u,10127u,10128u,393281u,584u,10130u,578u,548u,10121u,262205u,10u,10131u,10130u,
327809u,10u,10132u,10131u,10129u,393281u,584u,10133u,578u,548u,10121u,196670u,10133u,10132u,393281u,9u,
10134u,434u,2750u,548u,262205u,6u,10135u,10134u,262205u,6u,10136u,8966u,262205u,6u,10137u,9984u,
262205u,6u,10138u,8921u,327812u,6u,10139u,10137u,10138u,327808u,6u,10140u,10136u,10139u,262205u,6u,
10141u,10090u,327808u,6u,10142u,10140u,10141u,327808u,6u,10143u,10135u,10142u,327745u,9u,10144u,9562u,
548u,262205u,6u,10145u,10144u,262205u,6u,10146u,9973u,327808u,6u,10147u,10145u,10146u,393281u,584u,
10148u,578u,548u,10147u,262205u,10u,10149u,10148u,262205u,10u,10150u,10099u,327813u,10u,10151u,10149u,
10150u,393281u,584u,10152u,578u,548u,10143u,262205u,10u,10153u,10152u,327809u,10u,10154u,10153u,10151u,
393281u,584u,10155u,578u,548u,10143u,196670u,10155u,10154u,327745u,9u,10156u,9546u,548u,262205u,6u,
10157u,10156u,262205u,6u,10158u,10090u,327808u,6u,10159u,10157u,10158u,327745u,9u,10160u,9554u,548u,
262205u,6u,10161u,10160u,262205u,6u,10162u,9973u,327808u,6u,10163u,10161u,10162u,393281u,584u,10164u,
578u,548u,10163u,262205u,10u,10165u,10164u,262205u,6u,10166u,9015u,262205u,6u,10167u,9982u,262205u,
6u,10168u,8921u,327812u,6u,10169u,10167u,10168u,262205u,6u,10170u,10090u,327808u,6u,10171u,10169u,
10170u,327808u,6u,10172u,10166u,10171u,393281u,584u,10173u,578u,548u,10172u,262205u,10u,10174u,10173u,
327813u,10u,10175u,10165u,10174u,327745u,9u,10176u,9562u,548u,262205u,6u,10177u,10176u,262205u,6u,
10178u,9973u,327808u,6u,10179u,10177u,10178u,393281u,584u,10180u,578u,548u,10179u,262205u,10u,10181u,
10180u,262205u,6u,10182u,9015u,262205u,6u,10183u,9984u,262205u,6u,10184u,8921u,327812u,6u,10185u,
10183u,10184u,262205u,6u,10186u,10090u,327808u,6u,10187u,10185u,10186u,327808u,6u,10188u,10182u,10187u,
393281u,584u,10189u,578u,548u,10188u,262205u,10u,10190u,10189u,327813u,10u,10191u,10181u,10190u,327809u,
10u,10192u,10175u,10191u,393281u,584u,10193u,578u,548u,10159u,262205u,10u,10194u,10193u,327809u,10u,
10195u,10194u,10192u,393281u,584u,10196u,578u,548u,10159u,196670u,10196u,10195u,131321u,10094u,131320u,10094u,
262205u,6u,10197u,10090u,327808u,6u,10198u,10197u,551u,196670u,10090u,10198u,131321u,10091u,131320u,10093u,
262205u,57u,10200u,8953u,196670u,10199u,10200u,327737u,57u,10201u,61u,10199u,196855u,10203u,0u,262394u,
10201u,10202u,10203u,131320u,10202u,262205u,6u,10205u,8966u,262205u,6u,10206u,8971u,327808u,6u,10207u,
10205u,10206u,196670u,10204u,10207u,393281u,9u,10208u,434u,2750u,548u,262205u,6u,10209u,10208u,262205u,
6u,10210u,10204u,262205u,6u,10211u,9982u,327808u,6u,10212u,10210u,10211u,327808u,6u,10213u,10209u,
10212u,327745u,9u,10214u,9554u,548u,262205u,6u,10215u,10214u,262205u,6u,10216u,9973u,327808u,6u,
10217u,10215u,10216u,393281u,584u,10218u,578u,548u,10217u,262205u,10u,10219u,10218u,393281u,584u,10220u,
578u,548u,10213u,262205u,10u,10221u,10220u,327809u,10u,10222u,10221u,10219u,393281u,584u,10223u,578u,
548u,10213u,196670u,10223u,10222u,393281u,9u,10224u,434u,2750u,548u,262205u,6u,10225u,10224u,262205u,
6u,10226u,10204u,262205u,6u,10227u,9984u,327808u,6u,10228u,10226u,10227u,327808u,6u,10229u,10225u,
10228u,327745u,9u,10230u,9562u,548u,262205u,6u,10231u,10230u,262205u,6u,10232u,9973u,327808u,6u,
10233u,10231u,10232u,393281u,584u,10234u,578u,548u,10233u,262205u,10u,10235u,10234u,393281u,584u,10236u,
578u,548u,10229u,262205u,10u,10237u,10236u,327809u,10u,10238u,10237u,10235u,393281u,584u,10239u,578u,
548u,10229u,196670u,10239u,10238u,393281u,9u,10240u,434u,2750u,548u,262205u,6u,10241u,10240u,262205u,
6u,10242u,10204u,262205u,6u,10243u,8976u,327808u,6u,10244u,10242u,10243u,262205u,6u,10245u,9982u,
327808u,6u,10246u,10244u,10245u,327808u,6u,10247u,10241u,10246u,327745u,9u,10248u,9554u,548u,262205u,
6u,10249u,10248u,262205u,6u,10250u,9973u,327808u,6u,10251u,10249u,10250u,393281u,584u,10252u,578u,
548u,10251u,262205u,10u,10253u,10252u,393281u,584u,10254u,578u,548u,10247u,262205u,10u,10255u,10254u,
327809u,10u,10256u,10255u,10253u,393281u,584u,10257u,578u,548u,10247u,196670u,10257u,10256u,393281u,9u,
10258u,434u,2750u,548u,262205u,6u,10259u,10258u,262205u,6u,10260u,10204u,262205u,6u,10261u,8976u,
327808u,6u,10262u,10260u,10261u,262205u,6u,10263u,9984u,327808u,6u,10264u,10262u,10263u,327808u,6u,
10265u,10259u,10264u,327745u,9u,10266u,9562u,548u,262205u,6u,10267u,10266u,262205u,6u,10268u,9973u,
327808u,6u,10269u,10267u,10268u,393281u,584u,10270u,578u,548u,10269u,262205u,10u,10271u,10270u,393281u,
584u,10272u,578u,548u,10265u,262205u,10u,10273u,10272u,327809u,10u,10274u,10273u,10271u,393281u,584u,
10275u,578u,548u,10265u,196670u,10275u,10274u,131321u,10203u,131320u,10203u,131321u,9977u,131320u,9977u,262205u,
6u,10276u,9973u,327808u,6u,10277u,10276u,551u,196670u,9973u,10277u,131321u,9974u,131320u,9976u,262205u,
12u,10279u,9506u,196670u,10278u,10279u,262205u,12u,10281u,9546u,196670u,10280u,10281u,393273u,2u,10282u,
29u,10278u,10280u,262205u,12u,10283u,10278u,196670u,9506u,10283u,262205u,12u,10284u,10280u,196670u,9546u,
10284u,131321u,9520u,131320u,9520u,262205u,6u,10285u,9514u,327810u,6u,10286u,10285u,551u,196670u,9514u,
10286u,131321u,9517u,131320u,9519u,131326u,6723u,131320u,7664u,262205u,12u,10289u,431u,196670u,10288u,10289u,
262205u,12u,10291u,432u,196670u,10290u,10291u,327745u,9u,10293u,434u,551u,262205u,6u,10294u,10293u,
196670u,10292u,10294u,327745u,9u,10296u,434u,2490u,262205u,6u,10297u,10296u,196670u,10295u,10297u,327745u,
9u,10299u,434u,2404u,262205u,6u,10300u,10299u,196670u,10298u,10300u,262205u,6u,10301u,10292u,327859u,
57u,10302u,10301u,548u,262205u,6u,10303u,10295u,327859u,57u,10304u,10303u,548u,327846u,57u,10305u,
10302u,10304u,262205u,6u,10306u,10298u,327859u,57u,10307u,10306u,548u,327846u,57u,10308u,10305u,10307u,
196855u,10310u,0u,262394u,10308u,10309u,10310u,131320u,10309u,131326u,6465u,131320u,10310u,327745u,9u,10312u,
10288u,551u,262205u,6u,10313u,10312u,262205u,6u,10314u,10292u,262205u,6u,10315u,10295u,327812u,6u,
10316u,10314u,10315u,327851u,57u,10317u,10313u,10316u,196855u,10319u,0u,262394u,10317u,10318u,10319u,131320u,
10318u,131326u,6465u,131320u,10319u,327745u,9u,10321u,10290u,551u,262205u,6u,10322u,10321u,262205u,6u,
10323u,10292u,262205u,6u,10324u,10298u,327812u,6u,10325u,10323u,10324u,327851u,57u,10326u,10322u,10325u,
196855u,10328u,0u,262394u,10326u,10327u,10328u,131320u,10327u,131326u,6465u,131320u,10328u,327745u,58u,10331u,
434u,3118u,262205u,57u,10332u,10331u,196670u,10330u,10332u,327745u,9u,10334u,434u,2510u,262205u,6u,
10335u,10334u,196670u,10333u,10335u,262205u,6u,10337u,10333u,196670u,10336u,10337u,327737u,57u,10338u,65u,
10336u,262312u,57u,10339u,10338u,196855u,10341u,0u,262394u,10339u,10340u,10341u,131320u,10340u,131326u,6465u,
131320u,10341u,262205u,6u,10344u,10298u,262205u,6u,10345u,10295u,327812u,6u,10346u,10344u,10345u,196670u,
10343u,10346u,262205u,6u,10348u,10298u,262205u,6u,10349u,10298u,327812u,6u,10350u,10348u,10349u,196670u,
10347u,10350u,262205u,57u,10353u,10330u,196670u,10352u,10353u,327737u,57u,10354u,61u,10352u,196855u,10357u,
0u,262394u,10354u,10356u,10359u,131320u,10356u,262205u,6u,10358u,10298u,196670u,10355u,10358u,131321u,10357u,
131320u,10359u,196670u,10355u,548u,131321u,10357u,131320u,10357u,262205u,6u,10360u,10355u,196670u,10351u,10360u,
262205u,57u,10363u,10330u,196670u,10362u,10363u,327737u,57u,10364u,61u,10362u,196855u,10367u,0u,262394u,
10364u,10366u,10369u,131320u,10366u,262205u,6u,10368u,10298u,196670u,10365u,10368u,131321u,10367u,131320u,10369u,
196670u,10365u,548u,131321u,10367u,131320u,10367u,262205u,6u,10370u,10365u,196670u,10361u,10370u,262205u,6u,
10372u,10343u,262205u,6u,10373u,10347u,327808u,6u,10374u,10372u,10373u,262205u,6u,10375u,10351u,327808u,
6u,10376u,10374u,10375u,262205u,6u,10377u,10361u,327808u,6u,10378u,10376u,10377u,196670u,10371u,10378u,
327745u,9u,10379u,434u,2624u,262205u,6u,10380u,10379u,262205u,6u,10381u,10371u,327857u,57u,10382u,
10380u,10381u,196855u,10384u,0u,262394u,10382u,10383u,10384u,131320u,10383u,131326u,6465u,131320u,10384u,262205u,
6u,10387u,10333u,196670u,10386u,10387u,262205u,6u,10389u,10333u,262205u,6u,10390u,10343u,327808u,6u,
10391u,10389u,10390u,196670u,10388u,10391u,262205u,57u,10394u,10330u,196670u,10393u,10394u,327737u,57u,10395u,
61u,10393u,196855u,10398u,0u,262394u,10395u,10397u,10402u,131320u,10397u,262205u,6u,10399u,10388u,262205u,
6u,10400u,10347u,327808u,6u,10401u,10399u,10400u,196670u,10396u,10401u,131321u,10398u,131320u,10402u,196670u,
10396u,548u,131321u,10398u,131320u,10398u,262205u,6u,10403u,10396u,196670u,10392u,10403u,262205u,57u,10406u,
10330u,196670u,10405u,10406u,327737u,57u,10407u,61u,10405u,196855u,10410u,0u,262394u,10407u,10409u,10414u,
131320u,10409u,262205u,6u,10411u,10392u,262205u,6u,10412u,10351u,327808u,6u,10413u,10411u,10412u,196670u,
10408u,10413u,131321u,10410u,131320u,10414u,196670u,10408u,548u,131321u,10410u,131320u,10410u,262205u,6u,10415u,
10408u,196670u,10404u,10415u,262205u,6u,10417u,10292u,262205u,6u,10418u,10298u,327812u,6u,10419u,10417u,
10418u,262205u,7u,10421u,435u,196670u,10420u,10421u,196670u,10422u,10419u,196670u,10423u,630u,458809u,12u,
10424u,17u,10420u,10422u,10423u,262205u,7u,10425u,10420u,196670u,435u,10425u,196670u,10416u,10424u,262205u,
6u,10427u,10298u,262205u,7u,10429u,435u,196670u,10428u,10429u,196670u,10430u,10427u,196670u,10431u,630u,
458809u,12u,10432u,17u,10428u,10430u,10431u,262205u,7u,10433u,10428u,196670u,435u,10433u,196670u,10426u,
10432u,196670u,10434u,548u,131321u,10435u,131320u,10435u,262390u,10437u,10438u,0u,131321u,10439u,131320u,10439u,
262205u,6u,10440u,10434u,262205u,6u,10441u,10292u,327857u,57u,10442u,10440u,10441u,262394u,10442u,10436u,
10437u,131320u,10436u,327745u,9u,10444u,10288u,548u,262205u,6u,10445u,10444u,262205u,6u,10446u,10434u,
262205u,6u,10447u,10295u,327812u,6u,10448u,10446u,10447u,327808u,6u,10449u,10445u,10448u,196670u,10443u,
10449u,327745u,9u,10451u,10416u,548u,262205u,6u,10452u,10451u,262205u,6u,10453u,10434u,262205u,6u,
10454u,10298u,327812u,6u,10455u,10453u,10454u,327808u,6u,10456u,10452u,10455u,196670u,10450u,10456u,196670u,
10457u,548u,131321u,10458u,131320u,10458u,262390u,10460u,10461u,0u,131321u,10462u,131320u,10462u,262205u,6u,
10463u,10457u,262205u,6u,10464u,10298u,327857u,57u,10465u,10463u,10464u,262394u,10465u,10459u,10460u,131320u,
10459u,196670u,10466u,630u,262205u,6u,10468u,10386u,262205u,6u,10469u,10457u,262205u,6u,10470u,10295u,
327812u,6u,10471u,10469u,10470u,327808u,6u,10472u,10468u,10471u,196670u,10467u,10472u,262205u,6u,10474u,
10388u,262205u,6u,10475u,10457u,262205u,6u,10476u,10298u,327812u,6u,10477u,10475u,10476u,327808u,6u,
10478u,10474u,10477u,196670u,10473u,10478u,196670u,10479u,548u,131321u,10480u,131320u,10480u,262390u,10482u,10483u,
0u,131321u,10484u,131320u,10484u,262205u,6u,10485u,10479u,262205u,6u,10486u,10295u,327857u,57u,10487u,
10485u,10486u,262394u,10487u,10481u,10482u,131320u,10481u,262205u,6u,10488u,10467u,262205u,6u,10489u,10479u,
327808u,6u,10490u,10488u,10489u,393281u,584u,10491u,578u,548u,10490u,262205u,10u,10492u,10491u,262205u,
6u,10493u,10443u,262205u,6u,10494u,10479u,327808u,6u,10495u,10493u,10494u,393281u,584u,10496u,578u,
548u,10495u,262205u,10u,10497u,10496u,327813u,10u,10498u,10492u,10497u,262205u,10u,10499u,10466u,327809u,
10u,10500u,10499u,10498u,196670u,10466u,10500u,131321u,10483u,131320u,10483u,262205u,6u,10501u,10479u,327808u,
6u,10502u,10501u,551u,196670u,10479u,10502u,131321u,10480u,131320u,10482u,196670u,10503u,548u,131321u,10504u,
131320u,10504u,262390u,10506u,10507u,0u,131321u,10508u,131320u,10508u,262205u,6u,10509u,10503u,262205u,6u,
10510u,10298u,327857u,57u,10511u,10509u,10510u,262394u,10511u,10505u,10506u,131320u,10505u,262205u,6u,10512u,
10473u,262205u,6u,10513u,10503u,327808u,6u,10514u,10512u,10513u,393281u,584u,10515u,578u,548u,10514u,
262205u,10u,10516u,10515u,327745u,9u,10517u,10426u,548u,262205u,6u,10518u,10517u,262205u,6u,10519u,
10503u,327808u,6u,10520u,10518u,10519u,393281u,584u,10521u,578u,548u,10520u,262205u,10u,10522u,10521u,
327813u,10u,10523u,10516u,10522u,262205u,10u,10524u,10466u,327809u,10u,10525u,10524u,10523u,196670u,10466u,
10525u,131321u,10507u,131320u,10507u,262205u,6u,10526u,10503u,327808u,6u,10527u,10526u,551u,196670u,10503u,
10527u,131321u,10504u,131320u,10506u,262205u,6u,10529u,10392u,196670u,10528u,10529u,327737u,57u,10530u,65u,
10528u,196855u,10532u,0u,262394u,10530u,10531u,10532u,131320u,10531u,262205u,6u,10533u,10392u,262205u,6u,
10534u,10457u,327808u,6u,10535u,10533u,10534u,393281u,584u,10536u,578u,548u,10535u,262205u,10u,10537u,
10536u,262205u,10u,10538u,10466u,327809u,10u,10539u,10538u,10537u,196670u,10466u,10539u,131321u,10532u,131320u,
10532u,262205u,6u,10541u,10404u,196670u,10540u,10541u,327737u,57u,10542u,65u,10540u,196855u,10544u,0u,
262394u,10542u,10543u,10544u,131320u,10543u,262205u,6u,10545u,10404u,262205u,6u,10546u,10457u,327808u,6u,
10547u,10545u,10546u,393281u,584u,10548u,578u,548u,10547u,262205u,10u,10549u,10548u,262205u,10u,10550u,
10466u,327809u,10u,10551u,10550u,10549u,196670u,10466u,10551u,131321u,10544u,131320u,10544u,262205u,6u,10552u,
10450u,262205u,6u,10553u,10457u,327808u,6u,10554u,10552u,10553u,262205u,10u,10555u,10466u,393228u,10u,
10556u,1u,21u,10555u,393281u,584u,10557u,578u,548u,10554u,196670u,10557u,10556u,131321u,10461u,131320u,
10461u,262205u,6u,10558u,10457u,327808u,6u,10559u,10558u,551u,196670u,10457u,10559u,131321u,10458u,131320u,
10460u,196670u,10560u,548u,131321u,10561u,131320u,10561u,262390u,10563u,10564u,0u,131321u,10565u,131320u,10565u,
262205u,6u,10566u,10560u,262205u,6u,10567u,10298u,327857u,57u,10568u,10566u,10567u,262394u,10568u,10562u,
10563u,131320u,10562u,327745u,9u,10569u,10426u,548u,262205u,6u,10570u,10569u,262205u,6u,10571u,10560u,
327808u,6u,10572u,10570u,10571u,262205u,6u,10573u,10450u,262205u,6u,10574u,10560u,327808u,6u,10575u,
10573u,10574u,393281u,584u,10576u,578u,548u,10575u,262205u,10u,10577u,10576u,393281u,584u,10578u,578u,
548u,10572u,196670u,10578u,10577u,131321u,10564u,131320u,10564u,262205u,6u,10579u,10560u,327808u,6u,10580u,
10579u,551u,196670u,10560u,10580u,131321u,10561u,131320u,10563u,131321u,10438u,131320u,10438u,262205u,6u,10581u,
10434u,327808u,6u,10582u,10581u,551u,196670u,10434u,10582u,131321u,10435u,131320u,10437u,393281u,9u,10583u,
434u,2750u,551u,262205u,6u,10584u,10583u,327745u,9u,10585u,434u,2624u,262205u,6u,10586u,10585u,
327851u,57u,10587u,10584u,10586u,196855u,10589u,0u,262394u,10587u,10588u,10589u,131320u,10588u,327745u,19u,
10591u,434u,2750u,262205u,12u,10592u,10591u,196670u,10590u,10592u,327745u,9u,10594u,434u,2624u,262205u,
6u,10595u,10594u,196670u,10593u,10595u,196670u,10596u,630u,458809u,2u,10597u,24u,10590u,10593u,10596u,
262205u,12u,10598u,10590u,327745u,19u,10599u,434u,2750u,196670u,10599u,10598u,131321u,10589u,131320u,10589u,
262205u,12u,10601u,433u,196670u,10600u,10601u,327745u,9u,10603u,10288u,551u,262205u,6u,10604u,10603u,
196670u,10602u,10604u,196670u,10605u,630u,458809u,2u,10606u,24u,10600u,10602u,10605u,262205u,12u,10607u,
10600u,196670u,433u,10607u,262205u,6u,10609u,10298u,262205u,7u,10611u,435u,196670u,10610u,10611u,196670u,
10612u,10609u,196670u,10613u,630u,458809u,12u,10614u,17u,10610u,10612u,10613u,262205u,7u,10615u,10610u,
196670u,435u,10615u,196670u,10608u,10614u,262205u,6u,10617u,10292u,327810u,6u,10618u,10617u,551u,196670u,
10616u,10618u,131321u,10619u,131320u,10619u,262390u,10621u,10622u,0u,131321u,10623u,131320u,10623u,262205u,6u,
10624u,10616u,327855u,57u,10625u,10624u,548u,262394u,10625u,10620u,10621u,131320u,10620u,327745u,9u,10627u,
10288u,548u,262205u,6u,10628u,10627u,262205u,6u,10629u,10616u,262205u,6u,10630u,10295u,327812u,6u,
10631u,10629u,10630u,327808u,6u,10632u,10628u,10631u,196670u,10626u,10632u,262205u,6u,10634u,10616u,327853u,
57u,10635u,10634u,548u,196855u,10638u,0u,262394u,10635u,10637u,10646u,131320u,10637u,327745u,9u,10639u,
10416u,548u,262205u,6u,10640u,10639u,262205u,6u,10641u,10616u,327810u,6u,10642u,10641u,551u,262205u,
6u,10643u,10298u,327812u,6u,10644u,10642u,10643u,327808u,6u,10645u,10640u,10644u,196670u,10636u,10645u,
131321u,10638u,131320u,10646u,196670u,10636u,548u,131321u,10638u,131320u,10638u,262205u,6u,10647u,10636u,196670u,
10633u,10647u,262205u,6u,10649u,10298u,262205u,7u,10651u,435u,196670u,10650u,10651u,196670u,10652u,10649u,
196670u,10653u,630u,458809u,12u,10654u,17u,10650u,10652u,10653u,262205u,7u,10655u,10650u,196670u,435u,
10655u,196670u,10648u,10654u,196670u,10656u,548u,131321u,10657u,131320u,10657u,262390u,10659u,10660u,0u,131321u,
10661u,131320u,10661u,262205u,6u,10662u,10656u,262205u,6u,10663u,10298u,327857u,57u,10664u,10662u,10663u,
262394u,10664u,10658u,10659u,131320u,10658u,327745u,9u,10666u,10416u,548u,262205u,6u,10667u,10666u,262205u,
6u,10668u,10616u,262205u,6u,10669u,10298u,327812u,6u,10670u,10668u,10669u,262205u,6u,10671u,10656u,
327808u,6u,10672u,10670u,10671u,327808u,6u,10673u,10667u,10672u,393281u,584u,10674u,578u,548u,10673u,
262205u,10u,10675u,10674u,196670u,10665u,10675u,327745u,9u,10677u,10290u,548u,262205u,6u,10678u,10677u,
262205u,6u,10679u,10616u,262205u,6u,10680u,10298u,327812u,6u,10681u,10679u,10680u,262205u,6u,10682u,
10656u,327808u,6u,10683u,10681u,10682u,327808u,6u,10684u,10678u,10683u,393281u,584u,10685u,578u,548u,
10684u,262205u,10u,10686u,10685u,327745u,9u,10687u,10608u,548u,262205u,6u,10688u,10687u,262205u,6u,
10689u,10656u,327808u,6u,10690u,10688u,10689u,393281u,584u,10691u,578u,548u,10690u,262205u,10u,10692u,
10691u,327809u,10u,10693u,10686u,10692u,196670u,10676u,10693u,262205u,10u,10695u,10676u,262205u,10u,10696u,
10665u,262205u,10u,10697u,10665u,327813u,10u,10698u,10696u,10697u,327811u,10u,10699u,701u,10698u,327813u,
10u,10700u,10695u,10699u,196670u,10694u,10700u,262205u,6u,10702u,10386u,262205u,6u,10703u,10656u,262205u,
6u,10704u,10295u,327812u,6u,10705u,10703u,10704u,327808u,6u,10706u,10702u,10705u,196670u,10701u,10706u,
262205u,6u,10708u,10388u,262205u,6u,10709u,10656u,262205u,6u,10710u,10298u,327812u,6u,10711u,10709u,
10710u,327808u,6u,10712u,10708u,10711u,196670u,10707u,10712u,196670u,10713u,548u,131321u,10714u,131320u,10714u,
262390u,10716u,10717u,0u,131321u,10718u,131320u,10718u,262205u,6u,10719u,10713u,262205u,6u,10720u,10295u,
327857u,57u,10721u,10719u,10720u,262394u,10721u,10715u,10716u,131320u,10715u,393281u,9u,10722u,434u,2750u,
548u,262205u,6u,10723u,10722u,262205u,6u,10724u,10656u,262205u,6u,10725u,10295u,327812u,6u,10726u,
10724u,10725u,262205u,6u,10727u,10713u,327808u,6u,10728u,10726u,10727u,327808u,6u,10729u,10723u,10728u,
262205u,10u,10730u,10694u,262205u,6u,10731u,10626u,262205u,6u,10732u,10713u,327808u,6u,10733u,10731u,
10732u,393281u,584u,10734u,578u,548u,10733u,262205u,10u,10735u,10734u,327813u,10u,10736u,10730u,10735u,
393281u,584u,10737u,578u,548u,10729u,262205u,10u,10738u,10737u,327809u,10u,10739u,10738u,10736u,393281u,
584u,10740u,578u,548u,10729u,196670u,10740u,10739u,327745u,9u,10741u,433u,548u,262205u,6u,10742u,
10741u,262205u,6u,10743u,10616u,262205u,6u,10744u,10295u,327812u,6u,10745u,10743u,10744u,262205u,6u,
10746u,10713u,327808u,6u,10747u,10745u,10746u,327808u,6u,10748u,10742u,10747u,262205u,10u,10749u,10694u,
262205u,6u,10750u,10701u,262205u,6u,10751u,10713u,327808u,6u,10752u,10750u,10751u,393281u,584u,10753u,
578u,548u,10752u,262205u,10u,10754u,10753u,327813u,10u,10755u,10749u,10754u,393281u,584u,10756u,578u,
548u,10748u,262205u,10u,10757u,10756u,327809u,10u,10758u,10757u,10755u,393281u,584u,10759u,578u,548u,
10748u,196670u,10759u,10758u,131321u,10717u,131320u,10717u,262205u,6u,10760u,10713u,327808u,6u,10761u,10760u,
551u,196670u,10713u,10761u,131321u,10714u,131320u,10716u,196670u,10762u,548u,131321u,10763u,131320u,10763u,262390u,
10765u,10766u,0u,131321u,10767u,131320u,10767u,262205u,6u,10768u,10762u,262205u,6u,10769u,10298u,327857u,
57u,10770u,10768u,10769u,262394u,10770u,10764u,10765u,131320u,10764u,262205u,6u,10772u,10633u,327851u,57u,
10773u,10772u,548u,196855u,10776u,0u,262394u,10773u,10775u,10782u,131320u,10775u,262205u,6u,10777u,10633u,
262205u,6u,10778u,10762u,327808u,6u,10779u,10777u,10778u,393281u,584u,10780u,578u,548u,10779u,262205u,
10u,10781u,10780u,196670u,10774u,10781u,131321u,10776u,131320u,10782u,196670u,10774u,630u,131321u,10776u,131320u,
10776u,262205u,10u,10783u,10774u,196670u,10771u,10783u,393281u,9u,10784u,434u,2750u,548u,262205u,6u,
10785u,10784u,262205u,6u,10786u,10343u,262205u,6u,10787u,10656u,262205u,6u,10788u,10298u,327812u,6u,
10789u,10787u,10788u,327808u,6u,10790u,10786u,10789u,262205u,6u,10791u,10762u,327808u,6u,10792u,10790u,
10791u,327808u,6u,10793u,10785u,10792u,262205u,10u,10794u,10694u,262205u,10u,10795u,10771u,327813u,10u,
10796u,10794u,10795u,393281u,584u,10797u,578u,548u,10793u,262205u,10u,10798u,10797u,327809u,10u,10799u,
10798u,10796u,393281u,584u,10800u,578u,548u,10793u,196670u,10800u,10799u,327745u,9u,10801u,10648u,548u,
262205u,6u,10802u,10801u,262205u,6u,10803u,10762u,327808u,6u,10804u,10802u,10803u,262205u,10u,10805u,
10694u,262205u,6u,10806u,10707u,262205u,6u,10807u,10762u,327808u,6u,10808u,10806u,10807u,393281u,584u,
10809u,578u,548u,10808u,262205u,10u,10810u,10809u,327813u,10u,10811u,10805u,10810u,393281u,584u,10812u,
578u,548u,10804u,262205u,10u,10813u,10812u,327809u,10u,10814u,10813u,10811u,393281u,584u,10815u,578u,
548u,10804u,196670u,10815u,10814u,131321u,10766u,131320u,10766u,262205u,6u,10816u,10762u,327808u,6u,10817u,
10816u,551u,196670u,10762u,10817u,131321u,10763u,131320u,10765u,262205u,57u,10819u,10330u,196670u,10818u,10819u,
327737u,57u,10820u,61u,10818u,196855u,10822u,0u,262394u,10820u,10821u,10822u,131320u,10821u,393281u,9u,
10823u,434u,2750u,548u,262205u,6u,10824u,10823u,262205u,6u,10825u,10343u,262205u,6u,10826u,10347u,
327808u,6u,10827u,10825u,10826u,262205u,6u,10828u,10656u,327808u,6u,10829u,10827u,10828u,327808u,6u,
10830u,10824u,10829u,262205u,10u,10831u,10694u,393281u,584u,10832u,578u,548u,10830u,262205u,10u,10833u,
10832u,327809u,10u,10834u,10833u,10831u,393281u,584u,10835u,578u,548u,10830u,196670u,10835u,10834u,393281u,
9u,10836u,434u,2750u,548u,262205u,6u,10837u,10836u,262205u,6u,10838u,10343u,262205u,6u,10839u,
10347u,327808u,6u,10840u,10838u,10839u,262205u,6u,10841u,10351u,327808u,6u,10842u,10840u,10841u,262205u,
6u,10843u,10656u,327808u,6u,10844u,10842u,10843u,327808u,6u,10845u,10837u,10844u,262205u,10u,10846u,
10694u,393281u,584u,10847u,578u,548u,10845u,262205u,10u,10848u,10847u,327809u,10u,10849u,10848u,10846u,
393281u,584u,10850u,578u,548u,10845u,196670u,10850u,10849u,131321u,10822u,131320u,10822u,131321u,10660u,131320u,
10660u,262205u,6u,10851u,10656u,327808u,6u,10852u,10851u,551u,196670u,10656u,10852u,131321u,10657u,131320u,
10659u,262205u,12u,10854u,10608u,196670u,10853u,10854u,262205u,12u,10856u,10648u,196670u,10855u,10856u,393273u,
2u,10857u,29u,10853u,10855u,262205u,12u,10858u,10853u,196670u,10608u,10858u,262205u,12u,10859u,10855u,
196670u,10648u,10859u,131321u,10622u,131320u,10622u,262205u,6u,10860u,10616u,327810u,6u,10861u,10860u,551u,
196670u,10616u,10861u,131321u,10619u,131320u,10621u,131326u,6723u,131320u,7666u,65791u,65592u,327734u,2u,452u,
0u,440u,196663u,9u,441u,196663u,9u,442u,196663u,9u,443u,196663u,9u,444u,196663u,9u,
445u,196663u,9u,446u,196663u,9u,447u,196663u,9u,448u,196663u,439u,449u,196663u,58u,450u,
196663u,9u,451u,131320u,453u,262203u,8u,10874u,7u,262203u,19u,10879u,7u,262203u,19u,10887u,
7u,262203u,9u,10891u,7u,262203u,19u,10907u,7u,262203u,422u,10915u,7u,262203u,9u,10931u,
7u,262203u,58u,10934u,7u,262203u,19u,10937u,7u,262203u,19u,10939u,7u,262203u,19u,10941u,
7u,262203u,422u,10943u,7u,262203u,8u,10945u,7u,262203u,19u,10952u,7u,262203u,19u,10954u,
7u,262203u,422u,10956u,7u,262203u,8u,10958u,7u,262205u,6u,10866u,549u,327851u,57u,10867u,
10866u,548u,262205u,6u,10868u,547u,327851u,57u,10869u,10868u,548u,327846u,57u,10870u,10867u,10869u,
196855u,10872u,0u,262394u,10870u,10871u,10872u,131320u,10871u,65789u,131320u,10872u,262205u,6u,10875u,446u,
327745u,9u,10876u,449u,2510u,262205u,6u,10877u,10876u,393296u,7u,10878u,10875u,548u,10877u,196670u,
10874u,10878u,262205u,6u,10880u,441u,327745u,9u,10881u,449u,548u,262205u,6u,10882u,10881u,327745u,
9u,10883u,449u,551u,262205u,6u,10884u,10883u,327812u,6u,10885u,10882u,10884u,327760u,12u,10886u,
10880u,10885u,196670u,10879u,10886u,262205u,6u,10888u,444u,262205u,6u,10889u,443u,327851u,57u,10890u,
10889u,548u,196855u,10893u,0u,262394u,10890u,10892u,10899u,131320u,10892u,327745u,9u,10894u,449u,548u,
262205u,6u,10895u,10894u,327745u,9u,10896u,449u,551u,262205u,6u,10897u,10896u,327812u,6u,10898u,
10895u,10897u,196670u,10891u,10898u,131321u,10893u,131320u,10899u,327745u,9u,10900u,449u,548u,262205u,6u,
10901u,10900u,327745u,9u,10902u,449u,2490u,262205u,6u,10903u,10902u,327812u,6u,10904u,10901u,10903u,
196670u,10891u,10904u,131321u,10893u,131320u,10893u,262205u,6u,10905u,10891u,327760u,12u,10906u,10888u,10905u,
196670u,10887u,10906u,262205u,6u,10908u,443u,327745u,9u,10909u,449u,548u,262205u,6u,10910u,10909u,
327745u,9u,10911u,449u,2490u,262205u,6u,10912u,10911u,327812u,6u,10913u,10910u,10912u,327760u,12u,
10914u,10908u,10913u,196670u,10907u,10914u,262205u,6u,10916u,448u,327745u,9u,10917u,449u,548u,262205u,
6u,10918u,10917u,327745u,9u,10919u,449u,551u,262205u,6u,10920u,10919u,327745u,9u,10921u,449u,
2490u,262205u,6u,10922u,10921u,262205u,57u,10923u,450u,262205u,6u,10924u,442u,262205u,6u,10925u,
451u,262205u,6u,10926u,445u,262205u,6u,10927u,451u,327760u,12u,10928u,10926u,10927u,720976u,421u,
10929u,10916u,10918u,10920u,10922u,10923u,10924u,10925u,10928u,196670u,10915u,10929u,262205u,6u,10930u,447u,
262205u,6u,10932u,443u,196670u,10931u,10932u,327737u,57u,10933u,65u,10931u,196855u,10936u,0u,262394u,
10933u,10935u,10951u,131320u,10935u,262205u,12u,10938u,10879u,196670u,10937u,10938u,262205u,12u,10940u,10907u,
196670u,10939u,10940u,262205u,12u,10942u,10887u,196670u,10941u,10942u,262205u,421u,10944u,10915u,196670u,10943u,
10944u,262205u,7u,10946u,10874u,196670u,10945u,10946u,589881u,57u,10947u,436u,10937u,10939u,10941u,10943u,
10945u,262205u,12u,10948u,10941u,196670u,10887u,10948u,262205u,421u,10949u,10943u,196670u,10915u,10949u,262205u,
7u,10950u,10945u,196670u,10874u,10950u,196670u,10934u,10947u,131321u,10936u,131320u,10951u,262205u,12u,10953u,
10879u,196670u,10952u,10953u,262205u,12u,10955u,10887u,196670u,10954u,10955u,262205u,421u,10957u,10915u,196670u,
10956u,10957u,262205u,7u,10959u,10874u,196670u,10958u,10959u,524345u,57u,10960u,428u,10952u,10954u,10956u,
10958u,262205u,12u,10961u,10954u,196670u,10887u,10961u,262205u,421u,10962u,10956u,196670u,10915u,10962u,262205u,
7u,10963u,10958u,196670u,10874u,10963u,196670u,10934u,10960u,131321u,10936u,131320u,10936u,262205u,57u,10964u,
10934u,393385u,6u,10965u,10964u,551u,548u,262268u,10u,10966u,10965u,393281u,584u,10967u,578u,548u,
10930u,196670u,10967u,10966u,65789u,65592u,327734u,10u,457u,0u,37u,196663u,9u,454u,196663u,9u,
455u,196663u,9u,456u,131320u,458u,262203u,11u,10972u,7u,262203u,11u,11001u,7u,262203u,11u,
11030u,7u,262203u,11u,11057u,7u,262203u,11u,11086u,7u,262203u,11u,11090u,7u,262203u,11u,
11098u,7u,262205u,6u,10968u,455u,327812u,6u,10969u,10968u,3118u,196670u,455u,10969u,262205u,6u,
10970u,456u,327812u,6u,10971u,10970u,3118u,196670u,456u,10971u,262205u,6u,10973u,454u,262205u,6u,
10974u,455u,327808u,6u,10975u,10974u,2490u,327808u,6u,10976u,10973u,10975u,393281u,584u,10977u,578u,
548u,10976u,262205u,10u,10978u,10977u,262205u,6u,10979u,454u,262205u,6u,10980u,455u,327808u,6u,
10981u,10979u,10980u,393281u,584u,10982u,578u,548u,10981u,262205u,10u,10983u,10982u,327811u,10u,10984u,
10978u,10983u,458764u,10u,10985u,1u,40u,630u,10984u,262205u,6u,10986u,454u,262205u,6u,10987u,
455u,327808u,6u,10988u,10987u,2404u,327808u,6u,10989u,10986u,10988u,393281u,584u,10990u,578u,548u,
10989u,262205u,10u,10991u,10990u,262205u,6u,10992u,454u,262205u,6u,10993u,455u,327808u,6u,10994u,
10993u,551u,327808u,6u,10995u,10992u,10994u,393281u,584u,10996u,578u,548u,10995u,262205u,10u,10997u,
10996u,327811u,10u,10998u,10991u,10997u,458764u,10u,10999u,1u,40u,630u,10998u,327813u,10u,11000u,
10985u,10999u,196670u,10972u,11000u,262205u,6u,11002u,454u,262205u,6u,11003u,456u,327808u,6u,11004u,
11003u,2490u,327808u,6u,11005u,11002u,11004u,393281u,584u,11006u,578u,548u,11005u,262205u,10u,11007u,
11006u,262205u,6u,11008u,454u,262205u,6u,11009u,456u,327808u,6u,11010u,11008u,11009u,393281u,584u,
11011u,578u,548u,11010u,262205u,10u,11012u,11011u,327811u,10u,11013u,11007u,11012u,458764u,10u,11014u,
1u,40u,630u,11013u,262205u,6u,11015u,454u,262205u,6u,11016u,456u,327808u,6u,11017u,11016u,
2404u,327808u,6u,11018u,11015u,11017u,393281u,584u,11019u,578u,548u,11018u,262205u,10u,11020u,11019u,
262205u,6u,11021u,454u,262205u,6u,11022u,456u,327808u,6u,11023u,11022u,551u,327808u,6u,11024u,
11021u,11023u,393281u,584u,11025u,578u,548u,11024u,262205u,10u,11026u,11025u,327811u,10u,11027u,11020u,
11026u,458764u,10u,11028u,1u,40u,630u,11027u,327813u,10u,11029u,11014u,11028u,196670u,11001u,11029u,
262205u,6u,11031u,454u,262205u,6u,11032u,455u,327808u,6u,11033u,11032u,2490u,327808u,6u,11034u,
11031u,11033u,393281u,584u,11035u,578u,548u,11034u,262205u,10u,11036u,11035u,262205u,6u,11037u,454u,
262205u,6u,11038u,456u,327808u,6u,11039u,11038u,2490u,327808u,6u,11040u,11037u,11039u,393281u,584u,
11041u,578u,548u,11040u,262205u,10u,11042u,11041u,458764u,10u,11043u,1u,37u,11036u,11042u,262205u,
6u,11044u,454u,262205u,6u,11045u,455u,327808u,6u,11046u,11044u,11045u,393281u,584u,11047u,578u,
548u,11046u,262205u,10u,11048u,11047u,262205u,6u,11049u,454u,262205u,6u,11050u,456u,327808u,6u,
11051u,11049u,11050u,393281u,584u,11052u,578u,548u,11051u,262205u,10u,11053u,11052u,458764u,10u,11054u,
1u,40u,11048u,11053u,327811u,10u,11055u,11043u,11054u,458764u,10u,11056u,1u,40u,630u,11055u,
196670u,11030u,11056u,262205u,6u,11058u,454u,262205u,6u,11059u,455u,327808u,6u,11060u,11059u,2404u,
327808u,6u,11061u,11058u,11060u,393281u,584u,11062u,578u,548u,11061u,262205u,10u,11063u,11062u,262205u,
6u,11064u,454u,262205u,6u,11065u,456u,327808u,6u,11066u,11065u,2404u,327808u,6u,11067u,11064u,
11066u,393281u,584u,11068u,578u,548u,11067u,262205u,10u,11069u,11068u,458764u,10u,11070u,1u,37u,
11063u,11069u,262205u,6u,11071u,454u,262205u,6u,11072u,455u,327808u,6u,11073u,11072u,551u,327808u,
6u,11074u,11071u,11073u,393281u,584u,11075u,578u,548u,11074u,262205u,10u,11076u,11075u,262205u,6u,
11077u,454u,262205u,6u,11078u,456u,327808u,6u,11079u,11078u,551u,327808u,6u,11080u,11077u,11079u,
393281u,584u,11081u,578u,548u,11080u,262205u,10u,11082u,11081u,458764u,10u,11083u,1u,40u,11076u,
11082u,327811u,10u,11084u,11070u,11083u,458764u,10u,11085u,1u,40u,630u,11084u,196670u,11057u,11085u,
262205u,10u,11087u,11030u,262205u,10u,11088u,11057u,327813u,10u,11089u,11087u,11088u,196670u,11086u,11089u,
262205u,10u,11091u,10972u,262205u,10u,11092u,11001u,327809u,10u,11093u,11091u,11092u,262205u,10u,11094u,
11086u,327811u,10u,11095u,11093u,11094u,196670u,11090u,11095u,262205u,10u,11096u,11090u,327866u,57u,11097u,
11096u,630u,196855u,11100u,0u,262394u,11097u,11099u,11104u,131320u,11099u,262205u,10u,11101u,11086u,262205u,
10u,11102u,11090u,327816u,10u,11103u,11101u,11102u,196670u,11098u,11103u,131321u,11100u,131320u,11104u,196670u,
11098u,630u,131321u,11100u,131320u,11100u,262205u,10u,11105u,11098u,131326u,11105u,65592u,327734u,2u,471u,
0u,459u,196663u,9u,460u,196663u,9u,461u,196663u,9u,462u,196663u,9u,463u,196663u,9u,
464u,196663u,9u,465u,196663u,9u,466u,196663u,11u,467u,196663u,11u,468u,196663u,9u,469u,
196663u,58u,470u,131320u,472u,262203u,9u,11119u,7u,262203u,58u,11128u,7u,262203u,11u,11132u,
7u,262203u,9u,11146u,7u,262203u,58u,11154u,7u,262203u,11u,11165u,7u,262203u,9u,11172u,
7u,262203u,58u,11177u,7u,262203u,11u,11185u,7u,262203u,58u,11193u,7u,262203u,9u,11200u,
7u,262203u,9u,11209u,7u,262203u,9u,11210u,7u,262203u,11u,11222u,7u,262203u,58u,11261u,
7u,262203u,9u,11262u,7u,262203u,9u,11263u,7u,262203u,9u,11266u,7u,262203u,11u,11274u,
7u,262203u,9u,11279u,7u,262203u,9u,11291u,7u,262203u,58u,11298u,7u,262203u,58u,11299u,
7u,262203u,9u,11305u,7u,262203u,11u,11319u,7u,262203u,9u,11327u,7u,262203u,9u,11329u,
7u,262203u,9u,11331u,7u,262203u,58u,11342u,7u,262205u,6u,11108u,549u,327851u,57u,11109u,
11108u,548u,262205u,6u,11110u,547u,327851u,57u,11111u,11110u,548u,327846u,57u,11112u,11109u,11111u,
196855u,11114u,0u,262394u,11112u,11113u,11114u,131320u,11113u,65789u,131320u,11114u,262205u,6u,11116u,465u,
262268u,10u,11117u,548u,393281u,584u,11118u,578u,548u,11116u,196670u,11118u,11117u,196670u,11119u,548u,
131321u,11120u,131320u,11120u,262390u,11122u,11123u,0u,131321u,11124u,131320u,11124u,262205u,6u,11125u,11119u,
262205u,6u,11126u,466u,327857u,57u,11127u,11125u,11126u,262394u,11127u,11121u,11122u,131320u,11121u,262205u,
6u,11129u,461u,262205u,6u,11130u,11119u,327808u,6u,11131u,11129u,11130u,393281u,584u,11133u,578u,
548u,11131u,262205u,10u,11134u,11133u,196670u,11132u,11134u,327737u,57u,11135u,82u,11132u,196855u,11137u,
0u,262394u,11135u,11136u,11137u,131320u,11136u,262205u,6u,11138u,461u,262205u,6u,11139u,11119u,327808u,
6u,11140u,11138u,11139u,393281u,584u,11141u,578u,548u,11140u,262205u,10u,11142u,11141u,262205u,10u,
11143u,467u,327870u,57u,11144u,11142u,11143u,131321u,11137u,131320u,11137u,458997u,57u,11145u,11135u,11121u,
11144u,11136u,196670u,11128u,11145u,196670u,11146u,548u,131321u,11147u,131320u,11147u,262390u,11149u,11150u,0u,
131321u,11151u,131320u,11151u,262205u,6u,11152u,11146u,327857u,57u,11153u,11152u,3118u,262394u,11153u,11148u,
11149u,131320u,11148u,262205u,57u,11155u,11128u,196670u,11154u,11155u,327737u,57u,11156u,61u,11154u,196855u,
11158u,0u,262394u,11156u,11157u,11158u,131320u,11157u,262205u,6u,11159u,460u,262205u,6u,11160u,11119u,
327812u,6u,11161u,11160u,3118u,262205u,6u,11162u,11146u,327808u,6u,11163u,11161u,11162u,327808u,6u,
11164u,11159u,11163u,393281u,584u,11166u,578u,548u,11164u,262205u,10u,11167u,11166u,196670u,11165u,11167u,
327737u,57u,11168u,82u,11165u,131321u,11158u,131320u,11158u,458997u,57u,11169u,11156u,11148u,11168u,11157u,
196670u,11128u,11169u,131321u,11150u,131320u,11150u,262205u,6u,11170u,11146u,327808u,6u,11171u,11170u,551u,
196670u,11146u,11171u,131321u,11147u,131320u,11149u,262205u,6u,11173u,462u,196670u,11172u,11173u,327737u,57u,
11174u,65u,11172u,196855u,11176u,0u,262394u,11174u,11175u,11176u,131320u,11175u,262205u,57u,11178u,11128u,
196670u,11177u,11178u,327737u,57u,11179u,61u,11177u,196855u,11181u,0u,262394u,11179u,11180u,11181u,131320u,
11180u,262205u,6u,11182u,462u,262205u,6u,11183u,11119u,327808u,6u,11184u,11182u,11183u,393281u,584u,
11186u,578u,548u,11184u,262205u,10u,11187u,11186u,196670u,11185u,11187u,327737u,57u,11188u,82u,11185u,
131321u,11181u,131320u,11181u,458997u,57u,11189u,11179u,11175u,11188u,11180u,196670u,11128u,11189u,131321u,11176u,
131320u,11176u,262205u,6u,11190u,463u,262205u,6u,11191u,11119u,327808u,6u,11192u,11190u,11191u,262205u,
57u,11194u,11128u,196670u,11193u,11194u,327737u,57u,11195u,61u,11193u,393385u,10u,11196u,11195u,630u,
701u,393281u,584u,11197u,578u,548u,11192u,196670u,11197u,11196u,131321u,11123u,131320u,11123u,262205u,6u,
11198u,11119u,327808u,6u,11199u,11198u,551u,196670u,11119u,11199u,131321u,11120u,131320u,11122u,196670u,11200u,
548u,131321u,11201u,131320u,11201u,262390u,11203u,11204u,0u,131321u,11205u,131320u,11205u,262205u,6u,11206u,
11200u,262205u,6u,11207u,466u,327857u,57u,11208u,11206u,11207u,262394u,11208u,11202u,11203u,131320u,11202u,
196670u,11209u,3173u,196670u,11210u,548u,131321u,11211u,131320u,11211u,262390u,11213u,11214u,0u,131321u,11215u,
131320u,11215u,262205u,6u,11216u,11210u,262205u,6u,11217u,466u,327857u,57u,11218u,11216u,11217u,262394u,
11218u,11212u,11213u,131320u,11212u,262205u,6u,11219u,463u,262205u,6u,11220u,11210u,327808u,6u,11221u,
11219u,11220u,393281u,584u,11223u,578u,548u,11221u,262205u,10u,11224u,11223u,196670u,11222u,11224u,327737u,
57u,11225u,69u,11222u,262312u,57u,11226u,11225u,196855u,11228u,0u,262394u,11226u,11227u,11228u,131320u,
11227u,262205u,6u,11229u,11209u,327857u,57u,11230u,11229u,548u,262312u,57u,11231u,11230u,196855u,11233u,
0u,262394u,11231u,11232u,11233u,131320u,11232u,262205u,6u,11234u,461u,262205u,6u,11235u,11210u,327808u,
6u,11236u,11234u,11235u,393281u,584u,11237u,578u,548u,11236u,262205u,10u,11238u,11237u,262205u,6u,
11239u,461u,262205u,6u,11240u,11209u,327808u,6u,11241u,11239u,11240u,393281u,584u,11242u,578u,548u,
11241u,262205u,10u,11243u,11242u,327866u,57u,11244u,11238u,11243u,131321u,11233u,131320u,11233u,458997u,57u,
11245u,11230u,11227u,11244u,11232u,131321u,11228u,131320u,11228u,458997u,57u,11246u,11226u,11212u,11245u,11233u,
196855u,11248u,0u,262394u,11246u,11247u,11248u,131320u,11247u,262205u,6u,11249u,11210u,196670u,11209u,11249u,
131321u,11248u,131320u,11248u,131321u,11214u,131320u,11214u,262205u,6u,11250u,11210u,327808u,6u,11251u,11250u,
551u,196670u,11210u,11251u,131321u,11211u,131320u,11213u,262205u,6u,11252u,11209u,327857u,57u,11253u,11252u,
548u,196855u,11255u,0u,262394u,11253u,11254u,11255u,131320u,11254u,131321u,11203u,131320u,11255u,262205u,6u,
11257u,463u,262205u,6u,11258u,11209u,327808u,6u,11259u,11257u,11258u,393281u,584u,11260u,578u,548u,
11259u,196670u,11260u,701u,196670u,11261u,6465u,262205u,6u,11264u,462u,196670u,11263u,11264u,327737u,57u,
11265u,65u,11263u,196855u,11268u,0u,262394u,11265u,11267u,11277u,131320u,11267u,262205u,6u,11269u,462u,
262205u,6u,11270u,11209u,327808u,6u,11271u,11269u,11270u,393281u,584u,11272u,578u,548u,11271u,262205u,
10u,11273u,11272u,196670u,11274u,11273u,327737u,10u,11275u,73u,11274u,262254u,6u,11276u,11275u,196670u,
11266u,11276u,131321u,11268u,131320u,11277u,196670u,11266u,548u,131321u,11268u,131320u,11268u,262205u,6u,11278u,
11266u,196670u,11262u,11278u,196670u,11279u,548u,131321u,11280u,131320u,11280u,262390u,11282u,11283u,0u,131321u,
11284u,131320u,11284u,262205u,6u,11285u,11279u,262205u,6u,11286u,465u,393281u,584u,11287u,578u,548u,
11286u,262205u,10u,11288u,11287u,262268u,6u,11289u,11288u,327857u,57u,11290u,11285u,11289u,262394u,11290u,
11281u,11282u,131320u,11281u,262205u,6u,11292u,464u,262205u,6u,11293u,11279u,327808u,6u,11294u,11292u,
11293u,393281u,584u,11295u,578u,548u,11294u,262205u,10u,11296u,11295u,262254u,6u,11297u,11296u,196670u,
11291u,11297u,262205u,57u,11300u,470u,196670u,11299u,11300u,327737u,57u,11301u,61u,11299u,262312u,57u,
11302u,11301u,196855u,11304u,0u,262394u,11302u,11303u,11304u,131320u,11303u,262205u,6u,11306u,462u,196670u,
11305u,11306u,327737u,57u,11307u,65u,11305u,262312u,57u,11308u,11307u,131321u,11304u,131320u,11304u,458997u,
57u,11309u,11301u,11281u,11308u,11303u,262312u,57u,11310u,11309u,196855u,11312u,0u,262394u,11310u,11311u,
11312u,131320u,11311u,262205u,6u,11313u,11262u,262205u,6u,11314u,462u,262205u,6u,11315u,11291u,327808u,
6u,11316u,11314u,11315u,393281u,584u,11317u,578u,548u,11316u,262205u,10u,11318u,11317u,196670u,11319u,
11318u,327737u,10u,11320u,73u,11319u,262254u,6u,11321u,11320u,327850u,57u,11322u,11313u,11321u,131321u,
11312u,131320u,11312u,458997u,57u,11323u,11309u,11304u,11322u,11311u,196670u,11298u,11323u,262205u,57u,11324u,
11298u,196855u,11326u,0u,262394u,11324u,11325u,11326u,131320u,11325u,262205u,6u,11328u,460u,196670u,11327u,
11328u,262205u,6u,11330u,11209u,196670u,11329u,11330u,262205u,6u,11332u,11291u,196670u,11331u,11332u,458809u,
10u,11333u,457u,11327u,11329u,11331u,262205u,10u,11334u,468u,327866u,57u,11335u,11333u,11334u,131321u,
11326u,131320u,11326u,458997u,57u,11336u,11324u,11312u,11335u,11325u,196855u,11338u,0u,262394u,11336u,11337u,
11338u,131320u,11337u,196670u,11261u,6723u,131321u,11282u,131320u,11338u,131321u,11283u,131320u,11283u,262205u,6u,
11340u,11279u,327808u,6u,11341u,11340u,551u,196670u,11279u,11341u,131321u,11280u,131320u,11282u,262205u,57u,
11343u,11261u,196670u,11342u,11343u,327737u,57u,11344u,61u,11342u,196855u,11346u,0u,262394u,11344u,11345u,
11346u,131320u,11345u,131321u,11204u,131320u,11346u,262205u,6u,11348u,464u,262205u,6u,11349u,465u,393281u,
584u,11350u,578u,548u,11349u,262205u,10u,11351u,11350u,262268u,6u,11352u,11351u,327808u,6u,11353u,
11348u,11352u,262205u,6u,11354u,11209u,262255u,10u,11355u,11354u,393281u,584u,11356u,578u,548u,11353u,
196670u,11356u,11355u,262205u,6u,11357u,465u,262205u,6u,11358u,465u,393281u,584u,11359u,578u,548u,
11358u,262205u,10u,11360u,11359u,262268u,6u,11361u,11360u,327808u,6u,11362u,11361u,551u,262268u,10u,
11363u,11362u,393281u,584u,11364u,578u,548u,11357u,196670u,11364u,11363u,262205u,6u,11365u,469u,327853u,
57u,11366u,11365u,548u,196855u,11368u,0u,262394u,11366u,11367u,11368u,131320u,11367u,262205u,6u,11369u,
465u,393281u,584u,11370u,578u,548u,11369u,262205u,10u,11371u,11370u,262268u,6u,11372u,11371u,262205u,
6u,11373u,469u,327855u,57u,11374u,11372u,11373u,131321u,11368u,131320u,11368u,458997u,57u,11375u,11366u,
11346u,11374u,11367u,196855u,11377u,0u,262394u,11375u,11376u,11377u,131320u,11376u,131321u,11203u,131320u,11377u,
131321u,11204u,131320u,11204u,262205u,6u,11379u,11200u,327808u,6u,11380u,11379u,551u,196670u,11200u,11380u,
131321u,11201u,131320u,11203u,65789u,65592u,327734u,2u,476u,0u,473u,196663u,9u,474u,196663u,9u,
475u,131320u,477u,262203u,9u,11381u,7u,262205u,6u,11382u,549u,262205u,6u,11383u,550u,327812u,
6u,11384u,11382u,11383u,262205u,6u,11385u,547u,327808u,6u,11386u,11384u,11385u,196670u,11381u,11386u,
131321u,11387u,131320u,11387u,262390u,11389u,11390u,0u,131321u,11391u,131320u,11391u,262205u,6u,11392u,11381u,
262205u,6u,11393u,475u,327857u,57u,11394u,11392u,11393u,262394u,11394u,11388u,11389u,131320u,11388u,262205u,
6u,11395u,474u,262205u,6u,11396u,11381u,327808u,6u,11397u,11395u,11396u,393281u,584u,11398u,578u,
548u,11397u,196670u,11398u,630u,131321u,11390u,131320u,11390u,262205u,6u,11399u,550u,262205u,6u,11400u,
552u,327812u,6u,11401u,11399u,11400u,262205u,6u,11402u,11381u,327808u,6u,11403u,11402u,11401u,196670u,
11381u,11403u,131321u,11387u,131320u,11389u,65789u,65592u,327734u,2u,484u,0u,478u,196663u,9u,479u,
196663u,9u,480u,196663u,9u,481u,196663u,9u,482u,196663u,11u,483u,131320u,485u,262203u,9u,
11404u,7u,262203u,11u,11416u,7u,262203u,11u,11449u,7u,262203u,11u,11477u,7u,262203u,11u,
11491u,7u,262203u,11u,11492u,7u,262203u,11u,11519u,7u,262203u,11u,11526u,7u,262203u,11u,
11531u,7u,262203u,11u,11534u,7u,262203u,11u,11541u,7u,262203u,11u,11553u,7u,262203u,11u,
11566u,7u,262205u,6u,11405u,549u,262205u,6u,11406u,550u,327812u,6u,11407u,11405u,11406u,262205u,
6u,11408u,547u,327808u,6u,11409u,11407u,11408u,196670u,11404u,11409u,262205u,6u,11410u,11404u,262205u,
6u,11411u,481u,327855u,57u,11412u,11410u,11411u,196855u,11414u,0u,262394u,11412u,11413u,11414u,131320u,
11413u,65789u,131320u,11414u,262205u,6u,11417u,479u,262205u,6u,11418u,11404u,327808u,6u,11419u,11417u,
11418u,393281u,584u,11420u,578u,548u,11419u,262205u,10u,11421u,11420u,196670u,11416u,11421u,262205u,6u,
11422u,482u,196855u,11434u,0u,1507579u,11422u,11433u,0u,11423u,1u,11424u,2u,11425u,3u,11426u,
4u,11427u,5u,11428u,6u,11429u,7u,11430u,8u,11431u,9u,11432u,131320u,11433u,262205u,6u,
11580u,480u,262205u,6u,11581u,11404u,327808u,6u,11582u,11580u,11581u,262205u,10u,11583u,11416u,393281u,
584u,11584u,578u,548u,11582u,196670u,11584u,11583u,131321u,11434u,131320u,11423u,262205u,6u,11435u,480u,
262205u,6u,11436u,11404u,327808u,6u,11437u,11435u,11436u,262205u,10u,11438u,11416u,327866u,57u,11439u,
11438u,630u,262205u,10u,11440u,11416u,393385u,10u,11441u,11439u,11440u,630u,393281u,584u,11442u,578u,
548u,11437u,196670u,11442u,11441u,131321u,11434u,131320u,11424u,262205u,6u,11444u,480u,262205u,6u,11445u,
11404u,327808u,6u,11446u,11444u,11445u,262205u,10u,11447u,11416u,327866u,57u,11448u,11447u,630u,196855u,
11451u,0u,262394u,11448u,11450u,11453u,131320u,11450u,262205u,10u,11452u,11416u,196670u,11449u,11452u,131321u,
11451u,131320u,11453u,262205u,10u,11454u,483u,262205u,10u,11455u,11416u,327813u,10u,11456u,11454u,11455u,
196670u,11449u,11456u,131321u,11451u,131320u,11451u,262205u,10u,11457u,11449u,393281u,584u,11458u,578u,548u,
11446u,196670u,11458u,11457u,131321u,11434u,131320u,11425u,262205u,6u,11460u,480u,262205u,6u,11461u,11404u,
327808u,6u,11462u,11460u,11461u,262205u,10u,11463u,11416u,262271u,10u,11464u,11463u,393228u,10u,11465u,
1u,27u,11464u,327809u,10u,11466u,701u,11465u,327816u,10u,11467u,701u,11466u,393281u,584u,11468u,
578u,548u,11462u,196670u,11468u,11467u,131321u,11434u,131320u,11426u,262205u,6u,11470u,480u,262205u,6u,
11471u,11404u,327808u,6u,11472u,11470u,11471u,262205u,10u,11473u,11416u,393228u,10u,11474u,1u,21u,
11473u,393281u,584u,11475u,578u,548u,11472u,196670u,11475u,11474u,131321u,11434u,131320u,11427u,262205u,10u,
11478u,11416u,262271u,10u,11479u,11478u,393228u,10u,11480u,1u,27u,11479u,327809u,10u,11481u,701u,
11480u,327816u,10u,11482u,701u,11481u,196670u,11477u,11482u,262205u,6u,11483u,480u,262205u,6u,11484u,
11404u,327808u,6u,11485u,11483u,11484u,262205u,10u,11486u,11416u,262205u,10u,11487u,11477u,327813u,10u,
11488u,11486u,11487u,393281u,584u,11489u,578u,548u,11485u,196670u,11489u,11488u,131321u,11434u,131320u,11428u,
196670u,11491u,2971u,262205u,10u,11493u,11416u,262205u,10u,11494u,11416u,327813u,10u,11495u,11493u,11494u,
262205u,10u,11496u,11416u,327813u,10u,11497u,11495u,11496u,196670u,11492u,11497u,262205u,6u,11498u,480u,
262205u,6u,11499u,11404u,327808u,6u,11500u,11498u,11499u,262205u,10u,11501u,11416u,327813u,10u,11502u,
687u,11501u,262205u,10u,11503u,11491u,262205u,10u,11504u,11416u,262205u,10u,11505u,11492u,327813u,10u,
11506u,2973u,11505u,327809u,10u,11507u,11504u,11506u,327813u,10u,11508u,11503u,11507u,393228u,10u,11509u,
1u,21u,11508u,327809u,10u,11510u,701u,11509u,327813u,10u,11511u,11502u,11510u,393281u,584u,11512u,
578u,548u,11500u,196670u,11512u,11511u,131321u,11434u,131320u,11429u,262205u,6u,11514u,480u,262205u,6u,
11515u,11404u,327808u,6u,11516u,11514u,11515u,262205u,10u,11517u,11416u,327866u,57u,11518u,11517u,4461u,
196855u,11521u,0u,262394u,11518u,11520u,11523u,131320u,11520u,262205u,10u,11522u,11416u,196670u,11519u,11522u,
131321u,11521u,131320u,11523u,262205u,10u,11524u,11416u,393228u,10u,11525u,1u,27u,11524u,196670u,11526u,
11525u,327737u,10u,11527u,79u,11526u,196670u,11519u,11527u,131321u,11521u,131320u,11521u,262205u,10u,11528u,
11519u,393281u,584u,11529u,578u,548u,11516u,196670u,11529u,11528u,131321u,11434u,131320u,11430u,262205u,10u,
11532u,11416u,327866u,57u,11533u,11532u,4461u,196855u,11536u,0u,262394u,11533u,11535u,11538u,131320u,11535u,
262205u,10u,11537u,11416u,196670u,11534u,11537u,131321u,11536u,131320u,11538u,262205u,10u,11539u,11416u,393228u,
10u,11540u,1u,27u,11539u,196670u,11541u,11540u,327737u,10u,11542u,79u,11541u,196670u,11534u,11542u,
131321u,11536u,131320u,11536u,262205u,10u,11543u,11534u,196670u,11531u,11543u,262205u,6u,11544u,480u,262205u,
6u,11545u,11404u,327808u,6u,11546u,11544u,11545u,262205u,10u,11547u,11416u,262205u,10u,11548u,11531u,
393228u,10u,11549u,1u,21u,11548u,327813u,10u,11550u,11547u,11549u,393281u,584u,11551u,578u,548u,
11546u,196670u,11551u,11550u,131321u,11434u,131320u,11431u,262205u,10u,11554u,11416u,327809u,10u,11555u,11554u,
3067u,327816u,10u,11557u,11555u,11556u,196670u,11553u,11557u,262205u,6u,11558u,480u,262205u,6u,11559u,
11404u,327808u,6u,11560u,11558u,11559u,262205u,10u,11561u,11553u,458764u,10u,11562u,1u,40u,630u,
11561u,458764u,10u,11563u,1u,37u,701u,11562u,393281u,584u,11564u,578u,548u,11560u,196670u,11564u,
11563u,131321u,11434u,131320u,11432u,262205u,10u,11567u,11416u,327809u,10u,11568u,11567u,3067u,327816u,10u,
11569u,11568u,11556u,458764u,10u,11570u,1u,40u,630u,11569u,458764u,10u,11571u,1u,37u,701u,
11570u,196670u,11566u,11571u,262205u,6u,11572u,480u,262205u,6u,11573u,11404u,327808u,6u,11574u,11572u,
11573u,262205u,10u,11575u,11416u,262205u,10u,11576u,11566u,327813u,10u,11577u,11575u,11576u,393281u,584u,
11578u,578u,548u,11574u,196670u,11578u,11577u,131321u,11434u,131320u,11434u,65789u,65592u,327734u,2u,491u,
0u,108u,196663u,9u,486u,196663u,9u,487u,196663u,9u,488u,196663u,9u,489u,196663u,9u,
490u,131320u,492u,262203u,9u,11587u,7u,262203u,11u,11654u,7u,262203u,11u,11666u,7u,262205u,
6u,11588u,549u,262205u,6u,11589u,550u,327812u,6u,11590u,11588u,11589u,262205u,6u,11591u,547u,
327808u,6u,11592u,11590u,11591u,196670u,11587u,11592u,262205u,6u,11593u,11587u,262205u,6u,11594u,489u,
327855u,57u,11595u,11593u,11594u,196855u,11597u,0u,262394u,11595u,11596u,11597u,131320u,11596u,65789u,131320u,
11597u,262205u,6u,11599u,490u,196855u,11605u,0u,721147u,11599u,11604u,0u,11600u,1u,11601u,2u,
11602u,3u,11603u,131320u,11604u,262205u,6u,11680u,488u,262205u,6u,11681u,11587u,327808u,6u,11682u,
11680u,11681u,262205u,6u,11683u,486u,262205u,6u,11684u,11587u,327808u,6u,11685u,11683u,11684u,393281u,
584u,11686u,578u,548u,11685u,262205u,10u,11687u,11686u,393281u,584u,11688u,578u,548u,11682u,196670u,
11688u,11687u,131321u,11605u,131320u,11600u,262205u,6u,11606u,488u,262205u,6u,11607u,11587u,327808u,6u,
11608u,11606u,11607u,262205u,6u,11609u,486u,262205u,6u,11610u,11587u,327808u,6u,11611u,11609u,11610u,
393281u,584u,11612u,578u,548u,11611u,262205u,10u,11613u,11612u,262205u,6u,11614u,487u,262205u,6u,
11615u,11587u,327808u,6u,11616u,11614u,11615u,393281u,584u,11617u,578u,548u,11616u,262205u,10u,11618u,
11617u,327809u,10u,11619u,11613u,11618u,393281u,584u,11620u,578u,548u,11608u,196670u,11620u,11619u,131321u,
11605u,131320u,11601u,262205u,6u,11622u,488u,262205u,6u,11623u,11587u,327808u,6u,11624u,11622u,11623u,
262205u,6u,11625u,486u,262205u,6u,11626u,11587u,327808u,6u,11627u,11625u,11626u,393281u,584u,11628u,
578u,548u,11627u,262205u,10u,11629u,11628u,262205u,6u,11630u,487u,262205u,6u,11631u,11587u,327808u,
6u,11632u,11630u,11631u,393281u,584u,11633u,578u,548u,11632u,262205u,10u,11634u,11633u,327811u,10u,
11635u,11629u,11634u,393281u,584u,11636u,578u,548u,11624u,196670u,11636u,11635u,131321u,11605u,131320u,11602u,
262205u,6u,11638u,488u,262205u,6u,11639u,11587u,327808u,6u,11640u,11638u,11639u,262205u,6u,11641u,
486u,262205u,6u,11642u,11587u,327808u,6u,11643u,11641u,11642u,393281u,584u,11644u,578u,548u,11643u,
262205u,10u,11645u,11644u,262205u,6u,11646u,487u,262205u,6u,11647u,11587u,327808u,6u,11648u,11646u,
11647u,393281u,584u,11649u,578u,548u,11648u,262205u,10u,11650u,11649u,327813u,10u,11651u,11645u,11650u,
393281u,584u,11652u,578u,548u,11640u,196670u,11652u,11651u,131321u,11605u,131320u,11603u,262205u,6u,11655u,
487u,262205u,6u,11656u,11587u,327808u,6u,11657u,11655u,11656u,393281u,584u,11658u,578u,548u,11657u,
262205u,10u,11659u,11658u,196670u,11654u,11659u,262205u,6u,11660u,488u,262205u,6u,11661u,11587u,327808u,
6u,11662u,11660u,11661u,262205u,10u,11663u,11654u,393228u,10u,11664u,1u,4u,11663u,327866u,57u,
11665u,11664u,4545u,196855u,11668u,0u,262394u,11665u,11667u,11676u,131320u,11667u,262205u,6u,11669u,486u,
262205u,6u,11670u,11587u,327808u,6u,11671u,11669u,11670u,393281u,584u,11672u,578u,548u,11671u,262205u,
10u,11673u,11672u,262205u,10u,11674u,11654u,327816u,10u,11675u,11673u,11674u,196670u,11666u,11675u,131321u,
11668u,131320u,11676u,196670u,11666u,630u,131321u,11668u,131320u,11668u,262205u,10u,11677u,11666u,393281u,584u,
11678u,578u,548u,11662u,196670u,11678u,11677u,131321u,11605u,131320u,11605u,65789u,65592u,327734u,2u,500u,
0u,493u,196663u,9u,494u,196663u,9u,495u,196663u,9u,496u,196663u,9u,497u,196663u,9u,
498u,196663u,11u,499u,131320u,501u,262203u,9u,11691u,7u,262203u,11u,11703u,7u,262203u,11u,
11709u,7u,262203u,11u,11715u,7u,262203u,11u,11738u,7u,262203u,11u,11749u,7u,262203u,11u,
11757u,7u,262203u,11u,11772u,7u,262203u,11u,11773u,7u,262203u,11u,11777u,7u,262203u,11u,
11781u,7u,262203u,11u,11788u,7u,262203u,11u,11791u,7u,262203u,11u,11796u,7u,262203u,11u,
11819u,7u,262203u,11u,11822u,7u,262203u,11u,11829u,7u,262203u,11u,11832u,7u,262203u,11u,
11835u,7u,262205u,6u,11692u,549u,262205u,6u,11693u,550u,327812u,6u,11694u,11692u,11693u,262205u,
6u,11695u,547u,327808u,6u,11696u,11694u,11695u,196670u,11691u,11696u,262205u,6u,11697u,11691u,262205u,
6u,11698u,497u,327855u,57u,11699u,11697u,11698u,196855u,11701u,0u,262394u,11699u,11700u,11701u,131320u,
11700u,65789u,131320u,11701u,262205u,6u,11704u,494u,262205u,6u,11705u,11691u,327808u,6u,11706u,11704u,
11705u,393281u,584u,11707u,578u,548u,11706u,262205u,10u,11708u,11707u,196670u,11703u,11708u,262205u,6u,
11710u,495u,262205u,6u,11711u,11691u,327808u,6u,11712u,11710u,11711u,393281u,584u,11713u,578u,548u,
11712u,262205u,10u,11714u,11713u,196670u,11709u,11714u,196670u,11715u,701u,262205u,6u,11716u,498u,196855u,
11728u,0u,1507579u,11716u,11727u,0u,11717u,1u,11718u,2u,11719u,3u,11720u,4u,11721u,5u,
11722u,6u,11723u,7u,11724u,8u,11725u,9u,11726u,131320u,11727u,196670u,11715u,701u,131321u,11728u,
131320u,11717u,262205u,10u,11729u,11703u,327866u,57u,11730u,11729u,630u,393385u,10u,11731u,11730u,701u,
630u,196670u,11715u,11731u,131321u,11728u,131320u,11718u,262205u,10u,11733u,11703u,327866u,57u,11734u,11733u,
630u,262205u,10u,11735u,499u,393385u,10u,11736u,11734u,701u,11735u,196670u,11715u,11736u,131321u,11728u,
131320u,11719u,262205u,10u,11739u,11703u,262271u,10u,11740u,11739u,393228u,10u,11741u,1u,27u,11740u,
327809u,10u,11742u,701u,11741u,327816u,10u,11743u,701u,11742u,196670u,11738u,11743u,262205u,10u,11744u,
11738u,262205u,10u,11745u,11738u,327811u,10u,11746u,701u,11745u,327813u,10u,11747u,11744u,11746u,196670u,
11715u,11747u,131321u,11728u,131320u,11720u,262205u,10u,11750u,11703u,393228u,10u,11751u,1u,21u,11750u,
196670u,11749u,11751u,262205u,10u,11752u,11749u,262205u,10u,11753u,11749u,327813u,10u,11754u,11752u,11753u,
327811u,10u,11755u,701u,11754u,196670u,11715u,11755u,131321u,11728u,131320u,11721u,262205u,10u,11758u,11703u,
262271u,10u,11759u,11758u,393228u,10u,11760u,1u,27u,11759u,327809u,10u,11761u,701u,11760u,327816u,
10u,11762u,701u,11761u,196670u,11757u,11762u,262205u,10u,11763u,11757u,262205u,10u,11764u,11703u,262205u,
10u,11765u,11757u,327813u,10u,11766u,11764u,11765u,262205u,10u,11767u,11757u,327811u,10u,11768u,701u,
11767u,327813u,10u,11769u,11766u,11768u,327809u,10u,11770u,11763u,11769u,196670u,11715u,11770u,131321u,11728u,
131320u,11722u,196670u,11772u,2971u,262205u,10u,11774u,11703u,262205u,10u,11775u,11703u,327813u,10u,11776u,
11774u,11775u,196670u,11773u,11776u,262205u,10u,11778u,11773u,262205u,10u,11779u,11703u,327813u,10u,11780u,
11778u,11779u,196670u,11777u,11780u,262205u,10u,11782u,11772u,262205u,10u,11783u,11703u,262205u,10u,11784u,
11777u,327813u,10u,11785u,2973u,11784u,327809u,10u,11786u,11783u,11785u,327813u,10u,11787u,11782u,11786u,
196670u,11781u,11787u,262205u,10u,11789u,11781u,393228u,10u,11790u,1u,21u,11789u,196670u,11788u,11790u,
262205u,10u,11792u,11788u,262205u,10u,11793u,11788u,327813u,10u,11794u,11792u,11793u,327811u,10u,11795u,
701u,11794u,196670u,11791u,11795u,262205u,10u,11797u,11772u,262205u,10u,11798u,11773u,327813u,10u,11799u,
3043u,11798u,327809u,10u,11800u,701u,11799u,327813u,10u,11801u,11797u,11800u,196670u,11796u,11801u,262205u,
10u,11802u,11788u,327809u,10u,11803u,701u,11802u,327813u,10u,11804u,687u,11803u,262205u,10u,11805u,
11703u,327813u,10u,11806u,687u,11805u,262205u,10u,11807u,11791u,327813u,10u,11808u,11806u,11807u,262205u,
10u,11809u,11796u,327813u,10u,11810u,11808u,11809u,327809u,10u,11811u,11804u,11810u,196670u,11715u,11811u,
131321u,11728u,131320u,11723u,262205u,10u,11813u,11703u,262271u,10u,11814u,11813u,393228u,10u,11815u,1u,
27u,11814u,327809u,10u,11816u,701u,11815u,327816u,10u,11817u,701u,11816u,196670u,11715u,11817u,131321u,
11728u,131320u,11724u,262205u,10u,11820u,11703u,327866u,57u,11821u,11820u,4461u,196855u,11824u,0u,262394u,
11821u,11823u,11826u,131320u,11823u,262205u,10u,11825u,11703u,196670u,11822u,11825u,131321u,11824u,131320u,11826u,
262205u,10u,11827u,11703u,393228u,10u,11828u,1u,27u,11827u,196670u,11829u,11828u,327737u,10u,11830u,
79u,11829u,196670u,11822u,11830u,131321u,11824u,131320u,11824u,262205u,10u,11831u,11822u,196670u,11819u,11831u,
262205u,10u,11833u,11819u,393228u,10u,11834u,1u,21u,11833u,196670u,11832u,11834u,262205u,10u,11836u,
11703u,262271u,10u,11837u,11836u,393228u,10u,11838u,1u,27u,11837u,327809u,10u,11839u,701u,11838u,
327816u,10u,11840u,701u,11839u,196670u,11835u,11840u,262205u,10u,11841u,11832u,262205u,10u,11842u,11703u,
262205u,10u,11843u,11835u,327813u,10u,11844u,11842u,11843u,262205u,10u,11845u,11832u,262205u,10u,11846u,
11832u,327813u,10u,11847u,11845u,11846u,327811u,10u,11848u,701u,11847u,327813u,10u,11849u,11844u,11848u,
327809u,10u,11850u,11841u,11849u,196670u,11715u,11850u,131321u,11728u,131320u,11725u,262205u,10u,11852u,11703u,
327866u,57u,11854u,11852u,11853u,262205u,10u,11855u,11703u,327864u,57u,11856u,11855u,3067u,327847u,57u,
11857u,11854u,11856u,393385u,10u,11859u,11857u,11858u,630u,196670u,11715u,11859u,131321u,11728u,131320u,11726u,
262205u,10u,11861u,11703u,327868u,57u,11862u,11861u,11853u,196855u,11864u,0u,262394u,11862u,11863u,11865u,
131320u,11863u,196670u,11715u,630u,131321u,11864u,131320u,11865u,262205u,10u,11866u,11703u,327870u,57u,11867u,
11866u,3067u,196855u,11869u,0u,262394u,11867u,11868u,11870u,131320u,11868u,196670u,11715u,701u,131321u,11869u,
131320u,11870u,262205u,10u,11871u,11703u,327816u,10u,11872u,11871u,3067u,327809u,10u,11873u,11872u,687u,
196670u,11715u,11873u,131321u,11869u,131320u,11869u,131321u,11864u,131320u,11864u,131321u,11728u,131320u,11728u,262205u,
6u,11877u,496u,262205u,6u,11878u,11691u,327808u,6u,11879u,11877u,11878u,262205u,10u,11880u,11709u,
262205u,10u,11881u,11715u,327813u,10u,11882u,11880u,11881u,393281u,584u,11883u,578u,548u,11879u,196670u,
11883u,11882u,65789u,65592u,327734u,2u,509u,0u,129u,196663u,9u,502u,196663u,9u,503u,196663u,
9u,504u,196663u,9u,505u,196663u,9u,506u,196663u,9u,507u,196663u,9u,508u,131320u,510u,
262203u,9u,11884u,7u,262203u,11u,11896u,7u,262203u,11u,11902u,7u,262203u,11u,11908u,7u,
262203u,11u,11914u,7u,262205u,6u,11885u,549u,262205u,6u,11886u,550u,327812u,6u,11887u,11885u,
11886u,262205u,6u,11888u,547u,327808u,6u,11889u,11887u,11888u,196670u,11884u,11889u,262205u,6u,11890u,
11884u,262205u,6u,11891u,507u,327855u,57u,11892u,11890u,11891u,196855u,11894u,0u,262394u,11892u,11893u,
11894u,131320u,11893u,65789u,131320u,11894u,262205u,6u,11897u,502u,262205u,6u,11898u,11884u,327808u,6u,
11899u,11897u,11898u,393281u,584u,11900u,578u,548u,11899u,262205u,10u,11901u,11900u,196670u,11896u,11901u,
262205u,6u,11903u,503u,262205u,6u,11904u,11884u,327808u,6u,11905u,11903u,11904u,393281u,584u,11906u,
578u,548u,11905u,262205u,10u,11907u,11906u,196670u,11902u,11907u,262205u,6u,11909u,504u,262205u,6u,
11910u,11884u,327808u,6u,11911u,11909u,11910u,393281u,584u,11912u,578u,548u,11911u,262205u,10u,11913u,
11912u,196670u,11908u,11913u,196670u,11914u,4545u,262205u,6u,11915u,508u,196855u,11921u,0u,721147u,11915u,
11920u,0u,11916u,1u,11917u,2u,11918u,3u,11919u,131320u,11920u,262205u,6u,11995u,505u,262205u,
6u,11996u,11884u,327808u,6u,11997u,11995u,11996u,262205u,10u,11998u,11908u,393281u,584u,11999u,578u,
548u,11997u,196670u,11999u,11998u,262205u,6u,12000u,506u,262205u,6u,12001u,11884u,327808u,6u,12002u,
12000u,12001u,393281u,584u,12003u,578u,548u,12002u,196670u,12003u,630u,131321u,11921u,131320u,11916u,262205u,
6u,11922u,505u,262205u,6u,11923u,11884u,327808u,6u,11924u,11922u,11923u,262205u,10u,11925u,11908u,
393281u,584u,11926u,578u,548u,11924u,196670u,11926u,11925u,262205u,6u,11927u,506u,262205u,6u,11928u,
11884u,327808u,6u,11929u,11927u,11928u,262205u,10u,11930u,11908u,393281u,584u,11931u,578u,548u,11929u,
196670u,11931u,11930u,131321u,11921u,131320u,11917u,262205u,6u,11933u,505u,262205u,6u,11934u,11884u,327808u,
6u,11935u,11933u,11934u,262205u,10u,11936u,11908u,393281u,584u,11937u,578u,548u,11935u,196670u,11937u,
11936u,262205u,6u,11938u,506u,262205u,6u,11939u,11884u,327808u,6u,11940u,11938u,11939u,262205u,10u,
11941u,11908u,262271u,10u,11942u,11941u,393281u,584u,11943u,578u,548u,11940u,196670u,11943u,11942u,131321u,
11921u,131320u,11918u,262205u,6u,11945u,505u,262205u,6u,11946u,11884u,327808u,6u,11947u,11945u,11946u,
262205u,10u,11948u,11908u,262205u,10u,11949u,11902u,327813u,10u,11950u,11948u,11949u,393281u,584u,11951u,
578u,548u,11947u,196670u,11951u,11950u,262205u,6u,11952u,506u,262205u,6u,11953u,11884u,327808u,6u,
11954u,11952u,11953u,262205u,10u,11955u,11908u,262205u,10u,11956u,11896u,327813u,10u,11957u,11955u,11956u,
393281u,584u,11958u,578u,548u,11954u,196670u,11958u,11957u,131321u,11921u,131320u,11919u,262205u,10u,11960u,
11902u,393228u,10u,11961u,1u,4u,11960u,262205u,10u,11962u,11914u,327866u,57u,11963u,11961u,11962u,
196855u,11965u,0u,262394u,11963u,11964u,11985u,131320u,11964u,262205u,6u,11966u,505u,262205u,6u,11967u,
11884u,327808u,6u,11968u,11966u,11967u,262205u,10u,11969u,11908u,262205u,10u,11970u,11902u,327816u,10u,
11971u,11969u,11970u,393281u,584u,11972u,578u,548u,11968u,196670u,11972u,11971u,262205u,6u,11973u,506u,
262205u,6u,11974u,11884u,327808u,6u,11975u,11973u,11974u,262205u,10u,11976u,11908u,262271u,10u,11977u,
11976u,262205u,10u,11978u,11896u,327813u,10u,11979u,11977u,11978u,262205u,10u,11980u,11902u,262205u,10u,
11981u,11902u,327813u,10u,11982u,11980u,11981u,327816u,10u,11983u,11979u,11982u,393281u,584u,11984u,578u,
548u,11975u,196670u,11984u,11983u,131321u,11965u,131320u,11985u,262205u,6u,11986u,505u,262205u,6u,11987u,
11884u,327808u,6u,11988u,11986u,11987u,393281u,584u,11989u,578u,548u,11988u,196670u,11989u,630u,262205u,
6u,11990u,506u,262205u,6u,11991u,11884u,327808u,6u,11992u,11990u,11991u,393281u,584u,11993u,578u,
548u,11992u,196670u,11993u,630u,131321u,11965u,131320u,11965u,131321u,11921u,131320u,11921u,65789u,65592u,327734u,
2u,514u,0u,31u,196663u,9u,511u,196663u,9u,512u,196663u,9u,513u,131320u,515u,262203u,
9u,12006u,7u,196670u,12006u,548u,131321u,12007u,131320u,12007u,262390u,12009u,12010u,0u,131321u,12011u,
131320u,12011u,262205u,6u,12012u,12006u,262205u,6u,12013u,513u,327857u,57u,12014u,12012u,12013u,262394u,
12014u,12008u,12009u,131320u,12008u,262205u,6u,12015u,512u,262205u,6u,12016u,12006u,327808u,6u,12017u,
12015u,12016u,262205u,6u,12018u,511u,262205u,6u,12019u,12006u,327808u,6u,12020u,12018u,12019u,393281u,
584u,12021u,578u,548u,12020u,262205u,10u,12022u,12021u,393281u,584u,12023u,578u,548u,12017u,196670u,
12023u,12022u,131321u,12010u,131320u,12010u,262205u,6u,12024u,12006u,327808u,6u,12025u,12024u,551u,196670u,
12006u,12025u,131321u,12007u,131320u,12009u,65789u,65592u,327734u,2u,530u,0u,516u,196663u,9u,517u,
196663u,9u,518u,196663u,9u,519u,196663u,9u,520u,196663u,9u,521u,196663u,9u,522u,196663u,
9u,523u,196663u,9u,524u,196663u,9u,525u,196663u,58u,526u,196663u,58u,527u,196663u,11u,
528u,196663u,11u,529u,131320u,531u,262203u,9u,12026u,7u,262203u,9u,12035u,7u,262203u,11u,
12044u,7u,262203u,9u,12045u,7u,262203u,58u,12055u,7u,262203u,9u,12058u,7u,262203u,58u,
12077u,7u,262203u,9u,12080u,7u,196670u,12026u,548u,131321u,12027u,131320u,12027u,262390u,12029u,12030u,
0u,131321u,12031u,131320u,12031u,262205u,6u,12032u,12026u,262205u,6u,12033u,521u,327857u,57u,12034u,
12032u,12033u,262394u,12034u,12028u,12029u,131320u,12028u,196670u,12035u,548u,131321u,12036u,131320u,12036u,262390u,
12038u,12039u,0u,131321u,12040u,131320u,12040u,262205u,6u,12041u,12035u,262205u,6u,12042u,520u,327857u,
57u,12043u,12041u,12042u,262394u,12043u,12037u,12038u,131320u,12037u,196670u,12044u,630u,196670u,12045u,548u,
131321u,12046u,131320u,12046u,262390u,12048u,12049u,0u,131321u,12050u,131320u,12050u,262205u,6u,12051u,12045u,
262205u,6u,12052u,522u,327857u,57u,12053u,12051u,12052u,262394u,12053u,12047u,12048u,131320u,12047u,262205u,
6u,12054u,517u,262205u,57u,12056u,526u,196670u,12055u,12056u,327737u,57u,12057u,61u,12055u,196855u,
12060u,0u,262394u,12057u,12059u,12066u,131320u,12059u,262205u,6u,12061u,12045u,262205u,6u,12062u,12035u,
262205u,6u,12063u,523u,327812u,6u,12064u,12062u,12063u,327808u,6u,12065u,12061u,12064u,196670u,12058u,
12065u,131321u,12060u,131320u,12066u,262205u,6u,12067u,12035u,262205u,6u,12068u,12045u,262205u,6u,12069u,
523u,327812u,6u,12070u,12068u,12069u,327808u,6u,12071u,12067u,12070u,196670u,12058u,12071u,131321u,12060u,
131320u,12060u,262205u,6u,12072u,12058u,327808u,6u,12073u,12054u,12072u,393281u,584u,12074u,578u,548u,
12073u,262205u,10u,12075u,12074u,262205u,6u,12076u,518u,262205u,57u,12078u,527u,196670u,12077u,12078u,
327737u,57u,12079u,61u,12077u,196855u,12082u,0u,262394u,12079u,12081u,12088u,131320u,12081u,262205u,6u,
12083u,12026u,262205u,6u,12084u,12045u,262205u,6u,12085u,524u,327812u,6u,12086u,12084u,12085u,327808u,
6u,12087u,12083u,12086u,196670u,12080u,12087u,131321u,12082u,131320u,12088u,262205u,6u,12089u,12045u,262205u,
6u,12090u,12026u,262205u,6u,12091u,524u,327812u,6u,12092u,12090u,12091u,327808u,6u,12093u,12089u,
12092u,196670u,12080u,12093u,131321u,12082u,131320u,12082u,262205u,6u,12094u,12080u,327808u,6u,12095u,12076u,
12094u,393281u,584u,12096u,578u,548u,12095u,262205u,10u,12097u,12096u,327813u,10u,12098u,12075u,12097u,
262205u,10u,12099u,12044u,327809u,10u,12100u,12099u,12098u,196670u,12044u,12100u,131321u,12049u,131320u,12049u,
262205u,6u,12101u,12045u,327808u,6u,12102u,12101u,551u,196670u,12045u,12102u,131321u,12046u,131320u,12048u,
262205u,6u,12103u,519u,262205u,6u,12104u,12035u,262205u,6u,12105u,12026u,262205u,6u,12106u,525u,
327812u,6u,12107u,12105u,12106u,327808u,6u,12108u,12104u,12107u,327808u,6u,12109u,12103u,12108u,262205u,
10u,12110u,528u,262205u,10u,12111u,12044u,327813u,10u,12112u,12110u,12111u,262205u,10u,12113u,529u,
262205u,6u,12114u,519u,262205u,6u,12115u,12035u,262205u,6u,12116u,12026u,262205u,6u,12117u,525u,
327812u,6u,12118u,12116u,12117u,327808u,6u,12119u,12115u,12118u,327808u,6u,12120u,12114u,12119u,393281u,
584u,12121u,578u,548u,12120u,262205u,10u,12122u,12121u,327813u,10u,12123u,12113u,12122u,327809u,10u,
12124u,12112u,12123u,393281u,584u,12125u,578u,548u,12109u,196670u,12125u,12124u,131321u,12039u,131320u,12039u,
262205u,6u,12126u,12035u,327808u,6u,12127u,12126u,551u,196670u,12035u,12127u,131321u,12036u,131320u,12038u,
131321u,12030u,131320u,12030u,262205u,6u,12128u,12026u,327808u,6u,12129u,12128u,551u,196670u,12026u,12129u,
131321u,12027u,131320u,12029u,65789u,65592u,327734u,2u,542u,0u,532u,196663u,9u,533u,196663u,9u,
534u,196663u,9u,535u,196663u,9u,536u,196663u,9u,537u,196663u,9u,538u,196663u,9u,539u,
196663u,9u,540u,196663u,11u,541u,131320u,543u,262203u,9u,12130u,7u,262203u,9u,12139u,7u,
196670u,12130u,548u,131321u,12131u,131320u,12131u,262390u,12133u,12134u,0u,131321u,12135u,131320u,12135u,262205u,
6u,12136u,12130u,262205u,6u,12137u,537u,327857u,57u,12138u,12136u,12137u,262394u,12138u,12132u,12133u,
131320u,12132u,196670u,12139u,548u,131321u,12140u,131320u,12140u,262390u,12142u,12143u,0u,131321u,12144u,131320u,
12144u,262205u,6u,12145u,12139u,262205u,6u,12146u,536u,327857u,57u,12147u,12145u,12146u,262394u,12147u,
12141u,12142u,131320u,12141u,262205u,6u,12148u,535u,262205u,6u,12149u,12139u,262205u,6u,12150u,12130u,
262205u,6u,12151u,540u,327812u,6u,12152u,12150u,12151u,327808u,6u,12153u,12149u,12152u,327808u,6u,
12154u,12148u,12153u,262205u,10u,12155u,541u,262205u,6u,12156u,533u,262205u,6u,12157u,12139u,262205u,
6u,12158u,538u,327812u,6u,12159u,12157u,12158u,327808u,6u,12160u,12156u,12159u,393281u,584u,12161u,
578u,548u,12160u,262205u,10u,12162u,12161u,327813u,10u,12163u,12155u,12162u,262205u,6u,12164u,534u,
262205u,6u,12165u,12130u,262205u,6u,12166u,539u,327812u,6u,12167u,12165u,12166u,327808u,6u,12168u,
12164u,12167u,393281u,584u,12169u,578u,548u,12168u,262205u,10u,12170u,12169u,327813u,10u,12171u,12163u,
12170u,393281u,584u,12172u,578u,548u,12154u,262205u,10u,12173u,12172u,327809u,10u,12174u,12173u,12171u,
393281u,584u,12175u,578u,548u,12154u,196670u,12175u,12174u,131321u,12143u,131320u,12143u,262205u,6u,12176u,
12139u,327808u,6u,12177u,12176u,551u,196670u,12139u,12177u,131321u,12140u,131320u,12142u,131321u,12134u,131320u,
12134u,262205u,6u,12178u,12130u,327808u,6u,12179u,12178u,551u,196670u,12130u,12179u,131321u,12131u,131320u,
12133u,65789u,65592u,327734u,2u,544u,0u,3u,131320u,545u,262203u,9u,12189u,7u,262203u,9u,
12257u,7u,262203u,9u,12260u,7u,262203u,9u,12263u,7u,262203u,9u,12266u,7u,262203u,9u,
12269u,7u,262203u,9u,12272u,7u,262203u,9u,12275u,7u,262203u,9u,12278u,7u,262203u,11u,
12281u,7u,262203u,58u,12282u,7u,262203u,9u,12285u,7u,262203u,9u,12288u,7u,262203u,9u,
12291u,7u,262203u,9u,12294u,7u,262203u,9u,12297u,7u,262203u,9u,12300u,7u,262203u,9u,
12303u,7u,262203u,9u,12306u,7u,262203u,9u,12311u,7u,262203u,9u,12314u,7u,262203u,9u,
12317u,7u,262203u,9u,12320u,7u,262203u,9u,12323u,7u,262203u,9u,12331u,7u,262203u,9u,
12334u,7u,262203u,9u,12337u,7u,262203u,9u,12340u,7u,262203u,9u,12343u,7u,262203u,9u,
12346u,7u,262203u,9u,12349u,7u,262203u,9u,12352u,7u,262203u,9u,12355u,7u,262203u,58u,
12358u,7u,262203u,9u,12361u,7u,262203u,9u,12364u,7u,262203u,9u,12367u,7u,262203u,9u,
12370u,7u,262203u,9u,12373u,7u,262203u,9u,12376u,7u,262203u,9u,12379u,7u,262203u,9u,
12384u,7u,262203u,9u,12387u,7u,262203u,9u,12390u,7u,262203u,9u,12393u,7u,262203u,9u,
12396u,7u,262203u,9u,12399u,7u,262203u,9u,12402u,7u,262203u,9u,12407u,7u,262203u,9u,
12410u,7u,262203u,9u,12413u,7u,262203u,9u,12421u,7u,262203u,9u,12424u,7u,262203u,9u,
12427u,7u,262203u,9u,12430u,7u,262203u,9u,12433u,7u,262203u,58u,12436u,7u,262203u,9u,
12440u,7u,262203u,9u,12443u,7u,262203u,9u,12446u,7u,262203u,9u,12449u,7u,262203u,9u,
12452u,7u,262203u,9u,12455u,7u,262203u,9u,12458u,7u,262203u,9u,12461u,7u,262203u,9u,
12464u,7u,262203u,9u,12467u,7u,262203u,9u,12470u,7u,262203u,9u,12475u,7u,262203u,9u,
12478u,7u,262203u,9u,12481u,7u,262203u,9u,12484u,7u,262203u,9u,12487u,7u,262203u,9u,
12490u,7u,262203u,9u,12493u,7u,262203u,9u,12496u,7u,262203u,9u,12499u,7u,262203u,9u,
12502u,7u,262203u,9u,12505u,7u,262203u,9u,12519u,7u,262203u,9u,12522u,7u,262203u,9u,
12525u,7u,262203u,9u,12528u,7u,262203u,9u,12531u,7u,262203u,9u,12534u,7u,262203u,11u,
12537u,7u,262203u,58u,12538u,7u,262203u,58u,12539u,7u,262203u,9u,12542u,7u,262203u,9u,
12545u,7u,262203u,9u,12548u,7u,262203u,9u,12551u,7u,262203u,9u,12554u,7u,262203u,9u,
12557u,7u,262203u,211u,12562u,7u,262203u,9u,12599u,7u,262203u,9u,12602u,7u,262203u,9u,
12605u,7u,262203u,9u,12608u,7u,262203u,9u,12611u,7u,262203u,9u,12614u,7u,262203u,9u,
12617u,7u,262203u,9u,12620u,7u,262203u,211u,12623u,7u,262203u,9u,12625u,7u,262203u,9u,
12630u,7u,262203u,9u,12633u,7u,262203u,9u,12636u,7u,262203u,9u,12639u,7u,262203u,9u,
12642u,7u,262203u,236u,12647u,7u,262203u,9u,12663u,7u,262203u,9u,12666u,7u,262203u,9u,
12669u,7u,262203u,236u,12672u,7u,262203u,9u,12674u,7u,262203u,246u,12679u,7u,262203u,9u,
12709u,7u,262203u,9u,12712u,7u,262203u,9u,12715u,7u,262203u,9u,12718u,7u,262203u,246u,
12721u,7u,262203u,9u,12723u,7u,262203u,9u,12731u,7u,262203u,9u,12734u,7u,262203u,9u,
12737u,7u,262203u,9u,12740u,7u,262203u,9u,12743u,7u,262203u,9u,12746u,7u,262203u,58u,
12749u,7u,262203u,9u,12755u,7u,262203u,9u,12758u,7u,262203u,9u,12761u,7u,262203u,9u,
12764u,7u,262203u,9u,12767u,7u,262203u,9u,12770u,7u,262203u,9u,12773u,7u,262203u,9u,
12776u,7u,262203u,9u,12779u,7u,262203u,9u,12782u,7u,262203u,9u,12785u,7u,262203u,9u,
12788u,7u,262203u,58u,12791u,7u,262203u,283u,12794u,7u,262203u,9u,12823u,7u,262203u,9u,
12826u,7u,262203u,9u,12829u,7u,262203u,9u,12832u,7u,262203u,9u,12835u,7u,262203u,9u,
12838u,7u,262203u,283u,12841u,7u,262203u,9u,12843u,7u,262203u,9u,12860u,7u,262203u,9u,
12863u,7u,262203u,9u,12866u,7u,262203u,9u,12869u,7u,262203u,11u,12872u,7u,262203u,58u,
12873u,7u,262203u,58u,12874u,7u,262203u,58u,12875u,7u,262203u,9u,12881u,7u,262203u,9u,
12884u,7u,262203u,9u,12887u,7u,262203u,9u,12890u,7u,262203u,9u,12893u,7u,262203u,9u,
12896u,7u,262203u,9u,12899u,7u,262203u,58u,12902u,7u,262203u,318u,12905u,7u,262203u,9u,
12928u,7u,262203u,9u,12931u,7u,262203u,9u,12934u,7u,262203u,9u,12937u,7u,262203u,9u,
12940u,7u,262203u,9u,12943u,7u,262203u,318u,12946u,7u,262203u,11u,12948u,7u,262203u,58u,
12949u,7u,262203u,9u,12952u,7u,262203u,9u,12955u,7u,262203u,9u,12958u,7u,262203u,9u,
12961u,7u,262203u,9u,12964u,7u,262203u,339u,12969u,7u,262203u,9u,12998u,7u,262203u,9u,
13001u,7u,262203u,9u,13004u,7u,262203u,339u,13007u,7u,262203u,9u,13009u,7u,262203u,58u,
13012u,7u,262203u,9u,13013u,7u,262203u,339u,13018u,7u,262203u,9u,13044u,7u,262203u,9u,
13047u,7u,262203u,9u,13050u,7u,262203u,339u,13053u,7u,262203u,339u,13057u,7u,262203u,9u,
13083u,7u,262203u,9u,13086u,7u,262203u,9u,13089u,7u,262203u,339u,13092u,7u,262203u,9u,
13094u,7u,262203u,339u,13099u,7u,262203u,9u,13125u,7u,262203u,9u,13128u,7u,262203u,9u,
13131u,7u,262203u,9u,13134u,7u,262203u,339u,13137u,7u,262203u,9u,13139u,7u,262203u,339u,
13144u,7u,262203u,9u,13170u,7u,262203u,9u,13173u,7u,262203u,9u,13176u,7u,262203u,9u,
13179u,7u,262203u,339u,13182u,7u,262203u,339u,13186u,7u,262203u,9u,13212u,7u,262203u,9u,
13215u,7u,262203u,9u,13218u,7u,262203u,9u,13221u,7u,262203u,9u,13224u,7u,262203u,9u,
13227u,7u,262203u,339u,13230u,7u,262203u,9u,13232u,7u,262203u,9u,13235u,7u,262203u,339u,
13240u,7u,262203u,9u,13266u,7u,262203u,9u,13269u,7u,262203u,9u,13272u,7u,262203u,9u,
13275u,7u,262203u,9u,13278u,7u,262203u,9u,13281u,7u,262203u,9u,13284u,7u,262203u,9u,
13287u,7u,262203u,339u,13290u,7u,262203u,9u,13292u,7u,262203u,339u,13297u,7u,262203u,9u,
13326u,7u,262203u,9u,13329u,7u,262203u,9u,13332u,7u,262203u,9u,13335u,7u,262203u,9u,
13338u,7u,262203u,339u,13341u,7u,262203u,58u,13343u,7u,262203u,9u,13344u,7u,262203u,439u,
13349u,7u,262203u,9u,13366u,7u,262203u,9u,13369u,7u,262203u,9u,13372u,7u,262203u,9u,
13375u,7u,262203u,9u,13378u,7u,262203u,9u,13381u,7u,262203u,9u,13384u,7u,262203u,9u,
13387u,7u,262203u,439u,13390u,7u,262203u,58u,13392u,7u,262203u,9u,13393u,7u,262203u,9u,
13407u,7u,262203u,9u,13410u,7u,262203u,9u,13413u,7u,262203u,9u,13416u,7u,262203u,9u,
13419u,7u,262203u,9u,13422u,7u,262203u,9u,13425u,7u,262203u,11u,13428u,7u,262203u,11u,
13429u,7u,262203u,9u,13430u,7u,262203u,58u,13433u,7u,262203u,9u,13436u,7u,262203u,9u,
13439u,7u,262203u,9u,13447u,7u,262203u,9u,13450u,7u,262203u,9u,13453u,7u,262203u,9u,
13456u,7u,262203u,11u,13459u,7u,262203u,9u,13462u,7u,262203u,9u,13465u,7u,262203u,9u,
13468u,7u,262203u,9u,13471u,7u,262203u,9u,13474u,7u,262203u,9u,13482u,7u,262203u,9u,
13485u,7u,262203u,9u,13488u,7u,262203u,9u,13491u,7u,262203u,9u,13494u,7u,262203u,11u,
13497u,7u,262203u,9u,13500u,7u,262203u,9u,13503u,7u,262203u,9u,13506u,7u,262203u,9u,
13509u,7u,262203u,9u,13512u,7u,262203u,9u,13515u,7u,262203u,9u,13518u,7u,262203u,9u,
13523u,7u,262203u,9u,13526u,7u,262203u,9u,13529u,7u,262203u,9u,13546u,7u,262203u,9u,
13549u,7u,262203u,9u,13552u,7u,262203u,9u,13555u,7u,262203u,9u,13558u,7u,262203u,9u,
13561u,7u,262203u,9u,13564u,7u,262203u,9u,13567u,7u,262203u,9u,13570u,7u,262203u,58u,
13573u,7u,262203u,58u,13574u,7u,262203u,11u,13575u,7u,262203u,11u,13576u,7u,262203u,9u,
13582u,7u,262203u,9u,13585u,7u,262203u,9u,13588u,7u,262203u,9u,13591u,7u,262203u,9u,
13594u,7u,262203u,9u,13597u,7u,262203u,9u,13600u,7u,262203u,9u,13603u,7u,262203u,11u,
13606u,7u,393281u,12184u,12185u,12183u,548u,551u,262205u,6u,12186u,12185u,196670u,550u,12186u,393281u,
12184u,12187u,12183u,548u,2490u,262205u,6u,12188u,12187u,196670u,552u,12188u,196670u,12189u,548u,131321u,
12190u,131320u,12190u,262390u,12192u,12193u,0u,131321u,12194u,131320u,12194u,262205u,6u,12195u,12189u,262205u,
6u,12196u,550u,262205u,6u,12197u,552u,327812u,6u,12198u,12196u,12197u,327857u,57u,12199u,12195u,
12198u,262394u,12199u,12191u,12192u,131320u,12191u,262205u,6u,12200u,12189u,262205u,6u,12201u,550u,327815u,
6u,12202u,12200u,12201u,196670u,549u,12202u,262205u,6u,12203u,12189u,262205u,6u,12204u,550u,327819u,
6u,12205u,12203u,12204u,196670u,547u,12205u,393281u,12184u,12206u,12183u,548u,548u,262205u,6u,12207u,
12206u,196855u,12250u,0u,5570811u,12207u,12249u,0u,12208u,1u,12209u,2u,12210u,3u,12211u,4u,
12212u,5u,12213u,6u,12214u,7u,12215u,8u,12216u,9u,12217u,10u,12218u,11u,12219u,12u,
12220u,13u,12221u,14u,12222u,15u,12223u,16u,12224u,17u,12225u,18u,12226u,19u,12227u,20u,
12228u,21u,12229u,22u,12230u,23u,12231u,24u,12232u,25u,12233u,26u,12234u,27u,12235u,28u,
12236u,29u,12237u,30u,12238u,31u,12239u,32u,12240u,33u,12241u,34u,12242u,35u,12243u,36u,
12244u,37u,12245u,38u,12246u,39u,12247u,40u,12248u,131320u,12249u,131321u,12250u,131320u,12208u,393281u,
12184u,12251u,12183u,548u,2754u,262205u,6u,12252u,12251u,262268u,10u,12253u,12252u,393281u,12184u,12254u,
12183u,548u,3472u,262205u,6u,12255u,12254u,327851u,57u,12256u,12255u,548u,393281u,12184u,12258u,12183u,
548u,2404u,262205u,6u,12259u,12258u,196670u,12257u,12259u,393281u,12184u,12261u,12183u,548u,3118u,262205u,
6u,12262u,12261u,196670u,12260u,12262u,393281u,12184u,12264u,12183u,548u,2510u,262205u,6u,12265u,12264u,
196670u,12263u,12265u,393281u,12184u,12267u,12183u,548u,2624u,262205u,6u,12268u,12267u,196670u,12266u,12268u,
393281u,12184u,12270u,12183u,548u,2750u,262205u,6u,12271u,12270u,196670u,12269u,12271u,393281u,12184u,12273u,
12183u,548u,2409u,262205u,6u,12274u,12273u,196670u,12272u,12274u,393281u,12184u,12276u,12183u,548u,2395u,
262205u,6u,12277u,12276u,196670u,12275u,12277u,393281u,12184u,12279u,12183u,548u,3377u,262205u,6u,12280u,
12279u,196670u,12278u,12280u,196670u,12281u,12253u,196670u,12282u,12256u,917561u,2u,12283u,95u,12257u,12260u,
12263u,12266u,12269u,12272u,12275u,12278u,12281u,12282u,131321u,12250u,131320u,12209u,393281u,12184u,12286u,12183u,
548u,2404u,262205u,6u,12287u,12286u,196670u,12285u,12287u,393281u,12184u,12289u,12183u,548u,3118u,262205u,
6u,12290u,12289u,196670u,12288u,12290u,393281u,12184u,12292u,12183u,548u,2510u,262205u,6u,12293u,12292u,
196670u,12291u,12293u,393281u,12184u,12295u,12183u,548u,2624u,262205u,6u,12296u,12295u,196670u,12294u,12296u,
393281u,12184u,12298u,12183u,548u,2750u,262205u,6u,12299u,12298u,196670u,12297u,12299u,393281u,12184u,12301u,
12183u,548u,2409u,262205u,6u,12302u,12301u,196670u,12300u,12302u,393281u,12184u,12304u,12183u,548u,2395u,
262205u,6u,12305u,12304u,196670u,12303u,12305u,393281u,12184u,12307u,12183u,548u,3377u,262205u,6u,12308u,
12307u,196670u,12306u,12308u,786489u,2u,12309u,106u,12285u,12288u,12291u,12294u,12297u,12300u,12303u,12306u,
131321u,12250u,131320u,12210u,393281u,12184u,12312u,12183u,548u,2404u,262205u,6u,12313u,12312u,196670u,12311u,
12313u,393281u,12184u,12315u,12183u,548u,3118u,262205u,6u,12316u,12315u,196670u,12314u,12316u,393281u,12184u,
12318u,12183u,548u,2510u,262205u,6u,12319u,12318u,196670u,12317u,12319u,393281u,12184u,12321u,12183u,548u,
2624u,262205u,6u,12322u,12321u,196670u,12320u,12322u,393281u,12184u,12324u,12183u,548u,2750u,262205u,6u,
12325u,12324u,196670u,12323u,12325u,589881u,2u,12326u,114u,12311u,12314u,12317u,12320u,12323u,131321u,12250u,
131320u,12211u,393281u,12184u,12328u,12183u,548u,3472u,262205u,6u,12329u,12328u,327851u,57u,12330u,12329u,
548u,393281u,12184u,12332u,12183u,548u,2404u,262205u,6u,12333u,12332u,196670u,12331u,12333u,393281u,12184u,
12335u,12183u,548u,3118u,262205u,6u,12336u,12335u,196670u,12334u,12336u,393281u,12184u,12338u,12183u,548u,
2510u,262205u,6u,12339u,12338u,196670u,12337u,12339u,393281u,12184u,12341u,12183u,548u,2624u,262205u,6u,
12342u,12341u,196670u,12340u,12342u,393281u,12184u,12344u,12183u,548u,2750u,262205u,6u,12345u,12344u,196670u,
12343u,12345u,393281u,12184u,12347u,12183u,548u,2409u,262205u,6u,12348u,12347u,196670u,12346u,12348u,393281u,
12184u,12350u,12183u,548u,2395u,262205u,6u,12351u,12350u,196670u,12349u,12351u,393281u,12184u,12353u,12183u,
548u,3377u,262205u,6u,12354u,12353u,196670u,12352u,12354u,393281u,12184u,12356u,12183u,548u,2754u,262205u,
6u,12357u,12356u,196670u,12355u,12357u,196670u,12358u,12330u,917561u,2u,12359u,127u,12331u,12334u,12337u,
12340u,12343u,12346u,12349u,12352u,12355u,12358u,131321u,12250u,131320u,12212u,393281u,12184u,12362u,12183u,548u,
2404u,262205u,6u,12363u,12362u,196670u,12361u,12363u,393281u,12184u,12365u,12183u,548u,3118u,262205u,6u,
12366u,12365u,196670u,12364u,12366u,393281u,12184u,12368u,12183u,548u,2510u,262205u,6u,12369u,12368u,196670u,
12367u,12369u,393281u,12184u,12371u,12183u,548u,2624u,262205u,6u,12372u,12371u,196670u,12370u,12372u,393281u,
12184u,12374u,12183u,548u,2750u,262205u,6u,12375u,12374u,196670u,12373u,12375u,393281u,12184u,12377u,12183u,
548u,2409u,262205u,6u,12378u,12377u,196670u,12376u,12378u,393281u,12184u,12380u,12183u,548u,2395u,262205u,
6u,12381u,12380u,196670u,12379u,12381u,720953u,2u,12382u,137u,12361u,12364u,12367u,12370u,12373u,12376u,
12379u,131321u,12250u,131320u,12213u,393281u,12184u,12385u,12183u,548u,2404u,262205u,6u,12386u,12385u,196670u,
12384u,12386u,393281u,12184u,12388u,12183u,548u,3118u,262205u,6u,12389u,12388u,196670u,12387u,12389u,393281u,
12184u,12391u,12183u,548u,2510u,262205u,6u,12392u,12391u,196670u,12390u,12392u,393281u,12184u,12394u,12183u,
548u,2624u,262205u,6u,12395u,12394u,196670u,12393u,12395u,393281u,12184u,12397u,12183u,548u,2750u,262205u,
6u,12398u,12397u,196670u,12396u,12398u,393281u,12184u,12400u,12183u,548u,2409u,262205u,6u,12401u,12400u,
196670u,12399u,12401u,393281u,12184u,12403u,12183u,548u,2395u,262205u,6u,12404u,12403u,196670u,12402u,12404u,
720953u,2u,12405u,146u,12384u,12387u,12390u,12393u,12396u,12399u,12402u,131321u,12250u,131320u,12214u,393281u,
12184u,12408u,12183u,548u,2404u,262205u,6u,12409u,12408u,196670u,12407u,12409u,393281u,12184u,12411u,12183u,
548u,3118u,262205u,6u,12412u,12411u,196670u,12410u,12412u,393281u,12184u,12414u,12183u,548u,2510u,262205u,
6u,12415u,12414u,196670u,12413u,12415u,458809u,2u,12416u,151u,12407u,12410u,12413u,131321u,12250u,131320u,
12215u,393281u,12184u,12418u,12183u,548u,2409u,262205u,6u,12419u,12418u,327851u,57u,12420u,12419u,548u,
393281u,12184u,12422u,12183u,548u,2404u,262205u,6u,12423u,12422u,196670u,12421u,12423u,393281u,12184u,12425u,
12183u,548u,3118u,262205u,6u,12426u,12425u,196670u,12424u,12426u,393281u,12184u,12428u,12183u,548u,2510u,
262205u,6u,12429u,12428u,196670u,12427u,12429u,393281u,12184u,12431u,12183u,548u,2624u,262205u,6u,12432u,
12431u,196670u,12430u,12432u,393281u,12184u,12434u,12183u,548u,2750u,262205u,6u,12435u,12434u,196670u,12433u,
12435u,196670u,12436u,12420u,655417u,2u,12437u,160u,12421u,12424u,12427u,12430u,12433u,12436u,131321u,12250u,
131320u,12216u,393281u,12184u,12441u,12183u,548u,2404u,262205u,6u,12442u,12441u,196670u,12440u,12442u,393281u,
12184u,12444u,12183u,548u,3118u,262205u,6u,12445u,12444u,196670u,12443u,12445u,393281u,12184u,12447u,12183u,
548u,2510u,262205u,6u,12448u,12447u,196670u,12446u,12448u,393281u,12184u,12450u,12183u,548u,2624u,262205u,
6u,12451u,12450u,196670u,12449u,12451u,393281u,12184u,12453u,12183u,548u,2750u,262205u,6u,12454u,12453u,
196670u,12452u,12454u,393281u,12184u,12456u,12183u,548u,2409u,262205u,6u,12457u,12456u,196670u,12455u,12457u,
393281u,12184u,12459u,12183u,548u,2395u,262205u,6u,12460u,12459u,196670u,12458u,12460u,393281u,12184u,12462u,
12183u,548u,3377u,262205u,6u,12463u,12462u,196670u,12461u,12463u,393281u,12184u,12465u,12183u,548u,2754u,
262205u,6u,12466u,12465u,196670u,12464u,12466u,393281u,12184u,12468u,12183u,548u,3472u,262205u,6u,12469u,
12468u,196670u,12467u,12469u,393281u,12184u,12471u,12183u,548u,12439u,262205u,6u,12472u,12471u,196670u,12470u,
12472u,983097u,2u,12473u,174u,12440u,12443u,12446u,12449u,12452u,12455u,12458u,12461u,12464u,12467u,12470u,
131321u,12250u,131320u,12217u,393281u,12184u,12476u,12183u,548u,2404u,262205u,6u,12477u,12476u,196670u,12475u,
12477u,393281u,12184u,12479u,12183u,548u,3118u,262205u,6u,12480u,12479u,196670u,12478u,12480u,393281u,12184u,
12482u,12183u,548u,2510u,262205u,6u,12483u,12482u,196670u,12481u,12483u,393281u,12184u,12485u,12183u,548u,
2624u,262205u,6u,12486u,12485u,196670u,12484u,12486u,393281u,12184u,12488u,12183u,548u,2750u,262205u,6u,
12489u,12488u,196670u,12487u,12489u,393281u,12184u,12491u,12183u,548u,2409u,262205u,6u,12492u,12491u,196670u,
12490u,12492u,393281u,12184u,12494u,12183u,548u,2395u,262205u,6u,12495u,12494u,196670u,12493u,12495u,393281u,
12184u,12497u,12183u,548u,3377u,262205u,6u,12498u,12497u,196670u,12496u,12498u,393281u,12184u,12500u,12183u,
548u,2754u,262205u,6u,12501u,12500u,196670u,12499u,12501u,393281u,12184u,12503u,12183u,548u,3472u,262205u,
6u,12504u,12503u,196670u,12502u,12504u,393281u,12184u,12506u,12183u,548u,12439u,262205u,6u,12507u,12506u,
196670u,12505u,12507u,983097u,2u,12508u,187u,12475u,12478u,12481u,12484u,12487u,12490u,12493u,12496u,12499u,
12502u,12505u,131321u,12250u,131320u,12218u,393281u,12184u,12510u,12183u,548u,2395u,262205u,6u,12511u,12510u,
262268u,10u,12512u,12511u,393281u,12184u,12513u,12183u,548u,3377u,262205u,6u,12514u,12513u,327851u,57u,
12515u,12514u,548u,393281u,12184u,12516u,12183u,548u,2754u,262205u,6u,12517u,12516u,327851u,57u,12518u,
12517u,548u,393281u,12184u,12520u,12183u,548u,2404u,262205u,6u,12521u,12520u,196670u,12519u,12521u,393281u,
12184u,12523u,12183u,548u,3118u,262205u,6u,12524u,12523u,196670u,12522u,12524u,393281u,12184u,12526u,12183u,
548u,2510u,262205u,6u,12527u,12526u,196670u,12525u,12527u,393281u,12184u,12529u,12183u,548u,2624u,262205u,
6u,12530u,12529u,196670u,12528u,12530u,393281u,12184u,12532u,12183u,548u,2750u,262205u,6u,12533u,12532u,
196670u,12531u,12533u,393281u,12184u,12535u,12183u,548u,2409u,262205u,6u,12536u,12535u,196670u,12534u,12536u,
196670u,12537u,12512u,196670u,12538u,12515u,196670u,12539u,12518u,852025u,2u,12540u,199u,12519u,12522u,12525u,
12528u,12531u,12534u,12537u,12538u,12539u,131321u,12250u,131320u,12219u,393281u,12184u,12543u,12183u,548u,2404u,
262205u,6u,12544u,12543u,196670u,12542u,12544u,393281u,12184u,12546u,12183u,548u,3118u,262205u,6u,12547u,
12546u,196670u,12545u,12547u,393281u,12184u,12549u,12183u,548u,2510u,262205u,6u,12550u,12549u,196670u,12548u,
12550u,393281u,12184u,12552u,12183u,548u,2624u,262205u,6u,12553u,12552u,196670u,12551u,12553u,393281u,12184u,
12555u,12183u,548u,2750u,262205u,6u,12556u,12555u,196670u,12554u,12556u,393281u,12184u,12558u,12183u,548u,
2409u,262205u,6u,12559u,12558u,196670u,12557u,12559u,655417u,2u,12560u,208u,12542u,12545u,12548u,12551u,
12554u,12557u,131321u,12250u,131320u,12220u,393281u,12184u,12563u,12183u,548u,2754u,262205u,6u,12564u,12563u,
393281u,12184u,12565u,12183u,548u,3472u,262205u,6u,12566u,12565u,393281u,12184u,12567u,12183u,548u,12439u,
262205u,6u,12568u,12567u,393281u,12184u,12570u,12183u,548u,12569u,262205u,6u,12571u,12570u,393281u,12184u,
12573u,12183u,548u,12572u,262205u,6u,12574u,12573u,393281u,12184u,12576u,12183u,548u,12575u,262205u,6u,
12577u,12576u,393281u,12184u,12579u,12183u,548u,12578u,262205u,6u,12580u,12579u,393281u,12184u,12582u,12183u,
548u,12581u,262205u,6u,12583u,12582u,393281u,12184u,12585u,12183u,548u,12584u,262205u,6u,12586u,12585u,
393281u,12184u,12588u,12183u,548u,12587u,262205u,6u,12589u,12588u,393281u,12184u,12591u,12183u,548u,12590u,
262205u,6u,12592u,12591u,393281u,12184u,12594u,12183u,548u,12593u,262205u,6u,12595u,12594u,327851u,57u,
12596u,12595u,548u,983120u,210u,12597u,12564u,12566u,12568u,12571u,12574u,12577u,12580u,12583u,12586u,12589u,
12592u,12596u,196670u,12562u,12597u,393281u,12184u,12600u,12183u,548u,2404u,262205u,6u,12601u,12600u,196670u,
12599u,12601u,393281u,12184u,12603u,12183u,548u,3118u,262205u,6u,12604u,12603u,196670u,12602u,12604u,393281u,
12184u,12606u,12183u,548u,2510u,262205u,6u,12607u,12606u,196670u,12605u,12607u,393281u,12184u,12609u,12183u,
548u,2624u,262205u,6u,12610u,12609u,196670u,12608u,12610u,393281u,12184u,12612u,12183u,548u,2750u,262205u,
6u,12613u,12612u,196670u,12611u,12613u,393281u,12184u,12615u,12183u,548u,2409u,262205u,6u,12616u,12615u,
196670u,12614u,12616u,393281u,12184u,12618u,12183u,548u,2395u,262205u,6u,12619u,12618u,196670u,12617u,12619u,
393281u,12184u,12621u,12183u,548u,3377u,262205u,6u,12622u,12621u,196670u,12620u,12622u,262205u,210u,12624u,
12562u,196670u,12623u,12624u,393281u,12184u,12626u,12183u,548u,12598u,262205u,6u,12627u,12626u,196670u,12625u,
12627u,917561u,2u,12628u,223u,12599u,12602u,12605u,12608u,12611u,12614u,12617u,12620u,12623u,12625u,131321u,
12250u,131320u,12221u,393281u,12184u,12631u,12183u,548u,2404u,262205u,6u,12632u,12631u,196670u,12630u,12632u,
393281u,12184u,12634u,12183u,548u,3118u,262205u,6u,12635u,12634u,196670u,12633u,12635u,393281u,12184u,12637u,
12183u,548u,2510u,262205u,6u,12638u,12637u,196670u,12636u,12638u,393281u,12184u,12640u,12183u,548u,2624u,
262205u,6u,12641u,12640u,196670u,12639u,12641u,393281u,12184u,12643u,12183u,548u,2750u,262205u,6u,12644u,
12643u,196670u,12642u,12644u,589881u,2u,12645u,230u,12630u,12633u,12636u,12639u,12642u,131321u,12250u,131320u,
12222u,393281u,12184u,12648u,12183u,548u,2624u,262205u,6u,12649u,12648u,393281u,12184u,12650u,12183u,548u,
2750u,262205u,6u,12651u,12650u,393281u,12184u,12652u,12183u,548u,2409u,262205u,6u,12653u,12652u,393281u,
12184u,12654u,12183u,548u,2395u,262205u,6u,12655u,12654u,393281u,12184u,12656u,12183u,548u,3377u,262205u,
6u,12657u,12656u,393281u,12184u,12658u,12183u,548u,2754u,262205u,6u,12659u,12658u,393281u,12184u,12660u,
12183u,548u,3472u,262205u,6u,12661u,12660u,655440u,235u,12662u,12649u,12651u,12653u,12655u,12657u,12659u,
12661u,196670u,12647u,12662u,393281u,12184u,12664u,12183u,548u,2404u,262205u,6u,12665u,12664u,196670u,12663u,
12665u,393281u,12184u,12667u,12183u,548u,3118u,262205u,6u,12668u,12667u,196670u,12666u,12668u,393281u,12184u,
12670u,12183u,548u,2510u,262205u,6u,12671u,12670u,196670u,12669u,12671u,262205u,235u,12673u,12647u,196670u,
12672u,12673u,393281u,12184u,12675u,12183u,548u,12439u,262205u,6u,12676u,12675u,196670u,12674u,12676u,589881u,
2u,12677u,243u,12663u,12666u,12669u,12672u,12674u,131321u,12250u,131320u,12223u,393281u,12184u,12680u,12183u,
548u,2750u,262205u,6u,12681u,12680u,393281u,12184u,12682u,12183u,548u,2409u,262205u,6u,12683u,12682u,
393281u,12184u,12684u,12183u,548u,2395u,262205u,6u,12685u,12684u,393281u,12184u,12686u,12183u,548u,3377u,
262205u,6u,12687u,12686u,393281u,12184u,12688u,12183u,548u,2754u,262205u,6u,12689u,12688u,393281u,12184u,
12690u,12183u,548u,3472u,262205u,6u,12691u,12690u,393281u,12184u,12692u,12183u,548u,12439u,262205u,6u,
12693u,12692u,393281u,12184u,12694u,12183u,548u,12569u,262205u,6u,12695u,12694u,393281u,12184u,12696u,12183u,
548u,12572u,262205u,6u,12697u,12696u,393281u,12184u,12698u,12183u,548u,12575u,262205u,6u,12699u,12698u,
393281u,12184u,12700u,12183u,548u,12578u,262205u,6u,12701u,12700u,393281u,12184u,12702u,12183u,548u,12581u,
262205u,6u,12703u,12702u,327851u,57u,12704u,12703u,548u,393281u,12184u,12705u,12183u,548u,12584u,262205u,
6u,12706u,12705u,327851u,57u,12707u,12706u,548u,1048656u,245u,12708u,12681u,12683u,12685u,12687u,12689u,
12691u,12693u,12695u,12697u,12699u,12701u,12704u,12707u,196670u,12679u,12708u,393281u,12184u,12710u,12183u,548u,
2404u,262205u,6u,12711u,12710u,196670u,12709u,12711u,393281u,12184u,12713u,12183u,548u,3118u,262205u,6u,
12714u,12713u,196670u,12712u,12714u,393281u,12184u,12716u,12183u,548u,2510u,262205u,6u,12717u,12716u,196670u,
12715u,12717u,393281u,12184u,12719u,12183u,548u,2624u,262205u,6u,12720u,12719u,196670u,12718u,12720u,262205u,
245u,12722u,12679u,196670u,12721u,12722u,393281u,12184u,12724u,12183u,548u,12587u,262205u,6u,12725u,12724u,
196670u,12723u,12725u,655417u,2u,12726u,254u,12709u,12712u,12715u,12718u,12721u,12723u,131321u,12250u,131320u,
12224u,393281u,12184u,12728u,12183u,548u,2395u,262205u,6u,12729u,12728u,327851u,57u,12730u,12729u,548u,
393281u,12184u,12732u,12183u,548u,2404u,262205u,6u,12733u,12732u,196670u,12731u,12733u,393281u,12184u,12735u,
12183u,548u,3118u,262205u,6u,12736u,12735u,196670u,12734u,12736u,393281u,12184u,12738u,12183u,548u,2510u,
262205u,6u,12739u,12738u,196670u,12737u,12739u,393281u,12184u,12741u,12183u,548u,2624u,262205u,6u,12742u,
12741u,196670u,12740u,12742u,393281u,12184u,12744u,12183u,548u,2750u,262205u,6u,12745u,12744u,196670u,12743u,
12745u,393281u,12184u,12747u,12183u,548u,2409u,262205u,6u,12748u,12747u,196670u,12746u,12748u,196670u,12749u,
12730u,720953u,2u,12750u,264u,12731u,12734u,12737u,12740u,12743u,12746u,12749u,131321u,12250u,131320u,12225u,
393281u,12184u,12752u,12183u,548u,12572u,262205u,6u,12753u,12752u,327851u,57u,12754u,12753u,548u,393281u,
12184u,12756u,12183u,548u,2404u,262205u,6u,12757u,12756u,196670u,12755u,12757u,393281u,12184u,12759u,12183u,
548u,3118u,262205u,6u,12760u,12759u,196670u,12758u,12760u,393281u,12184u,12762u,12183u,548u,2510u,262205u,
6u,12763u,12762u,196670u,12761u,12763u,393281u,12184u,12765u,12183u,548u,2624u,262205u,6u,12766u,12765u,
196670u,12764u,12766u,393281u,12184u,12768u,12183u,548u,2750u,262205u,6u,12769u,12768u,196670u,12767u,12769u,
393281u,12184u,12771u,12183u,548u,2409u,262205u,6u,12772u,12771u,196670u,12770u,12772u,393281u,12184u,12774u,
12183u,548u,2395u,262205u,6u,12775u,12774u,196670u,12773u,12775u,393281u,12184u,12777u,12183u,548u,3377u,
262205u,6u,12778u,12777u,196670u,12776u,12778u,393281u,12184u,12780u,12183u,548u,2754u,262205u,6u,12781u,
12780u,196670u,12779u,12781u,393281u,12184u,12783u,12183u,548u,3472u,262205u,6u,12784u,12783u,196670u,12782u,
12784u,393281u,12184u,12786u,12183u,548u,12439u,262205u,6u,12787u,12786u,196670u,12785u,12787u,393281u,12184u,
12789u,12183u,548u,12569u,262205u,6u,12790u,12789u,196670u,12788u,12790u,196670u,12791u,12754u,1114169u,2u,
12792u,280u,12755u,12758u,12761u,12764u,12767u,12770u,12773u,12776u,12779u,12782u,12785u,12788u,12791u,131321u,
12250u,131320u,12226u,393281u,12184u,12795u,12183u,548u,2395u,262205u,6u,12796u,12795u,393281u,12184u,12797u,
12183u,548u,3377u,262205u,6u,12798u,12797u,393281u,12184u,12799u,12183u,548u,2754u,262205u,6u,12800u,
12799u,393281u,12184u,12801u,12183u,548u,3472u,262205u,6u,12802u,12801u,393281u,12184u,12803u,12183u,548u,
12439u,262205u,6u,12804u,12803u,393281u,12184u,12805u,12183u,548u,12569u,262205u,6u,12806u,12805u,393281u,
12184u,12807u,12183u,548u,12572u,262205u,6u,12808u,12807u,393281u,12184u,12809u,12183u,548u,12575u,262205u,
6u,12810u,12809u,393281u,12184u,12811u,12183u,548u,12578u,262205u,6u,12812u,12811u,393281u,12184u,12813u,
12183u,548u,12581u,262205u,6u,12814u,12813u,393281u,12184u,12815u,12183u,548u,12584u,262205u,6u,12816u,
12815u,393281u,12184u,12817u,12183u,548u,12587u,262205u,6u,12818u,12817u,393281u,12184u,12819u,12183u,548u,
12590u,262205u,6u,12820u,12819u,327851u,57u,12821u,12820u,548u,1048656u,282u,12822u,12796u,12798u,12800u,
12802u,12804u,12806u,12808u,12810u,12812u,12814u,12816u,12818u,12821u,196670u,12794u,12822u,393281u,12184u,12824u,
12183u,548u,2404u,262205u,6u,12825u,12824u,196670u,12823u,12825u,393281u,12184u,12827u,12183u,548u,3118u,
262205u,6u,12828u,12827u,196670u,12826u,12828u,393281u,12184u,12830u,12183u,548u,2510u,262205u,6u,12831u,
12830u,196670u,12829u,12831u,393281u,12184u,12833u,12183u,548u,2624u,262205u,6u,12834u,12833u,196670u,12832u,
12834u,393281u,12184u,12836u,12183u,548u,2750u,262205u,6u,12837u,12836u,196670u,12835u,12837u,393281u,12184u,
12839u,12183u,548u,2409u,262205u,6u,12840u,12839u,196670u,12838u,12840u,262205u,282u,12842u,12794u,196670u,
12841u,12842u,393281u,12184u,12844u,12183u,548u,12593u,262205u,6u,12845u,12844u,196670u,12843u,12845u,786489u,
2u,12846u,293u,12823u,12826u,12829u,12832u,12835u,12838u,12841u,12843u,131321u,12250u,131320u,12227u,393281u,
12184u,12848u,12183u,548u,2750u,262205u,6u,12849u,12848u,262268u,10u,12850u,12849u,393281u,12184u,12851u,
12183u,548u,2409u,262205u,6u,12852u,12851u,327851u,57u,12853u,12852u,548u,393281u,12184u,12854u,12183u,
548u,2395u,262205u,6u,12855u,12854u,327851u,57u,12856u,12855u,548u,393281u,12184u,12857u,12183u,548u,
3377u,262205u,6u,12858u,12857u,327851u,57u,12859u,12858u,548u,393281u,12184u,12861u,12183u,548u,2404u,
262205u,6u,12862u,12861u,196670u,12860u,12862u,393281u,12184u,12864u,12183u,548u,3118u,262205u,6u,12865u,
12864u,196670u,12863u,12865u,393281u,12184u,12867u,12183u,548u,2510u,262205u,6u,12868u,12867u,196670u,12866u,
12868u,393281u,12184u,12870u,12183u,548u,2624u,262205u,6u,12871u,12870u,196670u,12869u,12871u,196670u,12872u,
12850u,196670u,12873u,12853u,196670u,12874u,12856u,196670u,12875u,12859u,786489u,2u,12876u,304u,12860u,12863u,
12866u,12869u,12872u,12873u,12874u,12875u,131321u,12250u,131320u,12228u,393281u,12184u,12878u,12183u,548u,3377u,
262205u,6u,12879u,12878u,327851u,57u,12880u,12879u,548u,393281u,12184u,12882u,12183u,548u,2404u,262205u,
6u,12883u,12882u,196670u,12881u,12883u,393281u,12184u,12885u,12183u,548u,3118u,262205u,6u,12886u,12885u,
196670u,12884u,12886u,393281u,12184u,12888u,12183u,548u,2510u,262205u,6u,12889u,12888u,196670u,12887u,12889u,
393281u,12184u,12891u,12183u,548u,2624u,262205u,6u,12892u,12891u,196670u,12890u,12892u,393281u,12184u,12894u,
12183u,548u,2750u,262205u,6u,12895u,12894u,196670u,12893u,12895u,393281u,12184u,12897u,12183u,548u,2409u,
262205u,6u,12898u,12897u,196670u,12896u,12898u,393281u,12184u,12900u,12183u,548u,2395u,262205u,6u,12901u,
12900u,196670u,12899u,12901u,196670u,12902u,12880u,786489u,2u,12903u,315u,12881u,12884u,12887u,12890u,12893u,
12896u,12899u,12902u,131321u,12250u,131320u,12229u,393281u,12184u,12906u,12183u,548u,2395u,262205u,6u,12907u,
12906u,393281u,12184u,12908u,12183u,548u,3377u,262205u,6u,12909u,12908u,393281u,12184u,12910u,12183u,548u,
2754u,262205u,6u,12911u,12910u,393281u,12184u,12912u,12183u,548u,3472u,262205u,6u,12913u,12912u,327851u,
57u,12914u,12913u,548u,393281u,12184u,12915u,12183u,548u,12439u,262205u,6u,12916u,12915u,327851u,57u,
12917u,12916u,548u,393281u,12184u,12918u,12183u,548u,12569u,262205u,6u,12919u,12918u,327851u,57u,12920u,
12919u,548u,589904u,317u,12921u,12907u,12909u,12911u,12914u,12917u,12920u,196670u,12905u,12921u,393281u,12184u,
12922u,12183u,548u,12572u,262205u,6u,12923u,12922u,262268u,10u,12924u,12923u,393281u,12184u,12925u,12183u,
548u,12575u,262205u,6u,12926u,12925u,327851u,57u,12927u,12926u,548u,393281u,12184u,12929u,12183u,548u,
2404u,262205u,6u,12930u,12929u,196670u,12928u,12930u,393281u,12184u,12932u,12183u,548u,3118u,262205u,6u,
12933u,12932u,196670u,12931u,12933u,393281u,12184u,12935u,12183u,548u,2510u,262205u,6u,12936u,12935u,196670u,
12934u,12936u,393281u,12184u,12938u,12183u,548u,2624u,262205u,6u,12939u,12938u,196670u,12937u,12939u,393281u,
12184u,12941u,12183u,548u,2750u,262205u,6u,12942u,12941u,196670u,12940u,12942u,393281u,12184u,12944u,12183u,
548u,2409u,262205u,6u,12945u,12944u,196670u,12943u,12945u,262205u,317u,12947u,12905u,196670u,12946u,12947u,
196670u,12948u,12924u,196670u,12949u,12927u,852025u,2u,12950u,329u,12928u,12931u,12934u,12937u,12940u,12943u,
12946u,12948u,12949u,131321u,12250u,131320u,12230u,393281u,12184u,12953u,12183u,548u,2404u,262205u,6u,12954u,
12953u,196670u,12952u,12954u,393281u,12184u,12956u,12183u,548u,3118u,262205u,6u,12957u,12956u,196670u,12955u,
12957u,393281u,12184u,12959u,12183u,548u,2510u,262205u,6u,12960u,12959u,196670u,12958u,12960u,393281u,12184u,
12962u,12183u,548u,2624u,262205u,6u,12963u,12962u,196670u,12961u,12963u,393281u,12184u,12965u,12183u,548u,
2750u,262205u,6u,12966u,12965u,196670u,12964u,12966u,589881u,2u,12967u,336u,12952u,12955u,12958u,12961u,
12964u,131321u,12250u,131320u,12231u,393281u,12184u,12970u,12183u,548u,2624u,262205u,6u,12971u,12970u,393281u,
12184u,12972u,12183u,548u,2750u,262205u,6u,12973u,12972u,393281u,12184u,12974u,12183u,548u,2409u,262205u,
6u,12975u,12974u,393281u,12184u,12976u,12183u,548u,2395u,262205u,6u,12977u,12976u,393281u,12184u,12978u,
12183u,548u,3377u,262205u,6u,12979u,12978u,393281u,12184u,12980u,12183u,548u,2754u,262205u,6u,12981u,
12980u,327851u,57u,12982u,12981u,548u,393281u,12184u,12983u,12183u,548u,3472u,262205u,6u,12984u,12983u,
327851u,57u,12985u,12984u,548u,393281u,12184u,12986u,12183u,548u,12439u,262205u,6u,12987u,12986u,393281u,
12184u,12988u,12183u,548u,12569u,262205u,6u,12989u,12988u,393281u,12184u,12990u,12183u,548u,12572u,262205u,
6u,12991u,12990u,393281u,12184u,12992u,12183u,548u,12575u,262205u,6u,12993u,12992u,917584u,338u,12994u,
12971u,12973u,12975u,12977u,12979u,12982u,12985u,12987u,12989u,12991u,12993u,196670u,12969u,12994u,393281u,12184u,
12995u,12183u,548u,12581u,262205u,6u,12996u,12995u,327851u,57u,12997u,12996u,548u,393281u,12184u,12999u,
12183u,548u,2404u,262205u,6u,13000u,12999u,196670u,12998u,13000u,393281u,12184u,13002u,12183u,548u,3118u,
262205u,6u,13003u,13002u,196670u,13001u,13003u,393281u,12184u,13005u,12183u,548u,2510u,262205u,6u,13006u,
13005u,196670u,13004u,13006u,262205u,338u,13008u,12969u,196670u,13007u,13008u,393281u,12184u,13010u,12183u,548u,
12578u,262205u,6u,13011u,13010u,196670u,13009u,13011u,196670u,13012u,12997u,393281u,12184u,13014u,12183u,548u,
12584u,262205u,6u,13015u,13014u,196670u,13013u,13015u,720953u,2u,13016u,348u,12998u,13001u,13004u,13007u,
13009u,13012u,13013u,131321u,12250u,131320u,12232u,393281u,12184u,13019u,12183u,548u,2624u,262205u,6u,13020u,
13019u,393281u,12184u,13021u,12183u,548u,2750u,262205u,6u,13022u,13021u,393281u,12184u,13023u,12183u,548u,
2409u,262205u,6u,13024u,13023u,393281u,12184u,13025u,12183u,548u,2395u,262205u,6u,13026u,13025u,393281u,
12184u,13027u,12183u,548u,3377u,262205u,6u,13028u,13027u,393281u,12184u,13029u,12183u,548u,2754u,262205u,
6u,13030u,13029u,327851u,57u,13031u,13030u,548u,393281u,12184u,13032u,12183u,548u,3472u,262205u,6u,
13033u,13032u,327851u,57u,13034u,13033u,548u,393281u,12184u,13035u,12183u,548u,12439u,262205u,6u,13036u,
13035u,393281u,12184u,13037u,12183u,548u,12569u,262205u,6u,13038u,13037u,393281u,12184u,13039u,12183u,548u,
12572u,262205u,6u,13040u,13039u,393281u,12184u,13041u,12183u,548u,12575u,262205u,6u,13042u,13041u,917584u,
338u,13043u,13020u,13022u,13024u,13026u,13028u,13031u,13034u,13036u,13038u,13040u,13042u,196670u,13018u,13043u,
393281u,12184u,13045u,12183u,548u,2404u,262205u,6u,13046u,13045u,196670u,13044u,13046u,393281u,12184u,13048u,
12183u,548u,3118u,262205u,6u,13049u,13048u,196670u,13047u,13049u,393281u,12184u,13051u,12183u,548u,2510u,
262205u,6u,13052u,13051u,196670u,13050u,13052u,262205u,338u,13054u,13018u,196670u,13053u,13054u,524345u,2u,
13055u,355u,13044u,13047u,13050u,13053u,131321u,12250u,131320u,12233u,393281u,12184u,13058u,12183u,548u,2624u,
262205u,6u,13059u,13058u,393281u,12184u,13060u,12183u,548u,2750u,262205u,6u,13061u,13060u,393281u,12184u,
13062u,12183u,548u,2409u,262205u,6u,13063u,13062u,393281u,12184u,13064u,12183u,548u,2395u,262205u,6u,
13065u,13064u,393281u,12184u,13066u,12183u,548u,3377u,262205u,6u,13067u,13066u,393281u,12184u,13068u,12183u,
548u,2754u,262205u,6u,13069u,13068u,327851u,57u,13070u,13069u,548u,393281u,12184u,13071u,12183u,548u,
3472u,262205u,6u,13072u,13071u,327851u,57u,13073u,13072u,548u,393281u,12184u,13074u,12183u,548u,12439u,
262205u,6u,13075u,13074u,393281u,12184u,13076u,12183u,548u,12569u,262205u,6u,13077u,13076u,393281u,12184u,
13078u,12183u,548u,12572u,262205u,6u,13079u,13078u,393281u,12184u,13080u,12183u,548u,12575u,262205u,6u,
13081u,13080u,917584u,338u,13082u,13059u,13061u,13063u,13065u,13067u,13070u,13073u,13075u,13077u,13079u,13081u,
196670u,13057u,13082u,393281u,12184u,13084u,12183u,548u,2404u,262205u,6u,13085u,13084u,196670u,13083u,13085u,
393281u,12184u,13087u,12183u,548u,3118u,262205u,6u,13088u,13087u,196670u,13086u,13088u,393281u,12184u,13090u,
12183u,548u,2510u,262205u,6u,13091u,13090u,196670u,13089u,13091u,262205u,338u,13093u,13057u,196670u,13092u,
13093u,393281u,12184u,13095u,12183u,548u,12578u,262205u,6u,13096u,13095u,196670u,13094u,13096u,589881u,2u,
13097u,363u,13083u,13086u,13089u,13092u,13094u,131321u,12250u,131320u,12234u,393281u,12184u,13100u,12183u,548u,
2750u,262205u,6u,13101u,13100u,393281u,12184u,13102u,12183u,548u,2409u,262205u,6u,13103u,13102u,393281u,
12184u,13104u,12183u,548u,2395u,262205u,6u,13105u,13104u,393281u,12184u,13106u,12183u,548u,3377u,262205u,
6u,13107u,13106u,393281u,12184u,13108u,12183u,548u,2754u,262205u,6u,13109u,13108u,393281u,12184u,13110u,
12183u,548u,3472u,262205u,6u,13111u,13110u,327851u,57u,13112u,13111u,548u,393281u,12184u,13113u,12183u,
548u,12439u,262205u,6u,13114u,13113u,327851u,57u,13115u,13114u,548u,393281u,12184u,13116u,12183u,548u,
12569u,262205u,6u,13117u,13116u,393281u,12184u,13118u,12183u,548u,12572u,262205u,6u,13119u,13118u,393281u,
12184u,13120u,12183u,548u,12575u,262205u,6u,13121u,13120u,393281u,12184u,13122u,12183u,548u,12578u,262205u,
6u,13123u,13122u,917584u,338u,13124u,13101u,13103u,13105u,13107u,13109u,13112u,13115u,13117u,13119u,13121u,
13123u,196670u,13099u,13124u,393281u,12184u,13126u,12183u,548u,2404u,262205u,6u,13127u,13126u,196670u,13125u,
13127u,393281u,12184u,13129u,12183u,548u,3118u,262205u,6u,13130u,13129u,196670u,13128u,13130u,393281u,12184u,
13132u,12183u,548u,2510u,262205u,6u,13133u,13132u,196670u,13131u,13133u,393281u,12184u,13135u,12183u,548u,
2624u,262205u,6u,13136u,13135u,196670u,13134u,13136u,262205u,338u,13138u,13099u,196670u,13137u,13138u,393281u,
12184u,13140u,12183u,548u,12581u,262205u,6u,13141u,13140u,196670u,13139u,13141u,655417u,2u,13142u,372u,
13125u,13128u,13131u,13134u,13137u,13139u,131321u,12250u,131320u,12235u,393281u,12184u,13145u,12183u,548u,2750u,
262205u,6u,13146u,13145u,393281u,12184u,13147u,12183u,548u,2409u,262205u,6u,13148u,13147u,393281u,12184u,
13149u,12183u,548u,2395u,262205u,6u,13150u,13149u,393281u,12184u,13151u,12183u,548u,3377u,262205u,6u,
13152u,13151u,393281u,12184u,13153u,12183u,548u,2754u,262205u,6u,13154u,13153u,393281u,12184u,13155u,12183u,
548u,3472u,262205u,6u,13156u,13155u,327851u,57u,13157u,13156u,548u,393281u,12184u,13158u,12183u,548u,
12439u,262205u,6u,13159u,13158u,327851u,57u,13160u,13159u,548u,393281u,12184u,13161u,12183u,548u,12569u,
262205u,6u,13162u,13161u,393281u,12184u,13163u,12183u,548u,12572u,262205u,6u,13164u,13163u,393281u,12184u,
13165u,12183u,548u,12575u,262205u,6u,13166u,13165u,393281u,12184u,13167u,12183u,548u,12578u,262205u,6u,
13168u,13167u,917584u,338u,13169u,13146u,13148u,13150u,13152u,13154u,13157u,13160u,13162u,13164u,13166u,13168u,
196670u,13144u,13169u,393281u,12184u,13171u,12183u,548u,2404u,262205u,6u,13172u,13171u,196670u,13170u,13172u,
393281u,12184u,13174u,12183u,548u,3118u,262205u,6u,13175u,13174u,196670u,13173u,13175u,393281u,12184u,13177u,
12183u,548u,2510u,262205u,6u,13178u,13177u,196670u,13176u,13178u,393281u,12184u,13180u,12183u,548u,2624u,
262205u,6u,13181u,13180u,196670u,13179u,13181u,262205u,338u,13183u,13144u,196670u,13182u,13183u,589881u,2u,
13184u,380u,13170u,13173u,13176u,13179u,13182u,131321u,12250u,131320u,12236u,393281u,12184u,13187u,12183u,548u,
2395u,262205u,6u,13188u,13187u,393281u,12184u,13189u,12183u,548u,3377u,262205u,6u,13190u,13189u,393281u,
12184u,13191u,12183u,548u,2754u,262205u,6u,13192u,13191u,393281u,12184u,13193u,12183u,548u,3472u,262205u,
6u,13194u,13193u,393281u,12184u,13195u,12183u,548u,12439u,262205u,6u,13196u,13195u,393281u,12184u,13197u,
12183u,548u,12569u,262205u,6u,13198u,13197u,327851u,57u,13199u,13198u,548u,393281u,12184u,13200u,12183u,
548u,12572u,262205u,6u,13201u,13200u,327851u,57u,13202u,13201u,548u,393281u,12184u,13203u,12183u,548u,
12575u,262205u,6u,13204u,13203u,393281u,12184u,13205u,12183u,548u,12578u,262205u,6u,13206u,13205u,393281u,
12184u,13207u,12183u,548u,12581u,262205u,6u,13208u,13207u,393281u,12184u,13209u,12183u,548u,12584u,262205u,
6u,13210u,13209u,917584u,338u,13211u,13188u,13190u,13192u,13194u,13196u,13199u,13202u,13204u,13206u,13208u,
13210u,196670u,13186u,13211u,393281u,12184u,13213u,12183u,548u,2404u,262205u,6u,13214u,13213u,196670u,13212u,
13214u,393281u,12184u,13216u,12183u,548u,3118u,262205u,6u,13217u,13216u,196670u,13215u,13217u,393281u,12184u,
13219u,12183u,548u,2510u,262205u,6u,13220u,13219u,196670u,13218u,13220u,393281u,12184u,13222u,12183u,548u,
2624u,262205u,6u,13223u,13222u,196670u,13221u,13223u,393281u,12184u,13225u,12183u,548u,2750u,262205u,6u,
13226u,13225u,196670u,13224u,13226u,393281u,12184u,13228u,12183u,548u,2409u,262205u,6u,13229u,13228u,196670u,
13227u,13229u,262205u,338u,13231u,13186u,196670u,13230u,13231u,393281u,12184u,13233u,12183u,548u,12587u,262205u,
6u,13234u,13233u,196670u,13232u,13234u,393281u,12184u,13236u,12183u,548u,12590u,262205u,6u,13237u,13236u,
196670u,13235u,13237u,852025u,2u,13238u,392u,13212u,13215u,13218u,13221u,13224u,13227u,13230u,13232u,13235u,
131321u,12250u,131320u,12237u,393281u,12184u,13241u,12183u,548u,2754u,262205u,6u,13242u,13241u,393281u,12184u,
13243u,12183u,548u,3472u,262205u,6u,13244u,13243u,393281u,12184u,13245u,12183u,548u,12439u,262205u,6u,
13246u,13245u,393281u,12184u,13247u,12183u,548u,12569u,262205u,6u,13248u,13247u,393281u,12184u,13249u,12183u,
548u,12572u,262205u,6u,13250u,13249u,393281u,12184u,13251u,12183u,548u,12575u,262205u,6u,13252u,13251u,
327851u,57u,13253u,13252u,548u,393281u,12184u,13254u,12183u,548u,12578u,262205u,6u,13255u,13254u,327851u,
57u,13256u,13255u,548u,393281u,12184u,13257u,12183u,548u,12581u,262205u,6u,13258u,13257u,393281u,12184u,
13259u,12183u,548u,12584u,262205u,6u,13260u,13259u,393281u,12184u,13261u,12183u,548u,12587u,262205u,6u,
13262u,13261u,393281u,12184u,13263u,12183u,548u,12590u,262205u,6u,13264u,13263u,917584u,338u,13265u,13242u,
13244u,13246u,13248u,13250u,13253u,13256u,13258u,13260u,13262u,13264u,196670u,13240u,13265u,393281u,12184u,13267u,
12183u,548u,2404u,262205u,6u,13268u,13267u,196670u,13266u,13268u,393281u,12184u,13270u,12183u,548u,3118u,
262205u,6u,13271u,13270u,196670u,13269u,13271u,393281u,12184u,13273u,12183u,548u,2510u,262205u,6u,13274u,
13273u,196670u,13272u,13274u,393281u,12184u,13276u,12183u,548u,2624u,262205u,6u,13277u,13276u,196670u,13275u,
13277u,393281u,12184u,13279u,12183u,548u,2750u,262205u,6u,13280u,13279u,196670u,13278u,13280u,393281u,12184u,
13282u,12183u,548u,2409u,262205u,6u,13283u,13282u,196670u,13281u,13283u,393281u,12184u,13285u,12183u,548u,
2395u,262205u,6u,13286u,13285u,196670u,13284u,13286u,393281u,12184u,13288u,12183u,548u,3377u,262205u,6u,
13289u,13288u,196670u,13287u,13289u,262205u,338u,13291u,13240u,196670u,13290u,13291u,393281u,12184u,13293u,12183u,
548u,12593u,262205u,6u,13294u,13293u,196670u,13292u,13294u,917561u,2u,13295u,405u,13266u,13269u,13272u,
13275u,13278u,13281u,13284u,13287u,13290u,13292u,131321u,12250u,131320u,12238u,393281u,12184u,13298u,12183u,548u,
2409u,262205u,6u,13299u,13298u,393281u,12184u,13300u,12183u,548u,2395u,262205u,6u,13301u,13300u,393281u,
12184u,13302u,12183u,548u,3377u,262205u,6u,13303u,13302u,393281u,12184u,13304u,12183u,548u,2754u,262205u,
6u,13305u,13304u,393281u,12184u,13306u,12183u,548u,3472u,262205u,6u,13307u,13306u,393281u,12184u,13308u,
12183u,548u,12439u,262205u,6u,13309u,13308u,327851u,57u,13310u,13309u,548u,393281u,12184u,13311u,12183u,
548u,12569u,262205u,6u,13312u,13311u,327851u,57u,13313u,13312u,548u,393281u,12184u,13314u,12183u,548u,
12572u,262205u,6u,13315u,13314u,393281u,12184u,13316u,12183u,548u,12575u,262205u,6u,13317u,13316u,393281u,
12184u,13318u,12183u,548u,12578u,262205u,6u,13319u,13318u,393281u,12184u,13320u,12183u,548u,12581u,262205u,
6u,13321u,13320u,917584u,338u,13322u,13299u,13301u,13303u,13305u,13307u,13310u,13313u,13315u,13317u,13319u,
13321u,196670u,13297u,13322u,393281u,12184u,13323u,12183u,548u,12584u,262205u,6u,13324u,13323u,327851u,57u,
13325u,13324u,548u,393281u,12184u,13327u,12183u,548u,2404u,262205u,6u,13328u,13327u,196670u,13326u,13328u,
393281u,12184u,13330u,12183u,548u,3118u,262205u,6u,13331u,13330u,196670u,13329u,13331u,393281u,12184u,13333u,
12183u,548u,2510u,262205u,6u,13334u,13333u,196670u,13332u,13334u,393281u,12184u,13336u,12183u,548u,2624u,
262205u,6u,13337u,13336u,196670u,13335u,13337u,393281u,12184u,13339u,12183u,548u,2750u,262205u,6u,13340u,
13339u,196670u,13338u,13340u,262205u,338u,13342u,13297u,196670u,13341u,13342u,196670u,13343u,13325u,393281u,12184u,
13345u,12183u,548u,12587u,262205u,6u,13346u,13345u,196670u,13344u,13346u,786489u,2u,13347u,416u,13326u,
13329u,13332u,13335u,13338u,13341u,13343u,13344u,131321u,12250u,131320u,12239u,393281u,12184u,13350u,12183u,548u,
2754u,262205u,6u,13351u,13350u,393281u,12184u,13352u,12183u,548u,3472u,262205u,6u,13353u,13352u,393281u,
12184u,13354u,12183u,548u,12439u,262205u,6u,13355u,13354u,393281u,12184u,13356u,12183u,548u,12569u,262205u,
6u,13357u,13356u,393281u,12184u,13358u,12183u,548u,12572u,262205u,6u,13359u,13358u,393281u,12184u,13360u,
12183u,548u,12575u,262205u,6u,13361u,13360u,589904u,438u,13362u,13351u,13353u,13355u,13357u,13359u,13361u,
196670u,13349u,13362u,393281u,12184u,13363u,12183u,548u,12578u,262205u,6u,13364u,13363u,327851u,57u,13365u,
13364u,548u,393281u,12184u,13367u,12183u,548u,2404u,262205u,6u,13368u,13367u,196670u,13366u,13368u,393281u,
12184u,13370u,12183u,548u,3118u,262205u,6u,13371u,13370u,196670u,13369u,13371u,393281u,12184u,13373u,12183u,
548u,2510u,262205u,6u,13374u,13373u,196670u,13372u,13374u,393281u,12184u,13376u,12183u,548u,2624u,262205u,
6u,13377u,13376u,196670u,13375u,13377u,393281u,12184u,13379u,12183u,548u,2750u,262205u,6u,13380u,13379u,
196670u,13378u,13380u,393281u,12184u,13382u,12183u,548u,2409u,262205u,6u,13383u,13382u,196670u,13381u,13383u,
393281u,12184u,13385u,12183u,548u,2395u,262205u,6u,13386u,13385u,196670u,13384u,13386u,393281u,12184u,13388u,
12183u,548u,3377u,262205u,6u,13389u,13388u,196670u,13387u,13389u,262205u,438u,13391u,13349u,196670u,13390u,
13391u,196670u,13392u,13365u,393281u,12184u,13394u,12183u,548u,12581u,262205u,6u,13395u,13394u,196670u,13393u,
13395u,983097u,2u,13396u,452u,13366u,13369u,13372u,13375u,13378u,13381u,13384u,13387u,13390u,13392u,13393u,
131321u,12250u,131320u,12240u,393281u,12184u,13398u,12183u,548u,3377u,262205u,6u,13399u,13398u,262268u,10u,
13400u,13399u,393281u,12184u,13401u,12183u,548u,2754u,262205u,6u,13402u,13401u,262268u,10u,13403u,13402u,
393281u,12184u,13404u,12183u,548u,12439u,262205u,6u,13405u,13404u,327851u,57u,13406u,13405u,548u,393281u,
12184u,13408u,12183u,548u,2404u,262205u,6u,13409u,13408u,196670u,13407u,13409u,393281u,12184u,13411u,12183u,
548u,3118u,262205u,6u,13412u,13411u,196670u,13410u,13412u,393281u,12184u,13414u,12183u,548u,2510u,262205u,
6u,13415u,13414u,196670u,13413u,13415u,393281u,12184u,13417u,12183u,548u,2624u,262205u,6u,13418u,13417u,
196670u,13416u,13418u,393281u,12184u,13420u,12183u,548u,2750u,262205u,6u,13421u,13420u,196670u,13419u,13421u,
393281u,12184u,13423u,12183u,548u,2409u,262205u,6u,13424u,13423u,196670u,13422u,13424u,393281u,12184u,13426u,
12183u,548u,2395u,262205u,6u,13427u,13426u,196670u,13425u,13427u,196670u,13428u,13400u,196670u,13429u,13403u,
393281u,12184u,13431u,12183u,548u,3472u,262205u,6u,13432u,13431u,196670u,13430u,13432u,196670u,13433u,13406u,
983097u,2u,13434u,471u,13407u,13410u,13413u,13416u,13419u,13422u,13425u,13428u,13429u,13430u,13433u,131321u,
12250u,131320u,12241u,393281u,12184u,13437u,12183u,548u,2404u,262205u,6u,13438u,13437u,196670u,13436u,13438u,
393281u,12184u,13440u,12183u,548u,3118u,262205u,6u,13441u,13440u,196670u,13439u,13441u,393273u,2u,13442u,
476u,13436u,13439u,131321u,12250u,131320u,12242u,393281u,12184u,13444u,12183u,548u,2750u,262205u,6u,13445u,
13444u,262268u,10u,13446u,13445u,393281u,12184u,13448u,12183u,548u,2404u,262205u,6u,13449u,13448u,196670u,
13447u,13449u,393281u,12184u,13451u,12183u,548u,3118u,262205u,6u,13452u,13451u,196670u,13450u,13452u,393281u,
12184u,13454u,12183u,548u,2510u,262205u,6u,13455u,13454u,196670u,13453u,13455u,393281u,12184u,13457u,12183u,
548u,2624u,262205u,6u,13458u,13457u,196670u,13456u,13458u,196670u,13459u,13446u,589881u,2u,13460u,484u,
13447u,13450u,13453u,13456u,13459u,131321u,12250u,131320u,12243u,393281u,12184u,13463u,12183u,548u,2404u,262205u,
6u,13464u,13463u,196670u,13462u,13464u,393281u,12184u,13466u,12183u,548u,3118u,262205u,6u,13467u,13466u,
196670u,13465u,13467u,393281u,12184u,13469u,12183u,548u,2510u,262205u,6u,13470u,13469u,196670u,13468u,13470u,
393281u,12184u,13472u,12183u,548u,2624u,262205u,6u,13473u,13472u,196670u,13471u,13473u,393281u,12184u,13475u,
12183u,548u,2750u,262205u,6u,13476u,13475u,196670u,13474u,13476u,589881u,2u,13477u,491u,13462u,13465u,
13468u,13471u,13474u,131321u,12250u,131320u,12244u,393281u,12184u,13479u,12183u,548u,2409u,262205u,6u,13480u,
13479u,262268u,10u,13481u,13480u,393281u,12184u,13483u,12183u,548u,2404u,262205u,6u,13484u,13483u,196670u,
13482u,13484u,393281u,12184u,13486u,12183u,548u,3118u,262205u,6u,13487u,13486u,196670u,13485u,13487u,393281u,
12184u,13489u,12183u,548u,2510u,262205u,6u,13490u,13489u,196670u,13488u,13490u,393281u,12184u,13492u,12183u,
548u,2624u,262205u,6u,13493u,13492u,196670u,13491u,13493u,393281u,12184u,13495u,12183u,548u,2750u,262205u,
6u,13496u,13495u,196670u,13494u,13496u,196670u,13497u,13481u,655417u,2u,13498u,500u,13482u,13485u,13488u,
13491u,13494u,13497u,131321u,12250u,131320u,12245u,393281u,12184u,13501u,12183u,548u,2404u,262205u,6u,13502u,
13501u,196670u,13500u,13502u,393281u,12184u,13504u,12183u,548u,3118u,262205u,6u,13505u,13504u,196670u,13503u,
13505u,393281u,12184u,13507u,12183u,548u,2510u,262205u,6u,13508u,13507u,196670u,13506u,13508u,393281u,12184u,
13510u,12183u,548u,2624u,262205u,6u,13511u,13510u,196670u,13509u,13511u,393281u,12184u,13513u,12183u,548u,
2750u,262205u,6u,13514u,13513u,196670u,13512u,13514u,393281u,12184u,13516u,12183u,548u,2409u,262205u,6u,
13517u,13516u,196670u,13515u,13517u,393281u,12184u,13519u,12183u,548u,2395u,262205u,6u,13520u,13519u,196670u,
13518u,13520u,720953u,2u,13521u,509u,13500u,13503u,13506u,13509u,13512u,13515u,13518u,131321u,12250u,131320u,
12246u,393281u,12184u,13524u,12183u,548u,2404u,262205u,6u,13525u,13524u,196670u,13523u,13525u,393281u,12184u,
13527u,12183u,548u,3118u,262205u,6u,13528u,13527u,196670u,13526u,13528u,393281u,12184u,13530u,12183u,548u,
2510u,262205u,6u,13531u,13530u,196670u,13529u,13531u,458809u,2u,13532u,514u,13523u,13526u,13529u,131321u,
12250u,131320u,12247u,393281u,12184u,13534u,12183u,548u,3472u,262205u,6u,13535u,13534u,327851u,57u,13536u,
13535u,548u,393281u,12184u,13537u,12183u,548u,12439u,262205u,6u,13538u,13537u,327851u,57u,13539u,13538u,
548u,393281u,12184u,13540u,12183u,548u,12569u,262205u,6u,13541u,13540u,262268u,10u,13542u,13541u,393281u,
12184u,13543u,12183u,548u,12572u,262205u,6u,13544u,13543u,262268u,10u,13545u,13544u,393281u,12184u,13547u,
12183u,548u,2404u,262205u,6u,13548u,13547u,196670u,13546u,13548u,393281u,12184u,13550u,12183u,548u,3118u,
262205u,6u,13551u,13550u,196670u,13549u,13551u,393281u,12184u,13553u,12183u,548u,2510u,262205u,6u,13554u,
13553u,196670u,13552u,13554u,393281u,12184u,13556u,12183u,548u,2624u,262205u,6u,13557u,13556u,196670u,13555u,
13557u,393281u,12184u,13559u,12183u,548u,2750u,262205u,6u,13560u,13559u,196670u,13558u,13560u,393281u,12184u,
13562u,12183u,548u,2409u,262205u,6u,13563u,13562u,196670u,13561u,13563u,393281u,12184u,13565u,12183u,548u,
2395u,262205u,6u,13566u,13565u,196670u,13564u,13566u,393281u,12184u,13568u,12183u,548u,3377u,262205u,6u,
13569u,13568u,196670u,13567u,13569u,393281u,12184u,13571u,12183u,548u,2754u,262205u,6u,13572u,13571u,196670u,
13570u,13572u,196670u,13573u,13536u,196670u,13574u,13539u,196670u,13575u,13542u,196670u,13576u,13545u,1114169u,2u,
13577u,530u,13546u,13549u,13552u,13555u,13558u,13561u,13564u,13567u,13570u,13573u,13574u,13575u,13576u,131321u,
12250u,131320u,12248u,393281u,12184u,13579u,12183u,548u,2754u,262205u,6u,13580u,13579u,262268u,10u,13581u,
13580u,393281u,12184u,13583u,12183u,548u,2404u,262205u,6u,13584u,13583u,196670u,13582u,13584u,393281u,12184u,
13586u,12183u,548u,3118u,262205u,6u,13587u,13586u,196670u,13585u,13587u,393281u,12184u,13589u,12183u,548u,
2510u,262205u,6u,13590u,13589u,196670u,13588u,13590u,393281u,12184u,13592u,12183u,548u,2624u,262205u,6u,
13593u,13592u,196670u,13591u,13593u,393281u,12184u,13595u,12183u,548u,2750u,262205u,6u,13596u,13595u,196670u,
13594u,13596u,393281u,12184u,13598u,12183u,548u,2409u,262205u,6u,13599u,13598u,196670u,13597u,13599u,393281u,
12184u,13601u,12183u,548u,2395u,262205u,6u,13602u,13601u,196670u,13600u,13602u,393281u,12184u,13604u,12183u,
548u,3377u,262205u,6u,13605u,13604u,196670u,13603u,13605u,196670u,13606u,13581u,852025u,2u,13607u,542u,
13582u,13585u,13588u,13591u,13594u,13597u,13600u,13603u,13606u,131321u,12250u,131320u,12250u,131321u,12193u,131320u,
12193u,262205u,6u,13611u,12189u,327808u,6u,13612u,13611u,551u,196670u,12189u,13612u,131321u,12190u,131320u,
12192u,65789u,65592u,
};
#endif
}
