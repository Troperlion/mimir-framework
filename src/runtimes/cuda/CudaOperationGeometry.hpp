#pragma once
#include "Layers.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>

namespace CudaOperationGeometry {
struct Norm {
    int channels=0, spatial=0, groups=0;
    bool valid(const Layer& l, size_t count) {
        channels=l.in_channels; groups=l.type_enum==LayerType::InstanceNorm2d?channels:l.num_groups;
        const long long space=1LL*l.input_height*l.input_width;
        if (channels<=0 || groups<=0 || groups>channels || channels%groups ||
            l.input_height<=0 || l.input_width<=0 || space>INT_MAX ||
            1LL*channels*space>INT_MAX || count!=static_cast<size_t>(channels*space) || !(l.eps>0.f)) return false;
        spatial=space;
        if(l.type_enum==LayerType::InstanceNorm2d && l.affine && !l.use_bias)return false; // CPU forward always reads beta.
        return !l.affine || (l.getWeights() && l.getWeightsSize()>=static_cast<size_t>(channels)*(l.use_bias?2:1));
    }
};
struct Resize {
    int c=0, ih=0, iw=0, oh=0, ow=0, sh=1, sw=1;
    bool valid(const Layer& l, size_t count) {
        c=l.in_channels; ih=l.input_height>0?l.input_height:l.out_h; iw=l.input_width>0?l.input_width:l.out_w;
        if (c<=0 || ih<=0 || iw<=0 || 1LL*c*ih*iw>INT_MAX || count!=static_cast<size_t>(1LL*c*ih*iw)) return false;
        if (!std::isfinite(l.scale_h) || !std::isfinite(l.scale_w) || l.scale_h>=INT_MAX || l.scale_w>=INT_MAX) return false;
        const int explicit_h=l.output_height>0?l.output_height:l.out_h;
        const int explicit_w=l.output_width>0?l.output_width:l.out_w;
        if (l.type_enum==LayerType::UpsampleNearest) {
            sh=l.scale_h>0?static_cast<int>(std::lround(l.scale_h)):(explicit_h>0?std::max(1,explicit_h/ih):2);
            sw=l.scale_w>0?static_cast<int>(std::lround(l.scale_w)):(explicit_w>0?std::max(1,explicit_w/iw):2);
            if (sh<=0 || sw<=0 || 1LL*ih*sh>INT_MAX || 1LL*iw*sw>INT_MAX) return false;
            oh=ih*sh; ow=iw*sw;
        } else {
            const double height=double(ih)*std::max(0.f,l.scale_h), width=double(iw)*std::max(0.f,l.scale_w);
            if (height>INT_MAX || width>INT_MAX) return false;
            oh=explicit_h>0?explicit_h:std::max(1,static_cast<int>(std::lround(height)));
            ow=explicit_w>0?explicit_w:std::max(1,static_cast<int>(std::lround(width)));
        }
        return oh>0 && ow>0 && 1LL*c*oh*ow<=INT_MAX;
    }
    size_t outputSize() const { return static_cast<size_t>(c)*oh*ow; }
};
// CPU-compatible pooling geometry. Explicit dimensions avoid ambiguous legacy defaults.
struct Pool {
    int c=0,h=0,w=0,kh=0,kw=0,sh=0,sw=0,ph=0,pw=0,oh=0,ow=0;
    bool maximum=false, token=false;
    bool valid(const Layer& l,size_t n) {
        maximum=l.type_enum==LayerType::MaxPool1d || l.type_enum==LayerType::MaxPool2d;
        token=l.type_enum==LayerType::TokenMeanPool;
        if (token) {
            c=l.embed_dim; h=l.seq_len; w=1; kh=h; kw=1; sh=sw=1; ph=pw=0; oh=ow=1;
            return c>0 && h>0 && 1LL*c*h<=INT_MAX && n==static_cast<size_t>(1LL*c*h);
        }
        if((l.type_enum==LayerType::AvgPool2d || l.type_enum==LayerType::GlobalAvgPool2d || l.type_enum==LayerType::AdaptiveAvgPool2d) && l.in_channels<=0)return false;
        c=l.in_channels>0?l.in_channels:1;
        const bool one=l.type_enum==LayerType::MaxPool1d || l.type_enum==LayerType::AvgPool1d;
        h=one?static_cast<int>(n/c):l.input_height; w=one?1:l.input_width;
        if (h<=0 || w<=0 || 1LL*c*h*w>INT_MAX || n!=static_cast<size_t>(1LL*c*h*w)) return false;
        if (l.type_enum==LayerType::GlobalAvgPool2d || l.type_enum==LayerType::AdaptiveAvgPool2d) {
            kh=h; kw=w; sh=sw=1; ph=pw=0; oh=ow=1; return true;
        }
        kh=one?(l.kernel_h>0?l.kernel_h:2):l.get_kernel_h();
        sh=one?(l.stride_h>0?l.stride_h:kh):l.get_stride_h();
        ph=one?(l.pad_h>=0?l.pad_h:0):l.get_pad_h();
        kw=one?1:(maximum?kh:l.get_kernel_w());
        sw=one?1:(maximum?sh:l.get_stride_w());
        pw=one?0:(maximum?ph:l.get_pad_w());
        const long long nh=1LL*h+2LL*ph-kh,nw=1LL*w+2LL*pw-kw;
        if (kh<=0 || kw<=0 || sh<=0 || sw<=0 || ph<0 || pw<0 || nh<0 || nw<0 || nh/sh+1>INT_MAX || nw/sw+1>INT_MAX) return false;
        oh=nh/sh+1; ow=nw/sw+1;
        return 1LL*c*oh*ow<=INT_MAX;
    }
    size_t outputSize() const { return static_cast<size_t>(c)*oh*ow; }
};
// Data movement does not require a tensor arithmetic fallback.
inline bool movementType(LayerType t) {
    switch(t) {
        case LayerType::Identity: case LayerType::Flatten: case LayerType::Reshape:
        case LayerType::View: case LayerType::Squeeze: case LayerType::Unsqueeze:
        case LayerType::Transpose: case LayerType::Permute: case LayerType::Concat:
        case LayerType::Stack: case LayerType::Split: case LayerType::Chunk:
        case LayerType::Constant: case LayerType::ZeroPad2d: case LayerType::ReflectionPad2d:
        case LayerType::ReplicationPad2d: case LayerType::PixelShuffle: return true;
        default:return false;
    }
}
inline bool poolType(LayerType t) {
    switch(t) {
        case LayerType::MaxPool1d: case LayerType::MaxPool2d: case LayerType::AvgPool1d:
        case LayerType::AvgPool2d: case LayerType::GlobalAvgPool2d:
        case LayerType::AdaptiveAvgPool2d: case LayerType::TokenMeanPool:return true;
        default:return false;
    }
}
// Each output has one source index; split/concat are contiguous device copies.
struct Movement {
    std::vector<size_t> sizes;
    std::vector<int> shape,dims,strides;
    int c=0,h=0,w=0,oh=0,ow=0,ph=0,pw=0,shuffle=0,padding=0;
    bool valid(const Layer& l,const std::vector<const std::vector<float>*>& in) {
        if(in.empty() || !in[0] || in[0]->size()>INT_MAX) return false;
        const size_t n=in[0]->size();
        sizes={n};
        if(l.type_enum==LayerType::Constant) { sizes={l.getWeightsSize()};return l.getWeights() && sizes[0]>0 && sizes[0]<=INT_MAX; }
        if(l.type_enum==LayerType::Concat || l.type_enum==LayerType::Stack) {
            if(in.size()<2)return false;
            size_t total=0;
            for(auto* x:in) { if(!x || x->size()>INT_MAX || (l.type_enum==LayerType::Stack && x->size()!=n))return false;total+=x->size(); }
            sizes={total};return total<=INT_MAX;
        }
        if(l.type_enum==LayerType::Split || l.type_enum==LayerType::Chunk) {
            if(l.type_enum==LayerType::Split && !l.split_sizes.empty()) {
                sizes.clear();size_t total=0;for(int v:l.split_sizes){if(v<=0)return false;sizes.push_back(v);total+=v;}return total==n;
            }
            const int parts=l.type_enum==LayerType::Chunk?l.num_chunks:l.num_splits;
            if(parts<=0 || static_cast<size_t>(parts)>INT_MAX/sizeof(size_t) || (l.type_enum==LayerType::Split && n%parts))return false;
            sizes.assign(parts,n/parts);for(size_t i=0;i<n%parts;++i)++sizes[i];return true;
        }
        if(l.type_enum==LayerType::ZeroPad2d || l.type_enum==LayerType::ReflectionPad2d || l.type_enum==LayerType::ReplicationPad2d || l.type_enum==LayerType::PixelShuffle) {
            c=l.in_channels;h=l.input_height;w=l.input_width;
            if(c<=0 || h<=0 || w<=0 || 1LL*c*h*w>INT_MAX || n!=static_cast<size_t>(1LL*c*h*w))return false;
            if(l.type_enum==LayerType::PixelShuffle){
                if(!std::isfinite(l.scale_h) || l.scale_h<1 || l.scale_h>46340 || l.scale_h!=std::floor(l.scale_h))return false;
                shuffle=static_cast<int>(l.scale_h);if(c%(shuffle*shuffle) || 1LL*h*shuffle>INT_MAX || 1LL*w*shuffle>INT_MAX)return false;
                oh=h*shuffle;ow=w*shuffle;sizes={n};return true;
            }
            padding=l.type_enum==LayerType::ZeroPad2d?1:(l.type_enum==LayerType::ReflectionPad2d?2:3);
            ph=l.pad_h>=0?l.pad_h:1;pw=l.pad_w>=0?l.pad_w:ph;
            // CPU reflected/replicated forward uses one symmetric pad, backward uses pad_h/pad_w.
            if(padding!=1 && pw!=ph)return false;
            if(padding==1 && l.pad_w<0 && ph!=1)return false;
            if(1LL*h+2LL*ph>INT_MAX || 1LL*w+2LL*pw>INT_MAX)return false;
            oh=h+2*ph;ow=w+2*pw;if(1LL*c*oh*ow>INT_MAX)return false;sizes={static_cast<size_t>(c)*oh*ow};return true;
        }
        if(l.type_enum==LayerType::Transpose) { shape={l.in_features,l.out_features};dims={1,0}; }
        else if(l.type_enum==LayerType::Permute) { shape=l.shape.empty()?std::vector<int>{1,static_cast<int>(n)}:l.shape;dims=l.permute_dims; }
        else return true;
        if(shape.empty() || shape.size()!=dims.size())return false;
        std::vector<bool> seen(shape.size());size_t product=1;
        for(size_t i=0;i<shape.size();++i) {
            if(shape[i]<=0 || product>INT_MAX/static_cast<size_t>(shape[i]) || dims[i]<0 || dims[i]>=static_cast<int>(shape.size()) || seen[dims[i]])return false;
            product*=shape[i];seen[dims[i]]=true;
        }
        if(product!=n)return false;
        strides.assign(shape.size(),1);for(int i=static_cast<int>(shape.size())-2;i>=0;--i)strides[i]=strides[i+1]*shape[i+1];return true;
    }
};
struct Projection {
    // mode: embedding, bag, patch projection, bilinear.
    int mode=0,dim=0,vocab=0,pad=-1,rows=0,ids=0,features=0,second=0,prefix=0;
    size_t output=0,parameters=0;bool bias=false;
    bool valid(const Layer& l,const std::vector<const std::vector<float>*>& in){
        if(in.empty() || !in[0] || in[0]->empty() || in[0]->size()>INT_MAX || !l.getWeights())return false;
        ids=in[0]->size();
        if(l.type_enum==LayerType::Embedding || l.type_enum==LayerType::EmbeddingBag){
            mode=l.type_enum==LayerType::Embedding?1:2;vocab=std::max(1,l.vocab_size);pad=l.padding_idx;
            dim=std::max(1,mode==1?(l.embed_dim>0?l.embed_dim:l.out_features):l.embed_dim);
            rows=mode==1?ids:1;
            if(mode==2 && in.size()>1 && in[1] && !in[1]->empty()){if(in[1]->size()<2 || in[1]->size()>INT_MAX)return false;rows=in[1]->size()-1;}
            for(float v:*in[0])if(!std::isfinite(v) || double(v)<INT_MIN || double(v)>INT_MAX)return false;
            if(mode==2 && in.size()>1 && in[1])for(float v:*in[1])if(!std::isfinite(v) || double(v)<INT_MIN || double(v)>INT_MAX)return false;
            parameters=static_cast<size_t>(vocab)*dim;output=static_cast<size_t>(rows)*dim;
        }else if(l.type_enum==LayerType::PatchEmbed){
            mode=3;dim=l.embed_dim>0?l.embed_dim:l.out_features;rows=std::max(1,l.num_patches);features=std::max(1,l.patch_dim);
            const long long text=(1LL+std::max(1,l.seq_text))*dim;
            if(dim<=0 || text>INT_MAX || text+1LL*rows*features!=ids)return false;
            prefix=text;parameters=static_cast<size_t>(features)*dim+dim;output=text+static_cast<size_t>(rows)*dim;
            if(l.getWeightsSize()!=parameters)return false;
        }else if(l.type_enum==LayerType::Bilinear){
            mode=4;features=l.in_features;second=l.out_features;dim=l.embed_dim>0?l.embed_dim:(l.target_shape.empty()?0:l.target_shape[0]);
            if(features<=0 || second<=0 || dim<=0 || in.size()<2 || !in[1] || ids%features)return false;
            rows=ids/features;if(in[1]->size()!=static_cast<size_t>(rows)*second)return false;
            const long long per=1LL*features*second;if(per>INT_MAX || per*dim>INT_MAX)return false;
            parameters=per*dim;bias=l.use_bias;output=static_cast<size_t>(rows)*dim;parameters+=bias?dim:0;if(l.getWeightsSize()!=parameters)return false;
        }else return false;
        return output>0 && output<=INT_MAX && parameters<=INT_MAX && l.getWeightsSize()>=parameters;
    }
};
inline bool projectionType(LayerType t){return t==LayerType::Embedding || t==LayerType::EmbeddingBag || t==LayerType::PatchEmbed || t==LayerType::Bilinear;}
inline bool extendedConvType(LayerType t){return t==LayerType::Conv1d || t==LayerType::DepthwiseConv2d || t==LayerType::ConvTranspose2d;}
struct ExtendedConv {
    int mode=0,ic=0,oc=0,h=0,w=0,kh=0,kw=0,stride=0,pad=0,oh=0,ow=0;
    size_t parameters=0;bool bias=false;
    bool valid(const Layer& l,size_t n){
        if(!n || n>INT_MAX || !extendedConvType(l.type_enum))return false;
        mode=l.type_enum==LayerType::Conv1d?1:(l.type_enum==LayerType::DepthwiseConv2d?2:3);
        ic=l.in_channels>0?l.in_channels:1;oc=mode==2?ic:(l.out_channels>0?l.out_channels:1);
        h=mode==1?static_cast<int>(n/ic):l.input_height;w=mode==1?1:l.input_width;
        kh=l.kernel_h>0?l.kernel_h:3;kw=mode==1?1:kh;
        stride=mode==3?l.get_stride_h():(l.stride_h>0?l.stride_h:1);pad=mode==3?l.get_pad_h():(l.pad_h>=0?l.pad_h:0);
        // Raw legacy fields and getters must describe the same CPU forward/backward.
        if(kh!=l.get_kernel_h() || stride!=l.get_stride_h() || pad!=l.get_pad_h())return false;
        if(h<=0 || w<=0 || kh<=0 || stride<=0 || pad<0 || 1LL*ic*h*w>INT_MAX || n!=static_cast<size_t>(1LL*ic*h*w))return false;
        const long long nh=mode==3?(1LL*h-1)*stride-2LL*pad+kh:1LL*h+2LL*pad-kh;
        const long long nw=mode==1?1:(mode==3?(1LL*w-1)*stride-2LL*pad+kw:1LL*w+2LL*pad-kw);
        if(nh<0 || nw<0)return false;
        const long long a=mode==3?nh:nh/stride+1,b=mode==1?1:(mode==3?nw:nw/stride+1);
        if(a<=0 || b<=0 || a>INT_MAX || b>INT_MAX || 1LL*oc*a*b>INT_MAX)return false;
        oh=a;ow=b;
        parameters=static_cast<size_t>(oc)*(mode==2?1:ic)*kh*kw;bias=l.use_bias;
        return parameters<=INT_MAX && l.getWeights() && l.getWeightsSize()==parameters+(bias?oc:0);
    }
    size_t outputSize()const{return static_cast<size_t>(oc)*oh*ow;}
};
inline bool stochasticType(LayerType t){return t==LayerType::Dropout || t==LayerType::Dropout2d || t==LayerType::AlphaDropout || t==LayerType::Reparameterize;}
inline bool batchNormType(LayerType t){return t==LayerType::BatchNorm1d || t==LayerType::BatchNorm2d;}
struct BatchNorm {
    int channels=0,spatial=1,batch=0;bool affine=false,bias=false,running=false;
    bool valid(const Layer& l,size_t n){
        if(!n || n>INT_MAX || !(l.eps>0.f))return false;
        channels=l.in_channels>0?l.in_channels:l.out_channels;
        if(channels<=0 && !l.running_mean.empty())channels=l.running_mean.size();
        if(channels<=0 && l.affine && l.getWeights()){const size_t z=l.getWeightsSize();channels=l.use_bias && z%2==0?z/2:z;}
        if(channels<=0 || n%channels)return false;
        const long long area=1LL*l.input_height*l.input_width;
        spatial=l.type_enum==LayerType::BatchNorm2d && l.input_height>0 && l.input_width>0 && area<=INT_MAX?static_cast<int>(area):1;
        // Avoid the legacy CPU backward fallback choosing a different spatial layout.
        if(1LL*channels*spatial>INT_MAX || n%(static_cast<size_t>(channels)*spatial))return false;
        batch=n/(static_cast<size_t>(channels)*spatial);
        affine=l.affine;bias=affine && l.use_bias;
        if(affine && (!l.getWeights() || l.getWeightsSize()<static_cast<size_t>(channels)*(bias?2:1)))return false;
        running=l.track_running_stats && l.running_mean.size()==static_cast<size_t>(channels) && l.running_var.size()==static_cast<size_t>(channels);
        return batch>0;
    }
};
inline bool attentionType(LayerType t){return t==LayerType::SelfAttention || t==LayerType::MultiHeadAttention || t==LayerType::CrossAttention;}
struct Attention {
    int ed=0,ked=0,heads=0,q=0,k=0;bool cross=false,causal=false;int qbias=-1,obias=-1;
    size_t mainParameters=0,output=0;
    bool valid(const Layer& l,const std::vector<const std::vector<float>*>& in,bool backward=false){
        if(in.empty() || !in[0] || in[0]->empty() || in[0]->size()>INT_MAX)return false;
        cross=l.type_enum==LayerType::CrossAttention;causal=l.causal;heads=l.num_heads>0?l.num_heads:1;
        ed=l.embed_dim>0?l.embed_dim:(cross?(l.head_dim>0 && 1LL*l.head_dim*heads<=INT_MAX?l.head_dim*heads:0):in[0]->size());
        ked=cross && l.in_features>0?l.in_features:ed;
        if(ed<=0 || ked<=0 || ed%heads || in[0]->size()%ed || (backward && ked!=ed))return false;
        q=cross?in[0]->size()/ed:(l.seq_len>0?l.seq_len:1);k=q;
        if(!cross && in[0]->size()!=static_cast<size_t>(q)*ed)return false;
        if(cross){if(in.size()<2 || !in[1] || in[1]->empty() || in[1]->size()>INT_MAX || in[1]->size()%ked)return false;k=in[1]->size()/ked;}
        const long long matrix=1LL*ed*ed,kv=2LL*ked*ed,total=cross?2*matrix+kv:4*matrix;
        if(total>INT_MAX || 1LL*heads*q*k>INT_MAX || 1LL*q*ed>INT_MAX || 1LL*k*ed>INT_MAX/3)return false;
        mainParameters=total;output=static_cast<size_t>(q)*ed;
        if(!l.getWeights() || l.getWeightsSize()<mainParameters || l.getWeightsSize()>INT_MAX)return false;
        if(cross){if(l.getWeightsSize()>=mainParameters+ed)obias=mainParameters;}
        else if(l.getWeightsSize()>=mainParameters+4ULL*ed){qbias=mainParameters;obias=mainParameters+3*ed;}
        else if(l.getWeightsSize()>=mainParameters+ed)obias=mainParameters;
        return true;
    }
};
inline bool recurrentType(LayerType t){return t==LayerType::RNN || t==LayerType::GRU || t==LayerType::LSTM;}
struct Recurrent {
    int steps=0,input=0,hidden=0,gates=0;size_t parameters=0,workspace=0;
    bool valid(const Layer& l,size_t n){
        steps=l.seq_len;input=l.in_features;hidden=l.out_features;gates=l.type_enum==LayerType::RNN?1:(l.type_enum==LayerType::GRU?3:4);
        if(steps<=0 || input<=0 || hidden<=0 || 1LL*steps*input>INT_MAX || n!=static_cast<size_t>(1LL*steps*input) || 1LL*steps*hidden>INT_MAX)return false;
        const long long params=1LL*gates*hidden*(1LL*input+hidden+(l.use_bias?2:0)),space=12LL*steps*hidden+8LL*hidden;
        if(params>INT_MAX || space>INT_MAX || !l.getWeights() || l.getWeightsSize()<static_cast<size_t>(params) || l.getWeightsSize()>INT_MAX)return false;
        parameters=params;workspace=space;return true;
    }
    size_t outputSize()const{return static_cast<size_t>(steps)*hidden;}
};
struct Conv {
    int ic=0, oc=0, h=0, w=0, k=0, stride=0, pad=0, dilation=0, oh=0, ow=0;
    bool valid(const Layer& l, size_t count) {
        ic=l.in_channels>0?l.in_channels:1; oc=l.out_channels>0?l.out_channels:1;
        h=l.input_height; w=l.input_width; k=l.get_kernel_h(); stride=l.get_stride_h(); pad=l.get_pad_h();
        dilation=l.dilation_h!=1?l.dilation_h:l.dilation;
        if (h<=0 || w<=0 || k<=0 || stride<=0 || pad<0 || dilation<=0 ||
            1LL*ic*h*w>INT_MAX || count!=static_cast<size_t>(1LL*ic*h*w)) return false;
        const long long nh=1LL*h+2LL*pad-1LL*dilation*(k-1)-1, nw=1LL*w+2LL*pad-1LL*dilation*(k-1)-1;
        if (nh<0 || nw<0 || nh/stride+1>INT_MAX || nw/stride+1>INT_MAX) return false;
        oh=nh/stride+1; ow=nw/stride+1;
        const long long inner=1LL*ic*k*k, spatial=1LL*oh*ow;
        if (inner>INT_MAX || spatial>INT_MAX || inner*spatial>INT_MAX || 1LL*oc*inner>INT_MAX || 1LL*oc*spatial>INT_MAX) return false;
        return l.getWeights() && l.getWeightsSize()>=static_cast<size_t>(oc*inner)+(l.use_bias?oc:0);
    }
    size_t outputSize() const { return static_cast<size_t>(oc)*oh*ow; }
};
} // namespace CudaOperationGeometry
