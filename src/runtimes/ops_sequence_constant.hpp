#pragma once
#include "../Layers.hpp"
#include <algorithm>
#include <cmath>

// A positional table keeps its parameter/optimizer shape. Short contexts use a
// prefix; longer contexts interpolate its rows. Backward uses the exact adjoint.
namespace SequenceConstant {
inline bool enabled(const Layer& l) {
    return l.shape.size()==2 && l.shape[0]==-1 && l.embed_dim>0 && l.seq_len>0 &&
        l.shape[1]==l.embed_dim && l.getWeightsSize()%size_t(l.embed_dim)==0 && l.getWeightsSize()>0;
}
inline size_t outputSize(const Layer& l) {
    return enabled(l)?size_t(l.seq_len)*l.embed_dim:l.getWeightsSize();
}
template<class F> void rows(const Layer& l,F f) {
    const size_t dim=size_t(l.embed_dim), available=l.getWeightsSize()/dim;
    for(size_t row=0;row<size_t(l.seq_len);++row) {
        const double position=size_t(l.seq_len)<=available ? double(row) :
            double(row)*double(available-1)/double(std::max(1,l.seq_len-1));
        const size_t a=size_t(position),b=std::min(a+1,available-1);
        f(row,a,b,float(position-double(a)));
    }
}
inline bool forward(const Layer& l,std::vector<float>& output) {
    const auto* weights=l.getWeights();if(!weights)return false;
    output.resize(outputSize(l));
    if(!enabled(l)){std::copy(weights,weights+l.getWeightsSize(),output.begin());return true;}
    const size_t dim=size_t(l.embed_dim);
    rows(l,[&](size_t row,size_t a,size_t b,float t){for(size_t c=0;c<dim;++c)
        output[row*dim+c]=(1.f-t)*weights[a*dim+c]+t*weights[b*dim+c];});
    return true;
}
inline bool backward(Layer& l,const std::vector<float>& gradient) {
    if(gradient.size()!=outputSize(l))return false;
    if(!l.trainable_parameter)return true;
    if(l.grad_weights.size()!=l.getWeightsSize())l.grad_weights.assign(l.getWeightsSize(),0.f);
    if(!enabled(l)){for(size_t i=0;i<gradient.size();++i)l.grad_weights[i]+=gradient[i];return true;}
    const size_t dim=size_t(l.embed_dim);
    rows(l,[&](size_t row,size_t a,size_t b,float t){for(size_t c=0;c<dim;++c){
        l.grad_weights[a*dim+c]+=(1.f-t)*gradient[row*dim+c];
        l.grad_weights[b*dim+c]+=t*gradient[row*dim+c];}});
    return true;
}
}
