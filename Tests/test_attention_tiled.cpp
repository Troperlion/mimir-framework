#include "Layers.hpp"
#include "runtimes/cpu/LayerOps.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include <iostream>
#include <stdexcept>
static std::vector<float> values(size_t n,int mod) {
    std::vector<float> v(n);for(size_t i=0;i<n;++i)v[i]=float(int(i%mod)-mod/2)/32;return v;
}
static void check(bool cross,int queries,int keys,int dim,int kvdim,int heads,bool causal,bool bias) {
    auto input=values(size_t(queries)*dim,17),kvinput=values(size_t(keys)*kvdim,19);
    auto qw=values(size_t(dim)*dim,13),kvw=values(size_t(kvdim)*2*dim,11);
    auto qkvw=values(size_t(3)*dim*dim,13),ow=values(size_t(dim)*dim,7);
    auto qb=bias?values(3*dim,5):std::vector<float>{};
    auto ob=bias?values(dim,7):std::vector<float>{};
    std::vector<double> q(size_t(queries)*dim),k(size_t(keys)*dim),v(k.size());
    if(cross) {
        for(int t=0;t<queries;++t)for(int j=0;j<dim;++j)
            for(int c=0;c<dim;++c)q[size_t(t)*dim+j]+=double(input[size_t(t)*dim+c])*qw[size_t(c)*dim+j];
        for(int t=0;t<keys;++t)for(int j=0;j<dim;++j)for(int c=0;c<kvdim;++c) {
            k[size_t(t)*dim+j]+=double(kvinput[size_t(t)*kvdim+c])*kvw[size_t(c)*2*dim+j];
            v[size_t(t)*dim+j]+=double(kvinput[size_t(t)*kvdim+c])*kvw[size_t(c)*2*dim+dim+j];
        }
    } else {
        for(int t=0;t<queries;++t)for(int j=0;j<dim;++j) {
            double sums[3]={bias?qb[j]:0,bias?qb[dim+j]:0,bias?qb[2*dim+j]:0};
            for(int p=0;p<3;++p)for(int c=0;c<dim;++c)
                sums[p]+=double(input[size_t(t)*dim+c])*qkvw[size_t(p*dim+j)*dim+c];
            q[size_t(t)*dim+j]=sums[0];k[size_t(t)*dim+j]=sums[1];v[size_t(t)*dim+j]=sums[2];
        }
    }
    std::vector<double> attended(q.size()),scores(keys);
    const int hd=dim/heads;
    for(int h=0;h<heads;++h)for(int t=0;t<queries;++t) {
        for(int s=0;s<keys;++s) {
            double dot=0;for(int c=0;c<hd;++c)dot+=q[size_t(t)*dim+h*hd+c]*k[size_t(s)*dim+h*hd+c];
            scores[s]=(causal&&t<keys&&s>t)?-1e9:dot/std::sqrt(double(hd));
        }
        const double max=*std::max_element(scores.begin(),scores.end());double sum=0;
        for(double& score:scores){score=std::exp(score-max);sum+=score;}
        for(int c=0;c<hd;++c)for(int s=0;s<keys;++s)
            attended[size_t(t)*dim+h*hd+c]+=scores[s]/sum*v[size_t(s)*dim+h*hd+c];
    }
    auto actual=cross?LayerOps::cross_attention_forward(input,kvinput,qw,kvw,ow,ob,queries,keys,dim,kvdim,heads,causal)
                     :LayerOps::self_attention_forward(input,qkvw,ow,qb,ob,queries,dim,heads,causal);
    if(actual.size()!=q.size())throw std::runtime_error("Attention output shape");
    for(int t=0;t<queries;++t)for(int j=0;j<dim;++j) {
        double ref=bias?ob[j]:0;
        for(int c=0;c<dim;++c)ref+=attended[size_t(t)*dim+c]*ow[size_t(c)*dim+j];
        if(std::abs(actual[size_t(t)*dim+j]-ref)>3e-5*(1+std::abs(ref)))
            throw std::runtime_error("Attention differs from double reference");
    }
    Layer l("attention",cross?"CrossAttention":"SelfAttention",0);
    l.seq_len=queries;l.embed_dim=dim;l.num_heads=heads;l.causal=causal;l.in_features=kvdim;
    l.weights=cross?qw:qkvw;
    if(cross)l.weights.insert(l.weights.end(),kvw.begin(),kvw.end());
    l.weights.insert(l.weights.end(),ow.begin(),ow.end());
    if(!cross)l.weights.insert(l.weights.end(),qb.begin(),qb.end());
    l.weights.insert(l.weights.end(),ob.begin(),ob.end());
    CpuRuntime cpu;cpu.initialize(RuntimeConfig{});std::vector<std::vector<float>> output;
    std::vector<const std::vector<float>*> inputs{&input};if(cross)inputs.push_back(&kvinput);
    if(!cpu.forwardLayer(inputs,output,l,false)||output.size()!=1||output[0]!=actual)
        throw std::runtime_error("Attention dispatch differs from public helper");
}
int main() {
    for(bool causal:{false,true})for(bool bias:{false,true}) {
        check(false,1,1,3,3,1,causal,bias);
        check(false,127,127,15,15,3,causal,bias);
        check(false,128,128,8,8,2,causal,bias);
        check(false,129,129,15,15,3,causal,bias);
        check(false,257,257,8,8,2,causal,bias);
        check(true,129,131,15,7,3,causal,bias);
        check(true,257,129,8,11,2,causal,bias);
    }
    std::cout<<"ATTENTION_TILED_OK\n";
}
