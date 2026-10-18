#include "LossDispatch.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/portable/PortableDevice.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
namespace RuntimeLossGrad {
static void validateRelations(const Request& r) {
    if(!r.options.relations)return;
    for(const auto& relation:*r.options.relations) {
        if(relation.width<=0 || relation.height<=0 || relation.channels<=0 ||
           size_t(relation.width)*relation.height*relation.channels>r.count || !std::isfinite(relation.weight) || relation.weight<0)
            throw std::invalid_argument("Invalid relation supervision grid/weight");
        if(relation.weight==0)continue;
        for(const auto* box:{&relation.subject,&relation.object})
            if(!((*box)[0]>=0 && (*box)[1]>=0 && (*box)[2]<=1 && (*box)[3]<=1 && (*box)[2]>(*box)[0] && (*box)[3]>(*box)[1]))
                throw std::invalid_argument("Invalid relation region");
    }
}
bool valid(const Request& r) {
    return r.count && r.prediction && (r.operation==Operation::Pool || r.operation==Operation::PoolBackward || r.target);
}
bool deviceSizeFits(const Request& r) {
    if(!valid(r) || r.count>size_t(INT_MAX/32))return false;

    if(r.operation!=Operation::Pixel && (r.width<=0 || r.height<=0 || r.channels<=0 ||
       size_t(r.width)*r.height*r.channels!=r.count))return false;
    if(r.operation==Operation::Spectral && (uint64_t(r.width)*r.width+uint64_t(r.height)*r.height+12*r.count>size_t(INT_MAX/2)))return false;
    if(r.operation==Operation::SSIM && r.channels>INT_MAX/256)return false;
    if(r.operation==Operation::CrossEntropy && r.height>INT_MAX/256)return false;
    size_t entries=0;
    if(r.options.relations)for(const auto& rel:*r.options.relations) {
        if(rel.channels<=0 || rel.width<=0 || rel.height<=0 || size_t(rel.width)*rel.height*rel.channels>r.count)return false;
        if(rel.weight>0)entries+=size_t(rel.channels);
        if(entries>size_t(INT_MAX/64))return false;
    }
    if(12*r.count+26*entries>size_t(INT_MAX))return false;
    return true;
}
bool host(const Request& r,LossWithGrad& out) {
    if(!valid(r))return false;
    if(r.operation==Operation::CrossEntropy) {
        out={};out.grad.assign(r.count,0.f);int count=0;
        for(int token=0;token<r.height;++token){int label=int(r.target[token]);if(label>=0 && label<r.width)++count;}
        const float scale=count?1.f/float(count):0.f;
        for(int token=0;token<r.height;++token) {
            int label=int(r.target[token]);if(label<0 || label>=r.width)continue;
            size_t base=size_t(token)*r.width;float maximum=-std::numeric_limits<float>::infinity();
            for(int k=0;k<r.width;++k)maximum=std::max(maximum,r.prediction[base+k]);
            double sum=0.;for(int k=0;k<r.width;++k)sum+=std::exp(double(r.prediction[base+k]-maximum));
            out.loss+=double(maximum)+std::log(std::max(1e-30,sum))-r.prediction[base+size_t(label)];
            for(int k=0;k<r.width;++k)out.grad[base+k]=scale*float(std::exp(double(r.prediction[base+k]-maximum))/std::max(1e-30,sum));
            out.grad[base+size_t(label)]-=scale;
        }
        if(count)out.loss/=count;
        return true;
    }
    if(r.operation==Operation::Pixel){out=Host::pixel_loss_and_grad(r.prediction,r.target,r.count,r.type,r.options);return true;}
    std::vector<float> p(r.prediction,r.prediction+(r.operation==Operation::PoolBackward?size_t(std::max(1,r.width/2))*std::max(1,r.height/2)*r.channels:r.count));
    std::vector<float> t;if(r.target)t.assign(r.target,r.target+r.count);
    if(r.operation==Operation::SSIM){auto s=Host::ssim_global_hwc(p,t,r.width,r.height,r.channels,r.k1,r.k2,r.range);out.loss=s.loss;out.grad=std::move(s.grad);}
    else if(r.operation==Operation::Spectral)out=Host::spectral_dct_l1_hwc(p,t,r.width,r.height,r.channels);
    else if(r.operation==Operation::Pool){int w,h;Host::avgpool2x2_hwc(p,r.width,r.height,r.channels,out.grad,w,h);}
    else Host::avgpool2x2_back_hwc(p,r.width,r.height,r.channels,out.grad);
    return true;
}
std::vector<float> pack(const Request& r) {
    validateRelations(r);
    const size_t n=r.count;
    std::vector<float> m(r.operation==Operation::SSIM?std::max(12*n,9*n+size_t(r.channels)*8):12*n,0.f);
    const size_t input=r.operation==Operation::PoolBackward?size_t(std::max(1,r.width/2))*std::max(1,r.height/2)*r.channels:n;
    std::copy_n(r.prediction,input,m.data());
    if(r.target)std::copy_n(r.target,n,m.data()+n);
    for(size_t i=0;i<n;++i)m[2*n+i]=r.options.weights && r.options.weights->size()>=n?std::max(0.f,(*r.options.weights)[i]):1.f;
    if(r.operation==Operation::Pixel && r.options.relations) {
        for(size_t i=0;i<n;++i)m[5*n+i]=r.options.weights && r.options.weights->size()>i && (*r.options.weights)[i]<=0?0.f:1.f;
        size_t entries=0;
        for(const auto& relation:*r.options.relations) {
            if(relation.weight==0)continue;
            std::string predicate=relation.predicate;std::transform(predicate.begin(),predicate.end(),predicate.begin(),[](unsigned char c){return char(std::tolower(c));});
            const bool horizontal=predicate=="left_of" || predicate=="right_of" || predicate=="left" || predicate=="right";
            const bool vertical=predicate=="above" || predicate=="below" || predicate=="above_of" || predicate=="below_of";
            for(int channel=0;channel<relation.channels;++channel) {
                m.insert(m.end(),{float(relation.width),float(relation.height),float(relation.channels),relation.chw?1.f:0.f,
                    float(relation.weight/(double(r.options.relations->size())*relation.channels*(horizontal || vertical?2:3))),vertical?0.f:1.f,horizontal?0.f:1.f});
                for(double x:relation.subject)m.push_back(float(x));
                for(double x:relation.object)m.push_back(float(x));
                m.push_back(float(channel));++entries;
            }
        }
        m.resize(m.size()+10*entries,0.f);
    }
    if(r.operation==Operation::Spectral) {
        static thread_local std::map<std::pair<int,int>,std::vector<float>> coefficientCache;
        const auto key=std::make_pair(r.width,r.height);
        auto found=coefficientCache.find(key);
        if(found==coefficientCache.end()) {
            constexpr double pi=3.1415926535897932384626433832795;
            std::vector<float> coefficients;
            for(int length:{r.width,r.height}) {
                const double inv=1.0/double(length);
                for(int x=0;x<length;++x)for(int k=0;k<length;++k)
                    coefficients.push_back(float(std::cos((pi*(double(x)+.5)*double(k))*inv)));
                for(int k=0;k<length;++k)coefficients.push_back(float(std::sqrt((k==0?1.:2.)*inv)));
            }
            found=coefficientCache.emplace(key,std::move(coefficients)).first;
        }
        m.insert(m.end(),found->second.begin(),found->second.end());
    }
    return m;
}
std::vector<int> parameters(const Request& r) {
    const auto name=canonical_pixel_loss_name(r.type);
    int kind=0;if(name=="mae")kind=1;else if(name=="huber" || name=="smoothl1")kind=2;
    else if(name=="charbonnier")kind=3;else if(name=="gaussian_nll" || name=="nll_gaussian")kind=4;else if(name=="bce")kind=5;
    const size_t work=r.operation==Operation::SSIM?std::max(r.count,size_t(r.channels)*256):r.operation==Operation::CrossEntropy?std::max(r.count,size_t(r.height)*256):r.count;
    std::vector<int> args{0,int(work),int(r.operation),int(r.count),r.width,r.height,r.channels,kind,r.options.relation_only?1:0};
    for(float f:{std::max(1e-6f,r.options.huber_delta),std::max(1e-12f,r.options.charbonnier_eps),std::max(1e-6f,r.options.gaussian_nll_sigma),r.k1,r.k2,r.range}) {
        int bits;std::memcpy(&bits,&f,4);args.push_back(bits);
    }
    size_t entries=0;
    if(r.operation==Operation::Pixel && r.options.relations)for(const auto& rel:*r.options.relations)if(rel.weight>0 && rel.channels>0)entries+=size_t(rel.channels);
    args.push_back(int(entries));
    args[1]=int(std::max(size_t(args[1]),entries));
    return args;
}
std::vector<int> passes(const std::vector<int>& a) {
    switch(Operation(a[2])) {
        case Operation::Pixel:return a[15]?std::vector<int>{0,3,4,90,91}:std::vector<int>{0,90,91};
        case Operation::SSIM:return {0,1,2,90,91};
        case Operation::CrossEntropy:return {0,1,2,90,91};
        case Operation::Spectral:return {0,1,2,3,4,90,91};
        default:return {0};
    }
}
bool unpack(const Request& r,const std::vector<float>& m,LossWithGrad& out) {
    if(m.size()<12*r.count)return false;
    const size_t size=r.operation==Operation::Pool?size_t(std::max(1,r.width/2))*std::max(1,r.height/2)*r.channels:r.count;
    LossWithGrad candidate;candidate.grad.assign(m.begin()+3*r.count,m.begin()+3*r.count+size);
    if(r.operation!=Operation::Pool && r.operation!=Operation::PoolBackward)candidate.loss=m[4*r.count];
    if(r.operation==Operation::Pixel)candidate.relation_loss=m[11*r.count];
    if(!std::isfinite(candidate.loss) || !std::all_of(candidate.grad.begin(),candidate.grad.end(),[](float x){return std::isfinite(x);}))return false;
    out=std::move(candidate);return true;
}
bool portable(const Request& r,LossWithGrad& out,const char* backend,int index) {
    if(!deviceSizeFits(r))return false;
    static thread_local std::map<std::pair<std::string,int>,std::unique_ptr<PortableStaging::Device>> devices;
    auto& device=devices[{backend,index}];if(!device)device=PortableStaging::createLossDevice(backend,index);
    if(!device)return false;
    auto m=pack(r);if(!device->run(m,parameters(r))){device.reset();return false;}
    return unpack(r,m,out);
}
static LossWithGrad dispatch(const Request& r) {
    LossWithGrad result;
    if(!RuntimeRouter::instance().dispatchReconstructionLoss(r,result))throw std::runtime_error("No runtime available for reconstruction loss");
    return result;
}
LossWithGrad pixel_loss_and_grad(const float* p,const float* t,size_t n,const std::string& type,const PixelLossOptions& o) {
    if(!n || !p || !t)return Host::pixel_loss_and_grad(p,t,n,type,o);
    Request r;r.prediction=p;r.target=t;r.count=n;r.type=type;r.options=o;validateRelations(r);return dispatch(r);
}
LossWithGrad pixel_loss_and_grad(const std::vector<float>& p,const std::vector<float>& t,const std::string& type,const PixelLossOptions& o) {
    if(p.size()!=t.size())return {};
    return pixel_loss_and_grad(p.data(),t.data(),p.size(),type,o);
}
LossWithGrad text_cross_entropy(const float* logits,size_t logits_size,const std::vector<int>& ids,int seq,int vocab,int padding) {
    Request r;r.operation=Operation::CrossEntropy;r.width=std::max(2,vocab);r.height=std::max(1,seq);r.channels=1;r.count=size_t(r.width)*r.height;
    if(!logits || logits_size<r.count)throw std::invalid_argument("Invalid reconstruction logits");
    std::vector<float> targets(r.count,-1.f);
    for(size_t i=0;i<std::min(ids.size(),size_t(r.height));++i)if(ids[i]>=0 && ids[i]<r.width && (padding<0 || ids[i]!=padding))targets[i]=float(ids[i]);
    r.prediction=logits;r.target=targets.data();return dispatch(r);
}
static Request imageRequest(Operation op,const std::vector<float>& p,const std::vector<float>* t,int w,int h,int c) {
    Request r;r.operation=op;r.width=std::max(1,w);r.height=std::max(1,h);r.channels=std::max(1,c);
    r.count=size_t(r.width)*r.height*r.channels;r.prediction=p.data();r.target=t?t->data():nullptr;return r;
}
GlobalSSIM ssim_global_hwc(const std::vector<float>& p,const std::vector<float>& t,int w,int h,int c,float k1,float k2,float range) {
    auto r=imageRequest(Operation::SSIM,p,&t,w,h,c);r.k1=k1;r.k2=k2;r.range=range;
    if(p.size()<r.count || t.size()<r.count || c<=0)return Host::ssim_global_hwc(p,t,w,h,c,k1,k2,range);
    auto out=dispatch(r);return {out.loss,std::move(out.grad)};
}
LossWithGrad spectral_dct_l1_hwc(const std::vector<float>& p,const std::vector<float>& t,int w,int h,int c) {
    auto r=imageRequest(Operation::Spectral,p,&t,w,h,c);
    if(p.size()<r.count || t.size()<r.count)return Host::spectral_dct_l1_hwc(p,t,w,h,c);
    return dispatch(r);
}
void avgpool2x2_hwc(const std::vector<float>& p,int w,int h,int c,std::vector<float>& out,int& ow,int& oh) {
    auto r=imageRequest(Operation::Pool,p,nullptr,w,h,c);
    if(p.size()<r.count){Host::avgpool2x2_hwc(p,w,h,c,out,ow,oh);return;}
    ow=std::max(1,r.width/2);oh=std::max(1,r.height/2);out=dispatch(r).grad;
}
void avgpool2x2_back_hwc(const std::vector<float>& p,int w,int h,int c,std::vector<float>& out) {
    auto r=imageRequest(Operation::PoolBackward,p,nullptr,w,h,c);
    if(p.size()<size_t(std::max(1,r.width/2))*std::max(1,r.height/2)*r.channels){Host::avgpool2x2_back_hwc(p,w,h,c,out);return;}
    out=dispatch(r).grad;
}
}
namespace {
std::unordered_map<std::string,size_t>& lossCounts(){static thread_local std::unordered_map<std::string,size_t> counts;return counts;}
}
const std::unordered_map<std::string,size_t>& RuntimeRouter::reconstructionLossDispatchCounts()const{return lossCounts();}
bool RuntimeRouter::dispatchReconstructionLoss(const RuntimeLossGrad::Request& r,RuntimeLossGrad::LossWithGrad& out,AbstractRuntime** selected)const {
    if(selected)*selected=nullptr;
    if(!RuntimeLossGrad::valid(r))return false;
    ensureActivatedAndComposed();
    std::vector<AbstractRuntime*> candidates;
    // Standalone loss calls before a model exists retain the historical CPU path.
    static thread_local CpuRuntime standalone;
    if(runtime_priority_.empty()){if(!standalone.isInitialized())standalone.initialize(RuntimeConfig{});candidates.push_back(&standalone);}
    for(auto* rt:runtime_priority_)if(rt && rt->isInitialized() && !rt->config().disabled && rt->supportsReconstructionLoss() &&
       (std::string(rt->name())=="CPU" || RuntimeLossGrad::deviceSizeFits(r)))candidates.push_back(rt);
    if(candidates.empty())return false;
    using Key=std::tuple<int,size_t,int,int,int,std::string,bool,bool,size_t,std::vector<AbstractRuntime*>>;
    static thread_local std::map<Key,AbstractRuntime*> winners;
    Key key{int(r.operation),r.count,r.width,r.height,r.channels,RuntimeLossGrad::canonical_pixel_loss_name(r.type),r.options.weights!=nullptr,r.options.relation_only,r.options.relations?r.options.relations->size():0,candidates};
    auto it=winners.find(key);
    if(it==winners.end()) {
        double best=std::numeric_limits<double>::infinity();AbstractRuntime* winner=nullptr;
        for(auto* rt:candidates) {
            double elapsed=0;bool ok=true;
            for(int sample=0;sample<(candidates.size()>1?2:1);++sample){RuntimeLossGrad::LossWithGrad trial;auto start=std::chrono::steady_clock::now();if(!rt->reconstructionLoss(r,trial)){ok=false;break;}elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
            if(ok && elapsed<best){best=elapsed;winner=rt;}
        }
        if(!winner)return false;
        it=winners.emplace(key,winner).first;
        if(std::getenv("MIMIR_ACCEL_VERBOSE"))std::cerr<<"[reconstruction-loss] backend="<<winner->name()<<" op="<<int(r.operation)<<" elements="<<r.count<<" selection=measured_host_roundtrip\n";
    }
    std::vector<AbstractRuntime*> ordered{it->second};for(auto* rt:candidates)if(rt!=it->second)ordered.push_back(rt);
    for(auto* rt:ordered){RuntimeLossGrad::LossWithGrad trial;if(rt->reconstructionLoss(r,trial)){out=std::move(trial);it->second=rt;++lossCounts()[rt->name()];if(selected)*selected=rt;return true;}}
    return false;
}
