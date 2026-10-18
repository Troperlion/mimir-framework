#include "runtimes/optimizers/OptimizerDispatch.hpp"
#include "runtimes/tpu/TpuRuntime.hpp"
#include "Layers.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <limits>
#include <stdexcept>
#if defined(__linux__) || defined(__APPLE__)
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#ifndef MIMIR_TPU_WORKER
#define MIMIR_TPU_WORKER "src/runtimes/tpu/worker.py"
#endif
using json = nlohmann::json;
json Mimir::tpuLayerDescriptor(const Layer& l) {
    json j;
#define FIELD(f) j[#f] = l.f
    FIELD(in_features); FIELD(out_features); FIELD(in_channels); FIELD(out_channels);
    FIELD(input_height); FIELD(input_width); FIELD(output_height); FIELD(output_width);
    FIELD(kernel_h); FIELD(kernel_w); FIELD(stride_h); FIELD(stride_w); FIELD(pad_h); FIELD(pad_w);
    FIELD(dilation); FIELD(dilation_h); FIELD(eps); FIELD(num_groups); FIELD(affine);
    FIELD(use_bias); FIELD(track_running_stats); FIELD(running_mean); FIELD(running_var);
    FIELD(dropout_p); FIELD(vocab_size); FIELD(embed_dim); FIELD(padding_idx); FIELD(axis);
    FIELD(target_shape); FIELD(shape); FIELD(permute_dims); FIELD(split_sizes);
    FIELD(num_splits); FIELD(num_chunks); FIELD(seq_len); FIELD(num_heads); FIELD(head_dim);
    FIELD(causal); FIELD(scale_h); FIELD(scale_w); FIELD(out_h); FIELD(out_w);
    FIELD(leaky_relu_alpha); FIELD(seq_text); FIELD(num_patches); FIELD(patch_dim);
    FIELD(nms_iou_threshold); FIELD(nms_score_threshold); FIELD(nms_max_detections);
    FIELD(nms_class_agnostic); FIELD(trainable_parameter); FIELD(activation_param);
#undef FIELD
    j["type"] = l.type_enum==LayerType::PatchEmbed ? "PatchEmbed" :
        l.type_enum==LayerType::Reparameterize ? "Reparameterize" : LayerRegistry::type_to_string(l.type_enum);
    j["activation"] = static_cast<int>(l.activation);
    j["kh"] = l.get_kernel_h(); j["kw"] = l.get_kernel_w();
    j["sh"] = l.get_stride_h(); j["sw"] = l.get_stride_w();
    j["ph"] = l.get_pad_h(); j["pw"] = l.get_pad_w();
    return j;
}
namespace {
#if defined(__linux__) || defined(__APPLE__)
bool transfer(int fd, void* data, size_t size, bool writing) {
    auto* bytes = static_cast<char*>(data);
    // Includes compilation time; all waits are bounded even if the worker dies.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    while (size) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (ms <= 0) return false;
        pollfd p{fd, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
        const int ready = poll(&p, 1, static_cast<int>(ms));
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0 || !(p.revents & p.events)) return false;
        const ssize_t n = writing ? send(fd, bytes, size, MSG_NOSIGNAL) : recv(fd, bytes, size, 0);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n <= 0) return false;
        bytes += n; size -= static_cast<size_t>(n);
    }
    return true;
}
#endif
}
TpuRuntime::~TpuRuntime() { shutdown(); }
void TpuRuntime::closeWorker() {
#if defined(__linux__) || defined(__APPLE__)
    if (socket_ >= 0) close(socket_);
    if (child_ > 0) {
        // No unbounded wait on a stuck compiler or TPU driver.
        kill(child_, SIGKILL);
        while (waitpid(child_, nullptr, 0) < 0 && errno == EINTR) {}
    }
#endif
    socket_ = child_ = -1; initialized_ = false;
}
void TpuRuntime::shutdown() { std::lock_guard<std::mutex> lock(mutex_); closeWorker(); }
bool TpuRuntime::exchange(json request, const std::vector<const std::vector<float>*>& arrays,
                          json& response, std::vector<std::vector<float>>& result) {
#if defined(__linux__) || defined(__APPLE__)
    if (socket_ < 0) return false;
    request["sizes"] = json::array();
    for (const auto* a : arrays) { if (!a) return false; request["sizes"].push_back(a->size()); }
    std::string header = request.dump();
    uint32_t size = static_cast<uint32_t>(header.size());
    auto fail = [&]() { last_error_ = "TPU worker transport failed"; closeWorker(); return false; };
    if (!transfer(socket_, &size, sizeof(size), true) || !transfer(socket_, header.data(), header.size(), true)) return fail();
    for (const auto* a : arrays)
        if (!transfer(socket_, const_cast<float*>(a->data()), a->size()*sizeof(float), true)) return fail();
    if (!transfer(socket_, &size, sizeof(size), false) || size > 1024*1024) return fail();
    header.resize(size);
    if (!transfer(socket_, header.data(), size, false)) return fail();
    try {
        response = json::parse(header); result.clear();
        for (const auto& value : response.at("sizes")) {
            const uint64_t count = value.get<uint64_t>();
            if (count > (uint64_t(1)<<30)) return fail();
            result.emplace_back(static_cast<size_t>(count));
            if (!transfer(socket_, result.back().data(), result.back().size()*sizeof(float), false)) return fail();
        }
        if (!response.value("ok", false)) { last_error_ = response.value("error", "TPU execution failed"); return false; }
        return true;
    } catch (const std::exception& e) { last_error_ = e.what(); closeWorker(); return false; }
#else
    (void)request; (void)arrays; (void)response; (void)result; return false;
#endif
}
bool TpuRuntime::initialize(const RuntimeConfig& cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return true;
    config_ = cfg; last_error_.clear(); device_.clear();
    if (cfg.disabled) return false;
#if defined(__linux__) || defined(__APPLE__)
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)) { last_error_="socketpair failed"; return false; }
    fcntl(sockets[0], F_SETFD, FD_CLOEXEC); fcntl(sockets[1], F_SETFD, FD_CLOEXEC);
    const int flags=fcntl(sockets[0],F_GETFL,0);
    if(flags<0 || fcntl(sockets[0],F_SETFL,flags|O_NONBLOCK)<0) {
        close(sockets[0]);close(sockets[1]);last_error_="nonblocking TPU transport setup failed";return false;
    }
    const char* python = std::getenv("MIMIR_TPU_PYTHON");
    const char* worker = std::getenv("MIMIR_TPU_WORKER");
    const std::string executable = python && *python ? python : "python3";
    std::string script = worker && *worker ? worker : MIMIR_TPU_WORKER;
#if defined(__linux__)
    if ((!worker || !*worker) && !std::filesystem::is_regular_file(script)) {
        std::error_code error;
        const auto exe = std::filesystem::read_symlink("/proc/self/exe",error);
        if (!error) script=(exe.parent_path()/"../share/mimir/tpu/worker.py").lexically_normal().string();
    }
#endif
    const std::string device = std::to_string(cfg.device_index);
    const char* requested=std::getenv("MIMIR_TPU_TARGET");
    const std::string target=requested && *requested?requested:"auto";
    child_ = fork();
    if (child_ == 0) {
        close(sockets[0]);
        if (dup2(sockets[1], STDIN_FILENO) < 0 || dup2(sockets[1], STDOUT_FILENO) < 0) _exit(126);
        if (sockets[1] > STDOUT_FILENO) close(sockets[1]);
        execlp(executable.c_str(), executable.c_str(), "-u", script.c_str(), "--device", device.c_str(), "--target", target.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(sockets[1]); socket_=sockets[0];
    if (child_ < 0) { closeWorker(); last_error_="fork failed"; return false; }
    json response; std::vector<std::vector<float>> result;
    if (!exchange({{"command","hello"}}, {}, response, result) || response.value("platform", "") != "tpu") {
        if (last_error_.empty()) last_error_="worker did not confirm a TPU device";
        closeWorker(); return false;
    }
    device_=response.value("target", "TPU")+": "+response.value("device_kind", "")+
        " ["+response.value("device", "TPU")+"]"; initialized_=true;
    std::cerr << "[TPU] selected_runtime=TPU device=" << device_ << " execution=JAX/XLA/PJRT precision=F32\n";
    return true;
#else
    last_error_="TPU worker requires POSIX"; return false;
#endif
}
bool TpuRuntime::supportsForwardLayerType(LayerType t) const {
    return t >= LayerType::Conv2d && t < LayerType::UNKNOWN && t != LayerType::Lambda;
}
bool TpuRuntime::supportsBackwardLayerType(LayerType t) const { return supportsForwardLayerType(t); }
bool TpuRuntime::forward(const std::vector<const std::vector<float>*>& inputs,
                         std::vector<std::vector<float>>& outputs, const Layer& l,
                         bool training, bool fused, std::vector<uint8_t>* mask) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !supportsForwardLayerType(l.type_enum)) return false;
    for (auto* p : inputs) if (!p) return false;
    if (l.type_enum != LayerType::Constant && inputs.empty()) return false;
    std::vector<float> weights;
    if (l.getWeightsSize()) { if (!l.getWeights()) return false; weights.assign(l.getWeights(), l.getWeights()+l.getWeightsSize()); }
    auto arrays=inputs; arrays.push_back(&weights);
    json response; std::vector<std::vector<float>> result;
    if (!exchange({{"command","forward"},{"layer",Mimir::tpuLayerDescriptor(l)},{"training",training},
                   {"fused",fused},{"inputs",inputs.size()}},arrays,response,result)) throw std::runtime_error("TPU forward "+l.type+": "+last_error_);
    const size_t count=response.value("outputs",size_t(0));
    if (result.size()!=count+1 || count==0) return false;
    if (mask) { mask->clear(); for (float v:result.back()) mask->push_back(v!=0); }
    result.pop_back(); outputs=std::move(result); return true;
}
bool TpuRuntime::forwardLayer(const std::vector<const std::vector<float>*>& in,
    std::vector<std::vector<float>>& out,const Layer& l,bool training) { return forward(in,out,l,training,false,nullptr); }
bool TpuRuntime::forwardLayerWithContext(const std::vector<const std::vector<float>*>& in,
    std::vector<std::vector<float>>& out,const Layer& l,bool training,const RuntimeForwardContext& ctx) {
    return forward(in,out,l,training && (l.type_enum!=LayerType::Reparameterize || ctx.stochastic_latent),true,ctx.output_mask);
}
bool TpuRuntime::backwardLayer(const std::vector<const std::vector<float>*>& in,
    const std::vector<const std::vector<float>*>& go,std::vector<std::vector<float>>& gi,Layer& l,bool training) {
    return backwardLayerWithContext(in,go,gi,l,training,{});
}
bool TpuRuntime::backwardLayerWithContext(const std::vector<const std::vector<float>*>& in,
    const std::vector<const std::vector<float>*>& go,std::vector<std::vector<float>>& gi,Layer& l,bool training,const RuntimeBackwardContext& ctx) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !supportsBackwardLayerType(l.type_enum) || go.empty()) return false;
    std::vector<float> weights,mask,empty;
    if(l.getWeightsSize()) { if(!l.getWeights())return false; weights.assign(l.getWeights(),l.getWeights()+l.getWeightsSize()); }
    if(ctx.output_mask) for(uint8_t v:*ctx.output_mask)mask.push_back(v?1.f:0.f);
    auto arrays=in; arrays.push_back(&weights);
    arrays.insert(arrays.end(),go.begin(),go.end()); arrays.push_back(ctx.output?ctx.output:&empty); arrays.push_back(&mask);
    for(auto* p:arrays)if(!p)return false;
    json response; std::vector<std::vector<float>> result;
    if(!exchange({{"command","backward"},{"layer",Mimir::tpuLayerDescriptor(l)},{"training",training},
                 {"inputs",in.size()},{"grad_outputs",go.size()}},arrays,response,result))throw std::runtime_error("TPU backward "+l.type+": "+last_error_);
    const size_t count=response.value("grad_inputs",size_t(0));
    if(result.size()!=count+2 || result[count].size()!=weights.size())return false;
    auto& gw=result[count]; auto& gb=result[count+1];
    if(!gw.empty() && (l.type_enum!=LayerType::Constant || l.trainable_parameter)) {
        if(l.grad_weights.size()!=gw.size()) l.grad_weights.assign(gw.size(),0.f);
        for(size_t i=0;i<gw.size();++i)l.grad_weights[i]+=gw[i];
    }
    if(!gb.empty()) {
        if(l.grad_bias.size()!=gb.size())l.grad_bias.assign(gb.size(),0.f);
        for(size_t i=0;i<gb.size();++i)l.grad_bias[i]+=gb[i];
    }
    result.resize(count);gi=std::move(result);return true;
}
bool TpuRuntime::linearForward(const float* x,const float* w,const float* b,float* y,int batch,int i,int o) {
    if(!x||!w||!y||batch<=0||i<=0||o<=0)return false;
    Layer l("tpu/linear","Linear",0);l.in_features=i;l.out_features=o;l.seq_len=batch;l.use_bias=b!=nullptr;
    l.weights.assign(w,w+size_t(i)*o);if(b)l.weights.insert(l.weights.end(),b,b+o);
    std::vector<float> input(x,x+size_t(batch)*i);std::vector<std::vector<float>> out;
    if(!forwardLayer({&input},out,l,false)||out[0].size()!=size_t(batch)*o)return false;
    std::copy(out[0].begin(),out[0].end(),y);return true;
}
bool TpuRuntime::linearForwardTyped(const Mimir::TypedTensor& x,const Mimir::TypedTensor& w,
    const Mimir::TypedTensor* bias,Mimir::TypedTensor& out) {
    if(x.dtype()!=Mimir::DType::F32||w.dtype()!=Mimir::DType::F32||out.dtype()!=Mimir::DType::F32||
       x.shape().size()!=2||w.shape().size()!=2)return false;
    const int batch=x.shape()[0],in=x.shape()[1],outputs=w.shape()[0];
    if(w.shape()[1]!=in||out.shape()!=std::vector<int>({batch,outputs})||
       (bias&&(bias->dtype()!=Mimir::DType::F32||bias->shape()!=std::vector<int>({outputs}))))return false;
    return linearForward(static_cast<const float*>(x.data()),static_cast<const float*>(w.data()),
        bias?static_cast<const float*>(bias->data()):nullptr,static_cast<float*>(out.data()),batch,in,outputs);
}
bool TpuRuntime::mergeBranches(const std::vector<float>& a,const std::vector<float>& b,std::vector<float>& out,const Layer& l) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!initialized_)return false;
    json response;std::vector<std::vector<float>> result;
    if(!exchange({{"command","merge"},{"merge",static_cast<int>(l.merge_op)}},{&a,&b},response,result)||result.size()!=1)return false;
    out=std::move(result[0]);return true;
}
TpuRuntime* Mimir::tpuSessionRuntime() {
    static TpuRuntime runtime;
    static std::once_flag once;
    const char* enabled=std::getenv("MIMIR_TPU");
    if(!enabled||std::string(enabled)!="1")return nullptr;
    const auto cfg=RuntimeConfig::fromEnv("TPU");
    if(cfg.disabled)return nullptr;
    std::call_once(once,[&]() {
        if(!runtime.initialize(cfg))
            throw std::runtime_error("TPU requested but unavailable: "+runtime.lastError()+
                "; select a TPU Colab runtime and install jax[tpu]");
    });
    if(!runtime.isInitialized())throw std::runtime_error("TPU worker stopped: "+runtime.lastError());
    return &runtime;
}

bool TpuRuntime::optimizerStep(const RuntimeOptimizerRequest& request) {
    if(!supportsOptimizer() || (!optimizerDeviceSizeFits(request) || !optimizerRequestValid(request)))return false;
    const auto memory=optimizerPack(request);const auto args=optimizerParameters(request);
    std::vector<float> parameters(16);for(size_t i=0;i<parameters.size();++i)std::memcpy(&parameters[i],&args[3+i],4);
    std::vector<std::vector<float>> arrays;
    for(size_t i=0;i<4;++i)arrays.emplace_back(memory.begin()+i*request.count,memory.begin()+(i+1)*request.count);
    arrays.push_back(parameters);
    std::vector<const std::vector<float>*> pointers;for(const auto& a:arrays)pointers.push_back(&a);
    std::lock_guard<std::mutex> lock(mutex_);json response;std::vector<std::vector<float>> result;
    if(!exchange({{"command","optimizer"},{"kind",args[2]}},pointers,response,result) || result.size()!=3)return false;
    for(const auto& a:result)if(a.size()!=request.count)return false;
    auto updated=memory;
    std::copy(result[0].begin(),result[0].end(),updated.begin());
    std::copy(result[1].begin(),result[1].end(),updated.begin()+2*request.count);
    std::copy(result[2].begin(),result[2].end(),updated.begin()+3*request.count);
    return optimizerCommit(request,updated);
}

#include "runtimes/losses/LossDispatch.hpp"
bool TpuRuntime::reconstructionLoss(const RuntimeLossGrad::Request& r,RuntimeLossGrad::LossWithGrad& out) {
    if(!supportsReconstructionLoss() || !RuntimeLossGrad::deviceSizeFits(r))return false;
    auto memory=RuntimeLossGrad::pack(r);auto args=RuntimeLossGrad::parameters(r);
    std::vector<float> p(memory.begin(),memory.begin()+r.count),t(memory.begin()+r.count,memory.begin()+2*r.count),weights(memory.begin()+2*r.count,memory.begin()+3*r.count),params;
    for(size_t i=9;i<15;++i){float f;std::memcpy(&f,&args[i],4);params.push_back(f);}
    std::vector<float> coefficients(memory.begin()+12*r.count,memory.end()),mask(memory.begin()+5*r.count,memory.begin()+6*r.count);
    std::lock_guard<std::mutex> lock(mutex_);json response;std::vector<std::vector<float>> result;
    if(!exchange({{"command","reconstruction_loss"},{"op",args[2]},{"kind",args[7]},{"relation_only",args[8]!=0},{"width",r.width},{"height",r.height},{"channels",r.channels}},
                 {&p,&t,&weights,&params,&coefficients,&mask},response,result) || result.size()!=3 || result[0].size()!=1 || result[2].size()!=1)return false;
    const size_t expected=r.operation==RuntimeLossGrad::Operation::Pool?size_t(std::max(1,r.width/2))*std::max(1,r.height/2)*r.channels:r.count;
    if(result[1].size()!=expected)return false;
    RuntimeLossGrad::LossWithGrad candidate;candidate.loss=result[0][0];candidate.relation_loss=result[2][0];candidate.grad=std::move(result[1]);
    if(!std::isfinite(candidate.loss) || !std::all_of(candidate.grad.begin(),candidate.grad.end(),[](float x){return std::isfinite(x);}))return false;
    out=std::move(candidate);return true;
}
