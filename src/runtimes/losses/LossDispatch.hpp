#pragma once
#include "runtimes/ops_loss_and_grad.hpp"
#include <climits>
namespace RuntimeLossGrad {
// Internal only: the public reconstruction loss API remains unchanged.
enum class Operation { Pixel, SSIM, Pool, PoolBackward, Spectral, CrossEntropy };
struct Request {
    Operation operation=Operation::Pixel;
    const float* prediction=nullptr;
    const float* target=nullptr;
    size_t count=0;
    int width=1,height=1,channels=1;
    std::string type="mse";
    PixelLossOptions options;
    float k1=.01f,k2=.03f,range=2.f;
};
namespace Host {
LossWithGrad pixel_loss_and_grad(const float*,const float*,size_t,const std::string&,const PixelLossOptions&);
LossWithGrad pixel_loss_and_grad(const std::vector<float>&,const std::vector<float>&,const std::string&,const PixelLossOptions&);
GlobalSSIM ssim_global_hwc(const std::vector<float>&,const std::vector<float>&,int,int,int,float,float,float);
void avgpool2x2_hwc(const std::vector<float>&,int,int,int,std::vector<float>&,int&,int&);
void avgpool2x2_back_hwc(const std::vector<float>&,int,int,int,std::vector<float>&);
LossWithGrad spectral_dct_l1_hwc(const std::vector<float>&,const std::vector<float>&,int,int,int);
}
LossWithGrad text_cross_entropy(const float*,size_t,const std::vector<int>&,int,int,int);
bool valid(const Request&);
bool deviceSizeFits(const Request&);
bool host(const Request&,LossWithGrad&);
std::vector<float> pack(const Request&);
std::vector<int> parameters(const Request&);
bool unpack(const Request&,const std::vector<float>&,LossWithGrad&);
std::vector<int> passes(const std::vector<int>&);
bool portable(const Request&,LossWithGrad&,const char*,int);
}
