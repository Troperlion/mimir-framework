#include "Model.hpp"
#include "runtimes/optimizers/OptimizerDispatch.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
#ifdef MIMIR_TEST_OPENCL
bool optimizerOpenCLReferenceInitialize();
void optimizerOpenCLReferenceRun(std::vector<float>&, const std::vector<int>&);
#endif
int main(int argc,char** argv) {try {
    const bool device=argc>1 && std::string(argv[1])=="opencl";
#ifdef MIMIR_TEST_OPENCL
    if (device && !optimizerOpenCLReferenceInitialize()) return 77;
#else
    if (device) return 77;
#endif
    for (int kind=0;kind<9;++kind) for (size_t count:{size_t{1},size_t{257},size_t{4097}})
        for (int variant=0;variant<((kind==4 || kind==8)?3:2);++variant) {
            Optimizer optimizer;
            optimizer.type=static_cast<OptimizerType>(kind);
            optimizer.adafactor_beta1=variant==1?0.7f:0.f;
            optimizer.adafactor_relative_step=variant==1;
            optimizer.adafactor_scale_parameter=variant!=1;
            std::vector<float> w(count),g(count),m(count,.001f),v(count,.01f);
            for (size_t i=0;i<count;++i) {w[i]=std::sin(float(i+1))*.3f;g[i]=float(int(i%11)-5)*.02f;}
            if (variant==2 && kind==4) for (size_t i=0;i<count;++i) w[i]=1e25f*float(i%3+1);
            if (variant==2 && kind==8) {
                std::fill(g.begin(),g.end(),1e-30f);
                std::fill(m.begin(),m.end(),0.f);std::fill(v.begin(),v.end(),0.f);
                optimizer.weight_decay=0;
            }
            auto expected_w=w,expected_m=m,expected_v=v;
            for (int step=1;step<=12;++step) {
                optimizer.step=step;
                // Supply extra stale moments too: unused buffers must stay untouched.
                RuntimeOptimizerRequest request{&optimizer,.01f,w.data(),g.data(),m.data(),v.data(),count};
                RuntimeOptimizerRequest expected{&optimizer,.01f,expected_w.data(),g.data(),
                    expected_m.data(),expected_v.data(),count};
                optimizerHostStep(expected);
                const auto layout=optimizerDeviceLayout(request);
                auto packed=optimizerPack(request,true);
                auto parameters=optimizerParameters(request,true);
                require(parameters.size()==26,"compact metadata size");
                require(optimizerResultCount(parameters)==layout.result_count,"result prefix");
                require(packed.size()==layout.total_count,"workspace size");
                require(optimizerInputCount(parameters)==layout.result_count+count,"upload contains scratch");
                require(optimizerInputCount(parameters)<=packed.size(),"upload exceeds allocation");
                if (kind!=4 && kind!=8) require(layout.total_count==layout.result_count+count,"unused scratch retained");
                if (device) {
#ifdef MIMIR_TEST_OPENCL
                    optimizerOpenCLReferenceRun(packed,parameters);
#endif
                } else {
                    RuntimeOptimizerRequest mapped{&optimizer,.01f,packed.data(),packed.data()+layout.gradients,
                        (layout.moment_flags&1)?packed.data()+layout.first_moment:nullptr,
                        (layout.moment_flags&2)?packed.data()+layout.second_moment:nullptr,count};
                    optimizerHostStep(mapped);
                }
                // Corrupt the temporary suffix: committing must only use active results.
                std::fill(packed.begin()+layout.result_count,packed.end(),std::numeric_limits<float>::quiet_NaN());
                require(optimizerCommit(request,packed,true),"compact commit");
                auto compare=[&](const auto& actual,const auto& wanted) {
                    for (size_t i=0;i<count;++i) if (!std::isfinite(actual[i]) ||
                        std::abs(actual[i]-wanted[i])>4e-5f+5e-4f*std::abs(wanted[i])) {
                        std::cerr<<"kind="<<kind<<" count="<<count<<" variant="<<variant<<" step="<<step<<" index="<<i
                            <<" actual="<<actual[i]<<" expected="<<wanted[i]<<'\n';
                        throw std::runtime_error("compact optimizer parity");
                    }
                };
                compare(w,expected_w);compare(m,expected_m);compare(v,expected_v);
                auto unchanged=w;
                packed.pop_back();
                require(!optimizerCommit(request,packed,true) && w==unchanged,"truncated buffer modified weights");
            }
        }
    std::cout<<"COMPACT_OPTIMIZERS_OK all=9 states=preserved mode="<<(device?"opencl":"host")<<'\n';
    return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}}
