#include "test_utils.hpp"
#include "runtimes/tpu/TpuRuntime.hpp"
#include "Layers.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif
int main() {
    TpuRuntime rt;RuntimeConfig cfg;cfg.backend="TPU";cfg.disabled=true;
    TASSERT_TRUE(!rt.initialize(cfg));TASSERT_TRUE(!rt.isInitialized());
    TASSERT_TRUE(!rt.supportsForwardLayerType(LayerType::UNKNOWN));
    TASSERT_TRUE(!rt.supportsForwardLayerType(LayerType::Lambda));
    TASSERT_TRUE(rt.supportsForwardLayerType(LayerType::Conv2d));
    TASSERT_TRUE(rt.supportsBackwardLayerType(LayerType::PatchEmbed));
#if defined(__linux__) || defined(__APPLE__)
    // This is a transport mock, not a TPU numerical implementation.
    const auto path=std::filesystem::temp_directory_path()/("mimir-tpu-mock-"+std::to_string(getpid())+".py");
    struct Cleanup { std::filesystem::path p;~Cleanup(){std::filesystem::remove(p);} } cleanup{path};
    std::ofstream script(path);
    script<<R"PY(import array,json,struct,sys
r=sys.stdin.buffer;w=sys.stdout.buffer
while True:
    raw=r.read(4)
    if not raw:break
    req=json.loads(r.read(struct.unpack('<I',raw)[0]));arrays=[]
    for n in req['sizes']:
        a=array.array('f');a.frombytes(r.read(n*4));arrays.append(a)
    cmd=req['command'];out=[];meta={'ok':True}
    if cmd=='hello':meta.update(platform='cpu' if '--device' in sys.argv and sys.argv[sys.argv.index('--device')+1]=='1' else 'tpu',device='TRANSPORT_MOCK')
    elif cmd=='forward':
        if req['layer']['in_features']==99:
            meta.update(ok=False,error='injected TPU execution error')
        else:meta['outputs']=1;out=[arrays[0],array.array('f')]
    elif cmd=='backward':
        n=req['inputs'];meta['grad_inputs']=n
        out=[arrays[n+1]]+[array.array('f',[1.0]*len(arrays[n])),array.array('f')]
    meta['sizes']=[len(a) for a in out];header=json.dumps(meta).encode()
    w.write(struct.pack('<I',len(header)));w.write(header)
    for a in out:w.write(a.tobytes())
    w.flush()
)PY";
    script.close();
    const char* old_python=std::getenv("MIMIR_TPU_PYTHON");const std::string python=old_python?old_python:"";
    const char* old_worker=std::getenv("MIMIR_TPU_WORKER");const std::string worker=old_worker?old_worker:"";
    setenv("MIMIR_TPU_PYTHON","/nonexistent/mimir-python",1);cfg.disabled=false;
    TASSERT_TRUE(!rt.initialize(cfg));TASSERT_TRUE(!rt.lastError().empty());
    setenv("MIMIR_TPU_PYTHON","python3",1);setenv("MIMIR_TPU_WORKER",path.c_str(),1);
    cfg.device_index=1;TASSERT_TRUE(!rt.initialize(cfg)); // CPU handshake must be rejected.
    cfg.device_index=0;TASSERT_TRUE(rt.initialize(cfg));
    Layer layer("identity","Identity",0);std::vector<float> x={1,-2,.5f},g={.1f,.2f,.3f};
    std::vector<std::vector<float>> out,gi;
    TASSERT_TRUE(rt.forwardLayer({&x},out,layer,false));TASSERT_TRUE(out.size()==1 && out[0]==x);
    layer.weights={.7f,.8f};
    TASSERT_TRUE(rt.backwardLayer({&x},{&g},gi,layer,true));TASSERT_TRUE(gi.size()==1 && gi[0]==g);
    TASSERT_TRUE(layer.grad_weights==std::vector<float>({1,1}));
    TASSERT_TRUE(rt.backwardLayer({&x},{&g},gi,layer,true));
    TASSERT_TRUE(layer.grad_weights==std::vector<float>({2,2}));
    layer.in_features=99;bool rejected=false;
    try {rt.forwardLayer({&x},out,layer,false);}catch(const std::runtime_error&){rejected=true;}
    TASSERT_TRUE(rejected);rt.shutdown();TASSERT_TRUE(!rt.isInitialized());
    if(old_python)setenv("MIMIR_TPU_PYTHON",python.c_str(),1);else unsetenv("MIMIR_TPU_PYTHON");
    if(old_worker)setenv("MIMIR_TPU_WORKER",worker.c_str(),1);else unsetenv("MIMIR_TPU_WORKER");
#endif
    return 0;
}
