#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace PortableStaging {
// Kernels without independent output ownership still run their logical grid
// on one GPU invocation. These limits remain in force for that serial path.
// Keep them bounded; large operations must use a parallel kernel or router fallback.
inline constexpr size_t referenceMaxElements = 65536;
inline constexpr uint64_t referenceMaxWork = 4 * 1024 * 1024;

inline bool referenceInputsFit(size_t weights,
    const std::vector<const std::vector<float>*>& inputs) {
    if (weights > referenceMaxElements) return false;
    for (const auto* input : inputs) {
        if (!input) continue;
        if (input->size() > referenceMaxElements - weights) return false;
        weights += input->size();
    }
    return true;
}

inline bool referenceDispatchFits(size_t elements, const std::vector<int>& args) {
    if (elements > referenceMaxElements || args.size() < 3 ||
        args[1] <= 0 || args[2] <= 0) return false;
    const uint64_t grid = uint64_t(args[1]) * uint64_t(args[2]);
    if (args[0] == 11) { // row_norm_grad: each logical thread reduces rows
        return args.size() >= 9 && args[7] > 0 && args[8] > 0 &&
            grid <= referenceMaxWork / uint64_t(args[7]);
    }
    // Elementwise kernels have no full-tensor reduction per logical thread.
    switch (args[0]) {
        case 3: case 5: case 6: case 13: case 19: case 20: case 22:
        case 33: case 34: case 35: case 36: case 37:
        case 41: case 42: // positional table interpolation and its adjoint
            return grid <= referenceMaxWork / 16;
        case 38: // portable_copy loops over the staged tensor once
            return grid <= referenceMaxWork / (elements ? elements : 1);
        default: break;
    }
    if (args[0] >= 23 && args[0] <= 30) {
        const size_t geometry = args[0] <= 25 ? 6 : args[0] <= 27 ? 7 :
            args[0] == 28 ? 9 : args[0] == 29 ? 11 : 8;
        if (args.size() < geometry + 11) return false;
        const int ed = args[geometry], ked = args[geometry+1], heads = args[geometry+2];
        const int q = args[geometry+3], k = args[geometry+4];
        if (ed <= 0 || ked <= 0 || heads <= 0 || q <= 0 || k <= 0) return false;
        uint64_t loops = uint64_t(ed) + uint64_t(ked) + uint64_t(q) + uint64_t(k) + 4;
        if (args[0] == 24 || args[0] == 27)
            loops = uint64_t(k) * (uint64_t(ed / heads) + 4);
        return loops <= referenceMaxWork && grid <= referenceMaxWork / loops;
    }
    // Conservative bound for serial loops and per-thread reductions.
    if (grid > referenceMaxWork / (elements ? elements : 1)) return false;
    if (args[0] == 39) { // portable_gemm: pointers, m, n, k
        if (args.size() < 9) return false;
        uint64_t work = 1;
        for (size_t i = 6; i < 9; ++i) {
            if (args[i] <= 0 || uint64_t(args[i]) > referenceMaxWork / work) return false;
            work *= uint64_t(args[i]);
        }
    }
    return true;
}
// Only kernels with independent output ownership may use a parallel grid.
inline size_t parallelInvocationCount(const std::vector<int>& a) {
    if(a.size()<3 || a[1]<=0 || a[2]<=0)return 0;
    const int id=a[0];
    if(id==38)return a.size()>5 && a[5]>0?size_t(a[5]):0;
    if(id==39)return a.size()>7 && a[6]>0 && a[7]>0?((size_t(a[6])+15)/16)*((size_t(a[7])+15)/16)*256:0;
    if(id==40)return a.size()>7 && a[6]>0 && a[7]>0?size_t(a[6])*size_t(a[7]):0;
    switch(id){case 0:case 1:case 2:case 7:case 10:case 15:case 21:case 24:case 27:return size_t(a[2])*256;default:break;}
    if(id>=0 && id<=42 && id!=31 && id!=32)return size_t(a[1])*size_t(a[2]);
    return 0;
}
inline bool parallelDispatchFits(size_t elements,const std::vector<int>& a) {
    const size_t count=parallelInvocationCount(a);
    if(!count || count>16*1024*1024 || elements>64*1024*1024)return false;
    // Reduction workgroups cooperate over a bounded window. Flat kernels also
    // have a total work cap, especially for CAS scatter contention.
    const uint64_t limit=128*1024*1024;
    uint64_t loops=1,items=uint64_t(a[1])*a[2];
    auto positive=[&](size_t at)->uint64_t{return at<a.size() && a[at]>0?uint64_t(a[at]):0;};
    switch(a[0]){
      case 0:case 1:loops=positive(8);break;
      case 2:loops=positive(7);break;
      case 7:loops=positive(6);break;
      case 8:loops=1;break;
      case 9:loops=positive(8)*positive(8);break;
      case 10:loops=positive(8);break;
      case 11:loops=positive(7);break;
      case 12:
        if(a.size()<24)return false;
        loops=a[11]==1?1:a[11]==2?positive(16):a[11]==3?positive(17):positive(17)*positive(18);
        if(a[6]!=0 && positive(15)>65536)return false;
        break;
      case 14:loops=32;break;
      case 15:loops=positive(10)*positive(11);break;
      case 16:loops=positive(7);break;
      case 18:loops=positive(14)*positive(15)*(a.size()>9 && a[9]==2?1:positive(10));break;
      case 21:loops=positive(10)*positive(11);break;
      case 23:case 25:case 26:case 28:case 29:case 30:{
        size_t g=a[0]<=25?6:a[0]<=27?7:a[0]==28?9:a[0]==29?11:8;
        loops=std::max({positive(g),positive(g+1),positive(g+3),positive(g+4)});break;
      }
      case 24:case 27:{size_t g=a[0]==24?6:7;const auto heads=positive(g+2);if(!heads)return false;loops=positive(g+4)*(positive(g)/heads+4);break;}
      case 42:if(a.size()<9 || !positive(6) || positive(8)/positive(6)>65536)return false;break;
      case 5:if(!positive(6) || !positive(7) || positive(8)*positive(9)>4096*positive(6)*positive(7))return false;break;
      case 17:if(a.size()<16 || (a[15]!=0 && items>4096*positive(6)*positive(7)*positive(8)))return false;break;
      case 39:return a.size()>8 && a[8]>0 && a[8]<=65536 && positive(6)*positive(7)<=limit/positive(8);
      default:break;
    }
    if(!loops || loops>1024*1024 || items>limit/loops)return false;
    return true;
}

}
