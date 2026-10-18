#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace PortableStaging {
// Reference kernels run the entire logical grid on ONE GPU invocation.
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
}
