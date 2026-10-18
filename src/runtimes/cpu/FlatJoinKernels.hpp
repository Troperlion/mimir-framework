#pragma once
#include <vector>
#include <cstddef>

namespace RuntimeFlatJoin {
// Existing Concat/Stack semantics flatten inputs; the axis is metadata only.
// Borrow the inputs rather than materializing an owning copy of every tensor.
inline bool forward(const std::vector<const std::vector<float>*>& inputs,
                    std::vector<float>& output, bool require_equal_sizes) {
    if (inputs.empty() || !inputs.front()) return false;
    const size_t first_size = inputs.front()->size();
    size_t total = 0;
    for (const auto* input : inputs) {
        if (!input || (require_equal_sizes && input->size() != first_size)) return false;
        if (input->size() > output.max_size() - total) return false;
        total += input->size();
    }
    std::vector<float> joined;
    joined.reserve(total);
    for (const auto* input : inputs)
        joined.insert(joined.end(), input->begin(), input->end());
    output = std::move(joined);
    return true;
}
}
