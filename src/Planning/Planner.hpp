#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Layers.hpp"
#include "Planning/ExecutionCalibration.hpp"
#include "runtimes/AbstractRuntime.hpp"

namespace Mimir::Planning {

// --------------------------
// Tensor lifetime analysis
// --------------------------

struct TensorLifetime {
    // Indices of layers in the static schedule.
    // first_def: first layer that defines the value
    // last_use: last layer that reads the value
    int first_def = -1;
    int last_use = -1;

    // Metadata best-effort.
    std::string dtype = "float32";
    std::vector<int> shape;
};

// Very lightweight lifetime analysis for the current Model execution semantics:
// - schedule is the layer order (static scheduling)
// - routing is by named tensors (layer.inputs / layer.output)
// - multiple writes to the same name are treated as overwriting the same slot
//   (so we track lifetime per tensor name, not SSA versions)
// This is enough to drive safe "reuse once dead" decisions for scratch buffers.
inline std::unordered_map<std::string, TensorLifetime>
analyze_tensor_lifetimes(const std::vector<Layer>& layers) {
    std::unordered_map<std::string, TensorLifetime> lifetimes;

    auto get_inputs = [](const Layer& l) -> const std::vector<std::string>& {
        static const std::vector<std::string> kDefaultX = {"x"};
        return l.inputs.empty() ? kDefaultX : l.inputs;
    };

    for (int i = 0; i < static_cast<int>(layers.size()); ++i) {
        const Layer& layer = layers[static_cast<size_t>(i)];

        // Record uses
        for (const auto& in : get_inputs(layer)) {
            auto& lt = lifetimes[in];
            if (lt.first_def < 0) lt.first_def = 0; // assume defined at graph input
            lt.last_use = std::max(lt.last_use, i);
        }

        // Record def
        const std::string out = layer.output.empty() ? "x" : layer.output;
        auto& lt_out = lifetimes[out];
        if (lt_out.first_def < 0) lt_out.first_def = i;
        // If overwritten, lifetime restarts; keep first_def as earliest seen for this slot.
        lt_out.last_use = std::max(lt_out.last_use, i);
    }

    return lifetimes;
}

// --------------------------
// Fusion + execution plan
// --------------------------

enum class FusionKind : uint8_t {
    NONE = 0,
    CONV2D_RELU = 1,
    GENERIC_ACTIVATION = 2,
    GENERIC_SPLIT = 3,
    GENERIC_CHUNK = 4,
    GENERIC_ACTIVATION_SPLIT = 5,
    GENERIC_ACTIVATION_CHUNK = 6,
    GENERIC_UNARY_SHAPE = 7,
};

enum class PlannerMode : uint8_t { Legacy, Static, Cost };

struct PlannedOp {
    int layer_index = -1;
    FusionKind fusion = FusionKind::NONE;
};

struct ExecutionPlan {
    // Static scheduling: fixed op order
    std::vector<PlannedOp> ops;

    // Skip fused consumer layers at runtime.
    std::vector<uint8_t> skip_layer;

    // Per-layer fusion flags for cheap hot-path checks
    std::vector<uint8_t> fuse_relu_for_conv2d;

    // Generic inference-time fusions.
    std::vector<int> fuse_activation_consumer;
    std::vector<int> fuse_unary_consumer;
    std::vector<int> fuse_split_consumer;
    std::vector<uint8_t> fuse_split_kind; // 0=none, 1=Split, 2=Chunk

    // Generic chained fusion edges (producer -> fused consumer).
    // Allows fusing repeated adjacent layers (e.g. op->act->act->shape->shape).
    std::vector<int> fuse_chain_next;

    // Global-plan view. Historical fields above intentionally remain valid.
    struct TensorId {
        std::string name;
        size_t generation = 0;
        bool operator==(const TensorId& other) const {
            return generation == other.generation && name == other.name;
        }
        bool operator!=(const TensorId& other) const { return !(*this == other); }
    };
    struct TensorIdHash {
        size_t operator()(const TensorId& id) const {
            return std::hash<std::string>{}(id.name) ^
                   (std::hash<size_t>{}(id.generation) << 1u);
        }
    };
    enum class MemoryLocation : uint8_t { Host, Vulkan, OpenCL, CUDA, ROCm, FPGA, TPU };
    struct PlannedTensor {
        TensorId id;
        std::string name;
        size_t generation = 0;
        size_t bytes = 0;
        Mimir::DType dtype = Mimir::DType::UNKNOWN;
        std::vector<int> shape;
        size_t first_definition = 0;
        size_t last_use = 0;
        size_t backward_last_use = 0;
        RuntimeKind producer_runtime = RuntimeKind::CPU;
        MemoryLocation preferred_location = MemoryLocation::Host;
        size_t physical_buffer_id = std::numeric_limits<size_t>::max();
        bool persistent = false;
        bool parameter = false;
        bool graph_input = false;
        bool graph_output = false;
        bool required_for_backward = false;
    };
    struct PlannedLayer {
        size_t layer_index = 0;
        std::string layer_name;
        LayerType layer_type = LayerType::UNKNOWN;
        RuntimeKind runtime = RuntimeKind::CPU;
        RuntimeCapabilityLevel capability = RuntimeCapabilityLevel::Unsupported;
        std::vector<TensorId> inputs;
        TensorId output;
        bool skip = false;
        bool graph_fused = false;
        bool kernel_fused = false;
        std::optional<size_t> fused_into;
        size_t scratch_bytes = 0;
        std::vector<Calibration::Measurement> measured_costs;
        bool measured_host_roundtrip = false;
        double estimated_compute_cost = 0.0;
        double estimated_transfer_cost = 0.0;
    };
    struct PlannedTransfer {
        TensorId tensor;
        MemoryLocation from = MemoryLocation::Host;
        MemoryLocation to = MemoryLocation::Host;
        size_t bytes = 0;
        double estimated_cost = 0.0;
    };
    struct DeviceRegion {
        size_t id = 0;
        RuntimeKind runtime = RuntimeKind::CPU;
        std::vector<size_t> layer_indices;
    };
    struct PlannedFusionGroup {
        size_t producer = 0;
        std::vector<size_t> layers;
        bool graph_fused = false;
        bool kernel_fused = false;
    };
    struct Stats {
        size_t tensor_logical_bytes = 0;
        size_t tensor_physical_peak = 0;
        size_t buffer_reuse_saved = 0;
        size_t transfer_bytes = 0;
        size_t scratch_peak = 0;
        size_t buffer_reuse_count = 0;
        size_t buffer_reuse_bytes = 0;
    } stats;

    std::vector<PlannedLayer> layers;
    std::unordered_map<TensorId, PlannedTensor, TensorIdHash> tensors;
    std::vector<PlannedTransfer> transfers;
    std::vector<PlannedFusionGroup> fusion_groups;
    std::vector<DeviceRegion> regions;
    bool training = false;
    PlannerMode mode = PlannerMode::Static;

    bool empty() const { return ops.empty(); }
};

using TensorId = ExecutionPlan::TensorId;
using PlannedTensor = ExecutionPlan::PlannedTensor;
using PlannedLayer = ExecutionPlan::PlannedLayer;
using PlannedTransfer = ExecutionPlan::PlannedTransfer;
using PlannedFusionGroup = ExecutionPlan::PlannedFusionGroup;
using DeviceRegion = ExecutionPlan::DeviceRegion;
using MemoryLocation = ExecutionPlan::MemoryLocation;


inline const std::vector<std::string>& planner_inputs_for(const Layer& layer) {
    static const std::vector<std::string> kDefaultX = {"x"};
    return (layer.inputs.empty() && layer.type_enum != LayerType::Constant) ? kDefaultX : layer.inputs;
}

inline std::string planner_output_name_for(const Layer& layer) {
    return layer.output.empty() ? "x" : layer.output;
}

inline bool is_fusible_activation_layer(const Layer& layer) {
    switch (layer.type_enum) {
        case LayerType::ReLU:
        case LayerType::LeakyReLU:
        case LayerType::GELU:
        case LayerType::GEGLU:
        case LayerType::SiLU:
        case LayerType::Tanh:
        case LayerType::Sigmoid:
        case LayerType::Softmax:
        case LayerType::LogSoftmax:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish:
            return true;
        default:
            return false;
    }
}

inline bool is_fusible_split_layer(const Layer& layer) {
    return layer.type_enum == LayerType::Split || layer.type_enum == LayerType::Chunk;
}

inline bool is_fusible_unary_shape_layer(const Layer& layer) {
    switch (layer.type_enum) {
        case LayerType::Identity:
        case LayerType::Flatten:
        case LayerType::Reshape:
        case LayerType::View:
        case LayerType::Transpose:
        case LayerType::Permute:
        case LayerType::Squeeze:
        case LayerType::Unsqueeze:
            return true;
        default:
            return false;
    }
}

inline ExecutionPlan build_execution_plan_static(const std::vector<Layer>& layers, bool training,
                                                  bool allow_training_fusion = false,
                                                  bool fusion_enabled = true) {
    ExecutionPlan plan;
    plan.ops.reserve(layers.size());
    plan.skip_layer.assign(layers.size(), 0);
    plan.fuse_relu_for_conv2d.assign(layers.size(), 0);
    plan.fuse_activation_consumer.assign(layers.size(), -1);
    plan.fuse_unary_consumer.assign(layers.size(), -1);
    plan.fuse_split_consumer.assign(layers.size(), -1);
    plan.fuse_split_kind.assign(layers.size(), 0);
    plan.fuse_chain_next.assign(layers.size(), -1);

    std::unordered_map<std::string, int> tensor_use_count;
    for (const auto& layer : layers) {
        for (const auto& in : planner_inputs_for(layer)) {
            ++tensor_use_count[in];
        }
    }

    auto consumes_single_tensor = [](const Layer& consumer, const std::string& expected_input) -> bool {
        const auto& inputs = planner_inputs_for(consumer);
        return inputs.size() == 1 && inputs[0] == expected_input;
    };

    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];

        PlannedOp op;
        op.layer_index = static_cast<int>(i);
        op.fusion = FusionKind::NONE;

        if (plan.skip_layer[i] != 0) {
            plan.ops.push_back(op);
            continue;
        }

        const bool can_fuse_conv_relu = fusion_enabled && ((!training) || allow_training_fusion);
        const bool can_fuse_generic = fusion_enabled && !training;

        // Conservative by default: in training, fusions are disabled unless explicitly enabled.
        if (can_fuse_conv_relu && layer.type_enum == LayerType::Conv2d && layer.activation == ActivationType::RELU) {
            op.fusion = FusionKind::CONV2D_RELU;
            plan.fuse_relu_for_conv2d[i] = 1;
        }

        if (can_fuse_generic) {
            const std::string producer_out = planner_output_name_for(layer);
            const bool producer_has_single_consumer = tensor_use_count[producer_out] == 1;

            auto maybe_mark_split = [&](size_t producer_idx, size_t consumer_idx, bool after_activation) {
                const Layer& split_layer = layers[consumer_idx];
                plan.fuse_split_consumer[producer_idx] = static_cast<int>(consumer_idx);
                plan.fuse_split_kind[producer_idx] = (split_layer.type_enum == LayerType::Split) ? 1 : 2;
                plan.skip_layer[consumer_idx] = 1;
                plan.fuse_chain_next[producer_idx] = static_cast<int>(consumer_idx);
                if (after_activation) {
                    op.fusion = (split_layer.type_enum == LayerType::Split)
                        ? FusionKind::GENERIC_ACTIVATION_SPLIT
                        : FusionKind::GENERIC_ACTIVATION_CHUNK;
                } else {
                    op.fusion = (split_layer.type_enum == LayerType::Split)
                        ? FusionKind::GENERIC_SPLIT
                        : FusionKind::GENERIC_CHUNK;
                }
            };

            if (producer_has_single_consumer && (i + 1) < layers.size()) {
                int chain_cursor = static_cast<int>(i);
                std::string chain_output = producer_out;
                bool first_in_chain = true;

                while (true) {
                    const size_t next_idx = static_cast<size_t>(chain_cursor + 1);
                    if (next_idx >= layers.size()) break;

                    const Layer& next = layers[next_idx];
                    if (!consumes_single_tensor(next, chain_output)) break;
                    if (tensor_use_count[chain_output] != 1) break;

                    if (is_fusible_activation_layer(next)) {
                        plan.skip_layer[next_idx] = 1;
                        plan.fuse_chain_next[static_cast<size_t>(chain_cursor)] = static_cast<int>(next_idx);
                        if (first_in_chain) {
                            plan.fuse_activation_consumer[i] = static_cast<int>(next_idx);
                            if (op.fusion == FusionKind::NONE) {
                                op.fusion = FusionKind::GENERIC_ACTIVATION;
                            }
                        }
                        chain_cursor = static_cast<int>(next_idx);
                        chain_output = planner_output_name_for(next);
                        first_in_chain = false;
                        continue;
                    }

                    if (is_fusible_unary_shape_layer(next)) {
                        plan.skip_layer[next_idx] = 1;
                        plan.fuse_chain_next[static_cast<size_t>(chain_cursor)] = static_cast<int>(next_idx);
                        if (first_in_chain) {
                            plan.fuse_unary_consumer[i] = static_cast<int>(next_idx);
                            if (op.fusion == FusionKind::NONE) {
                                op.fusion = FusionKind::GENERIC_UNARY_SHAPE;
                            }
                        }
                        chain_cursor = static_cast<int>(next_idx);
                        chain_output = planner_output_name_for(next);
                        first_in_chain = false;
                        continue;
                    }

                    if (is_fusible_split_layer(next)) {
                        const bool after_activation = (op.fusion == FusionKind::GENERIC_ACTIVATION);
                        maybe_mark_split(static_cast<size_t>(chain_cursor), next_idx, after_activation);
                        // Keep the legacy root-producer metadata coherent even
                        // when the split is reached through a fused chain.
                        plan.fuse_split_consumer[i] = static_cast<int>(next_idx);
                        plan.fuse_split_kind[i] = (next.type_enum == LayerType::Split) ? 1 : 2;
                        break;
                    }

                    break;
                }
            }
        }

        plan.ops.push_back(op);
    }

    return plan;
}

inline RuntimeKind runtime_kind_from_name(const char* name) {
    const std::string value = name ? name : "";
    if (value == "CPU") return RuntimeKind::CPU;
    if (value == "VULKAN") return RuntimeKind::Vulkan;
    if (value == "OPENCL") return RuntimeKind::OpenCL;
    if (value == "CUDA") return RuntimeKind::CUDA;
    if (value == "ROCM") return RuntimeKind::ROCm;
    if (value == "TPU") return RuntimeKind::TPU;
    if (value == "FPGA") return RuntimeKind::FPGA;
    return RuntimeKind::Unknown;
}

inline MemoryLocation memory_location_for_runtime(const RuntimeKind runtime) {
    switch (runtime) {
        case RuntimeKind::Vulkan: return MemoryLocation::Vulkan;
        case RuntimeKind::OpenCL: return MemoryLocation::OpenCL;
        case RuntimeKind::CUDA: return MemoryLocation::CUDA;
        case RuntimeKind::ROCm: return MemoryLocation::ROCm;
        case RuntimeKind::TPU: return MemoryLocation::TPU;
        case RuntimeKind::FPGA: return MemoryLocation::FPGA;
        default: return MemoryLocation::Host;
    }
}

inline size_t planned_tensor_bytes(const Layer& layer) {
    const size_t item_size = Mimir::dtype_size_bytes(layer.dtype);
    if (item_size == 0 || layer.shape.empty()) return 0;
    size_t elements = 1;
    for (const int dimension : layer.shape) {
        if (dimension <= 0 || elements > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension)) {
            return 0;
        }
        elements *= static_cast<size_t>(dimension);
    }
    return elements * item_size;
}

inline ExecutionPlan build_execution_plan_global(
    const std::vector<Layer>& graph,
    const bool training,
    const PlannerMode mode = PlannerMode::Static,
    const bool fusion_enabled = true,
    const bool buffer_reuse = true
) {
    ExecutionPlan plan = build_execution_plan_static(graph, training, false, fusion_enabled);
    plan.training = training;
    plan.mode = mode;
    plan.layers.reserve(graph.size());

    std::unordered_map<std::string, TensorId> current;
    std::unordered_map<std::string, size_t> next_generation;
    auto ensure_input = [&](const std::string& name) -> TensorId {
        const auto found = current.find(name);
        if (found != current.end()) return found->second;
        TensorId id{name, 0};
        current.emplace(name, id);
        next_generation[name] = 1;
        PlannedTensor tensor;
        tensor.id = id;
        tensor.name = name;
        tensor.graph_input = true;
        tensor.persistent = true;
        tensor.first_definition = 0;
        tensor.last_use = 0;
        plan.tensors.emplace(id, std::move(tensor));
        return id;
    };

    for (size_t index = 0; index < graph.size(); ++index) {
        const Layer& layer = graph[index];
        PlannedLayer planned;
        planned.layer_index = index;
        planned.layer_name = layer.name;
        planned.layer_type = layer.type_enum;
        planned.skip = index < plan.skip_layer.size() && plan.skip_layer[index] != 0;
        planned.graph_fused = planned.skip ||
            (index < plan.fuse_chain_next.size() && plan.fuse_chain_next[index] >= 0);

        for (const std::string& input_name : planner_inputs_for(layer)) {
            const TensorId input = ensure_input(input_name);
            planned.inputs.push_back(input);
            auto& tensor = plan.tensors.at(input);
            tensor.last_use = std::max(tensor.last_use, index);
        }

        const std::string output_name = planner_output_name_for(layer);
        size_t generation = 0;
        const auto existing = current.find(output_name);
        if (existing != current.end()) generation = next_generation[output_name]++;
        else next_generation[output_name] = 1;
        const TensorId output{output_name, generation};
        current[output_name] = output;
        planned.output = output;

        PlannedTensor tensor;
        tensor.id = output;
        tensor.name = output_name;
        tensor.generation = generation;
        tensor.bytes = planned_tensor_bytes(layer);
        tensor.dtype = layer.dtype;
        tensor.shape = layer.shape;
        tensor.first_definition = index;
        tensor.last_use = index;
        tensor.required_for_backward = training;
        tensor.backward_last_use = training ? graph.size() * 2 - index : index;
        plan.stats.tensor_logical_bytes += tensor.bytes;
        plan.tensors.emplace(output, std::move(tensor));
        plan.layers.push_back(std::move(planned));
    }

    if (!plan.layers.empty()) {
        auto& output = plan.tensors.at(plan.layers.back().output);
        output.graph_output = true;
        output.persistent = true;
    }
    if (buffer_reuse) {
        // Applied below after the function definition becomes visible.
    }
    return plan;
}

inline void plan_host_buffer_reuse(ExecutionPlan& plan, const bool enabled) {
    struct Slot { size_t id; size_t bytes; size_t last_use; Mimir::DType dtype; };
    std::vector<Slot> slots;
    size_t next_slot = 0;
    std::vector<PlannedTensor*> ordered;
    ordered.reserve(plan.tensors.size());
    for (auto& entry : plan.tensors) ordered.push_back(&entry.second);
    std::sort(ordered.begin(), ordered.end(), [](const PlannedTensor* a, const PlannedTensor* b) {
        if (a->first_definition != b->first_definition) return a->first_definition < b->first_definition;
        if (a->name != b->name) return a->name < b->name;
        return a->generation < b->generation;
    });
    size_t physical = 0;
    for (PlannedTensor* tensor : ordered) {
        if (tensor->graph_input || tensor->parameter || tensor->persistent || tensor->bytes == 0) continue;
        Slot* selected = nullptr;
        if (enabled && !plan.training && !tensor->required_for_backward) {
            for (Slot& slot : slots) {
                if (slot.last_use < tensor->first_definition && slot.bytes >= tensor->bytes && slot.dtype == tensor->dtype) {
                    if (!selected || slot.bytes < selected->bytes) selected = &slot;
                }
            }
        }
        if (!selected) {
            slots.push_back({next_slot++, tensor->bytes, tensor->last_use, tensor->dtype});
            selected = &slots.back();
            physical += tensor->bytes;
        } else {
            ++plan.stats.buffer_reuse_count;
            plan.stats.buffer_reuse_saved += tensor->bytes;
        }
        tensor->physical_buffer_id = selected->id;
        selected->last_use = plan.training ? tensor->backward_last_use : tensor->last_use;
    }
    plan.stats.tensor_physical_peak = physical;
}

inline double estimated_transfer_cost(const size_t bytes, MemoryLocation from, MemoryLocation to) {
    if (from == to) return 0.0;
    return 5e-6 + static_cast<double>(bytes) / 12.0e9;
}

inline void apply_runtime_placement(ExecutionPlan& plan,
                                    const std::vector<AbstractRuntime*>& priority) {
    plan.transfers.clear();
    plan.regions.clear();
    plan.stats.transfer_bytes = 0;
    for (PlannedLayer& layer : plan.layers) layer.estimated_transfer_cost = 0.0;
    RuntimeKind previous = RuntimeKind::Unknown;
    for (PlannedLayer& layer : plan.layers) {
        AbstractRuntime* selected = nullptr;
        double selected_cost = std::numeric_limits<double>::infinity();
        layer.measured_host_roundtrip=false;
        bool have_measurements=false;
        if(plan.mode==PlannerMode::Cost)for(const auto& m:layer.measured_costs)
            if(std::find(priority.begin(),priority.end(),m.runtime)!=priority.end() && m.runtime->isInitialized() && m.configuration==Calibration::configuration(*m.runtime))have_measurements=true;
        for (AbstractRuntime* runtime : priority) {
            if (!runtime || !runtime->isInitialized()) continue;
            const auto capability = runtime->queryForwardCapability(layer.layer_type);
            if (runtimeCapabilityIsNative(capability)) {
                if (plan.mode != PlannerMode::Cost) {
                    selected = runtime;
                    layer.capability = capability;
                    break;
                }
                if(have_measurements){
                    for(const auto& m:layer.measured_costs)if(m.runtime==runtime && m.configuration==Calibration::configuration(*runtime) && m.seconds<selected_cost){
                        selected=runtime;selected_cost=m.seconds;layer.capability=capability;layer.estimated_compute_cost=m.seconds;layer.measured_host_roundtrip=true;
                    }
                    continue;
                }
                const RuntimeKind candidate_kind = runtime_kind_from_name(runtime->name());
                const MemoryLocation candidate_location = memory_location_for_runtime(candidate_kind);
                double transfer_cost = 0.0;
                size_t work_bytes = 0;
                for (const TensorId& input : layer.inputs) {
                    const auto& tensor = plan.tensors.at(input);
                    work_bytes += tensor.bytes;
                    transfer_cost += estimated_transfer_cost(
                        tensor.bytes, tensor.preferred_location, candidate_location);
                }
                const auto& output = plan.tensors.at(layer.output);
                work_bytes += output.bytes;
                transfer_cost += estimated_transfer_cost(output.bytes,candidate_location,MemoryLocation::Host);
                double throughput = 1.0e9;
                switch (candidate_kind) {
                    case RuntimeKind::CUDA:
                    case RuntimeKind::ROCm: throughput = 8.0e9; break;
                    case RuntimeKind::Vulkan: throughput = 4.0e9; break;
                    case RuntimeKind::OpenCL: throughput = 3.0e9; break;
                    case RuntimeKind::TPU: throughput = 8.0e9; break; // conservative, no measured TPU cost model
                    case RuntimeKind::FPGA: throughput = 2.0e9; break;
                    default: break;
                }
                const double compute_cost = static_cast<double>(std::max<size_t>(1, work_bytes)) / throughput;
                const double total = compute_cost + transfer_cost;
                if (total < selected_cost) {
                    selected_cost = total;
                    selected = runtime;
                    layer.capability = capability;
                    layer.estimated_compute_cost = compute_cost;
                }
            }
        }
        if (plan.mode != PlannerMode::Cost) {
            const size_t bytes = plan.tensors.at(layer.output).bytes;
            layer.estimated_compute_cost = static_cast<double>(std::max<size_t>(1, bytes)) / 1.0e9;
        }
        layer.runtime = selected ? runtime_kind_from_name(selected->name()) : RuntimeKind::CPU;
        const MemoryLocation destination = memory_location_for_runtime(layer.runtime);
        for (const TensorId& input : layer.inputs) {
            auto& tensor = plan.tensors.at(input);
            if (tensor.preferred_location != destination) {
                PlannedTransfer transfer{input, tensor.preferred_location, destination, tensor.bytes,
                                         layer.measured_host_roundtrip?0.0:estimated_transfer_cost(tensor.bytes, tensor.preferred_location, destination)};
                layer.estimated_transfer_cost += transfer.estimated_cost;
                plan.stats.transfer_bytes += transfer.bytes;
                plan.transfers.push_back(transfer);
            }
        }
        auto& output = plan.tensors.at(layer.output);
        output.producer_runtime = layer.runtime;
        output.preferred_location = plan.mode==PlannerMode::Cost?MemoryLocation::Host:destination;
        if(plan.mode==PlannerMode::Cost && destination!=MemoryLocation::Host){
            plan.transfers.push_back({output.id,destination,MemoryLocation::Host,output.bytes,layer.measured_host_roundtrip?0.0:estimated_transfer_cost(output.bytes,destination,MemoryLocation::Host)});
            plan.stats.transfer_bytes+=output.bytes;
        }
        if (plan.regions.empty() || previous != layer.runtime) {
            plan.regions.push_back({plan.regions.size(), layer.runtime, {}});
        }
        plan.regions.back().layer_indices.push_back(layer.layer_index);
        previous = layer.runtime;
    }
    if (!plan.layers.empty()) {
        auto& final_tensor = plan.tensors.at(plan.layers.back().output);
        if (final_tensor.preferred_location != MemoryLocation::Host) {
            const auto from = final_tensor.preferred_location;
            plan.transfers.push_back({final_tensor.id, from, MemoryLocation::Host, final_tensor.bytes,
                                      estimated_transfer_cost(final_tensor.bytes, from, MemoryLocation::Host)});
            plan.stats.transfer_bytes += final_tensor.bytes;
            plan.regions.push_back({plan.regions.size(), RuntimeKind::CPU, {}});
        }
    }
}

inline const char* runtime_kind_name(const RuntimeKind runtime) {
    switch (runtime) {
        case RuntimeKind::CPU: return "CPU";
        case RuntimeKind::Vulkan: return "Vulkan";
        case RuntimeKind::OpenCL: return "OpenCL";
        case RuntimeKind::CUDA: return "CUDA";
        case RuntimeKind::ROCm: return "ROCm";
        case RuntimeKind::TPU: return "TPU";
        case RuntimeKind::FPGA: return "FPGA";
        default: return "Unknown";
    }
}

inline const char* capability_level_name(const RuntimeCapabilityLevel capability) {
    switch (capability) {
        case RuntimeCapabilityLevel::Unsupported: return "Unsupported";
        case RuntimeCapabilityLevel::HostFallback: return "HostFallback";
        case RuntimeCapabilityLevel::Native: return "Native";
        case RuntimeCapabilityLevel::NativeOptimized: return "NativeOptimized";
        default: return "Unsupported";
    }
}

inline const char* memory_location_name(const MemoryLocation location) {
    switch (location) {
        case MemoryLocation::Host: return "Host";
        case MemoryLocation::Vulkan: return "Vulkan";
        case MemoryLocation::OpenCL: return "OpenCL";
        case MemoryLocation::CUDA: return "CUDA";
        case MemoryLocation::ROCm: return "ROCm";
        case MemoryLocation::TPU: return "TPU";
        case MemoryLocation::FPGA: return "FPGA";
        default: return "Unknown";
    }
}

inline std::string escape_json(const std::string& value) {
    std::ostringstream out;
    for (const char c : value) {
        if (c == '\\' || c == '"') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    return out.str();
}

inline std::string dump_execution_plan_text(const ExecutionPlan& plan) {
    std::ostringstream out;
    out << "[planner] layers=" << plan.layers.size() << " tensors=" << plan.tensors.size()
        << " transfers=" << plan.transfers.size() << " regions=" << plan.regions.size() << '\n';
    for (const PlannedLayer& layer : plan.layers) {
        out << "[planner] layer=" << layer.layer_index << " name=" << layer.layer_name
            << " runtime=" << runtime_kind_name(layer.runtime)
            << " capability=" << capability_level_name(layer.capability)
            << " output_location=" << memory_location_name(plan.tensors.at(layer.output).preferred_location)
            << " skip=" << layer.skip << " graph_fused=" << layer.graph_fused
            << " kernel_fused=" << layer.kernel_fused << " scratch=" << layer.scratch_bytes
            << " cost_basis=" << (layer.measured_host_roundtrip?"measured_host_roundtrip":"estimated")
            << " compute_cost=" << layer.estimated_compute_cost
            << " transfer_cost=" << layer.estimated_transfer_cost << '\n';
    }
    out << "[planner] tensor_logical_bytes=" << plan.stats.tensor_logical_bytes
        << " tensor_physical_peak=" << plan.stats.tensor_physical_peak
        << " buffer_reuse_saved=" << plan.stats.buffer_reuse_saved
        << " transfer_bytes=" << plan.stats.transfer_bytes
        << " scratch_peak=" << plan.stats.scratch_peak << '\n';
    for (const PlannedTransfer& transfer : plan.transfers) {
        out << "[planner] transfer tensor=" << transfer.tensor.name << '#' << transfer.tensor.generation
            << " from=" << memory_location_name(transfer.from)
            << " to=" << memory_location_name(transfer.to)
            << " bytes=" << transfer.bytes << " cost=" << transfer.estimated_cost << '\n';
    }
    return out.str();
}

inline std::string dump_execution_plan_json(const ExecutionPlan& plan) {
    std::ostringstream out;
    out << "{\"training\":" << (plan.training ? "true" : "false") << ",\"layers\":[";
    for (size_t i = 0; i < plan.layers.size(); ++i) {
        if (i) out << ',';
        const auto& layer = plan.layers[i];
        out << "{\"index\":" << layer.layer_index << ",\"name\":\"" << escape_json(layer.layer_name)
            << "\",\"runtime\":\"" << runtime_kind_name(layer.runtime)
            << "\",\"capability\":\"" << capability_level_name(layer.capability)
            << "\",\"output_location\":\""
            << memory_location_name(plan.tensors.at(layer.output).preferred_location)
            << "\",\"graph_fused\":" << (layer.graph_fused ? "true" : "false")
            << ",\"kernel_fused\":" << (layer.kernel_fused ? "true" : "false")
            << ",\"scratch_bytes\":" << layer.scratch_bytes
            << ",\"measured_host_roundtrip\":" << (layer.measured_host_roundtrip?"true":"false")
            << ",\"estimated_compute_cost\":" << std::setprecision(17) << layer.estimated_compute_cost
            << ",\"estimated_transfer_cost\":" << layer.estimated_transfer_cost << '}';
    }
    out << "],\"tensors\":[";
    std::vector<const PlannedTensor*> tensors;
    for (const auto& entry : plan.tensors) tensors.push_back(&entry.second);
    std::sort(tensors.begin(), tensors.end(), [](const PlannedTensor* a, const PlannedTensor* b) {
        return a->name == b->name ? a->generation < b->generation : a->name < b->name;
    });
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i) out << ',';
        out << "{\"name\":\"" << escape_json(tensors[i]->name) << "\",\"generation\":"
            << tensors[i]->generation << ",\"bytes\":" << tensors[i]->bytes << '}';
    }
    out << "],\"transfers\":[";
    for (size_t i = 0; i < plan.transfers.size(); ++i) {
        if (i) out << ',';
        const auto& transfer = plan.transfers[i];
        out << "{\"tensor\":\"" << escape_json(transfer.tensor.name)
            << "\",\"generation\":" << transfer.tensor.generation
            << ",\"from\":\"" << memory_location_name(transfer.from)
            << "\",\"to\":\"" << memory_location_name(transfer.to)
            << "\",\"bytes\":" << transfer.bytes
            << ",\"estimated_cost\":" << std::setprecision(17) << transfer.estimated_cost << '}';
    }
    out << "],\"regions\":[";
    for (size_t i = 0; i < plan.regions.size(); ++i) {
        if (i) out << ',';
        out << "{\"id\":" << plan.regions[i].id << ",\"runtime\":\""
            << runtime_kind_name(plan.regions[i].runtime) << "\",\"layers\":[";
        for (size_t j = 0; j < plan.regions[i].layer_indices.size(); ++j) {
            if (j) out << ',';
            out << plan.regions[i].layer_indices[j];
        }
        out << "]}";
    }
    out << "],\"stats\":{\"tensor_logical_bytes\":" << plan.stats.tensor_logical_bytes
        << ",\"tensor_physical_peak\":" << plan.stats.tensor_physical_peak
        << ",\"buffer_reuse_saved\":" << plan.stats.buffer_reuse_saved
        << ",\"transfer_bytes\":" << plan.stats.transfer_bytes
        << ",\"scratch_peak\":" << plan.stats.scratch_peak << "}}";
    return out.str();
}

struct BenchmarkRecord {
    LayerType layer_type = LayerType::UNKNOWN;
    RuntimeKind runtime = RuntimeKind::Unknown;
    Mimir::DType dtype = Mimir::DType::UNKNOWN;
    std::vector<int> shape;
    std::string parameters_key;
    double observed_seconds = 0.0;
};

class BenchmarkCache {
public:
    void record(const BenchmarkRecord& record) { records_[key(record)] = record.observed_seconds; }
    std::optional<double> lookup(const BenchmarkRecord& record) const {
        const auto found = records_.find(key(record));
        return found == records_.end() ? std::nullopt : std::optional<double>(found->second);
    }
private:
    static std::string key(const BenchmarkRecord& record) {
        std::ostringstream out;
        out << static_cast<int>(record.layer_type) << ':' << static_cast<int>(record.runtime) << ':'
            << static_cast<int>(record.dtype) << ':' << record.parameters_key;
        for (const int value : record.shape) out << ':' << value;
        return out.str();
    }
    std::unordered_map<std::string, double> records_;
};

inline std::string benchmark_parameters_key(const Layer& layer) {
    std::ostringstream out;
    out << "in=" << layer.in_features << ",out=" << layer.out_features
        << ",kernel=" << layer.get_kernel_h() << ",stride=" << layer.get_stride_h();
    return out.str();
}

inline AbstractRuntime* select_runtime_by_benchmark(const Layer& layer,
                                                     const PlannedTensor& tensor,
                                                     const std::vector<AbstractRuntime*>& candidates,
                                                     const BenchmarkCache& cache) {
    AbstractRuntime* best = nullptr;
    double best_time = std::numeric_limits<double>::infinity();
    for (AbstractRuntime* candidate : candidates) {
        if (!candidate || !candidate->isInitialized() ||
            !runtimeCapabilityIsNative(candidate->queryForwardCapability(layer.type_enum))) continue;
        BenchmarkRecord record{layer.type_enum, runtime_kind_from_name(candidate->name()), tensor.dtype,
                               tensor.shape, benchmark_parameters_key(layer), 0.0};
        const auto observed = cache.lookup(record);
        if (observed && *observed < best_time) { best_time = *observed; best = candidate; }
        else if (!best && !observed) best = candidate;
    }
    return best;
}

// --------------------------
// Memory planner (scratch)
// --------------------------

struct ScratchRequest {
    std::string tag;
    size_t min_bytes = 0;
};

// Best-effort scratch size planning for Conv2d fast path.
// Goal: allow callers to pre-warm a scratch pool and use shared tags.
struct Conv2dScratchPlan {
    size_t wT_bytes = 0;
    size_t xcol_bytes = 0;
    size_t c_bytes = 0;
};

inline Conv2dScratchPlan plan_conv2d_fastpath_scratch(const std::vector<Layer>& layers) {
    Conv2dScratchPlan out;

    // Mirror the fast-path heuristics (tile target ~32MB for Xcol).
    const size_t target_bytes = 32ULL * 1024ULL * 1024ULL;
    const size_t floats_budget = target_bytes / sizeof(float);

    for (const auto& layer : layers) {
        if (layer.type_enum != LayerType::Conv2d) continue;
        const int kernel_size = layer.get_kernel_h();
        const int in_channels = layer.in_channels;
        const int out_channels = layer.out_channels;
        if (kernel_size <= 0 || in_channels <= 0 || out_channels <= 0) continue;

        const int height = (layer.input_height > 0) ? layer.input_height : 64;
        const int width = (layer.input_width > 0) ? layer.input_width : 64;
        const int stride = layer.get_stride_h();
        const int padding = layer.get_pad_h();

        const int out_h = (height + 2 * padding - kernel_size) / std::max(1, stride) + 1;
        const int out_w = (width + 2 * padding - kernel_size) / std::max(1, stride) + 1;
        const int out_spatial = std::max(0, out_h) * std::max(0, out_w);
        const int K = in_channels * kernel_size * kernel_size;
        if (out_spatial <= 0 || K <= 0) continue;

        int tile_m = static_cast<int>(std::max<size_t>(256, std::min<size_t>(8192, floats_budget / static_cast<size_t>(K))));
        if (tile_m > out_spatial) tile_m = out_spatial;

        const size_t w_need = static_cast<size_t>(out_channels) * static_cast<size_t>(K);
        const size_t xcol_need = static_cast<size_t>(tile_m) * static_cast<size_t>(K);
        const size_t c_need = static_cast<size_t>(tile_m) * static_cast<size_t>(out_channels);

        out.wT_bytes = std::max(out.wT_bytes, w_need * sizeof(float));
        out.xcol_bytes = std::max(out.xcol_bytes, xcol_need * sizeof(float));
        out.c_bytes = std::max(out.c_bytes, c_need * sizeof(float));
    }

    return out;
}

inline void apply_global_scratch_plan(ExecutionPlan& plan, const std::vector<Layer>& layers) {
    const Conv2dScratchPlan scratch = plan_conv2d_fastpath_scratch(layers);
    const size_t peak = std::max({scratch.wT_bytes, scratch.xcol_bytes, scratch.c_bytes});
    plan.stats.scratch_peak = peak;
    for (size_t i = 0; i < plan.layers.size() && i < layers.size(); ++i) {
        if (layers[i].type_enum == LayerType::Conv2d ||
            layers[i].type_enum == LayerType::ConvTranspose2d ||
            layers[i].type_enum == LayerType::MatMul ||
            layers[i].type_enum == LayerType::BatchMatMul) {
            plan.layers[i].scratch_bytes = peak;
        }
    }
}

} // namespace Mimir::Planning
