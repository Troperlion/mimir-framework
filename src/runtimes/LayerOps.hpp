#pragma once

#include <string>
#include <vector>

#include "Layers.hpp"

namespace RuntimeLayerOps {

struct LayerParams {
	std::vector<float> weights;
	std::vector<float> bias;
	int in_features = 0;
	int out_features = 0;
	int kernel_size = 3;
	int stride = 1;
	int padding = 0;
	int dilation = 1;
	int groups = 1;
	bool use_hardware = true;
};

void setHardwareAcceleration(bool enable);
bool hardwareAccelerationEnabled();

void computeConv2D(const std::vector<float>& input, std::vector<float>& output,
				   const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
				   bool use_hardware = true);
void computeLinear(const std::vector<float>& input, std::vector<float>& output,
				   const LayerParams& params, bool use_hardware = true);
void computeMaxPool2D(const std::vector<float>& input, std::vector<float>& output,
					  int in_h, int in_w, int channels, int kernel_size, int stride,
					  bool use_hardware = true);
void computeAvgPool2D(const std::vector<float>& input, std::vector<float>& output,
					  int in_h, int in_w, int channels, int kernel_size, int stride,
					  bool use_hardware = true);
void computeActivation(std::vector<float>& data, const std::string& activation_type,
					   float param = 0.0f, bool use_hardware = true);
void computeBatchNorm(std::vector<float>& data, const std::vector<float>& gamma,
					  const std::vector<float>& beta, const std::vector<float>& running_mean,
					  const std::vector<float>& running_var, int batch_size, int channels,
					  int spatial_size, float eps = 1e-5f, bool training = false,
					  bool use_hardware = true);
void computeLayerNorm(std::vector<float>& data, const std::vector<float>& gamma,
					  const std::vector<float>& beta, int normalized_size,
					  float eps = 1e-5f, bool use_hardware = true);
bool prepareLayerNorm(const std::vector<float>& input, int normalized_size,
					  float eps, std::vector<float>& normalized);
void computeConvTranspose2D(const std::vector<float>& input, std::vector<float>& output,
							const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
							bool use_hardware = true);
void computeAttention(const std::vector<float>& query, const std::vector<float>& key,
					  const std::vector<float>& value, std::vector<float>& output,
					  int seq_len, int d_model, int num_heads, bool use_hardware = true);
void conv2dSame(const std::vector<float>& input, std::vector<float>& output,
				int width, int height, const std::vector<float>& kernel, int kernel_size);

void branchMerge(const std::vector<float>& branch1, const std::vector<float>& branch2,
				 std::vector<float>& output, MergeOperation merge_op,
				 bool use_hardware = true);
void branchSplit(const std::vector<float>& input,
				 std::vector<std::vector<float>>& outputs,
				 const std::vector<int>& split_sizes);

bool resolveUnaryOp(LayerType type, const Layer& layer, int& op_code, float& alpha);
bool resolveBinaryOp(LayerType type, int& op_code);

void unaryForwardHost(const std::vector<float>& input, std::vector<float>& output, int op_code, float alpha);
void binaryForwardHost(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& output, int op_code);

} // namespace RuntimeLayerOps
