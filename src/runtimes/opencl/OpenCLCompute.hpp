#ifndef __OPENCL_COMPUTE_HPP__
#define __OPENCL_COMPUTE_HPP__

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef ENABLE_OPENCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif

namespace OpenCLCompute {

// Moteur OpenCL minimal, pensé pour coexister avec Vulkan.
// Objectif immédiat: accélérer certaines opérations (ex: Linear) au runtime.
class ComputeEngine {
public:
    ComputeEngine() = default;
    ~ComputeEngine() { cleanup(); }

    ComputeEngine(const ComputeEngine&) = delete;
    ComputeEngine& operator=(const ComputeEngine&) = delete;

    bool initialize() {
#ifndef ENABLE_OPENCL
        initialized_ = false;
        return false;
#else
        if (initialized_) return true;

        cl_int err = CL_SUCCESS;
        cl_uint num_platforms = 0;
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            return false;
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) {
            return false;
        }

        // Choix: GPU d'abord, sinon CPU
        cl_platform_id chosen_platform = nullptr;
        cl_device_id chosen_device = nullptr;
        auto pick = [&](cl_device_type t) {
            for (cl_uint i = 0; i < num_platforms; ++i) {
                cl_uint dev_count = 0;
                if (clGetDeviceIDs(platforms[i], t, 0, nullptr, &dev_count) == CL_SUCCESS && dev_count > 0) {
                    std::vector<cl_device_id> devs(dev_count);
                    if (clGetDeviceIDs(platforms[i], t, dev_count, devs.data(), nullptr) == CL_SUCCESS) {
                        chosen_platform = platforms[i];
                        chosen_device = devs[0];
                        return true;
                    }
                }
            }
            return false;
        };
        if (!pick(CL_DEVICE_TYPE_GPU)) {
            (void)pick(CL_DEVICE_TYPE_CPU);
        }

        if (!chosen_device) {
            return false;
        }

        platform_ = chosen_platform;
        device_ = chosen_device;

        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS || !context_) {
            cleanup();
            return false;
        }

        queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
        if (err != CL_SUCCESS || !queue_) {
            cleanup();
            return false;
        }

        if (!buildProgramAndKernels()) {
            cleanup();
            return false;
        }

        initialized_ = true;
        return true;
#endif
    }

    bool isInitialized() const { return initialized_; }

    void cleanup() {
#ifdef ENABLE_OPENCL
        if (kernel_binary_) { clReleaseKernel(kernel_binary_); kernel_binary_ = nullptr; }
        if (kernel_unary_) { clReleaseKernel(kernel_unary_); kernel_unary_ = nullptr; }
        if (kernel_batch_matmul_) { clReleaseKernel(kernel_batch_matmul_); kernel_batch_matmul_ = nullptr; }
        if (kernel_matmul_) { clReleaseKernel(kernel_matmul_); kernel_matmul_ = nullptr; }
        if (kernel_linear_) { clReleaseKernel(kernel_linear_); kernel_linear_ = nullptr; }
        if (program_) { clReleaseProgram(program_); program_ = nullptr; }
        if (queue_) { clReleaseCommandQueue(queue_); queue_ = nullptr; }
        if (context_) { clReleaseContext(context_); context_ = nullptr; }
        platform_ = nullptr;
        device_ = nullptr;
#endif
        initialized_ = false;
    }

    // Linear: input [batch, in_f] ; weights [out_f, in_f] ; bias [out_f]
    // output [batch, out_f]
    bool linearForward(
        const float* input,
        const float* weights,
        const float* bias_or_null,
        float* output,
        int batch,
        int in_f,
        int out_f
    ) {
#ifndef ENABLE_OPENCL
        (void)input; (void)weights; (void)bias_or_null; (void)output; (void)batch; (void)in_f; (void)out_f;
        return false;
#else
        if (!initialized_ || !context_ || !queue_ || !kernel_linear_) return false;
        if (!input || !weights || !output) return false;
        if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

        cl_int err = CL_SUCCESS;
        const size_t bytes_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f) * sizeof(float);
        const size_t bytes_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f) * sizeof(float);
        const size_t bytes_b = static_cast<size_t>(out_f) * sizeof(float);
        const size_t bytes_out = static_cast<size_t>(batch) * static_cast<size_t>(out_f) * sizeof(float);

        // Note: allocation simple par appel. Optimisable via cache/pool ensuite.
        cl_mem buf_in = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_in, nullptr, &err);
        if (err != CL_SUCCESS || !buf_in) return false;
        cl_mem buf_w = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_w, nullptr, &err);
        if (err != CL_SUCCESS || !buf_w) { clReleaseMemObject(buf_in); return false; }
        cl_mem buf_b = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_b, nullptr, &err);
        if (err != CL_SUCCESS || !buf_b) { clReleaseMemObject(buf_in); clReleaseMemObject(buf_w); return false; }
        cl_mem buf_out = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bytes_out, nullptr, &err);
        if (err != CL_SUCCESS || !buf_out) { clReleaseMemObject(buf_in); clReleaseMemObject(buf_w); clReleaseMemObject(buf_b); return false; }

        err = clEnqueueWriteBuffer(queue_, buf_in, CL_TRUE, 0, bytes_in, input, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }
        err = clEnqueueWriteBuffer(queue_, buf_w, CL_TRUE, 0, bytes_w, weights, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        if (bias_or_null) {
            err = clEnqueueWriteBuffer(queue_, buf_b, CL_TRUE, 0, bytes_b, bias_or_null, 0, nullptr, nullptr);
        } else {
            zeroBiasScratch_.assign(static_cast<size_t>(out_f), 0.0f);
            err = clEnqueueWriteBuffer(queue_, buf_b, CL_TRUE, 0, bytes_b, zeroBiasScratch_.data(), 0, nullptr, nullptr);
        }
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        err  = clSetKernelArg(kernel_linear_, 0, sizeof(cl_mem), &buf_in);
        err |= clSetKernelArg(kernel_linear_, 1, sizeof(cl_mem), &buf_w);
        err |= clSetKernelArg(kernel_linear_, 2, sizeof(cl_mem), &buf_b);
        err |= clSetKernelArg(kernel_linear_, 3, sizeof(cl_mem), &buf_out);
        err |= clSetKernelArg(kernel_linear_, 4, sizeof(int), &batch);
        err |= clSetKernelArg(kernel_linear_, 5, sizeof(int), &in_f);
        err |= clSetKernelArg(kernel_linear_, 6, sizeof(int), &out_f);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        // Global 2D: (batch, out_f)
        const size_t global[2] = { static_cast<size_t>(batch), static_cast<size_t>(out_f) };
        err = clEnqueueNDRangeKernel(queue_, kernel_linear_, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        err = clEnqueueReadBuffer(queue_, buf_out, CL_TRUE, 0, bytes_out, output, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { release4(buf_in, buf_w, buf_b, buf_out); return false; }

        release4(buf_in, buf_w, buf_b, buf_out);
        return true;
#endif
    }

    // MatMul: C[M,N] = A[M,K] @ B[K,N]
    bool matmulForward(
        const float* a,
        const float* b,
        float* c,
        int M,
        int K,
        int N
    ) {
#ifndef ENABLE_OPENCL
        (void)a; (void)b; (void)c; (void)M; (void)K; (void)N;
        return false;
#else
    if (!initialized_ || !context_ || !queue_ || !kernel_matmul_) return false;
        if (!a || !b || !c) return false;
        if (M <= 0 || K <= 0 || N <= 0) return false;

    cl_int err = CL_SUCCESS;
    const size_t bytes_a = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);

    cl_mem buf_a = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_a, nullptr, &err);
    if (err != CL_SUCCESS || !buf_a) return false;
    cl_mem buf_b = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_b, nullptr, &err);
    if (err != CL_SUCCESS || !buf_b) { clReleaseMemObject(buf_a); return false; }
    cl_mem buf_c = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bytes_c, nullptr, &err);
    if (err != CL_SUCCESS || !buf_c) { clReleaseMemObject(buf_a); clReleaseMemObject(buf_b); return false; }

    err = clEnqueueWriteBuffer(queue_, buf_a, CL_TRUE, 0, bytes_a, a, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_c) clReleaseMemObject(buf_c); return false; }
    err = clEnqueueWriteBuffer(queue_, buf_b, CL_TRUE, 0, bytes_b, b, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_c) clReleaseMemObject(buf_c); return false; }

    err  = clSetKernelArg(kernel_matmul_, 0, sizeof(cl_mem), &buf_a);
    err |= clSetKernelArg(kernel_matmul_, 1, sizeof(cl_mem), &buf_b);
    err |= clSetKernelArg(kernel_matmul_, 2, sizeof(cl_mem), &buf_c);
    err |= clSetKernelArg(kernel_matmul_, 3, sizeof(int), &M);
    err |= clSetKernelArg(kernel_matmul_, 4, sizeof(int), &K);
    err |= clSetKernelArg(kernel_matmul_, 5, sizeof(int), &N);
    if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_c) clReleaseMemObject(buf_c); return false; }

        constexpr size_t kTile = 16;
        const size_t global[2] = {
            ((static_cast<size_t>(M) + kTile - 1) / kTile) * kTile,
            ((static_cast<size_t>(N) + kTile - 1) / kTile) * kTile
        };
        const size_t local[2] = { kTile, kTile };
        err = clEnqueueNDRangeKernel(queue_, kernel_matmul_, 2, nullptr, global, local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_c) clReleaseMemObject(buf_c); return false; }

    err = clEnqueueReadBuffer(queue_, buf_c, CL_TRUE, 0, bytes_c, c, 0, nullptr, nullptr);
    if (buf_a) clReleaseMemObject(buf_a);
    if (buf_b) clReleaseMemObject(buf_b);
    if (buf_c) clReleaseMemObject(buf_c);
    return err == CL_SUCCESS;
#endif
    }

    // BatchMatMul: C[B,M,N] = A[B,M,K] @ Bm[B,K,N]
    bool batchMatMulForward(
        const float* a,
        const float* b,
        float* c,
        int B,
        int M,
        int K,
        int N
    ) {
#ifndef ENABLE_OPENCL
        (void)a; (void)b; (void)c; (void)B; (void)M; (void)K; (void)N;
        return false;
#else
        if (!initialized_ || !context_ || !queue_ || !kernel_batch_matmul_) return false;
        if (!a || !b || !c) return false;
        if (B <= 0 || M <= 0 || K <= 0 || N <= 0) return false;

        cl_int err = CL_SUCCESS;
        const size_t bytes_a = static_cast<size_t>(B) * static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);
        const size_t bytes_b = static_cast<size_t>(B) * static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(float);
        const size_t bytes_c = static_cast<size_t>(B) * static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);

        cl_mem buf_a = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_a, nullptr, &err);
        if (err != CL_SUCCESS || !buf_a) return false;
        cl_mem buf_b = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes_b, nullptr, &err);
        if (err != CL_SUCCESS || !buf_b) { clReleaseMemObject(buf_a); return false; }
        cl_mem buf_c = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bytes_c, nullptr, &err);
        if (err != CL_SUCCESS || !buf_c) { clReleaseMemObject(buf_a); clReleaseMemObject(buf_b); return false; }

        err = clEnqueueWriteBuffer(queue_, buf_a, CL_TRUE, 0, bytes_a, a, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_c) clReleaseMemObject(buf_c); return false; }
        err = clEnqueueWriteBuffer(queue_, buf_b, CL_TRUE, 0, bytes_b, b, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_c) clReleaseMemObject(buf_c); return false; }

        err  = clSetKernelArg(kernel_batch_matmul_, 0, sizeof(cl_mem), &buf_a);
        err |= clSetKernelArg(kernel_batch_matmul_, 1, sizeof(cl_mem), &buf_b);
        err |= clSetKernelArg(kernel_batch_matmul_, 2, sizeof(cl_mem), &buf_c);
        err |= clSetKernelArg(kernel_batch_matmul_, 3, sizeof(int), &B);
        err |= clSetKernelArg(kernel_batch_matmul_, 4, sizeof(int), &M);
        err |= clSetKernelArg(kernel_batch_matmul_, 5, sizeof(int), &K);
        err |= clSetKernelArg(kernel_batch_matmul_, 6, sizeof(int), &N);
        if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_c) clReleaseMemObject(buf_c); return false; }

        constexpr size_t kTile = 16;
        const size_t global[3] = {
            static_cast<size_t>(B),
            ((static_cast<size_t>(M) + kTile - 1) / kTile) * kTile,
            ((static_cast<size_t>(N) + kTile - 1) / kTile) * kTile
        };
        const size_t local[3] = { 1, kTile, kTile };
        err = clEnqueueNDRangeKernel(queue_, kernel_batch_matmul_, 3, nullptr, global, local, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_c) clReleaseMemObject(buf_c); return false; }

        err = clEnqueueReadBuffer(queue_, buf_c, CL_TRUE, 0, bytes_c, c, 0, nullptr, nullptr);
        if (buf_a) clReleaseMemObject(buf_a);
        if (buf_b) clReleaseMemObject(buf_b);
        if (buf_c) clReleaseMemObject(buf_c);
        return err == CL_SUCCESS;
#endif
    }

    // Unary element-wise: out[i] = f(in[i])
    bool unaryForward(
        const float* in,
        float* out,
        int n,
        int op,
        float alpha
    ) {
#ifndef ENABLE_OPENCL
        (void)in; (void)out; (void)n; (void)op; (void)alpha;
        return false;
#else
        if (!initialized_ || !context_ || !queue_ || !kernel_unary_) return false;
        if (!in || !out || n <= 0) return false;

        cl_int err = CL_SUCCESS;
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        cl_mem buf_in = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_in) return false;
        cl_mem buf_out = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_out) { clReleaseMemObject(buf_in); return false; }

        err = clEnqueueWriteBuffer(queue_, buf_in, CL_TRUE, 0, bytes, in, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { if (buf_in) clReleaseMemObject(buf_in); if (buf_out) clReleaseMemObject(buf_out); return false; }

        err  = clSetKernelArg(kernel_unary_, 0, sizeof(cl_mem), &buf_in);
        err |= clSetKernelArg(kernel_unary_, 1, sizeof(cl_mem), &buf_out);
        err |= clSetKernelArg(kernel_unary_, 2, sizeof(int), &n);
        err |= clSetKernelArg(kernel_unary_, 3, sizeof(int), &op);
        err |= clSetKernelArg(kernel_unary_, 4, sizeof(float), &alpha);
        if (err != CL_SUCCESS) { if (buf_in) clReleaseMemObject(buf_in); if (buf_out) clReleaseMemObject(buf_out); return false; }

        const size_t global = static_cast<size_t>(n);
        err = clEnqueueNDRangeKernel(queue_, kernel_unary_, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { if (buf_in) clReleaseMemObject(buf_in); if (buf_out) clReleaseMemObject(buf_out); return false; }

        err = clEnqueueReadBuffer(queue_, buf_out, CL_TRUE, 0, bytes, out, 0, nullptr, nullptr);
        if (buf_in) clReleaseMemObject(buf_in);
        if (buf_out) clReleaseMemObject(buf_out);
        return err == CL_SUCCESS;
#endif
    }

    // Binary element-wise: out[i] = f(a[i], b[i])
    bool binaryForward(
        const float* a,
        const float* b,
        float* out,
        int n,
        int op,
        float alpha = 0.01f
    ) {
#ifndef ENABLE_OPENCL
        (void)a; (void)b; (void)out; (void)n; (void)op; (void)alpha;
        return false;
#else
        if (!initialized_ || !context_ || !queue_ || !kernel_binary_) return false;
        if (!a || !b || !out || n <= 0) return false;

        cl_int err = CL_SUCCESS;
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        cl_mem buf_a = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_a) return false;
        cl_mem buf_b = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_b) { clReleaseMemObject(buf_a); return false; }
        cl_mem buf_out = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_out) { clReleaseMemObject(buf_a); clReleaseMemObject(buf_b); return false; }

        err = clEnqueueWriteBuffer(queue_, buf_a, CL_TRUE, 0, bytes, a, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_out) clReleaseMemObject(buf_out); return false; }
        err = clEnqueueWriteBuffer(queue_, buf_b, CL_TRUE, 0, bytes, b, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_out) clReleaseMemObject(buf_out); return false; }

        err  = clSetKernelArg(kernel_binary_, 0, sizeof(cl_mem), &buf_a);
        err |= clSetKernelArg(kernel_binary_, 1, sizeof(cl_mem), &buf_b);
        err |= clSetKernelArg(kernel_binary_, 2, sizeof(cl_mem), &buf_out);
        err |= clSetKernelArg(kernel_binary_, 3, sizeof(int), &n);
        err |= clSetKernelArg(kernel_binary_, 4, sizeof(int), &op);
        err |= clSetKernelArg(kernel_binary_, 5, sizeof(float), &alpha);
        if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_out) clReleaseMemObject(buf_out); return false; }

        const size_t global = static_cast<size_t>(n);
        err = clEnqueueNDRangeKernel(queue_, kernel_binary_, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { if (buf_a) clReleaseMemObject(buf_a); if (buf_b) clReleaseMemObject(buf_b); if (buf_out) clReleaseMemObject(buf_out); return false; }

        err = clEnqueueReadBuffer(queue_, buf_out, CL_TRUE, 0, bytes, out, 0, nullptr, nullptr);
        if (buf_a) clReleaseMemObject(buf_a);
        if (buf_b) clReleaseMemObject(buf_b);
        if (buf_out) clReleaseMemObject(buf_out);
        return err == CL_SUCCESS;
#endif
    }

private:
#ifdef ENABLE_OPENCL
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_linear_ = nullptr;
    cl_kernel kernel_matmul_ = nullptr;
    cl_kernel kernel_batch_matmul_ = nullptr;
    cl_kernel kernel_unary_ = nullptr;
    cl_kernel kernel_binary_ = nullptr;

    std::vector<float> zeroBiasScratch_;

    static void release4(cl_mem a, cl_mem b, cl_mem c, cl_mem d) {
        if (a) clReleaseMemObject(a);
        if (b) clReleaseMemObject(b);
        if (c) clReleaseMemObject(c);
        if (d) clReleaseMemObject(d);
    }

    static void printBuildLog(cl_program prog, cl_device_id device) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size == 0) return;
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "OpenCL build log:\n" << log << "\n";
    }

    bool buildProgramAndKernels() {
        cl_int err = CL_SUCCESS;
        const char* src = R"CLC(
__kernel void linear_forward(
    __global const float* input,      // [batch, in_f]
    __global const float* weights,    // [out_f, in_f]
    __global const float* bias,       // [out_f]
    __global float* output,           // [batch, out_f]
    int batch,
    int in_f,
    int out_f
) {
    const int b = (int)get_global_id(0);
    const int o = (int)get_global_id(1);
    if (b >= batch || o >= out_f) return;
    float acc = bias[o];
    const int in_base = b * in_f;
    const int w_base = o * in_f;
    for (int k = 0; k < in_f; ++k) {
        acc += input[in_base + k] * weights[w_base + k];
    }
    output[b * out_f + o] = acc;
}

__kernel void matmul_forward(
    __global const float* A,   // [M, K]
    __global const float* B,   // [K, N]
    __global float* C,         // [M, N]
    int M,
    int K,
    int N
) {
#define TILE 16
    const int m = (int)get_global_id(0);
    const int n = (int)get_global_id(1);
    const int lm = (int)get_local_id(0);
    const int ln = (int)get_local_id(1);

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float acc = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int kA = t * TILE + ln;
        const int kB = t * TILE + lm;

        As[lm][ln] = (m < M && kA < K) ? A[m * K + kA] : 0.0f;
        Bs[lm][ln] = (kB < K && n < N) ? B[kB * N + n] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE; ++k) {
            acc += As[lm][k] * Bs[k][ln];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (m < M && n < N) {
        C[m * N + n] = acc;
    }
}

__kernel void batch_matmul_forward(
    __global const float* A,   // [B, M, K]
    __global const float* Bm,  // [B, K, N]
    __global float* C,         // [B, M, N]
    int batches,
    int M,
    int K,
    int N
) {
    const int b = (int)get_global_id(0);
    const int m = (int)get_global_id(1);
    const int n = (int)get_global_id(2);
    const int lm = (int)get_local_id(1);
    const int ln = (int)get_local_id(2);

    if (b >= batches) return;

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float acc = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;
    const int a_batch_base = b * (M * K);
    const int b_batch_base = b * (K * N);

    for (int t = 0; t < num_tiles; ++t) {
        const int kA = t * TILE + ln;
        const int kB = t * TILE + lm;

        As[lm][ln] = (m < M && kA < K) ? A[a_batch_base + m * K + kA] : 0.0f;
        Bs[lm][ln] = (kB < K && n < N) ? Bm[b_batch_base + kB * N + n] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE; ++k) {
            acc += As[lm][k] * Bs[k][ln];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (m < M && n < N) {
        C[b * (M * N) + m * N + n] = acc;
    }
}

__kernel void unary_forward(
    __global const float* in,
    __global float* out,
    int n,
    int op,
    float alpha
) {
    const int i = (int)get_global_id(0);
    if (i >= n) return;
    const float x = in[i];
    float y = x;

    switch (op) {
        case 0: y = fmax(x, 0.0f); break;                              // ReLU
        case 1: y = x > 0.0f ? x : alpha * x; break;                   // LeakyReLU
        case 2: y = 1.0f / (1.0f + exp(-x)); break;                    // Sigmoid
        case 3: y = tanh(x); break;                                    // Tanh
        case 4: { float s = 1.0f / (1.0f + exp(-x)); y = x * s; } break; // SiLU
        case 5: { float c = 0.7978845608f; float x3 = x*x*x; y = 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x3))); } break; // GELU
        case 6: y = x > 20.0f ? x : log(1.0f + exp(x)); break;        // Softplus
        case 7: { float sp = x > 20.0f ? x : log(1.0f + exp(x)); y = x * tanh(sp); } break; // Mish
        case 8: { float hs = (x + 3.0f) / 6.0f; y = clamp(hs, 0.0f, 1.0f); } break; // HardSigmoid
        case 9: { float hs = clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f); y = x * hs; } break; // HardSwish
        default: break;
    }
    out[i] = y;
}

__kernel void binary_forward(
    __global const float* a, __global const float* b,
    __global float* out, int n, int op, float alpha
) {
    const int i = (int)get_global_id(0);
    if (i >= n) return;
    const float x = a[i], g = b[i];
    float y = 0.0f;
    switch (op) {
        case 0: y = x + g; break;
        case 1: y = x - g; break;
        case 2: y = x * g; break;
        case 3: { float d = fabs(g) < 1e-8f ? (g >= 0.0f ? 1e-8f : -1e-8f) : g; y = x / d; } break;
        case 4: { float d = fabs(g) < 1e-8f ? (g >= 0.0f ? 1e-8f : -1e-8f) : g; y = -x / (d*d); } break;
        case 10: y = fmax(x,0.0f); break;
        case 11: y = x > 0.0f ? x : alpha*x; break;
        case 12: y = 1.0f/(1.0f+exp(-x)); break;
        case 13: y = tanh(x); break;
        case 14: y = x/(1.0f+exp(-x)); break;
        case 15: y = 0.5f*x*(1.0f+tanh(0.7978845608f*(x+0.044715f*x*x*x))); break;
        case 16: y = x > 20.0f ? x : log(1.0f+exp(x)); break;
        case 17: y = x*tanh(x > 20.0f ? x : log(1.0f+exp(x))); break;
        case 18: y = clamp((x+3.0f)/6.0f,0.0f,1.0f); break;
        case 19: y = x*clamp((x+3.0f)/6.0f,0.0f,1.0f); break;
        case 20: y = x > 0.0f ? g : 0.0f; break;
        case 21: y = g*(x > 0.0f ? 1.0f : alpha); break;
        case 22: { float s = 1.0f/(1.0f+exp(-x)); y = g*s*(1.0f-s); } break;
        case 23: { float t = tanh(x); y = g*(1.0f-t*t); } break;
        case 24: { float s = 1.0f/(1.0f+exp(-x)); y = g*(s+x*s*(1.0f-s)); } break;
        case 25: { float t = tanh(0.7978845608f*(x+0.044715f*x*x*x)); y = g*(0.5f*(1.0f+t)+0.5f*x*(1.0f-t*t)*0.7978845608f*(1.0f+0.134145f*x*x)); } break;
        case 26: y = g/(1.0f+exp(-x)); break;
        case 27: { float t = tanh(x > 20.0f ? x : log(1.0f+exp(x))); float s = 1.0f/(1.0f+exp(-x)); y = g*(t+x*(1.0f-t*t)*s); } break;
        case 28: y = x > -3.0f && x < 3.0f ? g/6.0f : 0.0f; break;
        case 29: y = g*(x <= -3.0f ? 0.0f : (x >= 3.0f ? 1.0f : x/3.0f+0.5f)); break;
    }
    out[i] = y;
}
)CLC";

        const size_t src_len = std::strlen(src);
        program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS || !program_) return false;

        err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            printBuildLog(program_, device_);
            return false;
        }

        kernel_linear_ = clCreateKernel(program_, "linear_forward", &err);
        if (err != CL_SUCCESS || !kernel_linear_) return false;
        kernel_matmul_ = clCreateKernel(program_, "matmul_forward", &err);
        if (err != CL_SUCCESS || !kernel_matmul_) return false;
        kernel_batch_matmul_ = clCreateKernel(program_, "batch_matmul_forward", &err);
        if (err != CL_SUCCESS || !kernel_batch_matmul_) return false;
        kernel_unary_ = clCreateKernel(program_, "unary_forward", &err);
        if (err != CL_SUCCESS || !kernel_unary_) return false;
        kernel_binary_ = clCreateKernel(program_, "binary_forward", &err);
        if (err != CL_SUCCESS || !kernel_binary_) return false;
        return true;
    }
#endif

    bool initialized_ = false;
};

} // namespace OpenCLCompute

#endif // __OPENCL_COMPUTE_HPP__
