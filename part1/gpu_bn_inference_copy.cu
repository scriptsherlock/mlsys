#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

// --------------------------------------------------------
// Warp reduction (float) 
// --------------------------------------------------------
__inline__ __device__ float warpReduceSum(float val) {
    // warpSize is 32 on all current CUDA GPUs
    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// --------------------------------------------------------
// Block reduction (float) 
// --------------------------------------------------------
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // up to 32 warps per block
    int lane = threadIdx.x % THREADS_PER_WARP;
    int wid  = threadIdx.x / THREADS_PER_WARP;

    // Reduce within warp
    val = warpReduceSum(val);

    // Write warp result to shared
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp loads the partial sums
    val = (threadIdx.x < (blockDim.x / THREADS_PER_WARP)) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void bn_partial_nchw(
    const float* __restrict__ x,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sqsum,
    int N, int C, int H, int W)
{
    int c = blockIdx.x;
    int n = blockIdx.y;
    if (c >= C || n >= N) return;

    int spatial = H * W;
    size_t stride_n = (size_t)C * spatial;
    size_t stride_c = (size_t)spatial;

    size_t base = (size_t)n * stride_n + (size_t)c * stride_c;

    float sum = 0.0f;
    float sq  = 0.0f;

    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        float v = x[base + i];
        sum += v;
        sq  += v * v;
    }

    sum = blockReduceSum(sum);
    sq  = blockReduceSum(sq);

    if (threadIdx.x == 0) {
        int idx = n * C + c;
        partial_sum[idx]   = sum;
        partial_sqsum[idx] = sq;
    }
}


__global__ void bn_stats_nchw(
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_sqsum,
    float* __restrict__ mean,
    float* __restrict__ var,
    int N, int C, int H, int W)
{
    int c = blockIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    float sq  = 0.0f;

    for (int n = 0; n < N; ++n) {
        int idx = n * C + c;
        sum += partial_sum[idx];
        sq  += partial_sqsum[idx];
    }

    float count = float(N) * float(H * W);
    float m  = sum / count;
    float ex2 = sq / count;
    float v  = ex2 - m * m;

    mean[c] = m;
    var[c]  = v;
}

__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    int N, int C, int H, int W,
    float eps)
{
    int c = blockIdx.x;
    int n = blockIdx.y;
    if (c >= C || n >= N) return;

    int spatial = H * W;
    size_t stride_n = (size_t)C * spatial;
    size_t stride_c = (size_t)spatial;

    size_t base = (size_t)n * stride_n + (size_t)c * stride_c;

    float m   = mean[c];
    float v   = var[c];
    float inv_std = rsqrtf(v + eps);

    float g = gamma[c];
    float b = beta[c];

    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        float xn = (x[base + i] - m) * inv_std;
        y[base + i] = g * xn + b;
    }
}

// --------------------------------------------------------
// Host Function
// --------------------------------------------------------
void solve(const float* input,
           const float* gamma,
           const float* beta,
           float* output,
           int N, int C, int H, int W, float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;

    int spatial = H * W;

    // Temp buffers: N*C partials, C stats
    float* d_partial_sum  = nullptr;
    float* d_partial_sq   = nullptr;
    float* d_mean         = nullptr;
    float* d_var          = nullptr;

    cudaMalloc(&d_partial_sum, N * C * sizeof(float));
    cudaMalloc(&d_partial_sq,  N * C * sizeof(float));
    cudaMalloc(&d_mean,        C * sizeof(float));
    cudaMalloc(&d_var,         C * sizeof(float));

    dim3 block(256, 1, 1);

    dim3 grid_partial(C, N, 1);
    bn_partial_nchw<<<grid_partial, block>>>(
        input, d_partial_sum, d_partial_sq,
        N, C, H, W
    );

    dim3 grid_stats(C, 1, 1);
    bn_stats_nchw<<<grid_stats, block>>>(
        d_partial_sum, d_partial_sq,
        d_mean, d_var,
        N, C, H, W
    );

    dim3 grid_bn(C, N, 1);
    gpu_bn_inference_nchw<<<grid_bn, block>>>(
        input, gamma, beta, output,
        d_mean, d_var,
        N, C, H, W, eps
    );

    cudaDeviceSynchronize();

    cudaFree(d_partial_sum);
    cudaFree(d_partial_sq);
    cudaFree(d_mean);
    cudaFree(d_var);
}
