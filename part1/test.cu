#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

__device__ float* g_mean;
__device__ float* g_invstd;

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % THREADS_PER_WARP;
    int wid  = threadIdx.x / THREADS_PER_WARP;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / THREADS_PER_WARP)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void bn_stats_multi(
    const float* __restrict__ x,
    int N, int C, int H, int W,
    float* __restrict__ sum,
    float* __restrict__ sumsq)
{
    int c = blockIdx.x;
    int bgroup = blockIdx.y;

    if (c >= C) return;

    int spatial = H * W;
    size_t stride_n = (size_t)C * spatial;
    size_t stride_c = (size_t)spatial;

    int blocks_per_channel = gridDim.y;
    int threads = blockDim.x;

    float local_sum   = 0.0f;
    float local_sumsq = 0.0f;

    bool use_vec4 = (spatial % 4 == 0);

    if (use_vec4) {
        int vec_len = spatial / 4;

        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            const float4* x4 = reinterpret_cast<const float4*>(x + base);

            for (int idx4 = bgroup * threads + threadIdx.x;
                 idx4 < vec_len;
                 idx4 += threads * blocks_per_channel)
            {
                float4 v = x4[idx4];
                float a = v.x, b = v.y, d = v.z, e = v.w;

                local_sum += a + b + d + e;
                local_sumsq += a*a + b*b + d*d + e*e;
            }
        }
    } else {
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;

            for (int i = bgroup * threads + threadIdx.x;
                 i < spatial;
                 i += threads * blocks_per_channel)
            {
                float v = x[base + i];
                local_sum += v;
                local_sumsq += v * v;
            }
        }
    }

    // Block-level reduction
    local_sum   = blockReduceSum(local_sum);
    local_sumsq = blockReduceSum(local_sumsq);

    if (threadIdx.x == 0) {
        atomicAdd(&sum[c],   local_sum);
        atomicAdd(&sumsq[c], local_sumsq);
    }
}

__global__ void bn_finalize_stats(
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    float* __restrict__ mean,
    float* __restrict__ invstd,
    int N, int C, int H, int W,
    float eps)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float count = float(N) * float(H * W);
    float s  = sum[c];
    float ss = sumsq[c];

    float m   = s / count;
    float ex2 = ss / count;
    float var = ex2 - m * m;
    if (var < 0.f) var = 0.f;

    mean[c]   = m;
    invstd[c] = rsqrtf(var + eps);
}


__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps) // eps unused here (stats already computed)
{
    int c = blockIdx.x;
    int bgroup = blockIdx.y;

    if (c >= C) return;

    int spatial = H * W;
    size_t stride_n = (size_t)C * spatial;
    size_t stride_c = (size_t)spatial;

    int blocks_per_channel = gridDim.y;
    int threads = blockDim.x;

    float mean   = g_mean[c];
    float invstd = g_invstd[c];
    float g      = gamma[c];
    float b      = beta[c];

    bool use_vec4 = (spatial % 4 == 0);

    if (use_vec4) {
        int vec_len = spatial / 4;

        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;

            const float4* x4 = reinterpret_cast<const float4*>(x + base);
            float4*       y4 = reinterpret_cast<float4*>(y + base);

            for (int idx4 = bgroup * threads + threadIdx.x;
                 idx4 < vec_len;
                 idx4 += threads * blocks_per_channel)
            {
                float4 vin = x4[idx4];

                float a = vin.x, b0 = vin.y, d = vin.z, e = vin.w;

                float xn0 = (a  - mean) * invstd;
                float xn1 = (b0 - mean) * invstd;
                float xn2 = (d  - mean) * invstd;
                float xn3 = (e  - mean) * invstd;

                float4 vout;
                vout.x = g * xn0 + b;
                vout.y = g * xn1 + b;
                vout.z = g * xn2 + b;
                vout.w = g * xn3 + b;

                y4[idx4] = vout;
            }
        }
    } else {
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;

            for (int i = bgroup * threads + threadIdx.x;
                 i < spatial;
                 i += threads * blocks_per_channel)
            {
                float v  = x[base + i];
                float xn = (v - mean) * invstd;
                y[base + i] = g * xn + b;
            }
        }
    }
}

void solve(const float* input,
           const float* gamma,
           const float* beta,
           float* output,
           int N, int C, int H, int W,
           float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;

    // Allocate temporary buffers
    float *d_sum, *d_sumsq, *d_mean, *d_invstd;
    cudaMalloc(&d_sum,    C * sizeof(float));
    cudaMalloc(&d_sumsq,  C * sizeof(float));
    cudaMalloc(&d_mean,   C * sizeof(float));
    cudaMalloc(&d_invstd, C * sizeof(float));

    cudaMemset(d_sum,   0, C * sizeof(float));
    cudaMemset(d_sumsq, 0, C * sizeof(float));

    // int blocks_per_channel = 8;
    int threads = 256;
    int elements_per_channel = N * spatial;
    int desired_blocks = (elements_per_channel + threads * 4 - 1) / (threads * 4);
    int blocks_per_channel = std::min(32, std::max(1, desired_blocks));
    dim3 block(256, 1, 1);
    dim3 grid_stats(C, blocks_per_channel, 1);

    bn_stats_multi<<<grid_stats, block>>>(input, N, C, H, W, d_sum, d_sumsq);

    int t2 = 128;
    int g2 = (C + t2 - 1) / t2;

    bn_finalize_stats<<<g2, t2>>>(d_sum, d_sumsq, d_mean, d_invstd, N, C, H, W, eps);

    // Link mean/invstd to device-global pointers
    cudaMemcpyToSymbol(g_mean,   &d_mean,   sizeof(float*));
    cudaMemcpyToSymbol(g_invstd, &d_invstd, sizeof(float*));

    dim3 grid_bn(C, blocks_per_channel, 1);

    gpu_bn_inference_nchw<<<grid_bn, block>>>(
        input, gamma, beta, output,
        N, C, H, W, eps
    );

    cudaDeviceSynchronize();

    cudaFree(d_sum);
    cudaFree(d_sumsq);
    cudaFree(d_mean);
    cudaFree(d_invstd);
}