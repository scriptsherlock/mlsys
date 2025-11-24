#include "solve.h"
#include <cuda_runtime.h>
#include <algorithm>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

__forceinline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__forceinline__ __device__ void blockReduceSum2(float& val1, float& val2) {
    static __shared__ float s1[32];
    static __shared__ float s2[32];
    int lane = threadIdx.x % THREADS_PER_WARP;
    int wid  = threadIdx.x / THREADS_PER_WARP;
    val1 = warpReduceSum(val1);
    val2 = warpReduceSum(val2);
    if (lane == 0) { s1[wid] = val1; s2[wid] = val2; }
    __syncthreads();
    if (wid == 0) {
        val1 = (threadIdx.x < (blockDim.x / THREADS_PER_WARP)) ? s1[lane] : 0.0f;
        val2 = (threadIdx.x < (blockDim.x / THREADS_PER_WARP)) ? s2[lane] : 0.0f;
        val1 = warpReduceSum(val1);
        val2 = warpReduceSum(val2);
    }
}

__global__ __launch_bounds__(256) 
void bn_fused_smem_final(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int spatial, float eps)
{
    extern __shared__ float s_data[]; // Holds entire channel
    // Separate shared mem for broadcast to avoid overwriting data
    __shared__ float s_scale;
    __shared__ float s_shift;

    int c = blockIdx.x; 
    if (c >= C) return;

    size_t stride_n = (size_t)C * spatial;
    size_t stride_c = (size_t)spatial;
    
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    
    bool aligned = ((size_t)x % 16 == 0) && (spatial % 4 == 0);
    
    // 1. LOAD
    if (aligned) {
        int vec_len = spatial / 4;
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            const float4* x4 = reinterpret_cast<const float4*>(x + base);
            
            int i = threadIdx.x;
            while (i < vec_len) {
                float4 v = x4[i];
                int idx = n * spatial + i * 4;
                // Unpack manually
                s_data[idx+0] = v.x; s_data[idx+1] = v.y; s_data[idx+2] = v.z; s_data[idx+3] = v.w;
                
                local_sum += v.x + v.y + v.z + v.w;
                local_sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
                i += blockDim.x;
            }
        }
    } else {
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
                float v = x[base + i];
                s_data[n * spatial + i] = v;
                local_sum += v;
                local_sumsq += v * v;
            }
        }
    }

    // 2. STATS
    blockReduceSum2(local_sum, local_sumsq);

    if (threadIdx.x == 0) {
        float count = (float)(N * spatial);
        float mean = local_sum / count;
        float var = (local_sumsq / count) - (mean * mean);
        float invstd = rsqrtf(max(var, 0.0f) + eps);
        
        // Write to dedicated shared variables
        s_scale = gamma[c] * invstd;
        s_shift = beta[c] - (mean * s_scale);
    }
    
    __syncthreads(); 
    
    // Read from dedicated shared variables
    float scale = s_scale;
    float shift = s_shift;

    // 3. NORMALIZE & WRITE
    bool out_aligned = ((size_t)y % 16 == 0) && (spatial % 4 == 0);

    if (out_aligned) {
        int vec_len = spatial / 4;
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            float4* y4 = reinterpret_cast<float4*>(y + base);

            int i = threadIdx.x;
            while (i < vec_len) {
                int idx = n * spatial + i * 4;
                float4 r;
                r.x = s_data[idx+0]; r.y = s_data[idx+1]; r.z = s_data[idx+2]; r.w = s_data[idx+3];
                
                r.x = r.x * scale + shift;
                r.y = r.y * scale + shift;
                r.z = r.z * scale + shift;
                r.w = r.w * scale + shift;
                
                y4[i] = r;
                i += blockDim.x;
            }
        }
    } else {
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
                y[base + i] = s_data[n * spatial + i] * scale + shift;
            }
        }
    }
}

__global__ void bn_stats_final(
    const float* __restrict__ x,
    int N, int C, int spatial,
    float* __restrict__ global_sum,
    float* __restrict__ global_sumsq)
{
    int c = blockIdx.x;
    int bgroup = blockIdx.y;
    int blocks_per_channel = gridDim.y;
    if (c >= C) return;

    size_t stride_n = (size_t)C * spatial;
    size_t stride_c = (size_t)spatial;
    
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    if (spatial % 4 == 0 && (size_t)x % 16 == 0) {
        int vec_len = spatial / 4;
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            const float4* x4 = reinterpret_cast<const float4*>(x + base);
            
            int idx = bgroup * blockDim.x + threadIdx.x;
            int step = blockDim.x * blocks_per_channel;

            // UNROLL 8x
            while (idx + 7 * step < vec_len) {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float4 v = x4[idx + u * step];
                    local_sum += v.x + v.y + v.z + v.w;
                    local_sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
                }
                idx += 8 * step;
            }

            for (; idx < vec_len; idx += step) {
                float4 v = x4[idx];
                local_sum += v.x + v.y + v.z + v.w;
                local_sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            }
        }
    } else {
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            for (int i = bgroup * blockDim.x + threadIdx.x; i < spatial; i += blockDim.x * blocks_per_channel) {
                float v = x[base + i];
                local_sum += v;
                local_sumsq += v * v;
            }
        }
    }

    blockReduceSum2(local_sum, local_sumsq);
    if (threadIdx.x == 0) {
        atomicAdd(&global_sum[c], local_sum);
        atomicAdd(&global_sumsq[c], local_sumsq);
    }
}


__global__ void bn_coeffs_fallback(
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ scale_out,
    float* __restrict__ shift_out,
    int count, int C, float eps)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s = sum[c], ss = sumsq[c];
    float mean = s / count;
    float var  = (ss / count) - (mean * mean);
    float invstd = rsqrtf(max(var, 0.0f) + eps);
    float g = gamma[c];
    scale_out[c] = invstd * g;
    shift_out[c] = beta[c] - (mean * scale_out[c]);
}

__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ shift_ptr,
    float* __restrict__ y,
    int N, int C, int spatial, int dummy, float dummy2)
{
    int c = blockIdx.x;
    int bgroup = blockIdx.y;
    int blocks_per_channel = gridDim.y;
    if (blocks_per_channel == 0) blocks_per_channel = 1;
    if (c >= C) return;

    size_t stride_n = (size_t)C * spatial;
    size_t stride_c = (size_t)spatial;

    float scale = scale_ptr[c];
    float shift = shift_ptr[c];

    if (spatial % 4 == 0 && (size_t)x % 16 == 0 && (size_t)y % 16 == 0) {
        int vec_len = spatial / 4;
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            const float4* x4 = reinterpret_cast<const float4*>(x + base);
            float4* y4 = reinterpret_cast<float4*>(y + base);
            
            int idx = bgroup * blockDim.x + threadIdx.x;
            int step = blockDim.x * blocks_per_channel;

            for (; idx < vec_len; idx += step) {
                float4 v = x4[idx];
                v.x = v.x * scale + shift; v.y = v.y * scale + shift;
                v.z = v.z * scale + shift; v.w = v.w * scale + shift;
                y4[idx] = v;
            }
        }
    } else {
        for (int n = 0; n < N; ++n) {
            size_t base = (size_t)n * stride_n + (size_t)c * stride_c;
            for (int i = bgroup * blockDim.x + threadIdx.x; i < spatial; i += blockDim.x * blocks_per_channel) {
                y[base + i] = x[base + i] * scale + shift;
            }
        }
    }
}

void solve(const float* input, const float* gamma, const float* beta, float* output, 
           int N, int C, int H, int W, float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;

    int spatial = H * W;
    size_t channel_bytes = (size_t)N * spatial * sizeof(float);
    

    bool use_smem_path = (channel_bytes <= 98304); 

    if (use_smem_path) {
        cudaFuncSetAttribute(bn_fused_smem_final, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        dim3 grid(C, 1, 1);
        dim3 block(256, 1, 1);
        bn_fused_smem_final<<<grid, block, channel_bytes>>>(input, gamma, beta, output, N, C, spatial, eps);
    } 
    else {
        float *d_sum, *d_sumsq, *d_scale, *d_shift;
        cudaMalloc(&d_sum,   C * sizeof(float));
        cudaMalloc(&d_sumsq, C * sizeof(float));
        cudaMalloc(&d_scale, C * sizeof(float));
        cudaMalloc(&d_shift, C * sizeof(float));
        cudaMemset(d_sum, 0, C * sizeof(float));
        cudaMemset(d_sumsq, 0, C * sizeof(float));

        int threads = 256;
        int total_elements_per_channel = N * spatial;
        int vec_elements = (total_elements_per_channel + 3) / 4;
        int blocks_needed = (vec_elements + threads - 1) / threads;
        
        int blocks_per_channel = std::min(64, std::max(1, 4096 / C));
        blocks_per_channel = std::min(blocks_per_channel, blocks_needed);

        dim3 grid_stats(C, blocks_per_channel, 1);
        bn_stats_final<<<grid_stats, threads>>>(input, N, C, spatial, d_sum, d_sumsq);

        int coeff_blocks = (C + 255) / 256;
        bn_coeffs_fallback<<<coeff_blocks, 256>>>(d_sum, d_sumsq, gamma, beta, d_scale, d_shift, N*spatial, C, eps);

        gpu_bn_inference_nchw<<<grid_stats, threads>>>(input, d_scale, d_shift, output, N, C, spatial, 0, 0.0f);

        cudaFree(d_sum); cudaFree(d_sumsq); cudaFree(d_scale); cudaFree(d_shift);
    }
    cudaDeviceSynchronize();
}