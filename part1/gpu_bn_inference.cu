#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

// --------------------------------------------------------
// Helper: Double Precision Warp Reduction
// (Splits double into two ints to ensure compatibility)
// --------------------------------------------------------
// __inline__ __device__ float shfl_down(float var, int delta) {
//     int2 temp = *reinterpret_cast<int2*>(&var); 
//     temp.x = __shfl_down_sync(FULL_MASK, temp.x, delta);
//     temp.y = __shfl_down_sync(FULL_MASK, temp.y, delta);
//     return *reinterpret_cast<double*>(&temp);
// }

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) 
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// __inline__ __device__ float warpReduceSum(float val) {
//     for (int offset = 16; offset > 0; offset >>= 1)
        
//     return val;
// }


// --------------------------------------------------------
// Helper: Block Reduction
// --------------------------------------------------------
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // Shared mem for 32 warps
    int lane = threadIdx.x % THREADS_PER_WARP;
    int wid = threadIdx.x / THREADS_PER_WARP;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / THREADS_PER_WARP)) ? shared[lane] : 0.0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// --------------------------------------------------------
// Kernel: Two-Pass Batch Norm (One Block Per Channel)
// --------------------------------------------------------
__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,      
    const float* __restrict__ gamma,  
    const float* __restrict__ beta,   
    float* __restrict__ y,            
    int N, int C, int H, int W,
    float eps)
{
    int c = blockIdx.x; 
    if (c >= C) return; // Safety check

    // Dimensions
    int spatial = H * W;
    // Use size_t to prevent overflow on large tensors
    size_t stride_n = (size_t)C * spatial; 
    size_t stride_c = (size_t)spatial;     
    float count = float(N * spatial);

    // Shared Memory for broadcasting stats
    __shared__ float s_mean;
    __shared__ float s_var;


    // === PASS 1: Calculate Mean ===
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        size_t offset = (size_t)n * stride_n + (size_t)c * stride_c;
        for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
            sum += (float)x[offset + i];
        }
    }
    
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) s_mean = sum / count;
    __syncthreads();

    // === PASS 2: Calculate Variance ===
    float sq_diff = 0.0f;
    float mean_val = s_mean;
    
    for (int n = 0; n < N; ++n) {
        size_t offset = (size_t)n * stride_n + (size_t)c * stride_c;
        for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
            float diff = (float)x[offset + i] - mean_val;
            sq_diff += diff * diff;
        }
    }

    sq_diff = blockReduceSum(sq_diff);
    
    if (threadIdx.x == 0) {
        s_var = sq_diff / count;
        
        // Pre-calculate transform factors
        // float inv_std = rsqrtf((float)s_var + eps);
    }
    __syncthreads();

    // === PASS 3: Normalize ===

    for (int n = 0; n < N; ++n) {
        size_t offset = (size_t)n * stride_n + (size_t)c * stride_c;
        for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
            float xn = ((float)x[offset + i] - (float)mean_val) * (float)(rsqrtf((float)s_var + eps));
            y[offset + i] = (float)(gamma[c] * xn + beta[c]);
        }
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

    // 1 Block Per Channel is the only safe way without changing the kernel signature.
    // It works perfectly for LLaMA/Mistral sizes.
    dim3 grid(C, 1, 1);
    dim3 block(256, 1, 1);

    gpu_bn_inference_nchw<<<grid, block>>>(input, gamma, beta, output, N, C, H, W, eps);
    
    // Essential: Wait for GPU to finish
    cudaDeviceSynchronize();
}