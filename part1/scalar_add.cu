#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

// Note: 1. External libraries are not allowed 
//       2. The solve function signature must keep unchanged 
//       3. Store the final result in the array y

__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ y,            // [N,C,H,W]
    int N, int C, int H, int W,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x

    int total_elements = N * C * H * W

    if (idx < total_elements)
    {
        y[idx] = x[idx] + eps;
    }
}

void solve(const float* input, // [N,C,H,W] (input shape: batch, channel, height, weight)  
           const float* gamma, // [C] 
           const float* beta, // [C]
           float* output, // [N,C,H,W]
           int N, int C, int H, int W, float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;

    // TODO: Kernel Configurabtions (dim3 block, dim3 grid)

    /*
    gpu_bn_inference_nchw<<<kernel_config1, kernel_config2, ...>>>(input, gamma, beta, output, N, C, H, W, eps);
    */
}

