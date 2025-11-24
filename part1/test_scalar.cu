#include <iostream>
#include <cuda_runtime.h>

// ------------------------------------------------------------
// Kernel: add scalar "eps" to every element in x
// ------------------------------------------------------------
__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;

    if (idx < total) {
        y[idx] = x[idx] + eps;
    }
}


// ------------------------------------------------------------
// Host launcher
// ------------------------------------------------------------
void solve(const float* input,
           const float* gamma,
           const float* beta,
           float* output,
           int N, int C, int H, int W,
           float eps)
{
    int total = N * C * H * W;
    if (total <= 0) return;

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    gpu_bn_inference_nchw<<<blocks, threads>>>(
        input, gamma, beta, output,
        N, C, H, W, eps
    );
}


// ------------------------------------------------------------
// MAIN — makes the program runnable
// ------------------------------------------------------------
int main() {
    int N = 1, C = 1, H = 2, W = 2;
    int total = N*C*H*W;

    // Host memory
    float h_input[]  = {1, 2, 3, 4};
    float h_output[4];

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input,  total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, h_input, total*sizeof(float),
               cudaMemcpyHostToDevice);

    // Call your solve() function
    solve(d_input, nullptr, nullptr, d_output,
          N, C, H, W,
          10.0f);     // scalar to add

    // Copy output back
    cudaMemcpy(h_output, d_output, total*sizeof(float),
               cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Output: ";
    for (int i = 0; i < total; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
