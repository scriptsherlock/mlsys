#include "solve.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Error Checking Macro
#define CHECK_CUDA_ERROR() do { \
    cudaError_t err = cudaPeekAtLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Constants
#define BLOCK_W 32
#define BLOCK_H 8

__global__ void gpu_dwconv2d_inference_nchw(
    const float* __restrict__ x,
    const float* __restrict__ k,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int KH, int KW,
    int OH, int OW,
    int pad_h, int pad_w,
    int stride_h, int stride_w)
{
    extern __shared__ float s_mem[];

    int in_tile_w = (BLOCK_W - 1) * stride_w + KW;
    int in_tile_h = (BLOCK_H - 1) * stride_h + KH;
    
    int total_nc = N * C;
    for (int nc = blockIdx.z; nc < total_nc; nc += gridDim.z) {
        int n = nc / C;
        int c = nc % C;

        int out_start_h = blockIdx.y * BLOCK_H;
        int out_start_w = blockIdx.x * BLOCK_W;

        int in_anchor_h = out_start_h * stride_h - pad_h;
        int in_anchor_w = out_start_w * stride_w - pad_w;

        size_t offset_x = (size_t)n * C * H * W + (size_t)c * H * W;
        size_t offset_k = (size_t)c * KH * KW;
        size_t offset_y = (size_t)n * C * OH * OW + (size_t)c * OH * OW;

        const float* x_ptr = x + offset_x;
        const float* k_ptr = k + offset_k;
        float* y_ptr = y + offset_y;

        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int num_threads = blockDim.x * blockDim.y;
        int tile_size = in_tile_h * in_tile_w;

        for (int i = tid; i < tile_size; i += num_threads) {
            int loc_y = i / in_tile_w;
            int loc_x = i % in_tile_w;
            int glob_h = in_anchor_h + loc_y;
            int glob_w = in_anchor_w + loc_x;

            float val = 0.0f;
            if (glob_h >= 0 && glob_h < H && glob_w >= 0 && glob_w < W) {
                val = x_ptr[glob_h * W + glob_w];
            }
            s_mem[i] = val;
        }
        __syncthreads();

        int thread_oh = threadIdx.y;
        int thread_ow = threadIdx.x;
        int global_oh = out_start_h + thread_oh;
        int global_ow = out_start_w + thread_ow;

        if (global_oh < OH && global_ow < OW) {
            float acc = 0.0f;
            int s_row_base = thread_oh * stride_h;
            int s_col_base = thread_ow * stride_w;

            #pragma unroll
            for (int kh = 0; kh < KH; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < KW; ++kw) {
                    float val_k = k_ptr[kh * KW + kw];
                    float val_x = s_mem[(s_row_base + kh) * in_tile_w + (s_col_base + kw)];
                    acc += val_x * val_k;
                }
            }
            y_ptr[global_oh * OW + global_ow] = acc;
        }
        __syncthreads();
    }
}

void solve(const float* x, const float* k, float* y, 
           int N, int C, int H, int W, 
           int KH, int KW, int pad_h, int pad_w, int stride_h, int stride_w)
{
    if (N <= 0 || C <= 0) return;
    
    int OH = (H + 2 * pad_h - KH) / stride_h + 1;
    int OW = (W + 2 * pad_w - KW) / stride_w + 1;
    if (OH <= 0 || OW <= 0) return;

    dim3 block(BLOCK_W, BLOCK_H);
    
    int grid_x = (OW + BLOCK_W - 1) / BLOCK_W;
    int grid_y = (OH + BLOCK_H - 1) / BLOCK_H;
    int grid_z = std::min(65535, N * C);

    dim3 grid(grid_x, grid_y, grid_z);

    int in_tile_w = (BLOCK_W - 1) * stride_w + KW;
    int in_tile_h = (BLOCK_H - 1) * stride_h + KH;
    size_t smem_size = in_tile_w * in_tile_h * sizeof(float);

    if (smem_size > 48 * 1024) {
        cudaError_t err = cudaFuncSetAttribute(
            gpu_dwconv2d_inference_nchw, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            smem_size);
            
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to set Dynamic Shared Memory: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    gpu_dwconv2d_inference_nchw<<<grid, block, smem_size>>>(
        x, k, y, N, C, H, W, KH, KW, OH, OW, pad_h, pad_w, stride_h, stride_w
    );
    
    CHECK_CUDA_ERROR();

    cudaDeviceSynchronize();
}