#include "solve.h"
#include <cuda_runtime.h>
#include <algorithm>

// ------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------
#define BLOCK_W 32
#define BLOCK_H 8

// ------------------------------------------------------------------------
// Kernel: Tiled Shared Memory Depthwise Convolution
// ------------------------------------------------------------------------
__global__ void gpu_dwconv2d_inference_nchw(
    const float* __restrict__ x,      // [N, C, H, W]
    const float* __restrict__ k,      // [C, 1, KH, KW]
    float* __restrict__ y,            // [N, C, OH, OW]
    int N, int C, int H, int W,
    int KH, int KW,
    int OH, int OW,
    int pad_h, int pad_w,
    int stride_h, int stride_w)
{
    // Dynamic Shared Memory to store the Input Tile
    extern __shared__ float s_mem[];

    // 1. Calculate Dimensions of the Input Tile needed for this Block
    // The block computes a (BLOCK_H x BLOCK_W) chunk of the output.
    // To do this, we need a larger chunk of the input dependent on Stride & Kernel.
    int in_tile_w = (BLOCK_W - 1) * stride_w + KW;
    int in_tile_h = (BLOCK_H - 1) * stride_h + KH;
    
    // 2. Grid-Stride Loop over (Batch, Channel) pairs
    // Flattened index nc = n * C + c
    int total_nc = N * C;
    for (int nc = blockIdx.z; nc < total_nc; nc += gridDim.z) {
        int n = nc / C;
        int c = nc % C;

        // 3. Calculate Output & Input Offsets
        // Output spatial offsets for this block
        int out_start_h = blockIdx.y * BLOCK_H;
        int out_start_w = blockIdx.x * BLOCK_W;

        // Corresponding Input spatial anchor (top-left corner of the sliding window)
        // h_in = h_out * stride - pad
        int in_anchor_h = out_start_h * stride_h - pad_h;
        int in_anchor_w = out_start_w * stride_w - pad_w;

        // Global Memory Pointers
        const float* x_ptr = x + (size_t)n * (C * H * W) + (size_t)c * (H * W);
        const float* k_ptr = k + (size_t)c * (KH * KW);
        float* y_ptr = y + (size_t)n * (C * OH * OW) + (size_t)c * (OH * OW);

        // 4. Cooperative Load: Input Tile -> Shared Memory
        // All threads in the block help load the input tile
        int tid = threadIdx.y * blockDim.x + threadIdx.x; // Linear thread ID (0..255)
        int num_threads = blockDim.x * blockDim.y;
        int tile_size = in_tile_h * in_tile_w;

        for (int i = tid; i < tile_size; i += num_threads) {
            int loc_y = i / in_tile_w; // Local Y in shared mem
            int loc_x = i % in_tile_w; // Local X in shared mem
            
            int glob_h = in_anchor_h + loc_y;
            int glob_w = in_anchor_w + loc_x;

            // Boundary Check (Padding)
            float val = 0.0f;
            if (glob_h >= 0 && glob_h < H && glob_w >= 0 && glob_w < W) {
                val = x_ptr[glob_h * W + glob_w];
            }
            s_mem[i] = val;
        }
        __syncthreads(); // Wait for load to finish

        // 5. Compute Output Pixel
        int thread_oh = threadIdx.y;
        int thread_ow = threadIdx.x;
        
        int global_oh = out_start_h + thread_oh;
        int global_ow = out_start_w + thread_ow;

        if (global_oh < OH && global_ow < OW) {
            float acc = 0.0f;

            // Sliding Window over the loaded Shared Memory
            // The input pixel corresponding to output (0,0) is at s_mem[0,0] (logically).
            // But we shifted indices.
            // For a thread (ty, tx), the top-left input pixel in s_mem is at:
            // y = ty * stride_h
            // x = tx * stride_w
            
            int s_row_base = thread_oh * stride_h;
            int s_col_base = thread_ow * stride_w;

            // Inner loops over kernel
            #pragma unroll
            for (int kh = 0; kh < KH; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < KW; ++kw) {
                    float val_k = k_ptr[kh * KW + kw];
                    // Read from shared memory
                    float val_x = s_mem[(s_row_base + kh) * in_tile_w + (s_col_base + kw)];
                    acc += val_x * val_k;
                }
            }
            
            // Write Output
            y_ptr[global_oh * OW + global_ow] = acc;
        }
        __syncthreads(); // Sync before loading next N/C batch (if any)
    }
}

// ------------------------------------------------------------------------
// Host Function
// ------------------------------------------------------------------------
void solve(const float* x, const float* k, float* y, 
           int N, int C, int H, int W, 
           int KH, int KW, int pad_h, int pad_w, int stride_h, int stride_w)
{
    // Calculate Output Dimensions
    int OH = (H + 2 * pad_h - KH) / stride_h + 1;
    int OW = (W + 2 * pad_w - KW) / stride_w + 1;

    if (N <= 0 || C <= 0 || OH <= 0 || OW <= 0) return;

    // Block Configuration: 32x8 (256 threads) covers a 32x8 output patch
    dim3 block(BLOCK_W, BLOCK_H);

    // Grid Configuration
    // X/Y: Spatial tiling
    // Z:  Batch * Channel (handled via grid-stride loop inside kernel to avoid limit)
    int grid_x = (OW + BLOCK_W - 1) / BLOCK_W;
    int grid_y = (OH + BLOCK_H - 1) / BLOCK_H;
    int grid_z = std::min(65535, N * C); // Cap to HW limit

    dim3 grid(grid_x, grid_y, grid_z);

    // Calculate Required Shared Memory Size per Block
    // Input tile size depends on block size, stride, and kernel size
    int in_tile_w = (BLOCK_W - 1) * stride_w + KW;
    int in_tile_h = (BLOCK_H - 1) * stride_h + KH;
    size_t smem_size = in_tile_w * in_tile_h * sizeof(float);

    // Ensure we request enough shared memory if > 48KB
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(gpu_dwconv2d_inference_nchw, 
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 
                             smem_size);
    }

    // Launch
    gpu_dwconv2d_inference_nchw<<<grid, block, smem_size>>>(
        x, k, y, N, C, H, W, KH, KW, OH, OW, pad_h, pad_w, stride_h, stride_w
    );
    
    cudaDeviceSynchronize();
}