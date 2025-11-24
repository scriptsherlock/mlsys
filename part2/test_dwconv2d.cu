
#include "solve.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <cinttypes>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(e)); std::abort(); } } while(0)

#define CURAND_CHECK(x) do { curandStatus_t s=(x); if(s!=CURAND_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuRAND error: %d\n", int(s)); std::abort(); } } while(0)

void cpu_dwconv2d_ref(const std::vector<float>& x,
                      const std::vector<float>& k,
                      std::vector<float>& y,
                      int N, int C, int H, int W,
                      int KH, int KW,
                      int pad_h, int pad_w,
                      int stride_h, int stride_w);

static void print_mat(const char* title, const std::vector<float>& t,
                      int N, int C, int H, int W) {
    int rows = N*C*H, cols = W;
    std::printf("%s (%d x %d):\n", title, rows, cols);
    for (int r = 0; r < rows; ++r) {
        int n = r / (C*H);
        int rem = r - n*(C*H);
        int c = rem / H;
        int h = rem - c*H;
        size_t base = (size_t)n*C*H*W + (size_t)c*H*W + (size_t)h*W;
        for (int w = 0; w < cols; ++w) std::printf("%8.4f ", t[base + (size_t)w]);
        std::printf("\n");
    }
    std::printf("\n");
}

struct CaseResult {
    const char* name;
    bool pass = false;
    bool skipped = false;
    std::string note;
};

struct DWCfg {
    const char* name;
    int N, C, H, W;
    int KH, KW;
    int pad_h, pad_w;
    int stride_h, stride_w;
};

static inline void out_dims(int H, int W, int KH, int KW,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int& OH, int& OW) {
    OH = (H + 2*pad_h - KH) / stride_h + 1;
    OW = (W + 2*pad_w - KW) / stride_w + 1;
}

static void fill_random_device_uniform(float* dptr, size_t n, unsigned long long seed) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateUniform(gen, dptr, (size_t)n));
    CURAND_CHECK(curandDestroyGenerator(gen));
}

static void fill_kernel_host(std::vector<float>& k, uint64_t seed) {
    std::mt19937 rng((uint32_t)seed);
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    for (auto& v : k) v = ud(rng);
}

static bool check_correctness(float* dX, float* dK, float* dY,
                              int N, int C, int H, int W,
                              int KH, int KW, int pad_h, int pad_w, int stride_h, int stride_w) {
    int OH, OW; out_dims(H,W,KH,KW,pad_h,pad_w,stride_h,stride_w,OH,OW);
    size_t in_elems  = (size_t)N*C*H*W;
    size_t ker_elems = (size_t)C*KH*KW;
    size_t out_elems = (size_t)N*C*OH*OW;

    std::vector<float> hX(in_elems), hK(ker_elems);
    CUDA_CHECK(cudaMemcpy(hX.data(), dX, in_elems*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hK.data(), dK, ker_elems*sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref;
    cpu_dwconv2d_ref(hX, hK, ref, N, C, H, W, KH, KW, pad_h, pad_w, stride_h, stride_w);

    solve(dX, dK, dY, N, C, H, W, KH, KW, pad_h, pad_w, stride_h, stride_w);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(out_elems);
    CUDA_CHECK(cudaMemcpy(out.data(), dY, out_elems*sizeof(float), cudaMemcpyDeviceToHost));

    const float atol = 1e-5f, rtol = 1e-5f;
    for (size_t i=0; i<out_elems; ++i) {
        float a = out[i], b = ref[i];
        if (!std::isfinite(a) || !std::isfinite(b)) return false;
        float diff = std::fabs(a - b);
        float thr  = atol + rtol * std::fabs(b);
        if (diff > thr) return false;
    }
    return true;
}

static float time_kernel(float *dX, float *dK, float *dY,
                         int N, int C, int H, int W,
                         int KH, int KW, int pad_h, int pad_w, int stride_h, int stride_w,
                         int warmup, int iters) {
    for (int i=0;i<warmup;++i) {
        solve(dX, dK, dY, N, C, H, W, KH, KW, pad_h, pad_w, stride_h, stride_w);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i=0;i<iters;++i) {
        solve(dX, dK, dY, N, C, H, W, KH, KW, pad_h, pad_w, stride_h, stride_w);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms=0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / (float)iters;
}

static bool run_case_fixed(const char* name,
                           int N, int C, int H, int W,
                           int KH, int KW,
                           int pad_h, int pad_w,
                           int stride_h, int stride_w,
                           const std::vector<float>& hX,
                           const std::vector<float>& hK) {
    std::vector<float> ref;
    cpu_dwconv2d_ref(hX, hK, ref, N, C, H, W, KH, KW, pad_h, pad_w, stride_h, stride_w);

    int OH, OW; out_dims(H,W,KH,KW,pad_h,pad_w,stride_h,stride_w,OH,OW);

    float *dX=nullptr, *dK=nullptr, *dY=nullptr;
    CUDA_CHECK(cudaMalloc(&dX, (size_t)N*C*H*W*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, (size_t)C*KH*KW*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, (size_t)N*C*OH*OW*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), (size_t)N*C*H*W*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), (size_t)C*KH*KW*sizeof(float), cudaMemcpyHostToDevice));

    solve(dX, dK, dY, N, C, H, W, KH, KW, pad_h, pad_w, stride_h, stride_w);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out((size_t)N*C*OH*OW);
    CUDA_CHECK(cudaMemcpy(out.data(), dY, out.size()*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dX); cudaFree(dK); cudaFree(dY);

    const float atol = 1e-5f, rtol = 1e-5f;
    bool ok = true; size_t bad = (size_t)-1;
    for (size_t i=0; i<out.size(); ++i) {
        float a = out[i], b = ref[i];
        if (!std::isfinite(a) || !std::isfinite(b)) { ok=false; bad=i; break; }
        float diff = std::fabs(a - b);
        float thr  = atol + rtol * std::fabs(b);
        if (diff > thr) { ok=false; bad=i; break; }
    }
    if (ok) {
        std::printf("[PASS] %s\n", name);
        return true;
    } else {
        std::printf("[FAIL] %s", name);
        if (bad != (size_t)-1) std::printf("  (first mismatch at idx %zu)\n", bad);
        else std::printf("\n");
        int OH2, OW2; out_dims(H,W,KH,KW,pad_h,pad_w,stride_h,stride_w,OH2,OW2);
        int total_out = N*C*OH2*OW2;
        if (total_out <= 64) {
            print_mat("Input X", hX, N, C, H, W);
            std::printf("Kernel (%d x %d):\n", C*KH, KW);
            for (int r=0; r<C*KH; ++r) {
                int c = r / KH;
                int kh = r - c*KH;
                size_t base = (size_t)c*KH*KW + (size_t)kh*KW;
                for (int w=0; w<KW; ++w) std::printf("%8.4f ", hK[base + (size_t)w]);
                std::printf("\n");
            }
            std::printf("\n");
            print_mat("CPU Ref (as (N*C*OH) x OW)", ref, N, C, OH2, OW2);
        }
        return false;
    }
}

int main() {
    const float eps_unused = 0.0f; // kept for parity with other harnesses
    const int warmup = 5;
    const int iters  = 10;
    const unsigned long long seed_all = 42ULL;

    int passed = 0, total = 0;

    std::vector<DWCfg> cfgs = {
        // MobileNetV1/V2 style 3x3 DW convs
        {"DW 3x3 s2 @112x112 C=32  (N=32)",  32,  32,112,112, 3,3, 1,1, 2,2},
        {"DW 3x3 s1 @56x56  C=64   (N=32)",  32,  64, 56, 56, 3,3, 1,1, 1,1},
        {"DW 3x3 s1 @14x14  C=1024 (N=8)",    8,1024, 14, 14, 3,3, 1,1, 1,1},

        // 5x5 depthwise for variety (less common but useful)
        {"DW 5x5 s1 @28x28  C=128  (N=16)",  16, 128, 28, 28, 5,5, 2,2, 1,1},
        {"DW 5x5 s2 @28x28  C=128  (N=16)",  16, 128, 28, 28, 5,5, 2,2, 2,2},
    };

    std::vector<CaseResult> results;

    for (const auto& cfg : cfgs) {
        int OH, OW; out_dims(cfg.H,cfg.W,cfg.KH,cfg.KW,cfg.pad_h,cfg.pad_w,cfg.stride_h,cfg.stride_w,OH,OW);
        size_t in_elems  = (size_t)cfg.N*cfg.C*cfg.H*cfg.W;
        size_t ker_elems = (size_t)cfg.C*cfg.KH*cfg.KW;
        size_t out_elems = (size_t)cfg.N*cfg.C*OH*OW;

        float *dX=nullptr, *dK=nullptr, *dY=nullptr;
        CUDA_CHECK(cudaMalloc(&dX, in_elems*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dK, ker_elems*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dY, out_elems*sizeof(float)));

        // Random X on device (seed 42), kernel on host (seed 42)
        fill_random_device_uniform(dX, in_elems, seed_all);
        std::vector<float> hK(ker_elems);
        fill_kernel_host(hK, seed_all);
        CUDA_CHECK(cudaMemcpy(dK, hK.data(), ker_elems*sizeof(float), cudaMemcpyHostToDevice));

        // Correctness (once, using current random tensors)
        bool ok = check_correctness(dX, dK, dY,
                                    cfg.N, cfg.C, cfg.H, cfg.W,
                                    cfg.KH, cfg.KW, cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w);
        results.push_back(CaseResult{cfg.name, ok, false, ok ? "" : "mismatch vs CPU reference"});

        // Timing
        float avg_ms = time_kernel(dX, dK, dY,
                                   cfg.N, cfg.C, cfg.H, cfg.W,
                                   cfg.KH, cfg.KW, cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w,
                                   warmup, iters);

        // MACs and bandwidth estimates
        double flops = 2.0 * (double)cfg.N * cfg.C * OH * OW * cfg.KH * cfg.KW;
        double tflops_per_s = (flops / (avg_ms * 1e-3)) / 1e12;
        double bytes = (double)in_elems*sizeof(float)
                     + (double)ker_elems*sizeof(float)
                     + (double)out_elems*sizeof(float);
        double gbps = (bytes / (avg_ms * 1e-3)) / 1e9;

        std::printf("\n==== Benchmark ====\n");
        std::printf("%s\n", cfg.name);
        std::printf("Shape: N=%d C=%d H=%d W=%d  KH=%d KW=%d  pad=(%d,%d) stride=(%d,%d)\n",
                    cfg.N,cfg.C,cfg.H,cfg.W,cfg.KH,cfg.KW,cfg.pad_h,cfg.pad_w,cfg.stride_h,cfg.stride_w);
        std::printf("Output: OH=%d OW=%d  elems(out)=%zu\n", OH, OW, out_elems);
        std::printf("Warmup=%d, Iters=%d\n", warmup, iters);
        std::printf("Avg time per iter: %.3f ms\n", avg_ms);
        std::printf("Throughput: %.2f TFLOPS/s\n", tflops_per_s);
        std::printf("Approx. BW: %.2f GB/s\n", gbps);

        cudaFree(dX); cudaFree(dK); cudaFree(dY);
    }

    std::printf("\n==== Summary (CNN DW) ====\n");
    int pass_cnt = 0;
    for (const auto& r : results) {
        if (r.skipped) {
            std::printf("[SKIP] %s  (%s)\n", r.name, r.note.c_str());
        } else if (r.pass) {
            std::printf("[PASS] %s\n", r.name);
            ++pass_cnt;
        } else {
            std::printf("[FAIL] %s  (%s)\n", r.name, r.note.c_str());
        }
    }

    std::printf("\n==== Deterministic Cases Summary ====\n");
    if (total > 0) {
        std::printf("Passed: %d / %d\n", passed, total);
    }
    if (results.size() > 0) {
        std::printf("CNN DW Passed: %d / %zu\n", pass_cnt, results.size());
    }
    return 0;
}
