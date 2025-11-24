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

void cpu_bn_inference_nchw_ref(const std::vector<float>& x,
                               const std::vector<float>& gamma,
                               const std::vector<float>& beta,
                               std::vector<float>& y,
                               int N, int C, int H, int W,
                               float eps);

struct CaseResult {
    const char* name;
    bool pass = false;
    bool skipped = false;
    std::string note;
};

struct BenchCfg {
    const char* name;
    int N, C, H, W;
};

static float time_solve_kernel(float *dX, float *dG, float *dB, float *dY,
                               int N, int C, int H, int W, float eps,
                               int warmup, int iters)
{
    for (int i=0;i<warmup;++i) {
        solve(dX, dG, dB, dY, N, C, H, W, eps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i=0;i<iters;++i) {
        solve(dX, dG, dB, dY, N, C, H, W, eps);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms=0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / (float)iters;
}

static void fill_random_device_uniform(float* dX, size_t n, unsigned long long seed)
{
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateUniform(gen, dX, (size_t)n));
    CURAND_CHECK(curandDestroyGenerator(gen));
}

static void fill_gamma_beta_host(std::vector<float>& g, std::vector<float>& b, uint64_t seed)
{
    std::mt19937 rng((uint32_t)seed);
    std::uniform_real_distribution<float> udg(0.5f, 1.5f);
    std::uniform_real_distribution<float> udb(-0.5f, 0.5f);
    for (auto &v : g) v = udg(rng);
    for (auto &v : b) v = udb(rng);
}

static bool check_correctness(float* dX, float* dG, float* dB, float* dY,
                              const std::vector<float>& hG, const std::vector<float>& hB,
                              int N, int C, int H, int W, float eps)
{
    size_t total = (size_t)N*C*H*W;

    std::vector<float> hX(total);
    CUDA_CHECK(cudaMemcpy(hX.data(), dX, total*sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref;
    cpu_bn_inference_nchw_ref(hX, hG, hB, ref, N, C, H, W, eps);

    solve(dX, dG, dB, dY, N, C, H, W, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(total);
    CUDA_CHECK(cudaMemcpy(out.data(), dY, total*sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i=0;i<total;++i) {
        if (!std::isfinite(out[i]) || std::fabs(out[i]-ref[i])>1e-5f) return false;
    }
    return true;
}

int main() {
    const float eps = 1e-5f;
    const int warmup = 5;
    const int iters = 10;
    const unsigned long long seed_all = 42ULL;

    std::vector<BenchCfg> cfgs = {
        {"ResNet-50 stem  (N=16,C=64,H=112,W=112)", 16,  64, 112, 112},
        {"MobileNetV2 early(N=32,C=32,H=112,W=112)", 32,  32, 112, 112},
        {"LLaMA/Mistral 7B (B=4,S=2048,H=4096) -> N=4,C=4096,H=1,W=2048", 4, 4096, 1, 2048},
        {"LLaMA 70B width (B=4,S=4096,H=8192) -> N=4,C=8192,H=1,W=4096", 4, 8192, 1, 4096},
    };


    std::vector<CaseResult> results;

    for (const auto& cfg : cfgs) {
        const size_t total = (size_t)cfg.N * cfg.C * cfg.H * cfg.W;

        float *dX=nullptr, *dG=nullptr, *dB=nullptr, *dY=nullptr;
        CUDA_CHECK(cudaMalloc(&dX, total*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dG, (size_t)cfg.C*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, (size_t)cfg.C*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dY, total*sizeof(float)));

        fill_random_device_uniform(dX, total, seed_all);

        std::vector<float> hG((size_t)cfg.C), hB((size_t)cfg.C);
        fill_gamma_beta_host(hG, hB, seed_all);
        CUDA_CHECK(cudaMemcpy(dG, hG.data(), (size_t)cfg.C*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), (size_t)cfg.C*sizeof(float), cudaMemcpyHostToDevice));

        bool ok = check_correctness(dX, dG, dB, dY, hG, hB, cfg.N, cfg.C, cfg.H, cfg.W, eps);
        results.push_back(CaseResult{
            cfg.name, ok, false, ok ? "" : "mismatch vs CPU reference"
        });

        float avg_ms = time_solve_kernel(dX, dG, dB, dY, cfg.N, cfg.C, cfg.H, cfg.W, eps, warmup, iters);

        double elems = (double)total;
        double elems_per_s = elems / (avg_ms * 1e-3);
        double bytes = (double)total*sizeof(float) /*X*/
                     + (double)cfg.C*sizeof(float) /*G*/
                     + (double)cfg.C*sizeof(float) /*B*/
                     + (double)total*sizeof(float) /*Y*/;
        double gbps = (bytes / (avg_ms * 1e-3)) / 1e9;

        std::printf("\n==== Benchmark ====\n");
        std::printf("%s\n", cfg.name);
        std::printf("Total elements: %zu\n", total);
        std::printf("Warmup=%d, Iters=%d\n", warmup, iters);
        std::printf("Avg time per iter: %.3f ms\n", avg_ms);
        std::printf("Throughput: %.2f G elems/s\n", elems_per_s/1e9);
        std::printf("Approx. BW: %.2f GB/s\n", gbps);

        cudaFree(dX); cudaFree(dG); cudaFree(dB); cudaFree(dY);
    }

    std::printf("\n==== Summary ====\n");
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
    std::printf("Passed: %d  Total: %zu\n", pass_cnt, results.size());
    return 0;
}
