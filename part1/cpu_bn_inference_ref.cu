#include <vector>
#include <cmath>

// input x: [N,C,H,W]  stat gamma: [C]  stat beta: [C]  output y: [N,C,H,W]
void cpu_bn_inference_nchw_ref(const std::vector<float>& x,
                        const std::vector<float>& gamma,
                        const std::vector<float>& beta,
                        std::vector<float>& y,
                        int N, int C, int H, int W,
                        float eps)
{
    const int NHW = N * H * W;
    y.assign((size_t)N*C*H*W, 0.0f);

    std::vector<double> mean(C, 0.0), var(C, 0.0);

    // mean over N*H*W per channel
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    size_t idx = (size_t)n*C*H*W + (size_t)c*H*W + (size_t)h*W + (size_t)w;
                    mean[c] += (double)x[idx];
                }
            }
        }
    }
    for (int c = 0; c < C; ++c) mean[c] /= (double)NHW;

    // variance over N*H*W per channel
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            double mc = mean[c];
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    size_t idx = (size_t)n*C*H*W + (size_t)c*H*W + (size_t)h*W + (size_t)w;
                    double d = (double)x[idx] - mc;
                    var[c] += d * d;
                }
            }
        }
    }
    for (int c = 0; c < C; ++c) var[c] /= (double)NHW;

    // normalize + affine
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            double mc = mean[c];
            double inv_std = 1.0 / std::sqrt(var[c] + (double)eps);
            double g = (double)gamma[c];
            double b = (double)beta[c];
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    size_t idx = (size_t)n*C*H*W + (size_t)c*H*W + (size_t)h*W + (size_t)w;
                    double xn = ((double)x[idx] - mc) * inv_std;
                    y[idx] = (float)(g * xn + b);
                }
            }
        }
    }
}

