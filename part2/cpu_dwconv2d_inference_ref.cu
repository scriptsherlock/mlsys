#include <vector>
#include <cmath>

// input x: [N,C,H, W], kernel k: [C,1,KH,KW] flattened as [C*KH*KW], output y: [N,C,OH,OW]
void cpu_dwconv2d_ref(const std::vector<float>& x,
                           const std::vector<float>& k,
                           std::vector<float>& y,
                           int N, int C, int H, int W,
                           int KH, int KW,
                           int pad_h, int pad_w,
                           int stride_h, int stride_w)
{
    int OH = (H + 2*pad_h - KH) / stride_h + 1;
    int OW = (W + 2*pad_w - KW) / stride_w + 1;
    y.assign((size_t)N * C * OH * OW, 0.0f);

    const int HW = H * W;
    const int CHW = C * HW;
    const int OHOW = OH * OW;
    const int COHOW = C * OHOW;

    for (int n=0; n<N; ++n) {
        for (int c=0; c<C; ++c) {
            for (int oh=0; oh<OH; ++oh) {
                for (int ow=0; ow<OW; ++ow) {
                    int h0 = oh * stride_h - pad_h;
                    int w0 = ow * stride_w - pad_w;
                    float acc = 0.0f;
                    for (int kh=0; kh<KH; ++kh) { // kernel height
                        int ih = h0 + kh;
                        if ((unsigned)ih >= (unsigned)H) continue;
                        for (int kw=0; kw<KW; ++kw) { // kernel width
                            int iw = w0 + kw;
                            if ((unsigned)iw >= (unsigned)W) continue;
                            float xv = x[(size_t)n*CHW + (size_t)c*HW + (size_t)ih*W + (size_t)iw];
                            float kv = k[(size_t)c*KH*KW + (size_t)kh*KW + (size_t)kw];
                            acc += xv * kv;
                        }
                    }
                    y[(size_t)n*COHOW + (size_t)c*OHOW + (size_t)oh*OW + (size_t)ow] = acc;
                }
            }
        }
    }
}

