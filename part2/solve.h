#pragma once

void solve(const float* input, const float* kernel, float* output,
           int N, int C, int H, int W,
           int KH, int KW,
           int pad_h, int pad_w,
           int stride_h, int stride_w);

