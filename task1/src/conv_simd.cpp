#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker, int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int T_H = 32;
    const int T_W = 32;

    for (int oys = 0; oys < H; oys += T_H) {
        for (int oxs = 0; oxs < W; oxs += T_W) {
            int oye = (oys + T_H > H) ? H : oys + T_H;
            int oxe = (oxs + T_W > W) ? W : oxs + T_W;

            for (int oy = oys; oy < oye; ++oy) {
                for (int ox = oxs; ox < oxe; ox += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            __m256 in1 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                            __m256 ker1 = _mm256_set1_ps(ker[ky * K + kx]);
                            acc = _mm256_fmadd_ps(in1, ker1, acc);
                        }
                    }
                    
                    _mm256_storeu_ps(&out[oy * W + ox], acc);
                }
            }
        }
    }
}