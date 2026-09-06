// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker, int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < 4 * (H / 4); oy += 4) {
        for (int ox = 0; ox < W; ox += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int kx = 0; kx < K; ++kx) {
                __m256 row0 = _mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + kx)]);
                __m256 row1 = _mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + kx)]);
                __m256 row2 = _mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + kx)]);
                __m256 row3 = _mm256_loadu_ps(&in[(oy + 3) * in_stride + (ox + kx)]);

                for (int ky = 0; ky < K; ++ky) {
                    __m256 ker_val = _mm256_set1_ps(ker[ky * K + kx]);
                    
                    acc0 = _mm256_fmadd_ps(row0, ker_val, acc0);
                    acc1 = _mm256_fmadd_ps(row1, ker_val, acc1);
                    acc2 = _mm256_fmadd_ps(row2, ker_val, acc2);
                    acc3 = _mm256_fmadd_ps(row3, ker_val, acc3);

                    row0 = row1;
                    row1 = row2;
                    row2 = row3;
                    
                    if (ky < K - 1) {
                        row3 = _mm256_loadu_ps(&in[(oy + 4 + ky) * in_stride + (ox + kx)]);
                    }
                }
            }

            _mm256_storeu_ps(&out[(oy + 0) * W + ox], acc0);
            _mm256_storeu_ps(&out[(oy + 1) * W + ox], acc1);
            _mm256_storeu_ps(&out[(oy + 2) * W + ox], acc2);
            _mm256_storeu_ps(&out[(oy + 3) * W + ox], acc3);
        }
    }

    for (int oy = 4 * (H / 4); oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 8) {
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