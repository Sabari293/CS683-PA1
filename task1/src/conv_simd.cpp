// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        
        for(int ox=8*(W/8);ox<W;ox+=1){
            out[oy*W+ox]=0.0f;
        }
        
        for (int ox = 0; ox <8*(W/8); ox+=8) {
            __m256 acc=_mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {    
                for (int kx = 0; kx < K; ++kx) {
                    __m256 in1=_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 ker1=_mm256_set1_ps(ker[ky*K+kx]);
                    acc=_mm256_fmadd_ps(in1,ker1,acc);
                }
            }
            _mm256_storeu_ps(&out[oy*W+ox],acc);
        }
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                float k = ker[ky * K + kx];
                for (int ox = 8*(W/8); ox < W; ++ox) {
                    out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * k;
                }
            }
        }
    }
}
