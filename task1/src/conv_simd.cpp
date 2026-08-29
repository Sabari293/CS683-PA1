// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < 4*(H/4); ++oy) {
        for (int ox = 0; ox < 8*(W/8); ox+=8) {
            _mm256_store_ps(&out[oy*W+ox],_mm256_setzero_ps());
        }
        for(int ox=8*(W/8);ox<W;ox+=1){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox <8*(W/8); ox+=8) {
                __m256 acc=_mm256_load_ps(&out[oy*W+ox]);
                for (int kx = 0; kx < K; ++kx) {
                    __m256 in1=_mm256_load_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 ker1=_mm256_set1_ps(ker[ky*K+kx]);
                    acc=_mm256_fmadd_ps(in1,ker1,acc);
                }
                _mm256_storeu_ps(&out[oy*W+ox],acc);
                
            }
            for (int ox = 8*(W/8); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
        oy++;
        for (int ox = 0; ox < 8*(W/8); ox+=8) {
            _mm256_store_ps(&out[oy*W+ox],_mm256_setzero_ps());
        }
        for(int ox=8*(W/8);ox<W;ox+=1){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox <8*(W/8); ox+=8) {
                __m256 acc=_mm256_load_ps(&out[oy*W+ox]);
                for (int kx = 0; kx < K; ++kx) {
                    __m256 in1=_mm256_load_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 ker1=_mm256_set1_ps(ker[ky*K+kx]);
                    acc=_mm256_fmadd_ps(in1,ker1,acc);
                }
                _mm256_storeu_ps(&out[oy*W+ox],acc);
                
            }
            for (int ox = 8*(W/8); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
        oy++;
        for (int ox = 0; ox < 8*(W/8); ox+=8) {
            _mm256_store_ps(&out[oy*W+ox],_mm256_setzero_ps());
        }
        for(int ox=8*(W/8);ox<W;ox+=1){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox <8*(W/8); ox+=8) {
                __m256 acc=_mm256_load_ps(&out[oy*W+ox]);
                for (int kx = 0; kx < K; ++kx) {
                    __m256 in1=_mm256_load_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 ker1=_mm256_set1_ps(ker[ky*K+kx]);
                    acc=_mm256_fmadd_ps(in1,ker1,acc);
                }
                _mm256_storeu_ps(&out[oy*W+ox],acc);
                
            }
            for (int ox = 8*(W/8); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
        oy++;
        for (int ox = 0; ox < 8*(W/8); ox+=8) {
            _mm256_store_ps(&out[oy*W+ox],_mm256_setzero_ps());
        }
        for(int ox=8*(W/8);ox<W;ox+=1){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox <8*(W/8); ox+=8) {
                __m256 acc=_mm256_load_ps(&out[oy*W+ox]);
                for (int kx = 0; kx < K; ++kx) {
                    __m256 in1=_mm256_load_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 ker1=_mm256_set1_ps(ker[ky*K+kx]);
                    acc=_mm256_fmadd_ps(in1,ker1,acc);
                }
                _mm256_storeu_ps(&out[oy*W+ox],acc);
                
            }
            for (int ox = 8*(W/8); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
    }
    for(int oy=4*(H/4);oy<H;oy++){
        for (int ox = 0; ox < 8*(W/8); ox+=8) {
            _mm256_store_ps(&out[oy*W+ox],_mm256_setzero_ps());
        }
        for(int ox=8*(W/8);ox<W;ox+=1){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox <8*(W/8); ox+=8) {
                __m256 acc=_mm256_load_ps(&out[oy*W+ox]);
                for (int kx = 0; kx < K; ++kx) {
                    __m256 in1=_mm256_load_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 ker1=_mm256_set1_ps(ker[ky*K+kx]);
                    acc=_mm256_fmadd_ps(in1,ker1,acc);
                }
                _mm256_storeu_ps(&out[oy*W+ox],acc);
                
            }
            for (int ox = 8*(W/8); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
    }
}
