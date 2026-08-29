// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < 8*(H/8); ++oy) {
        for (int ox = 0; ox < 8*(W/8); ox+=8) {
            _mm256_store_ps(&out[oy*W+ox],_mm256_setzero_ps());
            _mm256_store_ps(&out[(oy+1)*W+ox],_mm256_setzero_ps());
            _mm256_store_ps(&out[(oy+2)*W+ox],_mm256_setzero_ps());
            _mm256_store_ps(&out[(oy+3)*W+ox],_mm256_setzero_ps());
            oy+=4;
            _mm256_store_ps(&out[oy*W+ox],_mm256_setzero_ps());
            _mm256_store_ps(&out[(oy+1)*W+ox],_mm256_setzero_ps());
            _mm256_store_ps(&out[(oy+2)*W+ox],_mm256_setzero_ps());
            _mm256_store_ps(&out[(oy+3)*W+ox],_mm256_setzero_ps());
            oy-=4;
        }
        for(int ox=8*(W/8);ox<W;ox+=1){
            out[oy*W+ox]=0.0f;
            out[(oy+1)*W+ox]=0.0f;
            out[(oy+2)*W+ox]=0.0f;
            out[(oy+3)*W+ox]=0.0f;
            oy+=4;
            out[oy*W+ox]=0.0f;
            out[(oy+1)*W+ox]=0.0f;
            out[(oy+2)*W+ox]=0.0f;
            out[(oy+3)*W+ox]=0.0f;
            oy-=4;
        }
        for(int ox=0;ox<8*(W/8);ox+=8){
            __m256 acc0=_mm256_load_ps(&out[oy*W+ox]);
            __m256 acc1=_mm256_load_ps(&out[(oy+1)*W+ox]);
            __m256 acc2=_mm256_load_ps(&out[(oy+2)*W+ox]);
            __m256 acc3=_mm256_load_ps(&out[(oy+3)*W+ox]);
            oy+=4;
            __m256 acc4=_mm256_load_ps(&out[oy*W+ox]);
            __m256 acc5=_mm256_load_ps(&out[(oy+1)*W+ox]);
            __m256 acc6=_mm256_load_ps(&out[(oy+2)*W+ox]);
            __m256 acc7=_mm256_load_ps(&out[(oy+3)*W+ox]);
            oy-=4;

            for(int ky=0;ky<K;++ky){
                for(int kx=0;kx<K;++kx){
                    __m256 ker1=_mm256_set1_ps(ker[ky*K+kx]);
                    __m256 in1=_mm256_loadu_ps(&in[(oy+ky)*in_stride+(ox+kx)]);
                    acc0=_mm256_fmadd_ps(in1,ker1,acc0);

                    in1=_mm256_loadu_ps(&in[(oy+1+ky)*in_stride+(ox+kx)]);
                    acc1=_mm256_fmadd_ps(in1,ker1,acc1);
            
                    in1=_mm256_loadu_ps(&in[(oy+2+ky)*in_stride+(ox+kx)]);
                    acc2=_mm256_fmadd_ps(in1,ker1,acc2);

                    in1=_mm256_loadu_ps(&in[(oy+3+ky)*in_stride+(ox+kx)]);
                    acc3=_mm256_fmadd_ps(in1,ker1,acc3);

                    in1=_mm256_loadu_ps(&in[(oy+4+ky)*in_stride+(ox+kx)]);
                    acc4=_mm256_fmadd_ps(in1,ker1,acc4);

                    in1=_mm256_loadu_ps(&in[(oy+5+ky)*in_stride+(ox+kx)]);
                    acc5=_mm256_fmadd_ps(in1,ker1,acc5);

                    in1=_mm256_loadu_ps(&in[(oy+6+ky)*in_stride+(ox+kx)]);
                    acc6=_mm256_fmadd_ps(in1,ker1,acc6);

                    in1=_mm256_loadu_ps(&in[(oy+7+ky)*in_stride+(ox+kx)]);
                    acc7=_mm256_fmadd_ps(in1,ker1,acc7);
                }
            }

            _mm256_storeu_ps(&out[oy*W+ox],acc0);
            _mm256_storeu_ps(&out[(oy+1)*W+ox],acc1);
            _mm256_storeu_ps(&out[(oy+2)*W+ox],acc2);
            _mm256_storeu_ps(&out[(oy+3)*W+ox],acc3);
            _mm256_storeu_ps(&out[(oy+4)*W+ox],acc4);
            _mm256_storeu_ps(&out[(oy+5)*W+ox],acc5);
            _mm256_storeu_ps(&out[(oy+6)*W+ox],acc6);
            _mm256_storeu_ps(&out[(oy+7)*W+ox],acc7);
        }
        
        oy+=7;
    }
    for (int oy = 8*(H/8); oy < H; ++oy) {
        for (int ox = 0; ox < 8*(W/8); ox+=8) {
            _mm256_store_ps(&out[oy*W+ox],_mm256_setzero_ps());
        }
        for(int ox=8*(W/8);ox<W;ox+=1){
            out[oy*W+ox]=0.0f;
        }
        for (int ox = 0; ox <8*(W/8); ox+=8) {
            for (int ky = 0; ky < K; ++ky) {   
                __m256 acc=_mm256_load_ps(&out[oy*W+ox]);
                for (int kx = 0; kx < K; ++kx) {
                    __m256 in1=_mm256_load_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 ker1=_mm256_set1_ps(ker[ky*K+kx]);
                    acc=_mm256_fmadd_ps(in1,ker1,acc);
                }
                _mm256_storeu_ps(&out[oy*W+ox],acc);
                
            }
        }
    }
}
