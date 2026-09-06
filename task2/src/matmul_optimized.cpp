// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

float sum2561(__m256 p){
    float tmp[8];
    _mm256_storeu_ps(tmp,p);
    float su = 0.0f;
    su=(tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
    return su;
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    int dist = 64;
    int tile = 128;
    for(int ii = 0;ii<M;ii+=tile){
        int ie = (ii+tile<M?ii+tile:M);
        for(int jj = 0;jj<N;jj+=tile){
            int je = (jj+tile<N?jj+tile:N);
            for (int i = ii; i < ii+6*((ie-ii)/6); i+=6) {
                const float* a0 = A + static_cast<long>(i) * lda;
                const float* a1 = A + static_cast<long>(i+1) * lda;
                const float* a2 = A + static_cast<long>(i+2) * lda;
                const float* a3 = A + static_cast<long>(i+3) * lda;
                const float* a4 = A + static_cast<long>(i+4) * lda;
                const float* a5 = A + static_cast<long>(i+5) * lda;
                for (int j = jj; j < jj+2*((je-jj)/2); ++j) {
                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc02 = _mm256_setzero_ps();
                    __m256 acc03 = _mm256_setzero_ps();
                    __m256 acc04 = _mm256_setzero_ps();
                    __m256 acc05 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc12 = _mm256_setzero_ps();
                    __m256 acc13 = _mm256_setzero_ps();
                    __m256 acc14 = _mm256_setzero_ps();
                    __m256 acc15 = _mm256_setzero_ps();
                   
                    const float* b = B + static_cast<long>(j) * ldb;
                    const float* b01 = B+static_cast<long>(j+1)*ldb;
                    int p = 0;

                    for (; p < 16*((K)/16); p+=16) {
                        _mm_prefetch((const char*)&b[p + dist], _MM_HINT_T0);
                        _mm_prefetch((const char*)&b01[p + dist], _MM_HINT_T0);
                        __m256 a10 = _mm256_load_ps(&a0[p]); 
                        __m256 a11 = _mm256_load_ps(&a1[p]); 
                        __m256 a12 = _mm256_load_ps(&a2[p]); 
                        __m256 a13 = _mm256_load_ps(&a3[p]); 
                        __m256 a14 = _mm256_load_ps(&a4[p]); 
                        __m256 a15 = _mm256_load_ps(&a5[p]); 
                        __m256 a20 = _mm256_load_ps(&a0[p+8]); 
                        __m256 a21 = _mm256_load_ps(&a1[p+8]); 
                        __m256 a22 = _mm256_load_ps(&a2[p+8]); 
                        __m256 a23 = _mm256_load_ps(&a3[p+8]); 
                        __m256 a24 = _mm256_load_ps(&a4[p+8]); 
                        __m256 a25 = _mm256_load_ps(&a5[p+8]); 
                        __m256 b1 = _mm256_load_ps(&b[p]); 
                        __m256 b2 = _mm256_load_ps(&b01[p]);
                        __m256 b11 = _mm256_load_ps(&b[p+8]); 
                        __m256 b21 = _mm256_load_ps(&b01[p+8]);
                        acc00 = _mm256_fmadd_ps(a10,b1,acc00);
                        acc01 = _mm256_fmadd_ps(a11,b1,acc01);
                        acc02 = _mm256_fmadd_ps(a12,b1,acc02);
                        acc03 = _mm256_fmadd_ps(a13,b1,acc03);
                        acc04 = _mm256_fmadd_ps(a14,b1,acc04);
                        acc05 = _mm256_fmadd_ps(a15,b1,acc05);

                        acc10 = _mm256_fmadd_ps(a10,b2,acc10);
                        acc11 = _mm256_fmadd_ps(a11,b2,acc11);
                        acc12 = _mm256_fmadd_ps(a12,b2,acc12);
                        acc13 = _mm256_fmadd_ps(a13,b2,acc13);
                        acc14 = _mm256_fmadd_ps(a14,b2,acc14);
                        acc15 = _mm256_fmadd_ps(a15,b2,acc15);

                        acc00 = _mm256_fmadd_ps(a20,b11,acc00);
                        acc01 = _mm256_fmadd_ps(a21,b11,acc01);
                        acc02 = _mm256_fmadd_ps(a22,b11,acc02);
                        acc03 = _mm256_fmadd_ps(a23,b11,acc03);
                        acc04 = _mm256_fmadd_ps(a24,b11,acc04);
                        acc05 = _mm256_fmadd_ps(a25,b11,acc05);

                        acc10 = _mm256_fmadd_ps(a20,b21,acc10);
                        acc11 = _mm256_fmadd_ps(a21,b21,acc11);
                        acc12 = _mm256_fmadd_ps(a22,b21,acc12);
                        acc13 = _mm256_fmadd_ps(a23,b21,acc13);
                        acc14 = _mm256_fmadd_ps(a24,b21,acc14);
                        acc15 = _mm256_fmadd_ps(a25,b21,acc15);
                        

                    }
                    float s00 = sum2561(acc00);
                    float s10 = sum2561(acc01);
                    float s20 = sum2561(acc02);
                    float s30 = sum2561(acc03);
                    float s01 = sum2561(acc10);
                    float s11 = sum2561(acc11);
                    float s21 = sum2561(acc12);
                    float s31 = sum2561(acc13);
                    float s40 = sum2561(acc04);
                    float s50 = sum2561(acc05);
                    float s41 = sum2561(acc14);
                    float s51 = sum2561(acc15);
                    for(;p<K;p++){
                        s00 += (a0[p]*b[p]);
                        s10 += (a1[p]*b[p]);
                        s20 += (a2[p]*b[p]);
                        s30 += (a3[p]*b[p]);
                        s40 += (a4[p]*b[p]);
                        s50 += (a5[p]*b[p]);
                        s01 += (a0[p]*b01[p]);
                        s11 += (a1[p]*b01[p]);
                        s21 += (a2[p]*b01[p]);
                        s31 += (a3[p]*b01[p]);
                        s41 += (a4[p]*b01[p]);
                        s51 += (a5[p]*b01[p]);
                        
                    }
                    C[static_cast<long>(i) * ldc + j]=s00;
                    C[static_cast<long>(i+1) * ldc + j]=s10;
                    C[static_cast<long>(i+2) * ldc + j]=s20;
                    C[static_cast<long>(i+3) * ldc + j]=s30;
                    C[static_cast<long>(i+4) * ldc + j]=s40;
                    C[static_cast<long>(i+5) * ldc + j]=s50;
                
                    j++;
                    C[static_cast<long>(i) * ldc + j]=s01;
                    C[static_cast<long>(i+1) * ldc + j]=s11;
                    C[static_cast<long>(i+2) * ldc + j]=s21;
                    C[static_cast<long>(i+3) * ldc + j]=s31;   
                    C[static_cast<long>(i+4) * ldc + j]=s41;
                    C[static_cast<long>(i+5) * ldc + j]=s51;                
                    
                }
                for (int j =jj+ 2*((je-jj)/2); j < je; ++j) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    __m256 acc4 = _mm256_setzero_ps();
                    __m256 acc5 = _mm256_setzero_ps();
                    const float* b = B + static_cast<long>(j) * ldb;
                    int p = 0;
                    for (; p < 8*((K)/8); p+=8) {
                        __m256 a10 = _mm256_load_ps(&a0[p]); 
                        __m256 a11 = _mm256_load_ps(&a1[p]); 
                        __m256 a12 = _mm256_load_ps(&a2[p]); 
                        __m256 a13 = _mm256_load_ps(&a3[p]); 
                        __m256 a14 = _mm256_load_ps(&a4[p]); 
                        __m256 a15 = _mm256_load_ps(&a5[p]); 
                        __m256 b1 = _mm256_load_ps(&b[p]); 
                        acc0 = _mm256_fmadd_ps(a10,b1,acc0);
                        acc1 = _mm256_fmadd_ps(a11,b1,acc1);
                        acc2 = _mm256_fmadd_ps(a12,b1,acc2);
                        acc3 = _mm256_fmadd_ps(a13,b1,acc3);
                        acc4 = _mm256_fmadd_ps(a14,b1,acc4);
                        acc5 = _mm256_fmadd_ps(a15,b1,acc5);
                    }
                    float s0 = sum2561(acc0);
                    float s1 = sum2561(acc1);
                    float s2 = sum2561(acc2);
                    float s3 = sum2561(acc3);
                    float s4 = sum2561(acc4);
                    float s5 = sum2561(acc5);
                    for(;p<K;p++){
                        s0 += (a0[p]*b[p]);
                        s1 += (a1[p]*b[p]);
                        s2 += (a2[p]*b[p]);
                        s3 += (a3[p]*b[p]);
                        s4 += (a4[p]*b[p]);
                        s5 += (a5[p]*b[p]);
                    }
                    C[static_cast<long>(i) * ldc + j]=s0;
                    C[static_cast<long>(i+1) * ldc + j]=s1;
                    C[static_cast<long>(i+2) * ldc + j]=s2;
                    C[static_cast<long>(i+3) * ldc + j]=s3;
                    C[static_cast<long>(i+4) * ldc + j]=s4;
                    C[static_cast<long>(i+5) * ldc + j]=s5;
                    
                }
            }
            for (int i = ii+6*((ie-ii)/6); i < ie; ++i) {
                const float* a = A + static_cast<long>(i) * lda;
                for (int j = jj; j < je; ++j) {
                    __m256 acc = _mm256_setzero_ps();
                    const float* b = B + static_cast<long>(j) * ldb;
                    int p = 0;
                    for (; p < 8*((K)/8); p+=8) {
                        __m256 a1 = _mm256_load_ps(&a[p]); 
                        __m256 b1 = _mm256_load_ps(&b[p]); 
                        acc = _mm256_fmadd_ps(a1,b1,acc);
                    }
                    float tmp[8];
                    _mm256_storeu_ps(tmp,acc);
                    float s = 0.0f;
                    s+=(tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
                    for(;p<K;p++){
                        s += (a[p]*b[p]);
                    }
                    C[static_cast<long>(i) * ldc + j]=s;
                }
            }
        }
    }
}
