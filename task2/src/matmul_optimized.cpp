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
    for(int i=0;i<M;i++){
        float* c = C+static_cast<long>(i)*(ldc);
        __m256 zeros = _mm256_setzero_ps();
        for(int j = 0;j<8*(N/8);j+=8)_mm256_store_ps(&c[j],zeros);
        for(int j=8*(N/8);j<N;j++)c[j]=0.0f;
    }
    int tile = 256;
    
    for(int ii = 0;ii<M;ii+=tile){
        int ie = (ii+tile<M?ii+tile:M);
        for(int jj = 0;jj<N;jj+=tile){
            int je = (jj+tile<N?jj+tile:N);
            for (int i = ii; i < ii+4*((ie-ii)/4); i+=4) {
                const float* a0 = A + static_cast<long>(i) * lda;
                const float* a1 = A + static_cast<long>(i+1) * lda;
                const float* a2 = A + static_cast<long>(i+2) * lda;
                const float* a3 = A + static_cast<long>(i+3) * lda;

                for (int j = jj; j < jj+2*((je-jj)/2); ++j) {
                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc02 = _mm256_setzero_ps();
                    __m256 acc03 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc12 = _mm256_setzero_ps();
                    __m256 acc13 = _mm256_setzero_ps();
                    const float* b = B + static_cast<long>(j) * ldb;
                    const float* b01 = B+static_cast<long>(j+1)*ldb;
                    int p = 0;
                    for (; p < 8*((K)/8); p+=8) {
                        if (p + dist < K) {
                            _mm_prefetch((const char*)&a0[p + dist], _MM_HINT_T0);
                            _mm_prefetch((const char*)&a1[p + dist], _MM_HINT_T0);
                            _mm_prefetch((const char*)&a2[p + dist], _MM_HINT_T0);
                            _mm_prefetch((const char*)&a3[p + dist], _MM_HINT_T0);
                            _mm_prefetch((const char*)&b[p + dist], _MM_HINT_T0);
                            _mm_prefetch((const char*)&b01[p + dist], _MM_HINT_T0);
                        }
                        __m256 a10 = _mm256_load_ps(&a0[p]); 
                        __m256 a11 = _mm256_load_ps(&a1[p]); 
                        __m256 a12 = _mm256_load_ps(&a2[p]); 
                        __m256 a13 = _mm256_load_ps(&a3[p]); 
                        __m256 b1 = _mm256_load_ps(&b[p]); 
                        __m256 b2 = _mm256_load_ps(&b01[p]);
                        acc00 = _mm256_fmadd_ps(a10,b1,acc00);
                        acc01 = _mm256_fmadd_ps(a11,b1,acc01);
                        acc02 = _mm256_fmadd_ps(a12,b1,acc02);
                        acc03 = _mm256_fmadd_ps(a13,b1,acc03);

                        acc10 = _mm256_fmadd_ps(a10,b2,acc10);
                        acc11 = _mm256_fmadd_ps(a11,b2,acc11);
                        acc12 = _mm256_fmadd_ps(a12,b2,acc12);
                        acc13 = _mm256_fmadd_ps(a13,b2,acc13);
                    }
                    float s0 = sum2561(acc00);
                    float s1 = sum2561(acc01);
                    float s2 = sum2561(acc02);
                    float s3 = sum2561(acc03);
                    for(;p<K;p++){
                        s0 += (a0[p]*b[p]);
                        s1 += (a1[p]*b[p]);
                        s2 += (a2[p]*b[p]);
                        s3 += (a3[p]*b[p]);
                    }
                    C[static_cast<long>(i) * ldc + j]+=s0;
                    C[static_cast<long>(i+1) * ldc + j]+=s1;
                    C[static_cast<long>(i+2) * ldc + j]+=s2;
                    C[static_cast<long>(i+3) * ldc + j]+=s3;
                
                    s0 = sum2561(acc10);
                    s1 = sum2561(acc11);
                    s2 = sum2561(acc12);
                    s3 = sum2561(acc13);
                    for(;p<K;p++){
                        s0 += (a0[p]*b[p]);
                        s1 += (a1[p]*b[p]);
                        s2 += (a2[p]*b[p]);
                        s3 += (a3[p]*b[p]);
                    }
                    j++;
                    C[static_cast<long>(i) * ldc + j]+=s0;
                    C[static_cast<long>(i+1) * ldc + j]+=s1;
                    C[static_cast<long>(i+2) * ldc + j]+=s2;
                    C[static_cast<long>(i+3) * ldc + j]+=s3;
                    
                }
                for (int j =jj+ 2*((je-jj)/2); j < je; ++j) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    const float* b = B + static_cast<long>(j) * ldb;
                    int p = 0;
                    for (; p < 8*((K)/8); p+=8) {
                        __m256 a10 = _mm256_load_ps(&a0[p]); 
                        __m256 a11 = _mm256_load_ps(&a1[p]); 
                        __m256 a12 = _mm256_load_ps(&a2[p]); 
                        __m256 a13 = _mm256_load_ps(&a3[p]); 
                        __m256 b1 = _mm256_load_ps(&b[p]); 
                        acc0 = _mm256_fmadd_ps(a10,b1,acc0);
                        acc1 = _mm256_fmadd_ps(a11,b1,acc1);
                        acc2 = _mm256_fmadd_ps(a12,b1,acc2);
                        acc3 = _mm256_fmadd_ps(a13,b1,acc3);
                    }
                    float s0 = sum2561(acc0);
                    float s1 = sum2561(acc1);
                    float s2 = sum2561(acc2);
                    float s3 = sum2561(acc3);
                    for(;p<K;p++){
                        s0 += (a0[p]*b[p]);
                        s1 += (a1[p]*b[p]);
                        s2 += (a2[p]*b[p]);
                        s3 += (a3[p]*b[p]);
                    }
                    C[static_cast<long>(i) * ldc + j]+=s0;
                    C[static_cast<long>(i+1) * ldc + j]+=s1;
                    C[static_cast<long>(i+2) * ldc + j]+=s2;
                    C[static_cast<long>(i+3) * ldc + j]+=s3;
                    
                }
            }
            for (int i = ii+4*((ie-ii)/4); i < ie; ++i) {
                const float* a = A + static_cast<long>(i) * lda;
                for (int j = 0; j < N; ++j) {
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
                    C[static_cast<long>(i) * ldc + j]+=s;
                }
            }
        }
    }
}
