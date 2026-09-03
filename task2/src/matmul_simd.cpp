// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

float sum256(__m256 p){
    float tmp[8];
    _mm256_storeu_ps(tmp,p);
    float su = 0.0f;
    su=(tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
    return su;
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_load_ps(&a[p]); 
                __m256 b1 = _mm256_load_ps(&b[p]); 
                acc = _mm256_fmadd_ps(a1,b1,acc);
            }
            float s = 0.0f;
            s+=sum256(acc);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;
            
        }
    }
}
