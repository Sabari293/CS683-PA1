// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

inline float hsum(__m256 v) {
    __m128 l  = _mm256_castps256_ps128(v);
    __m128 h = _mm256_extractf128_ps(v, 1);
    l= _mm_add_ps(l, h);
    __m128 s = _mm_movehl_ps(l, l);
    l = _mm_add_ps(l, s);
    s = _mm_shuffle_ps(l, l, _MM_SHUFFLE(2, 3, 0, 1));
    l = _mm_add_ps(l, s);
    return _mm_cvtss_f32(l);
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
            float s = hsum(acc);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;
            
        }
    }
}
