#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    int dist1[] = {16, 32, 64, 128};
    int ii = 2;
    int dist = dist1[ii];
    int t = 0;
    // std::cout<<"prefetch dist: "<<dist<<" t: "<<t<<std::endl;
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* b = B + static_cast<long>(j) * ldb;
            _mm_prefetch(reinterpret_cast<const char*>(&b[0]), _MM_HINT_NTA);
            // _mm_prefetch(reinterpret_cast<const char*>(&a[0]), _MM_HINT_T0);
            for (int p = 0; p < K; p++) {
                if((p & 15) == 0){
                    // _mm_prefetch(reinterpret_cast<const char*>(&a[p + 16]), _MM_HINT_T1);
                    // _mm_prefetch(reinterpret_cast<const char*>(&a[p + 32]), _MM_HINT_T2);
                    _mm_prefetch(reinterpret_cast<const char*>(&b[p + dist]), static_cast<_mm_hint>(t));
                }
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }

}