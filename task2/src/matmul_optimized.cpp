// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include <iostream>

#include "matmul.h"

inline float hsum(__m256 v) {
    __m128 l = _mm256_castps256_ps128(v);
    __m128 h = _mm256_extractf128_ps(v, 1);
    l = _mm_add_ps(l, h);
    __m128 shuf = _mm_movehl_ps(l, l);
    l = _mm_add_ps(l, shuf);
    shuf = _mm_shuffle_ps(l, l, _MM_SHUFFLE(2, 3, 0, 1));
    l = _mm_add_ps(l, shuf);
    return _mm_cvtss_f32(l);
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    int prefetch_dist = 64;
    int tileH = 144;
    int tileW = 128;
    int kend = 8*(K/ 8);
    int t=3;
    for(int ii = 0;ii<M;ii+=tileH){
        int ie = (ii+tileH<M?ii+tileH:M);
        for(int jj = 0;jj<N;jj+=tileW){
            int je = (jj+tileW<N?jj+tileW:N);
            int iee = ii+ 3 * ((ie-ii) / 3);
            int jee = jj+4 * ((je-jj) / 4);
            for (int i = ii; i <iee; i += 3) {
                const float* a1 = A + static_cast<long>(i) * lda;
                const float* a2 = A + static_cast<long>(i + 1) * lda;
                const float* a3 = A + static_cast<long>(i + 2) * lda;

                    for (int j = jj; j < jee; j += 4) {
                        const float* b1 = B + static_cast<long>(j) * ldb;
                        const float* b2 = B + static_cast<long>(j + 1) * ldb;
                        const float* b3 = B + static_cast<long>(j + 2) * ldb;
                        const float* b4 = B + static_cast<long>(j + 3) * ldb;
                        
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();
                        __m256 acc4 = _mm256_setzero_ps();
                        __m256 acc5 = _mm256_setzero_ps();
                        __m256 acc6 = _mm256_setzero_ps();
                        __m256 acc7 = _mm256_setzero_ps();
                        __m256 acc8 = _mm256_setzero_ps();
                        __m256 acc9 = _mm256_setzero_ps();
                        __m256 acc10 = _mm256_setzero_ps();
                        __m256 acc11 = _mm256_setzero_ps();
                        __m256 acc12 = _mm256_setzero_ps();

                        for (int p = 0; p < kend; p += 8) {
                            if(p%16==0){
                                _mm_prefetch(reinterpret_cast<const char*>(&b1[p+prefetch_dist]),static_cast<_mm_hint>(t));
                                _mm_prefetch(reinterpret_cast<const char*>(&b2[p+prefetch_dist]),static_cast<_mm_hint>(t));
                                _mm_prefetch(reinterpret_cast<const char*>(&b3[p+prefetch_dist]),static_cast<_mm_hint>(t));
                                _mm_prefetch(reinterpret_cast<const char*>(&b4[p+prefetch_dist]),static_cast<_mm_hint>(t));
                            }
                            __m256 a01 = _mm256_loadu_ps(&a1[p]);
                            __m256 a02 = _mm256_loadu_ps(&a2[p]);
                            __m256 a03 = _mm256_loadu_ps(&a3[p]);
                            
                            
                            __m256 bt = _mm256_loadu_ps(&b1[p]);
                            acc1 = _mm256_fmadd_ps(a01, bt, acc1);
                            acc2 = _mm256_fmadd_ps(a02, bt, acc2);
                            acc3 = _mm256_fmadd_ps(a03, bt, acc3);
                            
                            
                            bt = _mm256_loadu_ps(&b2[p]);
                            acc4 = _mm256_fmadd_ps(a01, bt, acc4);
                            acc5 = _mm256_fmadd_ps(a02, bt, acc5);
                            acc6 = _mm256_fmadd_ps(a03, bt, acc6);

                            
                            bt = _mm256_loadu_ps(&b3[p]);
                            acc7 = _mm256_fmadd_ps(a01, bt, acc7);
                            acc8 = _mm256_fmadd_ps(a02, bt, acc8);
                            acc9 = _mm256_fmadd_ps(a03, bt, acc9);

                            
                            bt = _mm256_loadu_ps(&b4[p]);
                            acc10 = _mm256_fmadd_ps(a01, bt, acc10);
                            acc11 = _mm256_fmadd_ps(a02, bt, acc11);
                            acc12 = _mm256_fmadd_ps(a03, bt, acc12);
                        }
                        float s1 = hsum(acc1);
                        float s2 = hsum(acc4);
                        float s3 = hsum(acc7);
                        float s4 = hsum(acc10);
                        for (int p = kend; p < K; p++) {
                            s1 += a1[p] * b1[p];
                            s2 += a1[p] * b2[p];
                            s3 += a1[p] * b3[p];
                            s4 += a1[p] * b4[p];
                        }
                        C[static_cast<long>(i) * ldc + j] = s1;
                        C[static_cast<long>(i) * ldc + j + 1] = s2;
                        C[static_cast<long>(i) * ldc + j + 2] = s3;
                        C[static_cast<long>(i) * ldc + j + 3] = s4;
                        s1 = hsum(acc2);
                        s2 = hsum(acc5);
                        s3 = hsum(acc8);
                        s4 = hsum(acc11);
                        for (int p = kend; p < K; p++) {
                            s1 += a2[p] * b1[p];
                            s2 += a2[p] * b2[p];
                            s3 += a2[p] * b3[p];
                            s4 += a2[p] * b4[p];
                        }
                        C[static_cast<long>(i + 1) * ldc + j] = s1;
                        C[static_cast<long>(i + 1) * ldc + j + 1] = s2;
                        C[static_cast<long>(i + 1) * ldc + j + 2] = s3;
                        C[static_cast<long>(i + 1) * ldc + j + 3] = s4;
                        s1 = hsum(acc3);
                        s2 = hsum(acc6);
                        s3 = hsum(acc9);
                        s4 = hsum(acc12);
                        for (int p = kend; p < K; p++) {
                            s1 += a3[p] * b1[p];
                            s2 += a3[p] * b2[p];
                            s3 += a3[p] * b3[p];
                            s4 += a3[p] * b4[p];
                        }
                        C[static_cast<long>(i + 2) * ldc + j] = s1;
                        C[static_cast<long>(i + 2) * ldc + j + 1] = s2;
                        C[static_cast<long>(i + 2) * ldc + j + 2] = s3;
                        C[static_cast<long>(i + 2) * ldc + j + 3] = s4;
                    }

                    for (int j = jee; j < je; j += 1) {
                        const float* b = B + static_cast<long>(j) * ldb;
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();
                        for (int p = 0; p < kend; p += 8) {
                            __m256 bt = _mm256_loadu_ps(&b[p]);
                            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&a1[p]), bt, acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&a2[p]), bt, acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&a3[p]), bt, acc3);
                        }
                        float s1 = hsum(acc1);
                        float s2 = hsum(acc2);
                        float s3 = hsum(acc3);
                        for (int p = kend; p < K; p++) {
                            s1 += a1[p] * b[p];
                            s2 += a2[p] * b[p];
                            s3 += a3[p] * b[p];
                        }
                        C[static_cast<long>(i) * ldc + j] = s1;
                        C[static_cast<long>(i + 1) * ldc + j] = s2;
                        C[static_cast<long>(i + 2) * ldc + j] = s3;
                    }
                }

                for (int i = iee; i < ie; ++i) {
                    const float* a = A + static_cast<long>(i) * lda;

                    for (int j = jj; j < jee; j += 4) {
                        const float* b1 = B + static_cast<long>(j) * ldb;
                        const float* b2 = B + static_cast<long>(j + 1) * ldb;
                        const float* b3 = B + static_cast<long>(j + 2) * ldb;
                        const float* b4 = B + static_cast<long>(j + 3) * ldb;

                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();
                        __m256 acc4 = _mm256_setzero_ps();

                        for (int p = 0; p < kend; p += 8) {
                            __m256 a01 = _mm256_loadu_ps(&a[p]);

                            acc1 = _mm256_fmadd_ps(a01, _mm256_loadu_ps(&b1[p]), acc1);
                            acc2 = _mm256_fmadd_ps(a01, _mm256_loadu_ps(&b2[p]), acc2);
                            acc3 = _mm256_fmadd_ps(a01, _mm256_loadu_ps(&b3[p]), acc3);
                            acc4 = _mm256_fmadd_ps(a01, _mm256_loadu_ps(&b4[p]), acc4);
                        }

                        float s1 = hsum(acc1);
                        float s2 = hsum(acc2);
                        float s3 = hsum(acc3);
                        float s4 = hsum(acc4);

                        for (int p = kend; p < K; p++) {
                            s1 += a[p] * b1[p];
                            s2 += a[p] * b2[p];
                            s3 += a[p] * b3[p];
                            s4 += a[p] * b4[p];
                        }

                        C[static_cast<long>(i) * ldc + j] = s1;
                        C[static_cast<long>(i) * ldc + j + 1] = s2;
                        C[static_cast<long>(i) * ldc + j + 2] = s3;
                        C[static_cast<long>(i) * ldc + j + 3] = s4;
                    }

                    for (int j = jee; j < je; j += 1) {
                        const float* b = B + static_cast<long>(j) * ldb;
                        __m256 acc = _mm256_setzero_ps();
                        for (int p = 0; p < kend; p += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(&a[p]),_mm256_loadu_ps(&b[p]),acc);
                        float s=hsum(acc);
                        for (int p=kend;p<K;p++)s+=a[p]*b[p];
                        C[static_cast<long>(i) * ldc + j] = s;
                    }
            }
        }
    }
}