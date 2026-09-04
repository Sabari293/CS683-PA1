// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING
//
// Strategy:
//   - 3-level tiling over M, N, K to keep working sets in L1/L2 cache.
//   - Inner micro-kernel is the same 4×2 AVX2 FMA register tile as Stage 1.
//   - Software prefetch hints (_mm_prefetch with _MM_HINT_T0) are issued ahead of:
//       * A rows (prefetch_dist elements ahead along K)
//       * B rows (prefetch_dist elements ahead along K)
//       * Next B-row for the j+1 iteration ("B-next-row" prefetch at the j loop entry)
//
// Tile sizes and prefetch distance are tuned empirically:
//   - MC=64, NC=64: fits 64*64 = 4096 floats = 16 KB (well within 32 KB L1-D)
//   - KC=256:       the K panel of A (64*256 = 64 KB) + B (64*256 = 64 KB)
//                   fits in a 256 KB L2 cache.
//   - PREFETCH_DIST=16 elements = 64 bytes = 1 cache line ahead in the K dimension.
//
// Compile flags (pinned): -O2 -fno-tree-vectorize -mavx2 -mfma

#include <immintrin.h>
#include "matmul.h"

static inline float hsum256pf(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // Tile sizes.
    static constexpr int MC = 64;   // rows of A per macro-tile
    static constexpr int NC = 64;   // rows of B per macro-tile
    static constexpr int KC = 256;  // K-slice width per macro-tile
    static constexpr int PFETCH = 16; // prefetch distance in floats (64 bytes = 1 cache line)

    // Initialize C to 0 (K-tiling accumulates into C).
    for (int i = 0; i < M; ++i) {
        float* c = C + (long)i * ldc;
        for (int j = 0; j < N; ++j) c[j] = 0.0f;
    }

    // Macro-tiles: iterate over K first (so the A/B K-panels stay in cache), then N, then M.
    for (int kk = 0; kk < K; kk += KC) {
        int ke = (kk + KC < K) ? (kk + KC) : K;

        for (int ii = 0; ii < M; ii += MC) {
            int ie = (ii + MC < M) ? (ii + MC) : M;

            for (int jj = 0; jj < N; jj += NC) {
                int je = (jj + NC < N) ? (jj + NC) : N;

                // ---- inner micro-kernel: 4 A-rows × 2 B-rows tile ----------
                int i = ii;
                for (; i + 3 < ie; i += 4) {
                    const float* a0 = A + (long)i       * lda + kk;
                    const float* a1 = A + (long)(i + 1) * lda + kk;
                    const float* a2 = A + (long)(i + 2) * lda + kk;
                    const float* a3 = A + (long)(i + 3) * lda + kk;

                    int j = jj;
                    for (; j + 1 < je; j += 2) {
                        const float* b0 = B + (long)j       * ldb + kk;
                        const float* b1 = B + (long)(j + 1) * ldb + kk;

                        // Prefetch the next pair of B rows into L1.
                        if (j + 2 < je) {
                            _mm_prefetch((const char*)(B + (long)(j + 2) * ldb + kk), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (long)(j + 3) * ldb + kk), _MM_HINT_T0);
                        }

                        __m256 acc00 = _mm256_setzero_ps();
                        __m256 acc10 = _mm256_setzero_ps();
                        __m256 acc20 = _mm256_setzero_ps();
                        __m256 acc30 = _mm256_setzero_ps();
                        __m256 acc01 = _mm256_setzero_ps();
                        __m256 acc11 = _mm256_setzero_ps();
                        __m256 acc21 = _mm256_setzero_ps();
                        __m256 acc31 = _mm256_setzero_ps();

                        int klen = ke - kk;
                        int p = 0;
                        for (; p + 7 < klen; p += 8) {
                            // Prefetch ahead by PFETCH floats.
                            if (p + PFETCH < klen) {
                                _mm_prefetch((const char*)(a0 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(a1 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(a2 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(a3 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(b0 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(b1 + p + PFETCH), _MM_HINT_T0);
                            }

                            __m256 va0 = _mm256_loadu_ps(a0 + p);
                            __m256 va1 = _mm256_loadu_ps(a1 + p);
                            __m256 va2 = _mm256_loadu_ps(a2 + p);
                            __m256 va3 = _mm256_loadu_ps(a3 + p);
                            __m256 vb0 = _mm256_loadu_ps(b0 + p);
                            __m256 vb1 = _mm256_loadu_ps(b1 + p);

                            acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                            acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                            acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                            acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                            acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                            acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                            acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                            acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                        }

                        // Horizontal reduction.
                        float s00 = hsum256pf(acc00), s10 = hsum256pf(acc10);
                        float s20 = hsum256pf(acc20), s30 = hsum256pf(acc30);
                        float s01 = hsum256pf(acc01), s11 = hsum256pf(acc11);
                        float s21 = hsum256pf(acc21), s31 = hsum256pf(acc31);

                        // Scalar tail for K slice remainder.
                        for (; p < klen; ++p) {
                            float b0p = b0[p], b1p = b1[p];
                            s00 += a0[p] * b0p; s01 += a0[p] * b1p;
                            s10 += a1[p] * b0p; s11 += a1[p] * b1p;
                            s20 += a2[p] * b0p; s21 += a2[p] * b1p;
                            s30 += a3[p] * b0p; s31 += a3[p] * b1p;
                        }

                        // Accumulate into C (K-tiling means += not =).
                        C[(long)i       * ldc + j]     += s00;
                        C[(long)(i + 1) * ldc + j]     += s10;
                        C[(long)(i + 2) * ldc + j]     += s20;
                        C[(long)(i + 3) * ldc + j]     += s30;
                        C[(long)i       * ldc + j + 1] += s01;
                        C[(long)(i + 1) * ldc + j + 1] += s11;
                        C[(long)(i + 2) * ldc + j + 1] += s21;
                        C[(long)(i + 3) * ldc + j + 1] += s31;
                    }
                    // Scalar cleanup for odd N in tile.
                    for (; j < je; ++j) {
                        const float* b0 = B + (long)j * ldb + kk;
                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();
                        int klen = ke - kk;
                        int p = 0;
                        for (; p + 7 < klen; p += 8) {
                            __m256 vb = _mm256_loadu_ps(b0 + p);
                            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), vb, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), vb, acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + p), vb, acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + p), vb, acc3);
                        }
                        float s0 = hsum256pf(acc0), s1 = hsum256pf(acc1);
                        float s2 = hsum256pf(acc2), s3 = hsum256pf(acc3);
                        for (; p < klen; ++p) {
                            float bp = b0[p];
                            s0 += a0[p] * bp; s1 += a1[p] * bp;
                            s2 += a2[p] * bp; s3 += a3[p] * bp;
                        }
                        C[(long)i       * ldc + j] += s0;
                        C[(long)(i + 1) * ldc + j] += s1;
                        C[(long)(i + 2) * ldc + j] += s2;
                        C[(long)(i + 3) * ldc + j] += s3;
                    }
                }
                // Scalar tail for leftover M rows in tile.
                for (; i < ie; ++i) {
                    const float* a = A + (long)i * lda + kk;
                    for (int j = jj; j < je; ++j) {
                        const float* b = B + (long)j * ldb + kk;
                        __m256 acc = _mm256_setzero_ps();
                        int klen = ke - kk, p = 0;
                        for (; p + 7 < klen; p += 8)
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p),
                                                  _mm256_loadu_ps(b + p), acc);
                        float s = hsum256pf(acc);
                        for (; p < klen; ++p) s += a[p] * b[p];
                        C[(long)i * ldc + j] += s;
                    }
                }
            } // jj
        } // ii
    } // kk
}
