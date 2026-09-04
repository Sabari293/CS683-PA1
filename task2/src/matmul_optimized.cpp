// matmul_optimized.cpp  STAGE 3: ALL TECHNIQUES COMBINED  (GRADED)
//
// This kernel is injected verbatim into llama.cpp's CPU SGEMM path, so it must:
//   (a) be correct for arbitrary M, N, K and leading dimensions lda/ldb/ldc, and
//   (b) be as fast as possible, since its speedup over matmul_naive determines the score.
//
// Techniques applied (building on every lesson from the assignment):
//
//  1. LOOP REORDERING   – K-tiles iterate outermost so consecutive K-slices of A and B
//                         stay resident in L2/L1 while we sweep the M×N block.
//
//  2. CACHE TILING      – Three-level blocking (K × M × N macro-tiles) sizes the working
//                         set to fit in L2: two K-panels of size MC×KC and NC×KC together
//                         at ~128 KB.  MC=128, NC=128, KC=256.
//
//  3. SIMD (AVX2 FMA)   – Inner micro-kernel uses 8-wide AVX2 __m256 FMA.
//
//  4. REGISTER BLOCKING – Micro-kernel computes a 6 A-rows × 2 B-columns tile.
//                         6 accumulators per B-column × 2 B-columns = 12 __m256
//                         registers, leaving room for data and avoiding spills.
//
//  5. LOOP UNROLLING    – K loop unrolled 2× inside the SIMD block (2×8 = 16 K elements
//                         per inner iteration) to hide FMA latency (throughput = 2/cycle
//                         but latency = 4 cycles on modern Intel cores).
//
//  6. SOFTWARE PREFETCH – Each K-iteration prefetches the same A-rows and B-rows
//                         PFETCH elements ahead, placing data in L1 before it is
//                         consumed. Also prefetches the next j+2 B-rows ahead of the
//                         j loop to avoid B-row cold-miss stalls.
//
// Compile flags (pinned): -O2 -fno-tree-vectorize -mavx2 -mfma

#include <immintrin.h>
#include "matmul.h"

// Fast horizontal sum of an __m256 -> scalar.
static inline float hsum256_opt(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {

    // ---- Tile sizes -------------------------------------------------------
    // MC × KC  A-panel: 128 × 256 × 4 bytes = 128 KB  (fits in L2)
    // NC × KC  B-panel: 128 × 256 × 4 bytes = 128 KB  (fits in L2)
    // MC × NC  C-tile:  128 × 128 × 4 bytes =  64 KB  (fits in L2)
    static constexpr int MC = 128;
    static constexpr int NC = 128;
    static constexpr int KC = 256;
    // Prefetch distance in floats (2 cache lines = 32 floats = 128 bytes).
    static constexpr int PFETCH = 32;

    // ---- Initialize C to 0 ------------------------------------------------
    // (K-tiling accumulates with += so we need a clean slate.)
    for (int i = 0; i < M; ++i) {
        float* ci = C + (long)i * ldc;
        int j = 0;
        for (; j + 7 < N; j += 8) {
            _mm256_storeu_ps(ci + j, _mm256_setzero_ps());
        }
        for (; j < N; ++j) ci[j] = 0.0f;
    }

    // ---- Three-level tiled loops ------------------------------------------
    // Order: K-tile (kk), then M-tile (ii), then N-tile (jj).
    // This order keeps the active A-panel and B-panel in L2 while the inner
    // M×N micro-kernel sweeps the corresponding C-tile.
    for (int kk = 0; kk < K; kk += KC) {
        const int ke   = (kk + KC < K) ? kk + KC : K;
        const int klen = ke - kk;  // number of K-elements in this slice

        for (int ii = 0; ii < M; ii += MC) {
            const int ie = (ii + MC < M) ? ii + MC : M;

            for (int jj = 0; jj < N; jj += NC) {
                const int je = (jj + NC < N) ? jj + NC : N;

                // ==============================================================
                // Micro-kernel: 6 A-rows × 2 B-columns register tile
                // Each output element C[i][j] accumulates a K-slice dot product.
                // ==============================================================
                int i = ii;
                for (; i + 5 < ie; i += 6) {
                    const float* a0 = A + (long)i       * lda + kk;
                    const float* a1 = A + (long)(i + 1) * lda + kk;
                    const float* a2 = A + (long)(i + 2) * lda + kk;
                    const float* a3 = A + (long)(i + 3) * lda + kk;
                    const float* a4 = A + (long)(i + 4) * lda + kk;
                    const float* a5 = A + (long)(i + 5) * lda + kk;

                    // Prefetch first cache lines of all 6 A rows into L1.
                    _mm_prefetch((const char*)a0, _MM_HINT_T0);
                    _mm_prefetch((const char*)a1, _MM_HINT_T0);
                    _mm_prefetch((const char*)a2, _MM_HINT_T0);
                    _mm_prefetch((const char*)a3, _MM_HINT_T0);
                    _mm_prefetch((const char*)a4, _MM_HINT_T0);
                    _mm_prefetch((const char*)a5, _MM_HINT_T0);

                    int j = jj;
                    // 2-column tile with K-loop unrolled 2×.
                    for (; j + 1 < je; j += 2) {
                        const float* b0 = B + (long)j       * ldb + kk;
                        const float* b1 = B + (long)(j + 1) * ldb + kk;

                        // Prefetch next pair of B rows well ahead of use.
                        if (j + 2 < je) {
                            _mm_prefetch((const char*)(B + (long)(j + 2) * ldb + kk), _MM_HINT_T1);
                            _mm_prefetch((const char*)(B + (long)(j + 3) * ldb + kk), _MM_HINT_T1);
                        }

                        // 12 accumulator registers: 6 rows × 2 columns.
                        __m256 acc00 = _mm256_setzero_ps();
                        __m256 acc10 = _mm256_setzero_ps();
                        __m256 acc20 = _mm256_setzero_ps();
                        __m256 acc30 = _mm256_setzero_ps();
                        __m256 acc40 = _mm256_setzero_ps();
                        __m256 acc50 = _mm256_setzero_ps();
                        __m256 acc01 = _mm256_setzero_ps();
                        __m256 acc11 = _mm256_setzero_ps();
                        __m256 acc21 = _mm256_setzero_ps();
                        __m256 acc31 = _mm256_setzero_ps();
                        __m256 acc41 = _mm256_setzero_ps();
                        __m256 acc51 = _mm256_setzero_ps();

                        int p = 0;

                        // Unrolled 2×: consume 16 K-elements per iteration.
                        for (; p + 15 < klen; p += 16) {
                            // --- unroll iteration 0 (p+0..p+7) ---
                            if (p + PFETCH < klen) {
                                _mm_prefetch((const char*)(a0 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(a1 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(a2 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(a3 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(a4 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(a5 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(b0 + p + PFETCH), _MM_HINT_T0);
                                _mm_prefetch((const char*)(b1 + p + PFETCH), _MM_HINT_T0);
                            }
                            {
                                __m256 va0 = _mm256_loadu_ps(a0 + p);
                                __m256 va1 = _mm256_loadu_ps(a1 + p);
                                __m256 va2 = _mm256_loadu_ps(a2 + p);
                                __m256 va3 = _mm256_loadu_ps(a3 + p);
                                __m256 va4 = _mm256_loadu_ps(a4 + p);
                                __m256 va5 = _mm256_loadu_ps(a5 + p);
                                __m256 vb0 = _mm256_loadu_ps(b0 + p);
                                __m256 vb1 = _mm256_loadu_ps(b1 + p);

                                acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                                acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                                acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                                acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                                acc40 = _mm256_fmadd_ps(va4, vb0, acc40);
                                acc50 = _mm256_fmadd_ps(va5, vb0, acc50);
                                acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                                acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                                acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                                acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                                acc41 = _mm256_fmadd_ps(va4, vb1, acc41);
                                acc51 = _mm256_fmadd_ps(va5, vb1, acc51);
                            }
                            // --- unroll iteration 1 (p+8..p+15) ---
                            {
                                __m256 va0 = _mm256_loadu_ps(a0 + p + 8);
                                __m256 va1 = _mm256_loadu_ps(a1 + p + 8);
                                __m256 va2 = _mm256_loadu_ps(a2 + p + 8);
                                __m256 va3 = _mm256_loadu_ps(a3 + p + 8);
                                __m256 va4 = _mm256_loadu_ps(a4 + p + 8);
                                __m256 va5 = _mm256_loadu_ps(a5 + p + 8);
                                __m256 vb0 = _mm256_loadu_ps(b0 + p + 8);
                                __m256 vb1 = _mm256_loadu_ps(b1 + p + 8);

                                acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                                acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                                acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                                acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                                acc40 = _mm256_fmadd_ps(va4, vb0, acc40);
                                acc50 = _mm256_fmadd_ps(va5, vb0, acc50);
                                acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                                acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                                acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                                acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                                acc41 = _mm256_fmadd_ps(va4, vb1, acc41);
                                acc51 = _mm256_fmadd_ps(va5, vb1, acc51);
                            }
                        }
                        // Finish remaining 8-element chunks.
                        for (; p + 7 < klen; p += 8) {
                            __m256 va0 = _mm256_loadu_ps(a0 + p);
                            __m256 va1 = _mm256_loadu_ps(a1 + p);
                            __m256 va2 = _mm256_loadu_ps(a2 + p);
                            __m256 va3 = _mm256_loadu_ps(a3 + p);
                            __m256 va4 = _mm256_loadu_ps(a4 + p);
                            __m256 va5 = _mm256_loadu_ps(a5 + p);
                            __m256 vb0 = _mm256_loadu_ps(b0 + p);
                            __m256 vb1 = _mm256_loadu_ps(b1 + p);

                            acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                            acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                            acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                            acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                            acc40 = _mm256_fmadd_ps(va4, vb0, acc40);
                            acc50 = _mm256_fmadd_ps(va5, vb0, acc50);
                            acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                            acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                            acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                            acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                            acc41 = _mm256_fmadd_ps(va4, vb1, acc41);
                            acc51 = _mm256_fmadd_ps(va5, vb1, acc51);
                        }

                        // Horizontal reduction.
                        float s00 = hsum256_opt(acc00); float s01 = hsum256_opt(acc01);
                        float s10 = hsum256_opt(acc10); float s11 = hsum256_opt(acc11);
                        float s20 = hsum256_opt(acc20); float s21 = hsum256_opt(acc21);
                        float s30 = hsum256_opt(acc30); float s31 = hsum256_opt(acc31);
                        float s40 = hsum256_opt(acc40); float s41 = hsum256_opt(acc41);
                        float s50 = hsum256_opt(acc50); float s51 = hsum256_opt(acc51);

                        // Scalar tail: remaining K elements in this slice.
                        for (; p < klen; ++p) {
                            float b0p = b0[p], b1p = b1[p];
                            s00 += a0[p] * b0p; s01 += a0[p] * b1p;
                            s10 += a1[p] * b0p; s11 += a1[p] * b1p;
                            s20 += a2[p] * b0p; s21 += a2[p] * b1p;
                            s30 += a3[p] * b0p; s31 += a3[p] * b1p;
                            s40 += a4[p] * b0p; s41 += a4[p] * b1p;
                            s50 += a5[p] * b0p; s51 += a5[p] * b1p;
                        }

                        // Accumulate into C (K-tiling += semantics).
                        C[(long)i       * ldc + j]     += s00;
                        C[(long)(i + 1) * ldc + j]     += s10;
                        C[(long)(i + 2) * ldc + j]     += s20;
                        C[(long)(i + 3) * ldc + j]     += s30;
                        C[(long)(i + 4) * ldc + j]     += s40;
                        C[(long)(i + 5) * ldc + j]     += s50;
                        C[(long)i       * ldc + j + 1] += s01;
                        C[(long)(i + 1) * ldc + j + 1] += s11;
                        C[(long)(i + 2) * ldc + j + 1] += s21;
                        C[(long)(i + 3) * ldc + j + 1] += s31;
                        C[(long)(i + 4) * ldc + j + 1] += s41;
                        C[(long)(i + 5) * ldc + j + 1] += s51;
                    }

                    // --- scalar cleanup: odd final B-column in tile ----------
                    for (; j < je; ++j) {
                        const float* b0 = B + (long)j * ldb + kk;
                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();
                        __m256 acc4 = _mm256_setzero_ps();
                        __m256 acc5 = _mm256_setzero_ps();
                        int p = 0;
                        for (; p + 7 < klen; p += 8) {
                            __m256 vb = _mm256_loadu_ps(b0 + p);
                            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), vb, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), vb, acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + p), vb, acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + p), vb, acc3);
                            acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(a4 + p), vb, acc4);
                            acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(a5 + p), vb, acc5);
                        }
                        float s0 = hsum256_opt(acc0), s1 = hsum256_opt(acc1);
                        float s2 = hsum256_opt(acc2), s3 = hsum256_opt(acc3);
                        float s4 = hsum256_opt(acc4), s5 = hsum256_opt(acc5);
                        for (; p < klen; ++p) {
                            float bp = b0[p];
                            s0 += a0[p] * bp; s1 += a1[p] * bp;
                            s2 += a2[p] * bp; s3 += a3[p] * bp;
                            s4 += a4[p] * bp; s5 += a5[p] * bp;
                        }
                        C[(long)i       * ldc + j] += s0;
                        C[(long)(i + 1) * ldc + j] += s1;
                        C[(long)(i + 2) * ldc + j] += s2;
                        C[(long)(i + 3) * ldc + j] += s3;
                        C[(long)(i + 4) * ldc + j] += s4;
                        C[(long)(i + 5) * ldc + j] += s5;
                    }
                } // i (6-row blocks)

                // ==============================================================
                // Tail: leftover M rows (< 6) within the macro-tile.
                // Use 4-row, 2-row, then 1-row sub-kernels.
                // ==============================================================

                // --- 4-row tail ----------------------------------------------
                for (; i + 3 < ie; i += 4) {
                    const float* a0 = A + (long)i       * lda + kk;
                    const float* a1 = A + (long)(i + 1) * lda + kk;
                    const float* a2 = A + (long)(i + 2) * lda + kk;
                    const float* a3 = A + (long)(i + 3) * lda + kk;
                    int j = jj;
                    for (; j + 1 < je; j += 2) {
                        const float* b0 = B + (long)j       * ldb + kk;
                        const float* b1 = B + (long)(j + 1) * ldb + kk;
                        __m256 acc00 = _mm256_setzero_ps();
                        __m256 acc10 = _mm256_setzero_ps();
                        __m256 acc20 = _mm256_setzero_ps();
                        __m256 acc30 = _mm256_setzero_ps();
                        __m256 acc01 = _mm256_setzero_ps();
                        __m256 acc11 = _mm256_setzero_ps();
                        __m256 acc21 = _mm256_setzero_ps();
                        __m256 acc31 = _mm256_setzero_ps();
                        int p = 0;
                        for (; p + 7 < klen; p += 8) {
                            if (p + PFETCH < klen) {
                                _mm_prefetch((const char*)(a0 + p + PFETCH), _MM_HINT_T0);
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
                        float s00 = hsum256_opt(acc00), s01 = hsum256_opt(acc01);
                        float s10 = hsum256_opt(acc10), s11 = hsum256_opt(acc11);
                        float s20 = hsum256_opt(acc20), s21 = hsum256_opt(acc21);
                        float s30 = hsum256_opt(acc30), s31 = hsum256_opt(acc31);
                        for (; p < klen; ++p) {
                            float b0p = b0[p], b1p = b1[p];
                            s00 += a0[p] * b0p; s01 += a0[p] * b1p;
                            s10 += a1[p] * b0p; s11 += a1[p] * b1p;
                            s20 += a2[p] * b0p; s21 += a2[p] * b1p;
                            s30 += a3[p] * b0p; s31 += a3[p] * b1p;
                        }
                        C[(long)i       * ldc + j]     += s00;
                        C[(long)(i + 1) * ldc + j]     += s10;
                        C[(long)(i + 2) * ldc + j]     += s20;
                        C[(long)(i + 3) * ldc + j]     += s30;
                        C[(long)i       * ldc + j + 1] += s01;
                        C[(long)(i + 1) * ldc + j + 1] += s11;
                        C[(long)(i + 2) * ldc + j + 1] += s21;
                        C[(long)(i + 3) * ldc + j + 1] += s31;
                    }
                    for (; j < je; ++j) {
                        const float* b0 = B + (long)j * ldb + kk;
                        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
                        int p = 0;
                        for (; p + 7 < klen; p += 8) {
                            __m256 vb = _mm256_loadu_ps(b0 + p);
                            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), vb, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), vb, acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + p), vb, acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + p), vb, acc3);
                        }
                        float s0 = hsum256_opt(acc0), s1 = hsum256_opt(acc1);
                        float s2 = hsum256_opt(acc2), s3 = hsum256_opt(acc3);
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

                // --- 2-row tail ----------------------------------------------
                for (; i + 1 < ie; i += 2) {
                    const float* a0 = A + (long)i       * lda + kk;
                    const float* a1 = A + (long)(i + 1) * lda + kk;
                    int j = jj;
                    for (; j + 1 < je; j += 2) {
                        const float* b0 = B + (long)j       * ldb + kk;
                        const float* b1 = B + (long)(j + 1) * ldb + kk;
                        __m256 acc00 = _mm256_setzero_ps(), acc10 = _mm256_setzero_ps();
                        __m256 acc01 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
                        int p = 0;
                        for (; p + 7 < klen; p += 8) {
                            __m256 va0 = _mm256_loadu_ps(a0 + p);
                            __m256 va1 = _mm256_loadu_ps(a1 + p);
                            __m256 vb0 = _mm256_loadu_ps(b0 + p);
                            __m256 vb1 = _mm256_loadu_ps(b1 + p);
                            acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                            acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                            acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                            acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                        }
                        float s00 = hsum256_opt(acc00), s01 = hsum256_opt(acc01);
                        float s10 = hsum256_opt(acc10), s11 = hsum256_opt(acc11);
                        for (; p < klen; ++p) {
                            float b0p = b0[p], b1p = b1[p];
                            s00 += a0[p] * b0p; s01 += a0[p] * b1p;
                            s10 += a1[p] * b0p; s11 += a1[p] * b1p;
                        }
                        C[(long)i       * ldc + j]     += s00;
                        C[(long)(i + 1) * ldc + j]     += s10;
                        C[(long)i       * ldc + j + 1] += s01;
                        C[(long)(i + 1) * ldc + j + 1] += s11;
                    }
                    for (; j < je; ++j) {
                        const float* b0 = B + (long)j * ldb + kk;
                        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                        int p = 0;
                        for (; p + 7 < klen; p += 8) {
                            __m256 vb = _mm256_loadu_ps(b0 + p);
                            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), vb, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), vb, acc1);
                        }
                        float s0 = hsum256_opt(acc0), s1 = hsum256_opt(acc1);
                        for (; p < klen; ++p) {
                            float bp = b0[p];
                            s0 += a0[p] * bp; s1 += a1[p] * bp;
                        }
                        C[(long)i       * ldc + j] += s0;
                        C[(long)(i + 1) * ldc + j] += s1;
                    }
                }

                // --- 1-row tail (handles M % 6 == 1 or 3 or 5) --------------
                for (; i < ie; ++i) {
                    const float* a = A + (long)i * lda + kk;
                    for (int j = jj; j < je; ++j) {
                        const float* b = B + (long)j * ldb + kk;
                        __m256 acc = _mm256_setzero_ps();
                        int p = 0;
                        for (; p + 7 < klen; p += 8)
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p),
                                                  _mm256_loadu_ps(b + p), acc);
                        float s = hsum256_opt(acc);
                        for (; p < klen; ++p) s += a[p] * b[p];
                        C[(long)i * ldc + j] += s;
                    }
                }

            } // jj
        } // ii
    } // kk
}
