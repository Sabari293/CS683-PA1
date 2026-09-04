// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
//
// Strategy: register-tiled micro-kernel.
//   - Process 4 rows of A and 2 columns of B simultaneously (4×2 register tile).
//   - Each C[i][j] accumulator is one __m256 holding a partial dot-product over 8
//     consecutive K-elements at a time.  A single pass over K fills all 8 accumulators.
//   - Horizontal reduction at the end collapses each __m256 accumulator to a scalar.
//   - Scalar cleanup loops handle the leftover K elements (K % 8 != 0) and the leftover
//     rows / columns when M % 4 != 0 or N % 2 != 0.
//
// Compile flags (pinned): -O2 -fno-tree-vectorize -mavx2 -mfma

#include <immintrin.h>
#include "matmul.h"

// Horizontal sum of an __m256 register -> scalar float.
static inline float hsum256(__m256 v) {
    // Add upper 128 bits to lower 128 bits.
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    // Fold 4 -> 2 -> 1.
    __m128 shuf = _mm_movehdup_ps(sum);          // [1,1,3,3]
    __m128 sums = _mm_add_ps(sum, shuf);         // [0+1, _, 2+3, _]
    shuf        = _mm_movehl_ps(shuf, sums);     // [2+3, ...]
    sums        = _mm_add_ss(sums, shuf);        // [0+1+2+3, ...]
    return _mm_cvtss_f32(sums);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // --- 4 x 2 register-tiled micro-kernel -----------------------------------
    // Process blocks of 4 A-rows and 2 B-rows together.
    int i = 0;
    for (; i + 3 < M; i += 4) {
        const float* a0 = A + (long)i       * lda;
        const float* a1 = A + (long)(i + 1) * lda;
        const float* a2 = A + (long)(i + 2) * lda;
        const float* a3 = A + (long)(i + 3) * lda;

        int j = 0;
        // 2-column tile
        for (; j + 1 < N; j += 2) {
            const float* b0 = B + (long)j       * ldb;
            const float* b1 = B + (long)(j + 1) * ldb;

            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc21 = _mm256_setzero_ps();
            __m256 acc31 = _mm256_setzero_ps();

            int p = 0;
            // Main K loop: 8 floats per iteration (AVX2 register width).
            for (; p + 7 < K; p += 8) {
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
            float s00 = hsum256(acc00);
            float s10 = hsum256(acc10);
            float s20 = hsum256(acc20);
            float s30 = hsum256(acc30);
            float s01 = hsum256(acc01);
            float s11 = hsum256(acc11);
            float s21 = hsum256(acc21);
            float s31 = hsum256(acc31);

            // Scalar tail for K % 8.
            for (; p < K; ++p) {
                float b0p = b0[p];
                float b1p = b1[p];
                s00 += a0[p] * b0p; s01 += a0[p] * b1p;
                s10 += a1[p] * b0p; s11 += a1[p] * b1p;
                s20 += a2[p] * b0p; s21 += a2[p] * b1p;
                s30 += a3[p] * b0p; s31 += a3[p] * b1p;
            }

            C[(long)i       * ldc + j]     = s00;
            C[(long)(i + 1) * ldc + j]     = s10;
            C[(long)(i + 2) * ldc + j]     = s20;
            C[(long)(i + 3) * ldc + j]     = s30;
            C[(long)i       * ldc + j + 1] = s01;
            C[(long)(i + 1) * ldc + j + 1] = s11;
            C[(long)(i + 2) * ldc + j + 1] = s21;
            C[(long)(i + 3) * ldc + j + 1] = s31;
        }
        // Scalar cleanup for odd N.
        for (; j < N; ++j) {
            const float* b0 = B + (long)j * ldb;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int p = 0;
            for (; p + 7 < K; p += 8) {
                __m256 vb = _mm256_loadu_ps(b0 + p);
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), vb, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), vb, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + p), vb, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + p), vb, acc3);
            }
            float s0 = hsum256(acc0), s1 = hsum256(acc1);
            float s2 = hsum256(acc2), s3 = hsum256(acc3);
            for (; p < K; ++p) {
                float bp = b0[p];
                s0 += a0[p] * bp; s1 += a1[p] * bp;
                s2 += a2[p] * bp; s3 += a3[p] * bp;
            }
            C[(long)i       * ldc + j] = s0;
            C[(long)(i + 1) * ldc + j] = s1;
            C[(long)(i + 2) * ldc + j] = s2;
            C[(long)(i + 3) * ldc + j] = s3;
        }
    }

    // --- Scalar tail for leftover rows (M % 4) --------------------------------
    for (; i < M; ++i) {
        const float* a = A + (long)i * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + (long)j * ldb;
            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + 7 < K; p += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p),
                                      _mm256_loadu_ps(b + p), acc);
            }
            float s = hsum256(acc);
            for (; p < K; ++p) s += a[p] * b[p];
            C[(long)i * ldc + j] = s;
        }
    }
}
