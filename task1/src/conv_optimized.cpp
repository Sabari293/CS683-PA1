// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER  (GRADED)
//
// All optimisation techniques combined and tuned:
//
//  1. LOOP REORDERING   – inner loops are (ky, kx, ox) so each ker[ky][kx] is broadcast
//                         once across the full output row, maximising register reuse.
//
//  2. REGISTER BLOCKING – Process RB=8 output rows simultaneously. A single kernel
//                         element ker[ky][kx] is broadcast once and FMA'd into all 8
//                         row accumulators — cutting the broadcast cost by 8×.
//
//  3. SIMD (AVX2 FMA)   – 8 output columns computed per _mm256_fmadd_ps call.
//                         _mm256_loadu_ps handles any alignment.
//
//  4. LOOP UNROLLING    – kx unrolled 4× to keep FMA units busy and reduce loop overhead.
//                         For K=3, the full kernel (9 taps) runs in the kx-remainder path
//                         since 4-wide loop takes 0 iterations; the compiler unrolls 3-
//                         iteration loops under -O2.
//
//  5. TAIL HANDLING     – 4-row and 1-row sub-kernels handle H % 8 leftovers.
//
// Compile flags (pinned): -O2 -fno-tree-vectorize -mavx2 -mfma

#include <immintrin.h>
#include "convolution.h"

static constexpr int RB = 8;  // register-blocking factor (rows)

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // ---- 8-row register-blocked SIMD micro-kernel --------------------------
    int oy = 0;
    for (; oy + RB - 1 < H; oy += RB) {
        for (int ox = 0; ox < W; ox += 8) {
            // Software prefetch ahead (e.g. 16 floats = 64 bytes = 1 cache line)
            // We prefetch the current 8 rows that we are going to process next.
            _mm_prefetch((const char*)(in + (long)(oy + 0) * in_stride + ox + 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(in + (long)(oy + 1) * in_stride + ox + 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(in + (long)(oy + 2) * in_stride + ox + 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(in + (long)(oy + 3) * in_stride + ox + 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(in + (long)(oy + 4) * in_stride + ox + 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(in + (long)(oy + 5) * in_stride + ox + 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(in + (long)(oy + 6) * in_stride + ox + 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(in + (long)(oy + 7) * in_stride + ox + 32), _MM_HINT_T0);

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                // Base input pointers for all 8 output rows at this ky.
                const float* ip0 = in + (long)(oy + 0 + ky) * in_stride + ox;
                const float* ip1 = in + (long)(oy + 1 + ky) * in_stride + ox;
                const float* ip2 = in + (long)(oy + 2 + ky) * in_stride + ox;
                const float* ip3 = in + (long)(oy + 3 + ky) * in_stride + ox;
                const float* ip4 = in + (long)(oy + 4 + ky) * in_stride + ox;
                const float* ip5 = in + (long)(oy + 5 + ky) * in_stride + ox;
                const float* ip6 = in + (long)(oy + 6 + ky) * in_stride + ox;
                const float* ip7 = in + (long)(oy + 7 + ky) * in_stride + ox;

                const float* ker_row = ker + ky * K;
                int kx = 0;
                // 4-wide kx unroll.
                for (; kx + 3 < K; kx += 4) {
                    for (int dk = 0; dk < 4; ++dk) {
                        __m256 vk = _mm256_set1_ps(ker_row[kx + dk]);
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ip0 + kx + dk), vk, acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ip1 + kx + dk), vk, acc1);
                        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(ip2 + kx + dk), vk, acc2);
                        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(ip3 + kx + dk), vk, acc3);
                        acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(ip4 + kx + dk), vk, acc4);
                        acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(ip5 + kx + dk), vk, acc5);
                        acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(ip6 + kx + dk), vk, acc6);
                        acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(ip7 + kx + dk), vk, acc7);
                    }
                }
                // Scalar-tail kx (handles K=3,5 fully, and remainder for K>5).
                for (; kx < K; ++kx) {
                    __m256 vk = _mm256_set1_ps(ker_row[kx]);
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ip0 + kx), vk, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ip1 + kx), vk, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(ip2 + kx), vk, acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(ip3 + kx), vk, acc3);
                    acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(ip4 + kx), vk, acc4);
                    acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(ip5 + kx), vk, acc5);
                    acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(ip6 + kx), vk, acc6);
                    acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(ip7 + kx), vk, acc7);
                }
            }

            _mm256_storeu_ps(out + (long)(oy + 0) * W + ox, acc0);
            _mm256_storeu_ps(out + (long)(oy + 1) * W + ox, acc1);
            _mm256_storeu_ps(out + (long)(oy + 2) * W + ox, acc2);
            _mm256_storeu_ps(out + (long)(oy + 3) * W + ox, acc3);
            _mm256_storeu_ps(out + (long)(oy + 4) * W + ox, acc4);
            _mm256_storeu_ps(out + (long)(oy + 5) * W + ox, acc5);
            _mm256_storeu_ps(out + (long)(oy + 6) * W + ox, acc6);
            _mm256_storeu_ps(out + (long)(oy + 7) * W + ox, acc7);
        }
    }

    // ---- 4-row tail --------------------------------------------------------
    for (; oy + 3 < H; oy += 4) {
        for (int ox = 0; ox < W; ox += 8) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* ip0 = in + (long)(oy + 0 + ky) * in_stride + ox;
                const float* ip1 = in + (long)(oy + 1 + ky) * in_stride + ox;
                const float* ip2 = in + (long)(oy + 2 + ky) * in_stride + ox;
                const float* ip3 = in + (long)(oy + 3 + ky) * in_stride + ox;
                const float* kr = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 vk = _mm256_set1_ps(kr[kx]);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ip0 + kx), vk, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(ip1 + kx), vk, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(ip2 + kx), vk, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(ip3 + kx), vk, a3);
                }
            }
            _mm256_storeu_ps(out + (long)(oy + 0) * W + ox, a0);
            _mm256_storeu_ps(out + (long)(oy + 1) * W + ox, a1);
            _mm256_storeu_ps(out + (long)(oy + 2) * W + ox, a2);
            _mm256_storeu_ps(out + (long)(oy + 3) * W + ox, a3);
        }
    }

    // ---- 2-row tail --------------------------------------------------------
    for (; oy + 1 < H; oy += 2) {
        for (int ox = 0; ox < W; ox += 8) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* ip0 = in + (long)(oy + 0 + ky) * in_stride + ox;
                const float* ip1 = in + (long)(oy + 1 + ky) * in_stride + ox;
                const float* kr = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 vk = _mm256_set1_ps(kr[kx]);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ip0 + kx), vk, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(ip1 + kx), vk, a1);
                }
            }
            _mm256_storeu_ps(out + (long)(oy + 0) * W + ox, a0);
            _mm256_storeu_ps(out + (long)(oy + 1) * W + ox, a1);
        }
    }

    // ---- 1-row tail --------------------------------------------------------
    for (; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 8) {
            __m256 a0 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* ip0 = in + (long)(oy + ky) * in_stride + ox;
                const float* kr = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ip0 + kx),
                                         _mm256_set1_ps(kr[kx]), a0);
                }
            }
            _mm256_storeu_ps(out + (long)oy * W + ox, a0);
        }
    }
}
