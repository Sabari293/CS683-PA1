// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    int T = 128;
    int Tk = 128;
    int prefetch_dist = 32;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++)C[static_cast<long>(i) * ldc + j]=0.0f;
    }
    for(int sx=0;sx<M;sx+=T){
        for(int sy=0;sy<N;sy+=T){
            for(int k1=0;k1<K;k1+=Tk){
                int sxe=(sx+T<M?sx+T:M);
                int sye=(sy+T<N?sy+T:N);
                int ske=(k1+Tk<K?k1+Tk:K);
                for (int i = sx; i < sxe; ++i) {
                    const float* a = A + static_cast<long>(i) * lda;
                    for (int j = sy; j < sye; ++j) {
                        __m256 acc = _mm256_setzero_ps();
                        const float* b = B + static_cast<long>(j) * ldb;
                        _mm_prefetch(b+ldb,_MM_HINT_T0);
                        int p = k1;
                        for (; p < k1+8*((ske-k1)/8); p+=8) {
                            __m256 a1 = _mm256_load_ps(&a[p]); 
                            __m256 b1 = _mm256_load_ps(&b[p]); 
                            _mm_prefetch(reinterpret_cast<const char*>(&a[p+prefetch_dist]),_MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&b[p+prefetch_dist]),_MM_HINT_T0);
                            acc = _mm256_fmadd_ps(a1,b1,acc);
                        }
                        float tmp[8];
                        _mm256_storeu_ps(tmp,acc);
                        float s = 0.0f;
                        s+=(tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
                        for(;p<ske;p++){
                            s += (a[p]*b[p]);
                        }
                        C[static_cast<long>(i) * ldc + j]+=s;
                    }
                }
            }
        }
    }
}
