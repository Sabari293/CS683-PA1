// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    for (int i = 0; i < 4 * (M / 4); i += 4) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < 4 * (N / 4); j += 4) {

            const float* b1 = B + static_cast<long>(j) * ldb;
            const float* b2 = B + static_cast<long>(j+1) * ldb;
            const float* b3 = B + static_cast<long>(j+2) * ldb;
            const float* b4 = B + static_cast<long>(j+3) * ldb;

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();

            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_loadu_ps(&a[p]); 
                __m256 bt = _mm256_loadu_ps(&b1[p]); 
                acc1 = _mm256_fmadd_ps(a1,bt,acc1);
                bt = _mm256_loadu_ps(&b2[p]); 
                acc2 = _mm256_fmadd_ps(a1,bt,acc2);
                bt = _mm256_loadu_ps(&b3[p]); 
                acc3 = _mm256_fmadd_ps(a1,bt,acc3);
                bt = _mm256_loadu_ps(&b4[p]); 
                acc4 = _mm256_fmadd_ps(a1,bt,acc4);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc1);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b1[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;

            _mm256_storeu_ps(tmp,acc2);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b2[p]);
            }
            C[static_cast<long>(i) * ldc + j + 1]=s;

            _mm256_storeu_ps(tmp,acc3);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b3[p]);
            }
            C[static_cast<long>(i) * ldc + j + 2]=s;

            _mm256_storeu_ps(tmp,acc4);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b4[p]);
            }
            C[static_cast<long>(i) * ldc + j + 3]=s;
        }

        for (int j = 4 * (N / 4); j < N; j += 1) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_load_ps(&a[p]); 
                __m256 b1 = _mm256_load_ps(&b[p]); 
                acc = _mm256_fmadd_ps(a1,b1,acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;
        }

        i++;
        a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < 4 * (N / 4); j += 4) {

            const float* b1 = B + static_cast<long>(j) * ldb;
            const float* b2 = B + static_cast<long>(j+1) * ldb;
            const float* b3 = B + static_cast<long>(j+2) * ldb;
            const float* b4 = B + static_cast<long>(j+3) * ldb;

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();

            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_loadu_ps(&a[p]); 
                __m256 bt = _mm256_loadu_ps(&b1[p]); 
                acc1 = _mm256_fmadd_ps(a1,bt,acc1);
                bt = _mm256_loadu_ps(&b2[p]); 
                acc2 = _mm256_fmadd_ps(a1,bt,acc2);
                bt = _mm256_loadu_ps(&b3[p]); 
                acc3 = _mm256_fmadd_ps(a1,bt,acc3);
                bt = _mm256_loadu_ps(&b4[p]); 
                acc4 = _mm256_fmadd_ps(a1,bt,acc4);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc1);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b1[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;

            _mm256_storeu_ps(tmp,acc2);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b2[p]);
            }
            C[static_cast<long>(i) * ldc + j + 1]=s;

            _mm256_storeu_ps(tmp,acc3);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b3[p]);
            }
            C[static_cast<long>(i) * ldc + j + 2]=s;

            _mm256_storeu_ps(tmp,acc4);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b4[p]);
            }
            C[static_cast<long>(i) * ldc + j + 3]=s;
        }

        for (int j = 4 * (N / 4); j < N; j += 1) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_load_ps(&a[p]); 
                __m256 b1 = _mm256_load_ps(&b[p]); 
                acc = _mm256_fmadd_ps(a1,b1,acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;
        }

        i++;
        a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < 4 * (N / 4); j += 4) {

            const float* b1 = B + static_cast<long>(j) * ldb;
            const float* b2 = B + static_cast<long>(j+1) * ldb;
            const float* b3 = B + static_cast<long>(j+2) * ldb;
            const float* b4 = B + static_cast<long>(j+3) * ldb;

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();

            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_loadu_ps(&a[p]); 
                __m256 bt = _mm256_loadu_ps(&b1[p]); 
                acc1 = _mm256_fmadd_ps(a1,bt,acc1);
                bt = _mm256_loadu_ps(&b2[p]); 
                acc2 = _mm256_fmadd_ps(a1,bt,acc2);
                bt = _mm256_loadu_ps(&b3[p]); 
                acc3 = _mm256_fmadd_ps(a1,bt,acc3);
                bt = _mm256_loadu_ps(&b4[p]); 
                acc4 = _mm256_fmadd_ps(a1,bt,acc4);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc1);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b1[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;

            _mm256_storeu_ps(tmp,acc2);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b2[p]);
            }
            C[static_cast<long>(i) * ldc + j + 1]=s;

            _mm256_storeu_ps(tmp,acc3);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b3[p]);
            }
            C[static_cast<long>(i) * ldc + j + 2]=s;

            _mm256_storeu_ps(tmp,acc4);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b4[p]);
            }
            C[static_cast<long>(i) * ldc + j + 3]=s;
        }

        for (int j = 4 * (N / 4); j < N; j += 1) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_load_ps(&a[p]); 
                __m256 b1 = _mm256_load_ps(&b[p]); 
                acc = _mm256_fmadd_ps(a1,b1,acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;
        }

        i++;
        a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < 4 * (N / 4); j += 4) {

            const float* b1 = B + static_cast<long>(j) * ldb;
            const float* b2 = B + static_cast<long>(j+1) * ldb;
            const float* b3 = B + static_cast<long>(j+2) * ldb;
            const float* b4 = B + static_cast<long>(j+3) * ldb;

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();

            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_loadu_ps(&a[p]); 
                __m256 bt = _mm256_loadu_ps(&b1[p]); 
                acc1 = _mm256_fmadd_ps(a1,bt,acc1);
                bt = _mm256_loadu_ps(&b2[p]); 
                acc2 = _mm256_fmadd_ps(a1,bt,acc2);
                bt = _mm256_loadu_ps(&b3[p]); 
                acc3 = _mm256_fmadd_ps(a1,bt,acc3);
                bt = _mm256_loadu_ps(&b4[p]); 
                acc4 = _mm256_fmadd_ps(a1,bt,acc4);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc1);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b1[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;

            _mm256_storeu_ps(tmp,acc2);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b2[p]);
            }
            C[static_cast<long>(i) * ldc + j + 1]=s;

            _mm256_storeu_ps(tmp,acc3);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b3[p]);
            }
            C[static_cast<long>(i) * ldc + j + 2]=s;

            _mm256_storeu_ps(tmp,acc4);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b4[p]);
            }
            C[static_cast<long>(i) * ldc + j + 3]=s;
        }

        for (int j = 4 * (N / 4); j < N; j += 1) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_load_ps(&a[p]); 
                __m256 b1 = _mm256_load_ps(&b[p]); 
                acc = _mm256_fmadd_ps(a1,b1,acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;
        }
    }

    for (int i = 4 * (M / 4); i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < 4 * (N / 4); j += 4) {

            const float* b1 = B + static_cast<long>(j) * ldb;
            const float* b2 = B + static_cast<long>(j+1) * ldb;
            const float* b3 = B + static_cast<long>(j+2) * ldb;
            const float* b4 = B + static_cast<long>(j+3) * ldb;

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();

            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_loadu_ps(&a[p]); 
                __m256 bt = _mm256_loadu_ps(&b1[p]); 
                acc1 = _mm256_fmadd_ps(a1,bt,acc1);
                bt = _mm256_loadu_ps(&b2[p]); 
                acc2 = _mm256_fmadd_ps(a1,bt,acc2);
                bt = _mm256_loadu_ps(&b3[p]); 
                acc3 = _mm256_fmadd_ps(a1,bt,acc3);
                bt = _mm256_loadu_ps(&b4[p]); 
                acc4 = _mm256_fmadd_ps(a1,bt,acc4);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc1);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b1[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;

            _mm256_storeu_ps(tmp,acc2);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b2[p]);
            }
            C[static_cast<long>(i) * ldc + j + 1]=s;

            _mm256_storeu_ps(tmp,acc3);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b3[p]);
            }
            C[static_cast<long>(i) * ldc + j + 2]=s;

            _mm256_storeu_ps(tmp,acc4);
            s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b4[p]);
            }
            C[static_cast<long>(i) * ldc + j + 3]=s;
        }

        for (int j = 4 * (N / 4); j < N; j += 1) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < 8*(K/8); p+=8) {
                __m256 a1 = _mm256_load_ps(&a[p]); 
                __m256 b1 = _mm256_load_ps(&b[p]); 
                acc = _mm256_fmadd_ps(a1,b1,acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp,acc);
            float s = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]);
            for(int p = 8*(K/8);p<K;p++){
                s += (a[p]*b[p]);
            }
            C[static_cast<long>(i) * ldc + j]=s;
        }
    }
}