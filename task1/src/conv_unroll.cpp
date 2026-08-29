// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < 4*(H/4); ++oy) {
        for (int ox = 0; ox < 4*(W/4); ox+=4) {
            out[oy * W + ox] = 0.0f;
            out[oy*W+ox+1]=0.0f;
            out[oy*W+ox+2]=0.0f;
            out[oy*W+ox+3]=0.0f;
        }
        for(int ox=4*(W/4);ox<W;++ox){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox < 4*(W/4); ox+=4) {
                float acc = 0.0f;
                float acc1=0.0f;
                float acc2=0.0f;
                float acc3=0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + (ox +1+ kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox +2+ kx)] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + (ox +3+ kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
                
                out[oy * W + ox+1] += acc1;
                out[oy * W + ox+2] += acc2;
                out[oy * W + ox+3] += acc3;
            }
            for (int ox = 4*(W/4); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
        oy++;
        for (int ox = 0; ox < 4*(W/4); ox+=4) {
            out[oy * W + ox] = 0.0f;
            out[oy*W+ox+1]=0.0f;
            out[oy*W+ox+2]=0.0f;
            out[oy*W+ox+3]=0.0f;
        }
        for(int ox=4*(W/4);ox<W;++ox){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox < 4*(W/4); ox+=4) {
                float acc = 0.0f;
                float acc1=0.0f;
                float acc2=0.0f;
                float acc3=0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + (ox +1+ kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox +2+ kx)] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + (ox +3+ kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
                
                out[oy * W + ox+1] += acc1;
                out[oy * W + ox+2] += acc2;
                out[oy * W + ox+3] += acc3;
            }
            for (int ox = 4*(W/4); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
        oy++;
        for (int ox = 0; ox < 4*(W/4); ox+=4) {
            out[oy * W + ox] = 0.0f;
            out[oy*W+ox+1]=0.0f;
            out[oy*W+ox+2]=0.0f;
            out[oy*W+ox+3]=0.0f;
        }
        for(int ox=4*(W/4);ox<W;++ox){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox < 4*(W/4); ox+=4) {
                float acc = 0.0f;
                float acc1=0.0f;
                float acc2=0.0f;
                float acc3=0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + (ox +1+ kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox +2+ kx)] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + (ox +3+ kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
                
                out[oy * W + ox+1] += acc1;
                out[oy * W + ox+2] += acc2;
                out[oy * W + ox+3] += acc3;
            }
            for (int ox = 4*(W/4); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
        oy++;
        for (int ox = 0; ox < 4*(W/4); ox+=4) {
            out[oy * W + ox] = 0.0f;
            out[oy*W+ox+1]=0.0f;
            out[oy*W+ox+2]=0.0f;
            out[oy*W+ox+3]=0.0f;
        }
        for(int ox=4*(W/4);ox<W;++ox){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox < 4*(W/4); ox+=4) {
                float acc = 0.0f;
                float acc1=0.0f;
                float acc2=0.0f;
                float acc3=0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + (ox +1+ kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox +2+ kx)] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + (ox +3+ kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
                
                out[oy * W + ox+1] += acc1;
                out[oy * W + ox+2] += acc2;
                out[oy * W + ox+3] += acc3;
            }
            for (int ox = 4*(W/4); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
    }
    for (int oy = 4*(H/4); oy < H; ++oy) {
        for (int ox = 0; ox < 4*(W/4); ox+=4) {
            out[oy * W + ox] = 0.0f;
            out[oy*W+ox+1]=0.0f;
            out[oy*W+ox+2]=0.0f;
            out[oy*W+ox+3]=0.0f;
        }
        for(int ox=4*(W/4);ox<W;++ox){
            out[oy*W+ox]=0.0f;
        }
        for (int ky = 0; ky < K; ++ky) {    
            for (int ox = 0; ox < 4*(W/4); ox+=4) {
                float acc = 0.0f;
                float acc1=0.0f;
                float acc2=0.0f;
                float acc3=0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + (ox +1+ kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox +2+ kx)] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + (ox +3+ kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
                
                out[oy * W + ox+1] += acc1;
                out[oy * W + ox+2] += acc2;
                out[oy * W + ox+3] += acc3;
            }
            for (int ox = 4*(W/4); ox < W; ++ox) {
                float acc = 0.0f;
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
                out[oy * W + ox] += acc;
            }
        }
    }
}
// void conv_unroll1(const float* in, float* out, const float* ker,
//                  int H, int W, int K) {
//     // TODO(student): replace this placeholder with your unrolled implementation.
//     const int p = K / 2;
//     const int in_stride = W + 2 * p;  // padded row stride
//     for(int oy=0;oy<H;oy++){
//         for(int ox=0;ox<W;++ox){
//             out[oy*W+ox]=0.0f;
//         }
//     }
//     for (int ky = 0; ky < K; ++ky) { 
//         for (int kx = 0; kx < K; ++kx) {
//             float ker_val=ker[ky*K+kx];
//             for (int oy = 0; oy < 4*(H/4); ++oy) {           
//                 for (int ox = 0; ox < 4*(W/4); ++ox) {
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
                    
//                 }
//                 for(int ox=4*(W/4);ox<W;++ox){
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                 }
//                 oy++;
//                 for (int ox = 0; ox < 4*(W/4); ++ox) {
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
                    
//                 }
//                 for(int ox=4*(W/4);ox<W;++ox){
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                 }
//                 oy++;
//                 for (int ox = 0; ox < 4*(W/4); ++ox) {
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
                    
//                 }
//                 for(int ox=4*(W/4);ox<W;++ox){
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                 }
//                 oy++;
//                 for (int ox = 0; ox < 4*(W/4); ++ox) {
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
                    
//                 }
//                 for(int ox=4*(W/4);ox<W;++ox){
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                 }
//             }
//             for(int oy=4*(H/4);oy<H;oy++){
//                 for (int ox = 0; ox < 4*(W/4); ++ox) {
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                     ++ox;
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
                    
//                 }
//                 for(int ox=4*(W/4);ox<W;++ox){
//                     out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
//                 }
//             }
//         }
//     }
// }