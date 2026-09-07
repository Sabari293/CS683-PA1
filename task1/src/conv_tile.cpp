// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    int T_H=4;
    int T_W=1024;
    for(int oys = 0;oys<H ;oys+=T_H){
        for(int oxs = 0;oxs<W;oxs+=T_W){
            int oye = oys+T_H>H?H:oys+T_H;
            int oxe = oxs+T_W>W?W:oxs+T_W;
            for (int oy = oys; oy < oye; ++oy) {    
                for (int ox = oxs; ox < oxe; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
    
}