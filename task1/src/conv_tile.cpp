// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    for(int oy=0;oy<H;oy++){
        for(int ox=0;ox<W;ox++){
            out[oy*W+ox]=0.0f;
        }
    }
    for (int ky = 0; ky < K; ++ky) { 
        for (int kx = 0; kx < K; ++kx) {
            float ker_val=ker[ky*K+kx];
            for (int oy = 0; oy < H; ++oy) {           
                for (int ox = 0; ox < W; ++ox) {
                    out[oy*W+ox] += in[(oy + ky) * in_stride + (ox + kx)] * ker_val;
                }
            }
        }
    }
}
