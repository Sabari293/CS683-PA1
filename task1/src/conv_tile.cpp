// conv_tile.cpp
#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int TILE_H = 8;
    const int TILE_W = 2048;

    for (int by = 0; by < H; by += TILE_H) {
        int ey = std::min(by + TILE_H, H);
        for (int bx = 0; bx < W; bx += TILE_W) {
            int ex = std::min(bx + TILE_W, W);

            for (int oy = by; oy < ey; ++oy) {
                for (int ox = bx; ox < ex; ++ox) {
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
