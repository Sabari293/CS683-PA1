import subprocess
import re

template = """// conv_tile.cpp
#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int TILE_H = __TH__;
    const int TILE_W = __TW__;

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
"""

sizes = [(32, 512), (64, 512), (32, 1024), (64, 1024), (16, 2048), (32, 2048), (64, 2048),(4,8),(8,4),(6,2),(16,8),(32,64),(16,32),(8,2048),(16,1024),(8,1024)]

for th, tw in sizes:
    code = template.replace("__TH__", str(th)).replace("__TW__", str(tw))
    with open(r"./src/conv_tile.cpp", "w") as f:
        f.write(code)
    
    subprocess.run(["make"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = subprocess.run([r"./bin/conv", "tile"], capture_output=True, text=True)
    
    match = re.search(r"tile\s+yes\s+[\d.]+\s+[\d.]+\s+([\d.]+)x", result.stdout)
    if match:
        print(f"TILE_H={th}, TILE_W={tw} -> Speedup: {match.group(1)}x")
    else:
        print(f"TILE_H={th}, TILE_W={tw} -> Failed or no match")
