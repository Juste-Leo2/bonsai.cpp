// Test the conv weight permute independently
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include "safetensors.h"

int main(int argc, char** argv) {
    std::string model_path = "/home/leo/bonsai.cpp/models/flux2-vae.safetensors";
    std::string weight_name = "decoder.conv_in.weight";
    if (argc > 1) weight_name = argv[1];

    bonsai::SafetensorsFile st;
    st.open(model_path);

    const bonsai::SafeTensor * t = st.find(weight_name);
    if (!t) { std::cerr << "not found: " << weight_name << "\n"; return 1; }

    int64_t Co = t->shape[0];  // 512
    int64_t Ci = t->shape[1];  // 32
    int64_t kH = t->shape[2];  // 3
    int64_t kW = t->shape[3];  // 3
    std::cout << "weight shape: " << Co << "x" << Ci << "x" << kH << "x" << kW << "\n";

    // Allocate dst in ggml (kW, kH, Ci, Co) row-major
    std::vector<float> dst(Co * Ci * kH * kW, 0.0f);
    const float * src = reinterpret_cast<const float *>(t->data);

    // Manual permute: dst[kw, kh, ci, co] = src[co, ci, kh, kw]
    int64_t ne[4] = {kW, kH, Ci, Co};
    int64_t strides[4] = {ne[0], ne[0]*ne[1], ne[0]*ne[1]*ne[2], 1};
    for (int64_t co = 0; co < Co; ++co)
      for (int64_t ci = 0; ci < Ci; ++ci)
        for (int64_t kh = 0; kh < kH; ++kh)
          for (int64_t kw = 0; kw < kW; ++kw) {
            int64_t s = ((co * Ci + ci) * kH + kh) * kW + kw;
            int64_t d = ((kw * kH + kh) * Ci + ci) * Co + co;
            dst[d] = src[s];
          }

    // Print key positions
    auto print_pos = [&](const std::string& label, int64_t kw, int64_t kh, int64_t ci, int64_t co) {
        int64_t d = ((kw * kH + kh) * Ci + ci) * Co + co;
        std::cout << "  " << label << " dst[" << kw << "," << kh << "," << ci << "," << co
                  << "] mem[" << d << "] = " << dst[d] << "\n";
    };
    // PyTorch expectations
    // src[0,0,0,0] = -0.0086 → dst[0,0,0,0]
    // src[0,0,0,1] = 0.0070  → dst[1,0,0,0]
    // src[1,0,0,0] = -0.0016 → dst[0,0,0,1]
    // src[0,0,1,0] = -0.0091 → dst[0,1,0,0]
    // src[0,1,0,0] = 0.0220  → dst[0,0,1,0]
    // src[511,31,2,2] = 0.0107 → dst[2,2,31,511]
    std::cout << "Expected: -0.0086, 0.0070, -0.0016, -0.0091, 0.0220, 0.0107\n";
    print_pos("mem[0]    ", 0, 0, 0, 0);
    print_pos("mem[1]    ", 1, 0, 0, 0);
    print_pos("mem[12]   ", 0, 1, 0, 0);
    print_pos("mem[36]   ", 0, 0, 1, 0);
    print_pos("mem[2048] ", 0, 0, 0, 1);
    print_pos("mem[147455]", 2, 2, 31, 511);

    // Also verify by computing the expected dst value
    auto src_at = [&](int64_t co, int64_t ci, int64_t kh, int64_t kw) {
        int64_t s = ((co * Ci + ci) * kH + kh) * kW + kw;
        return src[s];
    };
    std::cout << "\nVerifying dst = src at permuted positions:\n";
    std::cout << "  dst[0,0,0,0] should = src[0,0,0,0] = " << src_at(0,0,0,0) << "\n";
    std::cout << "  dst[1,0,0,0] should = src[0,0,0,1] = " << src_at(0,0,0,1) << "\n";
    std::cout << "  dst[0,0,0,1] should = src[1,0,0,0] = " << src_at(1,0,0,0) << "\n";
    std::cout << "  dst[0,1,0,0] should = src[0,0,1,0] = " << src_at(0,0,1,0) << "\n";
    std::cout << "  dst[0,0,1,0] should = src[0,1,0,0] = " << src_at(0,1,0,0) << "\n";
    std::cout << "  dst[2,2,31,511] should = src[511,31,2,2] = " << src_at(511,31,2,2) << "\n";

    return 0;
}
