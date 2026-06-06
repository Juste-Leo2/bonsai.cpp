#pragma once

#include "ggml.h"

namespace bonsai {

struct Conv2d {
    ggml_tensor * w = nullptr;  // shape: [kW, kH, in, out]
    ggml_tensor * b = nullptr;  // shape: [out]
};

struct ResNet {
    Conv2d conv1;
    ggml_tensor * norm1_w = nullptr;
    ggml_tensor * norm1_b = nullptr;
    Conv2d conv2;
    ggml_tensor * norm2_w = nullptr;
    ggml_tensor * norm2_b = nullptr;
    Conv2d conv_shortcut;  // empty if not needed
};

struct Attention {
    ggml_tensor * group_norm_w = nullptr;
    ggml_tensor * group_norm_b = nullptr;
    ggml_tensor * to_q_w = nullptr;  // [in, out]
    ggml_tensor * to_q_b = nullptr;
    ggml_tensor * to_k_w = nullptr;
    ggml_tensor * to_k_b = nullptr;
    ggml_tensor * to_v_w = nullptr;
    ggml_tensor * to_v_b = nullptr;
    ggml_tensor * to_out_w = nullptr;
    ggml_tensor * to_out_b = nullptr;
    int n_heads = 8;
    int head_dim = 64;
};

struct UpBlock {
    std::vector<ResNet> resnets;        // 3 resnets
    Conv2d upsampler_conv;              // empty if no upsampler
    bool has_upsampler = false;
};

struct MidBlock {
    std::vector<ResNet> resnets;        // 2 resnets
    Attention attn;
};

struct VAEWeights {
    // post_quant_conv
    Conv2d post_quant_conv;

    // conv_in
    Conv2d conv_in;

    // mid block
    MidBlock mid_block;

    // 4 up blocks
    std::vector<UpBlock> up_blocks;     // size 4

    // conv_out
    ggml_tensor * conv_norm_out_w = nullptr;
    ggml_tensor * conv_norm_out_b = nullptr;
    Conv2d conv_out;
};

// Per-tensor weight loader result
struct WeightsView {
    // a 1D linear weight that must be stored as [in, out] in ggml
    // (PyTorch is [out, in])
    struct Linear {
        ggml_tensor * dst;  // pre-allocated in weights ctx
        const float * src;
        int64_t in_dim;
        int64_t out_dim;
    };
    std::vector<Linear> linears;

    // a 4D conv weight that must be stored as [kW, kH, in, out]
    // (PyTorch is [out, in, kH, kW])
    struct Conv {
        ggml_tensor * dst;
        const float * src;
        int64_t out_c, in_c, kh, kw;
    };
    std::vector<Conv> convs;

    // 1D weight or bias (no permutation)
    struct Vec {
        ggml_tensor * dst;
        const float * src;
        int64_t n;
    };
    std::vector<Vec> vecs;
};

}  // namespace bonsai
