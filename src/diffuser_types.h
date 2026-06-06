#pragma once

#include "ggml.h"

#include <string>
#include <vector>

namespace bonsai {

static constexpr int B1_0_BLOCK_SIZE = 32;
static constexpr int B1_0_BLOCK_BYTES = 6;

struct DiffuserParams {
    int in_channels = 128;
    int context_in_dim = 7680;
    int hidden_size = 3072;
    int num_heads = 24;
    int head_dim = 128;
    int depth = 5;
    int depth_single_blocks = 20;
    int axes_dim[4] = {32, 32, 32, 32};
    int theta = 10000;
    float mlp_ratio = 3.0f;
    bool use_guidance_embed = false;
    int mlp_hidden_dim = 0;

    DiffuserParams() {
        mlp_hidden_dim = int(hidden_size * mlp_ratio);
    }
};

struct B1Weights {
    ggml_tensor * data = nullptr;
    int in_dim = 0;
    int out_dim = 0;
};

struct FPWeights {
    ggml_tensor * data = nullptr;
};

struct DiffuserWeights {
    B1Weights img_in;
    FPWeights time_in_w1;
    FPWeights time_in_b1;
    FPWeights time_in_w2;
    FPWeights time_in_b2;
    B1Weights txt_in;
    B1Weights double_mod_img;
    B1Weights double_mod_txt;
    B1Weights single_mod;

    struct DoubleBlock {
        B1Weights attn_to_q;
        B1Weights attn_to_k;
        B1Weights attn_to_v;
        B1Weights attn_to_out;
        FPWeights attn_norm_q;
        FPWeights attn_norm_k;
        B1Weights attn_add_q;
        B1Weights attn_add_k;
        B1Weights attn_add_v;
        B1Weights attn_add_out;
        FPWeights attn_norm_added_q;
        FPWeights attn_norm_added_k;
        B1Weights ff_linear_in;
        B1Weights ff_linear_out;
        B1Weights ff_ctx_linear_in;
        B1Weights ff_ctx_linear_out;
    };
    std::vector<DoubleBlock> double_blocks;

    struct SingleBlock {
        B1Weights to_qkv_mlp_proj;
        B1Weights to_out;
        FPWeights norm_q;
        FPWeights norm_k;
    };
    std::vector<SingleBlock> single_blocks;

    B1Weights norm_out_linear;
    B1Weights proj_out;
};

} // namespace bonsai
