#include "ggml.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vae_types.h"

namespace bonsai {

// Reshape a 1D per-channel vector to a (1, 1, C, 1) tensor with contiguous
// strides in the activation context. The 1D source lives in the weights
// context, so we cannot simply reshape — the resulting view has
// strides inherited from the 1D source that confuse downstream broadcast
// ops. ggml_cont forces a real copy with proper strides.
//
// This shape is for adding a per-channel bias to a 4D NCHW tensor (where the
// channel dim is ne[2]). For a 2D tensor (e.g. the output of ggml_mul_mat),
// use column_1d instead.
static ggml_tensor * channel_1d(ggml_context * ctx, ggml_tensor * b) {
    ggml_tensor * v = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    return ggml_cont(ctx, v);
}

// Reshape a 1D vector of size C to (C, 1, 1, 1) so it broadcasts along the
// row dim (ne[0]) of a 2D tensor (the typical mul_mat output layout).
static ggml_tensor * column_1d(ggml_context * ctx, ggml_tensor * b) {
    ggml_tensor * v = ggml_reshape_4d(ctx, b, b->ne[0], 1, 1, 1);
    return ggml_cont(ctx, v);
}

// ResNet forward:
//   h = x
//   x = group_norm(x) * norm_w + norm_b
//   x = silu(x)
//   x = conv2d_3x3(x, conv1.w) + conv1.b
//   x = group_norm(x) * norm2_w + norm2_b
//   x = silu(x)
//   x = conv2d_3x3(x, conv2.w) + conv2.b
//   if has shortcut: h = conv2d_1x1(h, conv_shortcut.w) + conv_shortcut.b
//   return x + h
static ggml_tensor * build_resnet(ggml_context * ctx, ggml_tensor * x, const ResNet & r) {
    ggml_tensor * h = x;

    // norm1
    ggml_tensor * n1 = ggml_group_norm(ctx, x, 32, 1e-6f);
    n1 = ggml_mul(ctx, n1, channel_1d(ctx, r.norm1_w));
    n1 = ggml_add(ctx, n1, channel_1d(ctx, r.norm1_b));
    n1 = ggml_silu(ctx, n1);

    // conv1: 3x3, padding 1
    ggml_tensor * c1 = ggml_conv_2d_direct(ctx, r.conv1.w, n1, 1, 1, 1, 1, 1, 1);
    c1 = ggml_add(ctx, c1, channel_1d(ctx, r.conv1.b));

    // norm2
    ggml_tensor * n2 = ggml_group_norm(ctx, c1, 32, 1e-6f);
    n2 = ggml_mul(ctx, n2, channel_1d(ctx, r.norm2_w));
    n2 = ggml_add(ctx, n2, channel_1d(ctx, r.norm2_b));
    n2 = ggml_silu(ctx, n2);

    // conv2: 3x3, padding 1
    ggml_tensor * c2 = ggml_conv_2d_direct(ctx, r.conv2.w, n2, 1, 1, 1, 1, 1, 1);
    c2 = ggml_add(ctx, c2, channel_1d(ctx, r.conv2.b));

    // shortcut
    if (r.conv_shortcut.w != nullptr) {
        h = ggml_conv_2d_direct(ctx, r.conv_shortcut.w, h, 1, 1, 0, 0, 1, 1);
        h = ggml_add(ctx, h, channel_1d(ctx, r.conv_shortcut.b));
    }

    return ggml_add(ctx, c2, h);
}

// Self-attention block (naive, no multi-head split, single batch assumed):
//   h = x  (shape: [W, H, C, N] - ggml NCHW)
//   x = group_norm(x) * gn_w + gn_b
//   x = permute(0,1,3,2) -> [W, H, N, C]   (contiguous)
//   x = reshape_2d to [C, seq]  where seq = W*H*N
//   q = mul_mat(x, W_q) + b_q -> (ne0=C, ne1=seq)
//   k = mul_mat(x, W_k) + b_k
//   v = mul_mat(x, W_v) + b_v
//   scores = mul_mat(q, k) -> (ne0=seq, ne1=seq)  -- this is Q @ K^T
//   scores = scores * (1/sqrt(head_dim))
//   scores = softmax(scores) over ne[0]
//   ctx_attn = mul_mat(scores, transpose(v)) -> (ne0=C, ne1=seq)  -- scores @ V
//   out = mul_mat(ctx_attn, W_out) + b_out
//   out = transpose(out) -> (ne0=seq, ne1=C)
//   out = reshape_4d to (W, H, N, C)
//   out = permute(0,1,3,2) -> (W, H, C, N)   (contiguous)
//   return out + h
static ggml_tensor * build_attention(ggml_context * ctx, ggml_tensor * x, const Attention & a) {
    ggml_tensor * h = x;

    int64_t W = x->ne[0];
    int64_t H = x->ne[1];
    int64_t C = x->ne[2];
    int64_t N = x->ne[3];
    int64_t seq = W * H * N;

    // group_norm
    ggml_tensor * gn = ggml_group_norm(ctx, x, 32, 1e-6f);
    ggml_tensor * gw = channel_1d(ctx, a.group_norm_w);
    gn = ggml_mul(ctx, gn, gw);
    ggml_tensor * gb = channel_1d(ctx, a.group_norm_b);
    gn = ggml_add(ctx, gn, gb);

    // permute (W,H,C,N) -> (W,H,N,C), contiguous
    ggml_tensor * p = ggml_cont(ctx, ggml_permute(ctx, gn, 0, 1, 3, 2));

    // reshape to (ne0=C, ne1=seq)  -- "x" in our notation
    p = ggml_reshape_2d(ctx, p, C, seq);

    // Q, K, V  (mul_mat output is 2D: ne0=out=C, ne1=seq)
    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, p, a.to_q_w), column_1d(ctx, a.to_q_b));
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, p, a.to_k_w), column_1d(ctx, a.to_k_b));
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, p, a.to_v_w), column_1d(ctx, a.to_v_b));

    // Q @ K^T (math): ggml_mul_mat(A, B) = A^T @ B, and Q,K are stored as
    // (ne0=seq, ne1=C). To get Q @ K^T we need to transpose both so that
    // mul_mat(qT, kT) = (qT)^T @ kT = Q @ K^T, producing ne=(seq, seq).
    ggml_tensor * qT = ggml_cont(ctx, ggml_transpose(ctx, q));
    ggml_tensor * kT = ggml_cont(ctx, ggml_transpose(ctx, k));
    ggml_tensor * scores = ggml_mul_mat(ctx, qT, kT);
    // Python's AttnBlock is single-head with q shape (B, 1, seq, C), so SDPA
    // uses scale = 1/sqrt(C) where C is the embed_dim (= channel count here).
    float scale = 1.0f / std::sqrt(static_cast<float>(C));
    scores = ggml_scale(ctx, scores, scale);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);

    // softmax(QK^T) @ V (math): mul_mat(A, B) = A^T @ B, so we transpose
    // scores (and keep v as is) so that mul_mat(scoresT, v) = scores @ V,
    // producing ne=(seq, C).
    ggml_tensor * scoresT = ggml_cont(ctx, ggml_transpose(ctx, scores));
    ggml_tensor * ctx_attn = ggml_mul_mat(ctx, scoresT, v);

    // output projection: transpose ctx_attn so that
    // mul_mat(ctxT, to_out_w) = ctx_attn @ to_out_w, producing ne=(seq, C).
    ggml_tensor * ctxT = ggml_cont(ctx, ggml_transpose(ctx, ctx_attn));
    ggml_tensor * out = ggml_add(ctx, ggml_mul_mat(ctx, ctxT, a.to_out_w), column_1d(ctx, a.to_out_b));

    // transpose to (ne0=seq, ne1=C) then reshape to (W, H, N, C)
    out = ggml_cont(ctx, ggml_transpose(ctx, out));
    out = ggml_reshape_4d(ctx, out, W, H, N, C);

    // permute (W, H, N, C) -> (W, H, C, N) back to NCHW
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 1, 3, 2));

    return ggml_add(ctx, out, h);
}

    // Build the full decoder graph.
// x: input latent in NCHW (ggml: [W, H, 32, 1])
// returns: output RGB image in NCHW (ggml: [W, H, 3, 1]) with raw decoder values
//          (matches Python's flux2_vae.Decoder.forward, which does no post-processing;
//          sigmoid+clip is the visualization script's responsibility, not the decoder's)
ggml_tensor * build_decoder_graph(ggml_context * ctx, const VAEWeights & w, ggml_tensor * x) {
    // post_quant_conv: 1x1, 32 -> 32
    ggml_tensor * h = ggml_conv_2d_direct(ctx, w.post_quant_conv.w, x, 1, 1, 0, 0, 1, 1);
    h = ggml_add(ctx, h, channel_1d(ctx, w.post_quant_conv.b));

    // conv_in: 3x3, 32 -> 512, padding 1
    h = ggml_conv_2d_direct(ctx, w.conv_in.w, h, 1, 1, 1, 1, 1, 1);
    h = ggml_add(ctx, h, channel_1d(ctx, w.conv_in.b));

    // mid_block: matches Python Decoder.forward which is
    //   h = self.mid.block_1(h); h = self.mid.attn_1(h); h = self.mid.block_2(h)
    // i.e. attention is INTERLEAVED between the two resnets, not after them.
    h = build_resnet(ctx, h, w.mid_block.resnets[0]);
    h = build_attention(ctx, h, w.mid_block.attn);
    h = build_resnet(ctx, h, w.mid_block.resnets[1]);

    // up_blocks
    for (const UpBlock & ub : w.up_blocks) {
        for (const ResNet & r : ub.resnets) {
            h = build_resnet(ctx, h, r);
        }
        if (ub.has_upsampler) {
            h = ggml_upscale(ctx, h, 2, GGML_SCALE_MODE_NEAREST);
            h = ggml_conv_2d_direct(ctx, ub.upsampler_conv.w, h, 1, 1, 1, 1, 1, 1);
            h = ggml_add(ctx, h, channel_1d(ctx, ub.upsampler_conv.b));
        }
    }

    // conv_norm_out (group_norm, 32 groups)
    h = ggml_group_norm(ctx, h, 32, 1e-6f);
    h = ggml_mul(ctx, h, channel_1d(ctx, w.conv_norm_out_w));
    h = ggml_add(ctx, h, channel_1d(ctx, w.conv_norm_out_b));
    h = ggml_silu(ctx, h);

    // conv_out: 3x3, 128 -> 3
    h = ggml_conv_2d_direct(ctx, w.conv_out.w, h, 1, 1, 1, 1, 1, 1);
    h = ggml_add(ctx, h, channel_1d(ctx, w.conv_out.b));

    return h;
}

}  // namespace bonsai
