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

// Self-attention block (single-head, single-batch, ggml NCHW input):
//   h = x                                       (W, H, C, N)
//   x = group_norm(x) * gn_w + gn_b
//   x = permute(1, 2, 0, 3) + cont              -> (C, W, H, N) contiguous, C is inner dim
//   x = reshape_2d to (C, seq)                  (matches PyTorch rearrange to (seq, C) with C inner)
//   q = mul_mat(W_q, x) + b_q                   (ne0=C, ne1=seq)  -- weight as first arg
//   k = mul_mat(W_k, x) + b_k
//   v = mul_mat(W_v, x) + b_v
//   scores = mul_mat(k, q)                      (ne0=seq=keys, ne1=seq=queries) -- k first so ne0=keys
//   scores = scores * (1/sqrt(C))
//   scores = softmax(scores) over ne[0]         (normalizes over keys -- correct for SDPA)
//   vT   = transpose(v)
//   ctx  = mul_mat(scores, vT)                  (ne0=seq, ne1=C) -- math (C, seq), needs swap
//   ctxT = transpose(ctx)                       (ne0=C, ne1=seq) -- math (seq, C)
//   out  = mul_mat(W_out, ctxT) + b_out         (ne0=C, ne1=seq) -- math (seq, C)
//   out  = reshape_4d to (C, W, H, N)
//   out  = permute(2, 0, 1, 3) + cont           -> (W, H, C, N) NCHW
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

    // permute (W,H,C,N) -> (C,W,H,N), contiguous. After this, the memory is
    // laid out with C as the inner (fastest-varying) dim, matching PyTorch's
    // rearrange(q, "b c h w -> b 1 (h w) c") which puts c as innermost.
    // ggml_permute args are the NEW position of each OLD dim: axis0=new_pos_of_ne0,
    // so to put old ne0 (W) at new position 1, axis0=1; old ne1 (H) at new pos 2,
    // axis1=2; old ne2 (C) at new pos 0, axis2=0; old ne3 (N) stays, axis3=3.
    ggml_tensor * p = ggml_cont(ctx, ggml_permute(ctx, gn, 1, 2, 0, 3));

    // reshape to (ne0=C, ne1=seq)  -- "x" in our notation
    p = ggml_reshape_2d(ctx, p, C, seq);

    // Q, K, V  -- weight as first arg so result has ne0=C (the channel dim).
    // In ggml, ggml_mul_mat(A, B) gives result.ne[0] = A->ne[1], so with
    // A=W (ne0=in_dim=C, ne1=out_dim=C) and B=p (ne0=C, ne1=seq), the
    // result is (ne0=C, ne1=seq). This is the right shape for the per-channel
    // bias (column_1d reshapes the bias to (C,1,1,1) which broadcasts along
    // ne[1] from 1 to seq, adding the channel-bias to every position).
    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, a.to_q_w, p), column_1d(ctx, a.to_q_b));
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, a.to_k_w, p), column_1d(ctx, a.to_k_b));
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, a.to_v_w, p), column_1d(ctx, a.to_v_b));

    // Q @ K^T (math): Q,K are stored as (ne0=C, ne1=seq). For mul_mat(A,B) the
    // contraction is over ne[0] and result.ne[0] = A->ne[1]. To get the math
    // (seq, seq) scores and have ne[0]=keys (so the default softmax on ne[0]
    // normalizes over the keys, matching SDPA), we want A.ne[1]=keys, so we
    // pass k as the first arg: mul_mat(k, q) -> ne0=k.ne[1]=seq=keys, ne1=seq.
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    // Python's AttnBlock is single-head with q shape (B, 1, seq, C), so SDPA
    // uses scale = 1/sqrt(C) where C is the embed_dim (= channel count here).
    float scale = 1.0f / std::sqrt(static_cast<float>(C));
    scores = ggml_scale(ctx, scores, scale);
    // ggml_soft_max_ext normalizes over ne[0]; with our setup ne[0]=keys. ✓
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);

    // softmax(QK^T) @ V (math): scores is (ne0=seq=keys, ne1=seq=queries),
    // v is (ne0=C, ne1=seq). To compute scores @ V via mul_mat(A, B) we need
    // A.ne[0] = B.ne[0] for the contraction. We transpose v to get vT with
    // ne[0]=seq, then mul_mat(scores, vT) contracts on ne[0]=seq. The result
    // is (ne0=seq, ne1=C) which corresponds to math (C, seq) — transposed
    // relative to what we want (math (seq, C)), so we transpose once more.
    ggml_tensor * vT = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * ctx_attn = ggml_mul_mat(ctx, scores, vT);
    ggml_tensor * ctxT = ggml_cont(ctx, ggml_transpose(ctx, ctx_attn));

    // output projection: ctxT is (ne0=C, ne1=seq). With W_out also (ne0=C, ne1=C),
    // mul_mat(W_out, ctxT) gives (ne0=C, ne1=seq) — math (seq, C). ✓
    ggml_tensor * out = ggml_add(ctx, ggml_mul_mat(ctx, a.to_out_w, ctxT), column_1d(ctx, a.to_out_b));

    // out is (ne0=C, ne1=seq). Reshape to (C, W, H, N) preserving the
    // (C, W, H, N) row-major memory order, then permute to NCHW.
    // ggml_permute args = new position of each old dim: old C (ne0) at new
    // pos 2 -> axis0=2; old W (ne1) at new pos 0 -> axis1=0; old H (ne2) at
    // new pos 1 -> axis2=1; old N (ne3) stays -> axis3=3.
    out = ggml_reshape_4d(ctx, out, C, W, H, N);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 2, 0, 1, 3));

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
