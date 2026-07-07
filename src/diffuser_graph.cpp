#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "diffuser_types.h"
#include "diffuser_graph.h"
#include "b1_0_kernel.h"

namespace bonsai {

static ggml_tensor * column_1d(ggml_context * ctx, ggml_tensor * b) {
    ggml_tensor * v = ggml_reshape_4d(ctx, b, b->ne[0], 1, 1, 1);
    return ggml_cont(ctx, v);
}

static ggml_tensor * silu_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * h = ggml_mul_mat(ctx, w, x);
    if (b) { h = ggml_add(ctx, h, column_1d(ctx, b)); }
    return ggml_silu(ctx, h);
}

struct ModSplit {
    ggml_tensor * shift;
    ggml_tensor * scale;
    ggml_tensor * gate;
};

static void split_mod(ggml_context * ctx, ggml_tensor * mod, int hidden_size, int batch, ModSplit & s1, ModSplit & s2) {
    int H = hidden_size;
    ggml_tensor * parts[6];
    for (int i = 0; i < 6; i++) {
        parts[i] = ggml_view_2d(ctx, mod, H, 1, mod->nb[1], i * H * sizeof(float));
    }
    s1 = {parts[0], parts[1], parts[2]};
    s2 = {parts[3], parts[4], parts[5]};
}

static void split_mod_single(ggml_context * ctx, ggml_tensor * mod, int hidden_size, int batch, ModSplit & s) {
    int H = hidden_size;
    s.shift = ggml_view_2d(ctx, mod, H, 1, mod->nb[1], 0 * H * sizeof(float));
    s.scale = ggml_view_2d(ctx, mod, H, 1, mod->nb[1], 1 * H * sizeof(float));
    s.gate  = ggml_view_2d(ctx, mod, H, 1, mod->nb[1], 2 * H * sizeof(float));
}

struct QKV {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
};

struct QKVProj {
    ggml_tensor * qkv;
    ggml_tensor * mlp;
};

static QKVProj single_qkv(ggml_context * ctx, ggml_tensor * x, int hidden_size, int mlp_hd, int n_threads, B1LinearUserData & ud, const DiffuserWeights::SingleBlock & blk) {
    ggml_tensor * h = b1_linear(ctx, x, blk.to_qkv_mlp_proj, n_threads, ud);
    int qkv_dim = 3 * hidden_size;
    int mlp_dim = mlp_hd * 2;
    QKVProj res;
    res.qkv = ggml_view_2d(ctx, h, qkv_dim, h->ne[1], h->nb[1], 0);
    res.mlp = ggml_view_2d(ctx, h, mlp_dim, h->ne[1], h->nb[1], qkv_dim * sizeof(float));
    return res;
}

static ggml_tensor * rms_norm_qk(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int head_dim) {
    int n_heads = x->ne[1];
    int seq     = x->ne[2];
    x = ggml_cont(ctx, ggml_reshape_3d(ctx, x, head_dim, n_heads, seq));
    x = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_tensor * w_view = ggml_reshape_3d(ctx, w, head_dim, 1, 1);
    x = ggml_mul(ctx, x, ggml_repeat(ctx, w_view, x));
    return ggml_cont(ctx, x);
}

static QKV split_qkv(ggml_context * ctx, ggml_tensor * qkv, int hidden_size, int num_heads, int seq) {
    int head_dim = hidden_size / num_heads;
    QKV r;

    auto view_h = [&](ggml_tensor * t, int offset) {
        return ggml_view_2d(ctx, t, hidden_size, seq, t->nb[1], offset * hidden_size * sizeof(float));
    };

    ggml_tensor * q_t = view_h(qkv, 0);
    ggml_tensor * k_t = view_h(qkv, 1);
    ggml_tensor * v_t = view_h(qkv, 2);

    auto reshape_heads = [&](ggml_tensor * t) {
        return ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, t), head_dim, num_heads, seq));
    };

    r.q = reshape_heads(q_t);
    r.k = reshape_heads(k_t);
    r.v = reshape_heads(v_t);
    return r;
}

static QKV split_qkv_cat(ggml_context * ctx, ggml_tensor * q_t, ggml_tensor * k_t, ggml_tensor * v_t, int head_dim, int num_heads, int seq) {
    QKV r;
    auto reshape = [&](ggml_tensor * t) {
        return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, t, head_dim, num_heads, seq), 0, 2, 1, 3));
    };
    r.q = reshape(q_t);
    r.k = reshape(k_t);
    r.v = reshape(v_t);
    return r;
}

// Flux 2D RoPE implementation
// Reference (Python): rope(pos, dim, theta) returns [..., dim/2, 2, 2]
//   out = stack([cos, -sin, sin, cos], dim=-1)
//   out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
// apply_rope(xq, xk, freqs_cis):
//   xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
//   xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
//   result: [a*cos - b*sin, a*sin + b*cos] for (a, b) = (q_even, q_odd)
//
// C++ implementation: precompute cos[d, s] and sin[d, s] (d in [0, head_dim/2)).
// Use a custom op to apply the 2D rotation in-place per (d, h, s) pair.

static ggml_tensor * build_rope_freqs_table(ggml_context * ctx, const int * axes_dim, int n_axes, float theta) {
    int max_half = 0;
    for (int i = 0; i < n_axes; i++) {
        max_half = std::max(max_half, axes_dim[i] / 2);
    }
    ggml_tensor * freqs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, max_half, n_axes);
    freqs->data = malloc(max_half * n_axes * sizeof(float));
    float * data = (float *)freqs->data;
    for (int a = 0; a < n_axes; a++) {
        int ax_half = axes_dim[a] / 2;
        for (int j = 0; j < ax_half; j++) {
            float scale = (float)(j * 2) / (float)axes_dim[a];
            data[a * max_half + j] = 1.0f / powf(theta, scale);
        }
    }
    return freqs;
}

static void build_rope_cos_sin(
    ggml_context * ctx,
    ggml_tensor * ids,
    ggml_tensor * freqs_table,
    const int * axes_dim,
    int n_axes,
    ggml_tensor ** cos_out,
    ggml_tensor ** sin_out)
{
    // ids: [n_axes, seq] (ne[0]=n_axes, ne[1]=seq, column-major)
    //   data layout: ids[a + s*n_axes] for axis a, token s
    // freqs_table: [max_half, n_axes]
    //   data layout: freqs[d + a*max_half] for freq d, axis a
    // For each axis a:
    //   axis_ids:  [1, seq] (view with stride n_axes*sizeof(float))
    //   axis_freqs: [1, ax_half] (view)
    //   angles_a:  [ax_half, seq] = mul(axis_freqs, axis_ids)
    // Concat axes -> [head_dim/2, seq]
    int seq = ids->ne[1];

    ggml_tensor * all_angles = nullptr;
    for (int a = 0; a < n_axes; a++) {
        int ax_half = axes_dim[a] / 2;
        // axis_ids: stride between rows = n_axes*sizeof(float), offset = a*sizeof(float)
        ggml_tensor * axis_ids = ggml_view_2d(ctx, ids, 1, seq,
            (size_t)ids->ne[0] * sizeof(float), (size_t)a * sizeof(float));
        ggml_tensor * axis_ids_c = ggml_cont(ctx, axis_ids);
        ggml_tensor * axis_freqs = ggml_view_1d(ctx, freqs_table, ax_half,
            (size_t)a * freqs_table->ne[0] * sizeof(float));
        ggml_tensor * axis_ids_2d  = ggml_reshape_2d(ctx, axis_ids_c, 1,     seq);
        ggml_tensor * axis_freqs_2d = ggml_reshape_2d(ctx, axis_freqs, 1,     ax_half);
        ggml_tensor * angles = ggml_mul_mat(ctx, axis_freqs_2d, axis_ids_2d);
        if (all_angles == nullptr) {
            all_angles = angles;
        } else {
            all_angles = ggml_concat(ctx, all_angles, angles, 0);
        }
    }
    *cos_out = ggml_cont(ctx, ggml_cos(ctx, all_angles));
    *sin_out = ggml_cont(ctx, ggml_sin(ctx, all_angles));
}

static void apply_rope_2d(
    ggml_context * ctx,
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * cos_t,
    ggml_tensor * sin_t,
    int head_dim,
    int n_threads,
    std::vector<Rope2DUserData> & rope_ud,
    ggml_tensor ** q_out,
    ggml_tensor ** k_out)
{
    int n_heads = q->ne[1];
    int seq     = q->ne[2];
    rope_ud.push_back({head_dim, n_heads, seq});
    *q_out = rope_2d_fwd(ctx, q, cos_t, sin_t, rope_ud.back());
    rope_ud.push_back({head_dim, n_heads, seq});
    *k_out = rope_2d_fwd(ctx, k, cos_t, sin_t, rope_ud.back());
}

static ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int seq_q, int seq_k, int head_dim) {
    int n_heads = q->ne[1];

    ggml_tensor * q_2d = ggml_cont(ctx, ggml_reshape_2d(ctx, q, head_dim, seq_q * n_heads));
    ggml_tensor * k_2d = ggml_cont(ctx, ggml_reshape_2d(ctx, k, head_dim, seq_k * n_heads));
    ggml_tensor * v_2d = ggml_cont(ctx, ggml_reshape_2d(ctx, v, head_dim, seq_k * n_heads));

    ggml_tensor * scores = ggml_mul_mat(ctx, k_2d, q_2d);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    scores = ggml_scale(ctx, scores, scale);

    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);

    ggml_tensor * attn = ggml_mul_mat(ctx, v_2d, scores);

    attn = ggml_cont(ctx, ggml_reshape_4d(ctx, attn, head_dim, seq_q, n_heads, 1));
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
    attn = ggml_cont(ctx, ggml_reshape_2d(ctx, attn, head_dim * n_heads, seq_q));

    return attn;
}

static ggml_tensor * mlp_act(ggml_context * ctx, ggml_tensor * mlp) {
    return ggml_swiglu(ctx, mlp);
}

DiffuserGraph build_diffuser_graph(
    ggml_context * ctx,
    const DiffuserParams & params,
    const DiffuserWeights & weights,
    int img_tokens,
    int txt_tokens,
    int batch,
    int n_threads)
{
    DiffuserGraph result;
    int H = params.hidden_size;
    int C = params.in_channels;
    int ctx_dim = params.context_in_dim;
    int n_heads = params.num_heads;
    int head_dim = params.head_dim;
    int mlp_hd = params.mlp_hidden_dim;
    int total_tokens = img_tokens + txt_tokens;

    // Pre-reserve userdata vectors to prevent reallocation invalidating
    // pointers passed to ggml_custom_4d as userdata.
    // Counts: 12 * params.depth (double blocks) +
    //         2 * params.depth_single_blocks (single blocks)
    result.b1_ud.reserve(12 * params.depth + 2 * params.depth_single_blocks + 16);
    // Each apply_rope_2d call pushes 2 entries (one for q, one for k).
    // Double block: 2 apply_rope_2d calls (img + txt) * 2 = 4 pushes per block.
    // Single block: 1 apply_rope_2d call * 2 = 2 pushes per block.
    result.rope_ud.reserve(4 * params.depth + 2 * params.depth_single_blocks + 16);

    result.graph = ggml_new_graph_custom(ctx, 65536, false);

    result.img_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, params.in_channels, img_tokens * batch);
    result.txt_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, params.context_in_dim, txt_tokens * batch);
    result.timestep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    result.img_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, img_tokens * batch);
    result.txt_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, txt_tokens * batch);

    ggml_set_input(result.img_in);
    ggml_set_input(result.txt_in);
    ggml_set_input(result.timestep);
    ggml_set_input(result.img_ids);
    ggml_set_input(result.txt_ids);

    int ud_idx = 0;
    auto b1 = [&](ggml_tensor * act, const B1Weights & w) -> ggml_tensor * {
        result.b1_ud.push_back({w.in_dim, w.out_dim});
        (void)ud_idx;
        return b1_linear(ctx, act, w, n_threads, result.b1_ud.back());
    };

    ggml_tensor * w_img = ggml_reshape_2d(ctx, weights.img_in.data, C, H);
    ggml_tensor * h_img = ggml_mul_mat(ctx, w_img, result.img_in);
    ggml_tensor * w_txt = ggml_reshape_2d(ctx, weights.txt_in.data, ctx_dim, H);
    ggml_tensor * h_txt = ggml_mul_mat(ctx, w_txt, result.txt_in);

    ggml_tensor * te = ggml_timestep_embedding(ctx, result.timestep, 256, 10000);
    ggml_format_name(te, "te_emb");
    ggml_tensor * w1_2d = ggml_reshape_2d(ctx, weights.time_in_w1.data, 256, H);
    te = ggml_mul_mat(ctx, w1_2d, te);
    ggml_format_name(te, "te_w1");
    if (weights.time_in_b1.data) te = ggml_add(ctx, te, column_1d(ctx, weights.time_in_b1.data));
    ggml_format_name(te, "te_pre_silu");
    te = ggml_silu(ctx, te);
    ggml_format_name(te, "te_post_silu");
    ggml_tensor * w2_2d = ggml_reshape_2d(ctx, weights.time_in_w2.data, H, H);
    ggml_tensor * vec = ggml_mul_mat(ctx, w2_2d, te);
    ggml_format_name(vec, "vec");
    if (weights.time_in_b2.data) vec = ggml_add(ctx, vec, column_1d(ctx, weights.time_in_b2.data));
    ggml_format_name(vec, "vec_final");

    ggml_tensor * vec_silu = ggml_silu(ctx, vec);
    ggml_tensor * w_mod_img = ggml_reshape_2d(ctx, weights.double_mod_img.data, H, 6 * H);
    ggml_tensor * mod_img_raw = ggml_mul_mat(ctx, w_mod_img, vec_silu);
    ggml_tensor * w_mod_txt = ggml_reshape_2d(ctx, weights.double_mod_txt.data, H, 6 * H);
    ggml_tensor * mod_txt_raw = ggml_mul_mat(ctx, w_mod_txt, vec_silu);

    int B = batch;
    (void)B;

    ModSplit img_mod1, img_mod2;
    split_mod(ctx, mod_img_raw, H, 1, img_mod1, img_mod2);

    ModSplit txt_mod1, txt_mod2;
    split_mod(ctx, mod_txt_raw, H, 1, txt_mod1, txt_mod2);

    int n_axes = 4;
    int axes_dim[4] = {32, 32, 32, 32};

    ggml_tensor * freqs_table = build_rope_freqs_table(ctx, axes_dim, n_axes, (float)params.theta);

    ggml_tensor * cos_img = nullptr;
    ggml_tensor * sin_img = nullptr;
    ggml_tensor * cos_txt = nullptr;
    ggml_tensor * sin_txt = nullptr;
    build_rope_cos_sin(ctx, result.img_ids, freqs_table, axes_dim, n_axes, &cos_img, &sin_img);
    build_rope_cos_sin(ctx, result.txt_ids, freqs_table, axes_dim, n_axes, &cos_txt, &sin_txt);

    ggml_tensor * cos_combined = ggml_concat(ctx, cos_txt, cos_img, 1);
    ggml_tensor * sin_combined = ggml_concat(ctx, sin_txt, sin_img, 1);

    ggml_tensor * one_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * ones = ggml_fill(ctx, one_tensor, 1.0f);

    (void)ones;

    for (int d = 0; d < params.depth; d++) {
        const auto & db = weights.double_blocks[d];

        debug_check(ctx, h_img, "h_img_pre", nullptr);
        debug_check(ctx, h_txt, "h_txt_pre", nullptr);

        ggml_tensor * img_n = ggml_norm(ctx, h_img, 1e-6f);
        ggml_tensor * txt_n = ggml_norm(ctx, h_txt, 1e-6f);

        img_n = ggml_add(ctx, ggml_mul(ctx, img_n, ggml_add(ctx, ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod1.scale, H, 0)), img_n), ones)), ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod1.shift, H, 0)), img_n));

        txt_n = ggml_add(ctx, ggml_mul(ctx, txt_n, ggml_add(ctx, ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.scale, H, 0)), txt_n), ones)), ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.shift, H, 0)), txt_n));

        ggml_tensor * img_q = b1(img_n, db.attn_to_q);
        img_q = debug_check(ctx, img_q, "img_q_first", nullptr);
        ggml_tensor * img_k = b1(img_n, db.attn_to_k);
        ggml_tensor * img_v = b1(img_n, db.attn_to_v);

        ggml_tensor * txt_q = b1(txt_n, db.attn_add_q);
        ggml_tensor * txt_k = b1(txt_n, db.attn_add_k);
        ggml_tensor * txt_v = b1(txt_n, db.attn_add_v);

        img_q = ggml_cont(ctx, ggml_reshape_3d(ctx, img_q, head_dim, n_heads, img_tokens));
        img_k = ggml_cont(ctx, ggml_reshape_3d(ctx, img_k, head_dim, n_heads, img_tokens));
        img_v = ggml_cont(ctx, ggml_reshape_3d(ctx, img_v, head_dim, n_heads, img_tokens));

        txt_q = ggml_cont(ctx, ggml_reshape_3d(ctx, txt_q, head_dim, n_heads, txt_tokens));
        txt_k = ggml_cont(ctx, ggml_reshape_3d(ctx, txt_k, head_dim, n_heads, txt_tokens));
        txt_v = ggml_cont(ctx, ggml_reshape_3d(ctx, txt_v, head_dim, n_heads, txt_tokens));

        img_q = rms_norm_qk(ctx, img_q, db.attn_norm_q.data, head_dim);
        img_k = rms_norm_qk(ctx, img_k, db.attn_norm_k.data, head_dim);
        txt_q = rms_norm_qk(ctx, txt_q, db.attn_norm_added_q.data, head_dim);
        txt_k = rms_norm_qk(ctx, txt_k, db.attn_norm_added_k.data, head_dim);

        debug_check(ctx, img_q, "img_q_normed", nullptr);
        debug_check(ctx, img_k, "img_k_normed", nullptr);

        ggml_tensor * img_q_r = nullptr;
        ggml_tensor * img_k_r = nullptr;
        ggml_tensor * txt_q_r = nullptr;
        ggml_tensor * txt_k_r = nullptr;
        apply_rope_2d(ctx, img_q, img_k, cos_img, sin_img, head_dim, n_threads, result.rope_ud, &img_q_r, &img_k_r);
        apply_rope_2d(ctx, txt_q, txt_k, cos_txt, sin_txt, head_dim, n_threads, result.rope_ud, &txt_q_r, &txt_k_r);
        img_q = img_q_r;
        img_k = img_k_r;
        txt_q = txt_q_r;
        txt_k = txt_k_r;

        debug_check(ctx, img_q, "img_q_roped", nullptr);
        debug_check(ctx, img_k, "img_k_roped", nullptr);

        int img_t = img_tokens;
        int txt_t = txt_tokens;
        ggml_tensor * q = ggml_concat(ctx, txt_q, img_q, 2);
        ggml_tensor * k = ggml_concat(ctx, txt_k, img_k, 2);
        ggml_tensor * v = ggml_concat(ctx, txt_v, img_v, 2);

        ggml_tensor * q_3d = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * k_3d = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * v_3d = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        ggml_tensor * attn;
#if 1 // FLASH ATTENTION
        ggml_tensor * k_3d_f16 = k_3d;
        ggml_tensor * v_3d_f16 = v_3d;
        if (k_3d_f16->type == GGML_TYPE_F32) k_3d_f16 = ggml_cast(ctx, k_3d_f16, GGML_TYPE_F16);
        if (v_3d_f16->type == GGML_TYPE_F32) v_3d_f16 = ggml_cast(ctx, v_3d_f16, GGML_TYPE_F16);
        
        attn = ggml_flash_attn_ext(ctx, q_3d, k_3d_f16, v_3d_f16, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        
        attn = debug_check(ctx, attn, "flash_attn_raw", nullptr);
        attn = ggml_reshape_2d(ctx, attn, H, txt_t + img_t);
#else
        ggml_tensor * scores = ggml_mul_mat(ctx, k_3d, q_3d);
        scores = debug_check(ctx, scores, "db_scores_raw", nullptr);
        scores = ggml_scale_inplace(ctx, scores, scale);
        scores = debug_check(ctx, scores, "scores_scaled", nullptr);
        scores = ggml_soft_max_inplace(ctx, scores);
        debug_check(ctx, scores, "scores_softmaxed", nullptr);

        ggml_tensor * v_3d_t = ggml_cont(ctx, ggml_permute(ctx, v_3d, 1, 0, 2, 3));
        attn = ggml_mul_mat(ctx, v_3d_t, scores);
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx, attn, H, txt_t + img_t);
#endif

        ggml_tensor * img_attn_out = ggml_view_2d(ctx, attn, H, img_t, attn->nb[1], txt_t * H * sizeof(float));
        ggml_tensor * txt_attn_out = ggml_view_2d(ctx, attn, H, txt_t, attn->nb[1], 0);

        txt_attn_out = b1(txt_attn_out, db.attn_add_out);

        img_attn_out = debug_check(ctx, img_attn_out, "img_attn_out_before_proj", nullptr);
        img_attn_out = b1(img_attn_out, db.attn_to_out);
        img_attn_out = debug_check(ctx, img_attn_out, "img_attn_out_after_proj", nullptr);
        
        ggml_tensor * mod_gate = ggml_view_1d(ctx, img_mod1.gate, H, 0);
        h_img = ggml_add(ctx, h_img, ggml_mul(ctx, ggml_repeat(ctx, ggml_cont(ctx, mod_gate), img_attn_out), img_attn_out));
        h_txt = ggml_add(ctx, h_txt, ggml_mul(ctx, ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.gate, H, 0)), txt_attn_out), txt_attn_out));

        ggml_tensor * img_ff_n = ggml_norm(ctx, h_img, 1e-6f);
        ggml_tensor * txt_ff_n = ggml_norm(ctx, h_txt, 1e-6f);

        img_ff_n = ggml_add(ctx, ggml_mul(ctx, img_ff_n, ggml_add(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.scale, H, 0))), img_ff_n), ones)), ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.shift, H, 0))), img_ff_n));
        txt_ff_n = ggml_add(ctx, ggml_mul(ctx, txt_ff_n, ggml_add(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.scale, H, 0))), txt_ff_n), ones)), ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.shift, H, 0))), txt_ff_n));

        ggml_tensor * img_mlp_in = b1(img_ff_n, db.ff_linear_in);
        ggml_tensor * txt_mlp_in = b1(txt_ff_n, db.ff_ctx_linear_in);

        ggml_tensor * img_mlp_a = ggml_swiglu(ctx, img_mlp_in);
        img_mlp_a = b1(img_mlp_a, db.ff_linear_out);

        ggml_tensor * txt_mlp_a = ggml_swiglu(ctx, txt_mlp_in);
        txt_mlp_a = b1(txt_mlp_a, db.ff_ctx_linear_out);

        h_img = ggml_add(ctx, h_img, ggml_mul(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.gate, H, 0))), img_mlp_a), img_mlp_a));
        h_txt = ggml_add(ctx, h_txt, ggml_mul(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.gate, H, 0))), txt_mlp_a), txt_mlp_a));
    }

    ggml_tensor * combined = ggml_concat(ctx, h_txt, h_img, 1);

    ggml_tensor * w_single = ggml_reshape_2d(ctx, weights.single_mod.data, H, 3 * H);
    ggml_tensor * mod_single_raw = ggml_mul_mat(ctx, w_single, vec_silu);
    ModSplit single_mod;
    split_mod_single(ctx, mod_single_raw, H, 1, single_mod);

    for (int s = 0; s < params.depth_single_blocks; s++) {
        const auto & sb = weights.single_blocks[s];

        ggml_tensor * x_n = ggml_norm(ctx, combined, 1e-6f);
        x_n = ggml_add(ctx, ggml_mul(ctx, x_n, ggml_add(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, single_mod.scale, H, 0))), x_n), ones)), ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, single_mod.shift, H, 0))), x_n));

        ggml_tensor * proj_all = b1(x_n, sb.to_qkv_mlp_proj);
        proj_all = debug_check(ctx, proj_all, "sb_proj_all", nullptr);
        int qkv_dim = 3 * H;
        ggml_tensor * qkv_t = ggml_view_2d(ctx, proj_all, qkv_dim, total_tokens, proj_all->nb[1], 0);
        ggml_tensor * mlp_t = ggml_view_2d(ctx, proj_all, mlp_hd * 2, total_tokens, proj_all->nb[1], qkv_dim * sizeof(float));

        QKV qkv = split_qkv(ctx, qkv_t, H, n_heads, total_tokens);

        qkv.q = rms_norm_qk(ctx, qkv.q, sb.norm_q.data, head_dim);
        qkv.q = debug_check(ctx, qkv.q, "sb_qkv_q_norm", nullptr);
        qkv.k = rms_norm_qk(ctx, qkv.k, sb.norm_k.data, head_dim);
        qkv.k = debug_check(ctx, qkv.k, "sb_qkv_k_norm", nullptr);

        ggml_tensor * q_r = nullptr;
        ggml_tensor * k_r = nullptr;
        apply_rope_2d(ctx, qkv.q, qkv.k, cos_combined, sin_combined, head_dim, n_threads, result.rope_ud, &q_r, &k_r);
        qkv.q = q_r;
        qkv.k = k_r;

        int seq = total_tokens;
        ggml_tensor * q_3d_a = ggml_cont(ctx, ggml_permute(ctx, qkv.q, 0, 2, 1, 3));
        ggml_tensor * k_3d_a = ggml_cont(ctx, ggml_permute(ctx, qkv.k, 0, 2, 1, 3));
        ggml_tensor * v_3d_a = ggml_cont(ctx, ggml_permute(ctx, qkv.v, 0, 2, 1, 3));

        float scale_s = 1.0f / std::sqrt(static_cast<float>(head_dim));

        ggml_tensor * s_attn;
#if 1 // FLASH ATTENTION
        ggml_tensor * k_3d_a_f16 = k_3d_a;
        ggml_tensor * v_3d_a_f16 = v_3d_a;
        if (k_3d_a_f16->type == GGML_TYPE_F32) k_3d_a_f16 = ggml_cast(ctx, k_3d_a_f16, GGML_TYPE_F16);
        if (v_3d_a_f16->type == GGML_TYPE_F32) v_3d_a_f16 = ggml_cast(ctx, v_3d_a_f16, GGML_TYPE_F16);
        
        s_attn = ggml_flash_attn_ext(ctx, q_3d_a, k_3d_a_f16, v_3d_a_f16, nullptr, scale_s, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(s_attn, GGML_PREC_F32);
        
        s_attn = debug_check(ctx, s_attn, "sb_flash_attn_raw", nullptr);
        s_attn = ggml_reshape_2d(ctx, s_attn, H, seq);
#else
        ggml_tensor * s_scores = ggml_mul_mat(ctx, k_3d_a, q_3d_a);
        s_scores = debug_check(ctx, s_scores, "s_scores_raw", nullptr);
        s_scores = ggml_scale_inplace(ctx, s_scores, scale_s);
        s_scores = debug_check(ctx, s_scores, "s_scores_scaled", nullptr);
        s_scores = ggml_soft_max_inplace(ctx, s_scores);

        ggml_tensor * v_3d_a_t = ggml_cont(ctx, ggml_permute(ctx, v_3d_a, 1, 0, 2, 3));
        s_attn = ggml_mul_mat(ctx, v_3d_a_t, s_scores);
        s_attn = ggml_cont(ctx, ggml_permute(ctx, s_attn, 0, 2, 1, 3));
        s_attn = ggml_reshape_2d(ctx, s_attn, H, seq);
#endif

        ggml_tensor * s_mlp_a = mlp_act(ctx, mlp_t);
        ggml_tensor * s_cat = ggml_concat(ctx, s_attn, s_mlp_a, 0);

        ggml_tensor * s_out = b1(s_cat, sb.to_out);

        combined = ggml_add(ctx, combined, ggml_mul(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, single_mod.gate, H, 0))), s_out), s_out));
    }

    ggml_tensor * final_img = ggml_view_2d(ctx, combined, H, img_tokens, combined->nb[1], txt_tokens * H * sizeof(float));

    ggml_tensor * fn = ggml_norm(ctx, final_img, 1e-6f);
    ggml_tensor * vec_silu_out = ggml_silu(ctx, vec);
    ggml_tensor * w_norm_out = ggml_reshape_2d(ctx, weights.norm_out_linear.data, H, 2 * H);
    ggml_tensor * mod_out_raw = ggml_mul_mat(ctx, w_norm_out, vec_silu_out);
    ggml_tensor * mod_out_shift = ggml_view_2d(ctx, mod_out_raw, H, 1, mod_out_raw->nb[1], 0);
    ggml_tensor * mod_out_scale = ggml_view_2d(ctx, mod_out_raw, H, 1, mod_out_raw->nb[1], H * sizeof(float));

    fn = ggml_add(ctx, ggml_mul(ctx, fn, ggml_add(ctx, ggml_repeat(ctx, column_1d(ctx, mod_out_scale), fn), ones)), ggml_repeat(ctx, column_1d(ctx, mod_out_shift), fn));

    ggml_tensor * w_proj_out = ggml_reshape_2d(ctx, weights.proj_out.data, H, C);
    result.out = ggml_mul_mat(ctx, w_proj_out, fn);
    ggml_set_output(result.out);

    ggml_build_forward_expand(result.graph, result.out);

    return result;
}

} // namespace bonsai
