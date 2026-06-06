#include "ggml.h"

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
    int seq = x->ne[1];
    int n_heads = x->ne[0] / head_dim;
    x = ggml_cont(ctx, ggml_reshape_3d(ctx, x, head_dim, seq, n_heads));
    x = ggml_rms_norm(ctx, x, 1e-6f);
    x = ggml_mul(ctx, x, ggml_repeat(ctx, column_1d(ctx, w), x));
    return ggml_cont(ctx, ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2]));
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
        return ggml_cont(ctx, ggml_reshape_3d(ctx, t, head_dim, num_heads, seq));
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

static ggml_tensor * apply_rope_2d(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * pe, int head_dim, int theta) {
    int n_dims = head_dim;
    int mode = GGML_ROPE_TYPE_NEOX;
    int sections[4] = {0};

    q = ggml_rope_ext(ctx, q, pe, nullptr, n_dims, mode, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pe, nullptr, n_dims, mode, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    return q;
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

static ggml_tensor * mlp_act(ggml_context * ctx, ggml_tensor * mlp, int mlp_hd) {
    ggml_tensor * g = ggml_view_2d(ctx, mlp, mlp_hd, mlp->ne[1], mlp->nb[1], 0);
    ggml_tensor * x = ggml_view_2d(ctx, mlp, mlp_hd, mlp->ne[1], mlp->nb[1], mlp_hd * sizeof(float));
    return ggml_mul(ctx, ggml_silu(ctx, g), x);
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
    int n_heads = params.num_heads;
    int head_dim = params.head_dim;
    int mlp_hd = params.mlp_hidden_dim;
    int total_tokens = img_tokens + txt_tokens;

    result.graph = ggml_new_graph_custom(ctx, 65536, false);

    result.img_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, params.in_channels, img_tokens * batch);
    result.txt_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, params.context_in_dim, txt_tokens * batch);
    result.timestep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_input(result.img_in);
    ggml_set_input(result.txt_in);
    ggml_set_input(result.timestep);

    int ud_idx = 0;
    auto b1 = [&](ggml_tensor * act, const B1Weights & w) -> ggml_tensor * {
        result.b1_ud.push_back({w.in_dim, w.out_dim});
        (void)ud_idx;
        return b1_linear(ctx, act, w, n_threads, result.b1_ud.back());
    };

    ggml_tensor * h_img = b1(result.img_in, weights.img_in);
    ggml_tensor * h_txt = b1(result.txt_in, weights.txt_in);

    ggml_tensor * te = ggml_timestep_embedding(ctx, result.timestep, 256, 10000);
    te = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, weights.time_in_w1.data, te), column_1d(ctx, weights.time_in_b1.data)));
    ggml_tensor * vec = ggml_add(ctx, ggml_mul_mat(ctx, weights.time_in_w2.data, te), column_1d(ctx, weights.time_in_b2.data));

    ggml_tensor * mod_img_raw = b1(vec, weights.double_mod_img);
    ggml_tensor * mod_txt_raw = b1(vec, weights.double_mod_txt);

    int B = batch;
    (void)B;

    ModSplit img_mod1, img_mod2;
    split_mod(ctx, mod_img_raw, H, 1, img_mod1, img_mod2);

    ModSplit txt_mod1, txt_mod2;
    split_mod(ctx, mod_txt_raw, H, 1, txt_mod1, txt_mod2);

    std::vector<ggml_tensor *> pe_cache;
    (void)pe_cache;

    for (int d = 0; d < params.depth; d++) {
        const auto & db = weights.double_blocks[d];

        ggml_tensor * img_n = ggml_rms_norm(ctx, h_img, 1e-6f);
        ggml_tensor * txt_n = ggml_rms_norm(ctx, h_txt, 1e-6f);

        img_n = ggml_add(ctx, ggml_mul(ctx, img_n, ggml_add(ctx, ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod1.scale, H, 0)), img_n), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1))), ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod1.shift, H, 0)), img_n));

        txt_n = ggml_add(ctx, ggml_mul(ctx, txt_n, ggml_add(ctx, ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.scale, H, 0)), txt_n), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1))), ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.shift, H, 0)), txt_n));

        ggml_tensor * img_q = b1(img_n, db.attn_to_q);
        ggml_tensor * img_k = b1(img_n, db.attn_to_k);
        ggml_tensor * img_v = b1(img_n, db.attn_to_v);

        ggml_tensor * txt_q = b1(txt_n, db.attn_add_q);
        ggml_tensor * txt_k = b1(txt_n, db.attn_add_k);
        ggml_tensor * txt_v = b1(txt_n, db.attn_add_v);

        img_q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, img_q, head_dim, n_heads, img_tokens), 0, 2, 1, 3));
        img_k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, img_k, head_dim, n_heads, img_tokens), 0, 2, 1, 3));
        img_v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, img_v, head_dim, n_heads, img_tokens), 0, 2, 1, 3));

        int img_t = img_tokens;
        int txt_t = txt_tokens;

        ggml_tensor * pe_ctx_fake = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim / 2, txt_t);

        txt_q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, txt_q, head_dim, n_heads, txt_t), 0, 2, 1, 3));
        txt_k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, txt_k, head_dim, n_heads, txt_t), 0, 2, 1, 3));
        txt_v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, txt_v, head_dim, n_heads, txt_t), 0, 2, 1, 3));

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_heads, txt_t + img_t, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_heads, txt_t + img_t, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_heads, txt_t + img_t, 1);

        q = ggml_cpy(ctx, txt_q, ggml_view_4d(ctx, q, head_dim, n_heads, txt_t, 1, q->nb[1], q->nb[2], q->nb[3], 0));
        q = ggml_cpy(ctx, img_q, ggml_view_4d(ctx, q, head_dim, n_heads, img_t, 1, q->nb[1], q->nb[2], q->nb[3], txt_t * head_dim * sizeof(float)));

        k = ggml_cpy(ctx, txt_k, ggml_view_4d(ctx, k, head_dim, n_heads, txt_t, 1, k->nb[1], k->nb[2], k->nb[3], 0));
        k = ggml_cpy(ctx, img_k, ggml_view_4d(ctx, k, head_dim, n_heads, img_t, 1, k->nb[1], k->nb[2], k->nb[3], txt_t * head_dim * sizeof(float)));

        v = ggml_cpy(ctx, txt_v, ggml_view_4d(ctx, v, head_dim, n_heads, txt_t, 1, v->nb[1], v->nb[2], v->nb[3], 0));
        v = ggml_cpy(ctx, img_v, ggml_view_4d(ctx, v, head_dim, n_heads, img_t, 1, v->nb[1], v->nb[2], v->nb[3], txt_t * head_dim * sizeof(float)));

        (void)pe_ctx_fake;

        ggml_tensor * q_2d = ggml_cont(ctx, ggml_reshape_2d(ctx, q, head_dim, (txt_t + img_t) * n_heads));
        ggml_tensor * k_2d = ggml_cont(ctx, ggml_reshape_2d(ctx, k, head_dim, (txt_t + img_t) * n_heads));
        ggml_tensor * v_2d = ggml_cont(ctx, ggml_reshape_2d(ctx, v, head_dim, (txt_t + img_t) * n_heads));

        ggml_tensor * scores = ggml_mul_mat(ctx, k_2d, q_2d);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);

        ggml_tensor * attn = ggml_mul_mat(ctx, v_2d, scores);
        attn = ggml_cont(ctx, ggml_reshape_4d(ctx, attn, head_dim, txt_t + img_t, n_heads, 1));
        attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
        attn = ggml_cont(ctx, ggml_reshape_2d(ctx, attn, H, txt_t + img_t));

        ggml_tensor * img_attn_out = ggml_view_2d(ctx, attn, H, img_t, attn->nb[1], txt_t * H * sizeof(float));
        ggml_tensor * txt_attn_out = ggml_view_2d(ctx, attn, H, txt_t, attn->nb[1], 0);

        img_attn_out = b1(img_attn_out, db.attn_to_out);
        txt_attn_out = b1(txt_attn_out, db.attn_add_out);

        h_img = ggml_add(ctx, h_img, ggml_mul(ctx, ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod1.gate, H, 0)), img_attn_out), img_attn_out));
        h_txt = ggml_add(ctx, h_txt, ggml_mul(ctx, ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.gate, H, 0)), txt_attn_out), txt_attn_out));

        ggml_tensor * img_ff_n = ggml_rms_norm(ctx, h_img, 1e-6f);
        ggml_tensor * txt_ff_n = ggml_rms_norm(ctx, h_txt, 1e-6f);

        img_ff_n = ggml_add(ctx, ggml_mul(ctx, img_ff_n, ggml_add(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.scale, H, 0))), img_ff_n), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1))), ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.shift, H, 0))), img_ff_n));
        txt_ff_n = ggml_add(ctx, ggml_mul(ctx, txt_ff_n, ggml_add(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.scale, H, 0))), txt_ff_n), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1))), ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.shift, H, 0))), txt_ff_n));

        ggml_tensor * img_mlp_in = b1(img_ff_n, db.ff_linear_in);
        ggml_tensor * txt_mlp_in = b1(txt_ff_n, db.ff_ctx_linear_in);

        int mlp_s = mlp_hd * 2;
        ggml_tensor * img_mlp_g = ggml_view_2d(ctx, img_mlp_in, mlp_hd, img_t, img_mlp_in->nb[1], 0);
        ggml_tensor * img_mlp_x = ggml_view_2d(ctx, img_mlp_in, mlp_hd, img_t, img_mlp_in->nb[1], mlp_hd * sizeof(float));
        ggml_tensor * img_mlp_a = ggml_mul(ctx, ggml_silu(ctx, img_mlp_g), img_mlp_x);
        img_mlp_a = b1(img_mlp_a, db.ff_linear_out);

        ggml_tensor * txt_mlp_g = ggml_view_2d(ctx, txt_mlp_in, mlp_hd, txt_t, txt_mlp_in->nb[1], 0);
        ggml_tensor * txt_mlp_x = ggml_view_2d(ctx, txt_mlp_in, mlp_hd, txt_t, txt_mlp_in->nb[1], mlp_hd * sizeof(float));
        ggml_tensor * txt_mlp_a = ggml_mul(ctx, ggml_silu(ctx, txt_mlp_g), txt_mlp_x);
        txt_mlp_a = b1(txt_mlp_a, db.ff_ctx_linear_out);

        h_img = ggml_add(ctx, h_img, ggml_mul(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.gate, H, 0))), img_mlp_a), img_mlp_a));
        h_txt = ggml_add(ctx, h_txt, ggml_mul(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.gate, H, 0))), txt_mlp_a), txt_mlp_a));
    }

    ggml_tensor * combined = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, (txt_tokens + img_tokens));
    combined = ggml_cpy(ctx, h_txt, ggml_view_2d(ctx, combined, H, txt_tokens, combined->nb[1], 0));
    combined = ggml_cpy(ctx, h_img, ggml_view_2d(ctx, combined, H, img_tokens, combined->nb[1], txt_tokens * H * sizeof(float)));

    ggml_tensor * mod_single_raw = b1(vec, weights.single_mod);
    ModSplit single_mod;
    split_mod_single(ctx, mod_single_raw, H, 1, single_mod);

    for (int s = 0; s < params.depth_single_blocks; s++) {
        const auto & sb = weights.single_blocks[s];

        ggml_tensor * x_n = ggml_rms_norm(ctx, combined, 1e-6f);
        x_n = ggml_add(ctx, ggml_mul(ctx, x_n, ggml_add(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, single_mod.scale, H, 0))), x_n), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1))), ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, single_mod.shift, H, 0))), x_n));

        ggml_tensor * proj_all = b1(x_n, sb.to_qkv_mlp_proj);
        int qkv_dim = 3 * H;
        ggml_tensor * qkv_t = ggml_view_2d(ctx, proj_all, qkv_dim, total_tokens, proj_all->nb[1], 0);
        ggml_tensor * mlp_t = ggml_view_2d(ctx, proj_all, mlp_hd * 2, total_tokens, proj_all->nb[1], qkv_dim * sizeof(float));

        QKV qkv = split_qkv(ctx, qkv_t, H, n_heads, total_tokens);

        int seq = total_tokens;
        ggml_tensor * q_2d_a = ggml_cont(ctx, ggml_permute(ctx, qkv.q, 0, 2, 1, 3));
        ggml_tensor * k_2d_a = ggml_cont(ctx, ggml_permute(ctx, qkv.k, 0, 2, 1, 3));
        ggml_tensor * v_2d_a = ggml_cont(ctx, ggml_permute(ctx, qkv.v, 0, 2, 1, 3));

        q_2d_a = ggml_cont(ctx, ggml_reshape_2d(ctx, q_2d_a, head_dim, seq * n_heads));
        k_2d_a = ggml_cont(ctx, ggml_reshape_2d(ctx, k_2d_a, head_dim, seq * n_heads));
        v_2d_a = ggml_cont(ctx, ggml_reshape_2d(ctx, v_2d_a, head_dim, seq * n_heads));

        ggml_tensor * s_scores = ggml_mul_mat(ctx, k_2d_a, q_2d_a);
        float scale_s = 1.0f / std::sqrt(static_cast<float>(head_dim));
        s_scores = ggml_scale(ctx, s_scores, scale_s);
        s_scores = ggml_soft_max_ext(ctx, s_scores, nullptr, 1.0f, 0.0f);

        ggml_tensor * s_attn = ggml_mul_mat(ctx, v_2d_a, s_scores);
        s_attn = ggml_cont(ctx, ggml_reshape_4d(ctx, s_attn, head_dim, seq, n_heads, 1));
        s_attn = ggml_permute(ctx, s_attn, 0, 2, 1, 3);
        s_attn = ggml_cont(ctx, ggml_reshape_2d(ctx, s_attn, H, seq));

        ggml_tensor * s_mlp_a = mlp_act(ctx, mlp_t, mlp_hd);
        ggml_tensor * s_cat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H + mlp_hd, seq);
        s_cat = ggml_cpy(ctx, s_attn, ggml_view_2d(ctx, s_cat, H, seq, s_cat->nb[1], 0));
        s_cat = ggml_cpy(ctx, s_mlp_a, ggml_view_2d(ctx, s_cat, mlp_hd, seq, s_cat->nb[1], H * sizeof(float)));

        ggml_tensor * s_out = b1(s_cat, sb.to_out);

        combined = ggml_add(ctx, combined, ggml_mul(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, single_mod.gate, H, 0))), s_out), s_out));
    }

    ggml_tensor * final_img = ggml_view_2d(ctx, combined, H, img_tokens, combined->nb[1], txt_tokens * H * sizeof(float));

    ggml_tensor * fn = ggml_rms_norm(ctx, final_img, 1e-6f);
    ggml_tensor * mod_out_raw = b1(vec, weights.norm_out_linear);
    ggml_tensor * mod_out_shift = ggml_view_2d(ctx, mod_out_raw, H, 1, mod_out_raw->nb[1], 0);
    ggml_tensor * mod_out_scale = ggml_view_2d(ctx, mod_out_raw, H, 1, mod_out_raw->nb[1], H * sizeof(float));

    fn = ggml_add(ctx, ggml_mul(ctx, fn, ggml_add(ctx, ggml_repeat(ctx, column_1d(ctx, mod_out_scale), fn), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1))), ggml_repeat(ctx, column_1d(ctx, mod_out_shift), fn));

    result.out = b1(fn, weights.proj_out);
    ggml_set_output(result.out);

    ggml_build_forward_expand(result.graph, result.out);

    return result;
}

} // namespace bonsai
