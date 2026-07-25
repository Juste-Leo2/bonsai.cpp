#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "diffuser_types.h"
#include "b1_0_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace bonsai;

bool bonsai::g_bonsai_debug = false;

static void f32_write(const char *path, const ggml_tensor *t) {
    FILE *fp = fopen(path, "wb");
    fwrite(t->data, 1, ggml_nbytes(t), fp);
    fclose(fp);
}

static ggml_tensor * column_1d(ggml_context * ctx, ggml_tensor * b) {
    ggml_tensor * v = ggml_reshape_4d(ctx, b, b->ne[0], 1, 1, 1);
    return ggml_cont(ctx, v);
}

static ggml_tensor * load_f32(ggml_context *ctx, const char *path, int ne0, int ne1) {
    ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    t->data = malloc(ne0 * ne1 * sizeof(float));
    size_t expected = ne0 * ne1 * sizeof(float);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", path); abort(); }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size != expected) { fprintf(stderr, "FATAL: %s size %zu != %zu\n", path, size, expected); abort(); }
    fread(t->data, 1, size, f);
    fclose(f);
    return t;
}

static ggml_tensor * load_f32_1d(ggml_context *ctx, const char *path, int n) {
    ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    t->data = malloc(n * sizeof(float));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", path); abort(); }
    fread(t->data, sizeof(float), n, f);
    fclose(f);
    return t;
}

static ggml_tensor * load_b1_weight(ggml_context *ctx, const char *path, int in_dim, int out_dim) {
    const int block_size = 32;
    const int block_bytes = 6;
    int n_blocks = in_dim / block_size;
    size_t nbytes = out_dim * n_blocks * block_bytes;
    ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, nbytes, 1);
    t->data = malloc(nbytes);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", path); abort(); }
    fread(t->data, 1, nbytes, f);
    fclose(f);
    return t;
}


struct ModSplit {
    ggml_tensor * shift;
    ggml_tensor * scale;
    ggml_tensor * gate;
};

static void split_mod(ggml_context * ctx, ggml_tensor * mod, int H, ModSplit & s1, ModSplit & s2) {
    ggml_tensor * parts[6];
    for (int i = 0; i < 6; i++) {
        parts[i] = ggml_view_2d(ctx, mod, H, 1, mod->nb[1], i * H * sizeof(float));
    }
    s1 = {parts[0], parts[1], parts[2]};
    s2 = {parts[3], parts[4], parts[5]};
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
    rope_ud.push_back({0x524F5045, head_dim, n_heads, seq});
    *q_out = rope_2d_fwd(ctx, q, cos_t, sin_t, rope_ud.back());
    rope_ud.push_back({0x524F5045, head_dim, n_heads, seq});
    *k_out = rope_2d_fwd(ctx, k, cos_t, sin_t, rope_ud.back());
}

int main(int argc, char **argv) {
    int H = 3072, C = 128, ctx_dim = 7680, n_heads = 24, head_dim = 128;
    int img_tokens = 4096, txt_tokens = 512;
    int mlp_hd = H * 3;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--threads" && i+1<argc) n_threads = std::stoi(argv[++i]);
    }

    fprintf(stderr, "Init context...\n");
    struct ggml_init_params cparams = { 256ULL*1024*1024, NULL, true };
    ggml_context *ctx = ggml_init(cparams);

    // ==== Load inputs ====
    fprintf(stderr, "Loading inputs...\n");
    ggml_tensor *h_img_in = load_f32(ctx, "h_img.bin", H, img_tokens);
    ggml_tensor *h_txt_in = load_f32(ctx, "h_txt.bin", H, txt_tokens);
    ggml_tensor *mod_img = load_f32(ctx, "mod_img.bin", 1, 6*H);
    ggml_tensor *mod_txt = load_f32(ctx, "mod_txt.bin", 1, 6*H);
    ggml_tensor *cos_img = load_f32(ctx, "cos_img.bin", 64, img_tokens);
    ggml_tensor *sin_img = load_f32(ctx, "sin_img.bin", 64, img_tokens);
    ggml_tensor *cos_txt = load_f32(ctx, "cos_txt.bin", 64, txt_tokens);
    ggml_tensor *sin_txt = load_f32(ctx, "sin_txt.bin", 64, txt_tokens);

    // ==== Load weights ====
    fprintf(stderr, "Loading weights...\n");
    B1Weights db_to_q       = { load_b1_weight(ctx, "w_attn.to_q.weight", H, H), H, H };
    B1Weights db_to_k       = { load_b1_weight(ctx, "w_attn.to_k.weight", H, H), H, H };
    B1Weights db_to_v       = { load_b1_weight(ctx, "w_attn.to_v.weight", H, H), H, H };
    B1Weights db_to_out     = { load_b1_weight(ctx, "w_attn.to_out.0.weight", H, H), H, H };
    B1Weights db_add_q      = { load_b1_weight(ctx, "w_attn.add_q_proj.weight", H, H), H, H };
    B1Weights db_add_k      = { load_b1_weight(ctx, "w_attn.add_k_proj.weight", H, H), H, H };
    B1Weights db_add_v      = { load_b1_weight(ctx, "w_attn.add_v_proj.weight", H, H), H, H };
    B1Weights db_add_out    = { load_b1_weight(ctx, "w_attn.to_add_out.weight", H, H), H, H };
    B1Weights db_ff_in      = { load_b1_weight(ctx, "w_ff.linear_in.weight", H, mlp_hd*2), H, mlp_hd*2 };
    B1Weights db_ff_out     = { load_b1_weight(ctx, "w_ff.linear_out.weight", mlp_hd, H), mlp_hd, H };
    B1Weights db_ff_ctx_in  = { load_b1_weight(ctx, "w_ff_context.linear_in.weight", H, mlp_hd*2), H, mlp_hd*2 };
    B1Weights db_ff_ctx_out = { load_b1_weight(ctx, "w_ff_context.linear_out.weight", mlp_hd, H), mlp_hd, H };

    ggml_tensor *norm_q   = load_f32_1d(ctx, "w_attn.norm_q.weight", head_dim);
    ggml_tensor *norm_k   = load_f32_1d(ctx, "w_attn.norm_k.weight", head_dim);
    ggml_tensor *norm_aq  = load_f32_1d(ctx, "w_attn.norm_added_q.weight", head_dim);
    ggml_tensor *norm_ak  = load_f32_1d(ctx, "w_attn.norm_added_k.weight", head_dim);

    // ==== Setup user data ====
    std::vector<B1LinearUserData> b1_ud;
    std::vector<Rope2DUserData> rope_ud;
    b1_ud.reserve(16);
    rope_ud.reserve(8);

    auto b1 = [&](ggml_tensor * act, B1Weights & w) -> ggml_tensor * {
        b1_ud.push_back({0x31423142, w.in_dim, w.out_dim});
        return b1_linear(ctx, act, w, n_threads, b1_ud.back());
    };

    ggml_tensor *h_img = h_img_in;
    ggml_tensor *h_txt = h_txt_in;

    ModSplit img_mod1, img_mod2;
    split_mod(ctx, mod_img, H, img_mod1, img_mod2);
    ModSplit txt_mod1, txt_mod2;
    split_mod(ctx, mod_txt, H, txt_mod1, txt_mod2);

    // ==== Build double block graph ====
    fprintf(stderr, "Building graph...\n");

    ggml_tensor *one_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor *ones = ggml_fill(ctx, one_tensor, 1.0f);

    // --- Attention sub-block ---
    ggml_tensor * img_n = ggml_norm(ctx, h_img, 1e-6f);
    ggml_tensor * txt_n = ggml_norm(ctx, h_txt, 1e-6f);

    img_n = ggml_add(ctx,
        ggml_mul(ctx, img_n, ggml_add(ctx,
            ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod1.scale, H, 0)), img_n), ones)),
        ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod1.shift, H, 0)), img_n));

    txt_n = ggml_add(ctx,
        ggml_mul(ctx, txt_n, ggml_add(ctx,
            ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.scale, H, 0)), txt_n), ones)),
        ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.shift, H, 0)), txt_n));

    ggml_tensor * img_q = b1(img_n, db_to_q);
    ggml_tensor * img_k = b1(img_n, db_to_k);
    ggml_tensor * img_v = b1(img_n, db_to_v);

    ggml_tensor * txt_q = b1(txt_n, db_add_q);
    ggml_tensor * txt_k = b1(txt_n, db_add_k);
    ggml_tensor * txt_v = b1(txt_n, db_add_v);

    img_q = ggml_cont(ctx, ggml_reshape_3d(ctx, img_q, head_dim, n_heads, img_tokens));
    img_k = ggml_cont(ctx, ggml_reshape_3d(ctx, img_k, head_dim, n_heads, img_tokens));
    img_v = ggml_cont(ctx, ggml_reshape_3d(ctx, img_v, head_dim, n_heads, img_tokens));

    txt_q = ggml_cont(ctx, ggml_reshape_3d(ctx, txt_q, head_dim, n_heads, txt_tokens));
    txt_k = ggml_cont(ctx, ggml_reshape_3d(ctx, txt_k, head_dim, n_heads, txt_tokens));
    txt_v = ggml_cont(ctx, ggml_reshape_3d(ctx, txt_v, head_dim, n_heads, txt_tokens));

    img_q = rms_norm_qk(ctx, img_q, norm_q, head_dim);
    img_k = rms_norm_qk(ctx, img_k, norm_k, head_dim);
    txt_q = rms_norm_qk(ctx, txt_q, norm_aq, head_dim);
    txt_k = rms_norm_qk(ctx, txt_k, norm_ak, head_dim);

    ggml_tensor * img_q_r = nullptr, *img_k_r = nullptr, *txt_q_r = nullptr, *txt_k_r = nullptr;
    apply_rope_2d(ctx, img_q, img_k, cos_img, sin_img, head_dim, n_threads, rope_ud, &img_q_r, &img_k_r);
    apply_rope_2d(ctx, txt_q, txt_k, cos_txt, sin_txt, head_dim, n_threads, rope_ud, &txt_q_r, &txt_k_r);
    img_q = img_q_r; img_k = img_k_r; txt_q = txt_q_r; txt_k = txt_k_r;

    int img_t = img_tokens, txt_t = txt_tokens;
    ggml_tensor * q = ggml_concat(ctx, txt_q, img_q, 2);
    ggml_tensor * k = ggml_concat(ctx, txt_k, img_k, 2);
    ggml_tensor * v = ggml_concat(ctx, txt_v, img_v, 2);

    ggml_tensor * q_3d = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_3d = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_3d = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    ggml_tensor * attn;
#if 1
    ggml_tensor * k_3d_f16 = k_3d;
    ggml_tensor * v_3d_f16 = v_3d;
    if (k_3d_f16->type == GGML_TYPE_F32) k_3d_f16 = ggml_cast(ctx, k_3d_f16, GGML_TYPE_F16);
    if (v_3d_f16->type == GGML_TYPE_F32) v_3d_f16 = ggml_cast(ctx, v_3d_f16, GGML_TYPE_F16);
    attn = ggml_flash_attn_ext(ctx, q_3d, k_3d_f16, v_3d_f16, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(ctx, attn, H, txt_t + img_t);
#else
    ggml_tensor * scores = ggml_mul_mat(ctx, k_3d, q_3d);
    scores = ggml_scale_inplace(ctx, scores, scale);
    scores = ggml_soft_max_inplace(ctx, scores);
    ggml_tensor * v_3d_t = ggml_cont(ctx, ggml_permute(ctx, v_3d, 1, 0, 2, 3));
    attn = ggml_mul_mat(ctx, v_3d_t, scores);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, H, txt_t + img_t);
#endif

    ggml_tensor * img_attn_out = ggml_view_2d(ctx, attn, H, img_t, attn->nb[1], txt_t * H * sizeof(float));
    ggml_tensor * txt_attn_out = ggml_view_2d(ctx, attn, H, txt_t, attn->nb[1], 0);

    txt_attn_out = b1(txt_attn_out, db_add_out);
    img_attn_out = b1(img_attn_out, db_to_out);

    ggml_tensor * mod_gate = ggml_view_1d(ctx, img_mod1.gate, H, 0);
    h_img = ggml_add(ctx, h_img, ggml_mul(ctx, ggml_repeat(ctx, ggml_cont(ctx, mod_gate), img_attn_out), img_attn_out));
    h_txt = ggml_add(ctx, h_txt, ggml_mul(ctx, ggml_repeat(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod1.gate, H, 0)), txt_attn_out), txt_attn_out));

    // --- FF sub-block ---
    ggml_tensor * img_ff_n = ggml_norm(ctx, h_img, 1e-6f);
    ggml_tensor * txt_ff_n = ggml_norm(ctx, h_txt, 1e-6f);

    img_ff_n = ggml_add(ctx,
        ggml_mul(ctx, img_ff_n, ggml_add(ctx,
            ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.scale, H, 0))), img_ff_n), ones)),
        ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.shift, H, 0))), img_ff_n));

    txt_ff_n = ggml_add(ctx,
        ggml_mul(ctx, txt_ff_n, ggml_add(ctx,
            ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.scale, H, 0))), txt_ff_n), ones)),
        ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.shift, H, 0))), txt_ff_n));

    ggml_tensor * img_mlp_in = b1(img_ff_n, db_ff_in);
    ggml_tensor * txt_mlp_in = b1(txt_ff_n, db_ff_ctx_in);

    ggml_tensor * img_mlp_a = ggml_swiglu(ctx, img_mlp_in);
    img_mlp_a = b1(img_mlp_a, db_ff_out);

    ggml_tensor * txt_mlp_a = ggml_swiglu(ctx, txt_mlp_in);
    txt_mlp_a = b1(txt_mlp_a, db_ff_ctx_out);

    h_img = ggml_add(ctx, h_img, ggml_mul(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, img_mod2.gate, H, 0))), img_mlp_a), img_mlp_a));
    h_txt = ggml_add(ctx, h_txt, ggml_mul(ctx, ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, txt_mod2.gate, H, 0))), txt_mlp_a), txt_mlp_a));

    ggml_set_output(h_img);
    ggml_set_output(h_txt);

    // ==== Compute ====
    fprintf(stderr, "Building forward graph...\n");
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 65536, false);
    ggml_build_forward_expand(graph, h_img);
    ggml_build_forward_expand(graph, h_txt);

    fprintf(stderr, "Init backend...\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    fprintf(stderr, "Computing...\n");
    ggml_backend_graph_compute(backend, graph);
    fprintf(stderr, "Done!\n");

    f32_write("h_img_out.bin", h_img);
    f32_write("h_txt_out.bin", h_txt);

    fprintf(stderr, "Saved outputs: h_img=[%lld,%lld] (%lld bytes) h_txt=[%lld,%lld] (%lld bytes)\n",
        (long long)h_img->ne[0], (long long)h_img->ne[1], (long long)ggml_nbytes(h_img),
        (long long)h_txt->ne[0], (long long)h_txt->ne[1], (long long)ggml_nbytes(h_txt));

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
