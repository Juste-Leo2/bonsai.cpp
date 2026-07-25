/**
 * WebGPU diffuser component tests: GPU vs CPU for all diffuser pipeline ops
 * Tests: input_proj, timestep+modulation, double_block, single_block, output_head
 * Uses reduced dimensions for speed. Full-dim tests in test_webgpu_ops.cpp
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include "diffuser_types.h"
#include "b1_0_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace bonsai;
namespace bonsai { bool g_bonsai_debug = false; }

// ═══ Helpers ═══

static float cosine(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) dot += (double)a[i]*b[i], na += (double)a[i]*a[i], nb += (double)b[i]*b[i];
    return (float)(dot / (sqrt(fmax(na, 1e-30)) * sqrt(fmax(nb, 1e-30))));
}

static void print_stats(const char *label, const float *v, int n) {
    double sum = 0, sum2 = 0;
    float vmin = v[0], vmax = v[0];
    for (int i = 0; i < n; i++) {
        sum += v[i];
        sum2 += (double)v[i] * v[i];
        if (v[i] < vmin) vmin = v[i];
        if (v[i] > vmax) vmax = v[i];
    }
    float mean = (float)(sum / n);
    float std  = (float)sqrt(sum2 / n - mean * mean);
    fprintf(stdout, "  %s mu=%.4f std=%.4f min=%.4f max=%.4f\n", label, mean, std, vmin, vmax);
    fflush(stdout);
}

static std::vector<uint8_t> pack_b1(const float *wgt, int out_dim, int in_dim) {
    int nb = in_dim / 32, rs = nb * 6;
    std::vector<uint8_t> out(out_dim * rs);
    for (int r = 0; r < out_dim; r++)
        for (int b = 0; b < nb; b++) {
            float scale = 0;
            for (int i = 0; i < 32; i++) scale += fabsf(wgt[r * in_dim + b * 32 + i]);
            scale /= 32.0f;
            uint16_t sb = ggml_fp32_to_fp16(scale);
            memcpy(&out[r * rs + b * 6], &sb, 2);
            uint32_t bits = 0;
            for (int i = 0; i < 32; i++)
                if (wgt[r * in_dim + b * 32 + i] >= 0.0f) bits |= (1u << i);
            memcpy(&out[r * rs + b * 6 + 2], &bits, 4);
        }
    return out;
}

static std::vector<float> gen_rand(int n, int seed) {
    srand(seed);
    std::vector<float> v(n);
    for (auto &x : v) x = (float)rand() / RAND_MAX * 2 - 1;
    return v;
}

static ggml_tensor *column_1d(ggml_context *ctx, ggml_tensor *b) {
    return ggml_cont(ctx, ggml_reshape_4d(ctx, b, b->ne[0], 1, 1, 1));
}

static ggml_tensor *rms_norm_qk(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, int hd) {
    int nh = x->ne[1], sq = x->ne[2];
    x = ggml_cont(ctx, ggml_reshape_3d(ctx, x, hd, nh, sq));
    x = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_tensor *wv = ggml_reshape_3d(ctx, w, hd, 1, 1);
    return ggml_cont(ctx, ggml_mul(ctx, x, ggml_repeat(ctx, wv, x)));
}

// ═══ Test: input_proj ═══
static bool test_input_proj(ggml_backend_t cpu, ggml_backend_t gpu) {
    int H = 1024, C = 64, ctxd = 512, it = 256, tt = 64;
    fprintf(stdout, "\n--- input_proj H=%d C=%d ctx=%d ---\n", H, C, ctxd); fflush(stdout);

    auto w_img_f = gen_rand(H * C, 100);
    auto w_txt_f = gen_rand(H * ctxd, 101);
    auto act     = gen_rand(C * it, 102);
    auto emb     = gen_rand(ctxd * tt, 103);
    auto b1_img  = pack_b1(w_img_f.data(), H, C);
    auto b1_txt  = pack_b1(w_txt_f.data(), H, ctxd);

    int n_h_img = H * it, n_h_txt = H * tt;
    auto run = [&](ggml_backend_t be) -> std::pair<std::vector<float>, std::vector<float>> {
        struct ggml_init_params gp = { 256*1024*1024, NULL, true };
        ggml_context *ctx = ggml_init(gp);
        ggml_tensor *w_i = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)b1_img.size(), 1);
        ggml_tensor *w_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)b1_txt.size(), 1);
        ggml_tensor *a_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, it);
        ggml_tensor *e_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ctxd, tt);
        B1LinearUserData ud1 = {0x31423142, C, H}, ud2 = {0x31423142, ctxd, H};
        B1Weights bw1 = {w_i, C, H}, bw2 = {w_t, ctxd, H};
        ggml_tensor *h_i = b1_linear(ctx, a_t, bw1, 4, ud1);
        ggml_tensor *h_t = b1_linear(ctx, e_t, bw2, 4, ud2);
        ggml_cgraph *g = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(g, h_i);
        ggml_build_forward_expand(g, h_t);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(w_i, b1_img.data(), 0, b1_img.size());
        ggml_backend_tensor_set(w_t, b1_txt.data(), 0, b1_txt.size());
        ggml_backend_tensor_set(a_t, act.data(), 0, act.size() * sizeof(float));
        ggml_backend_tensor_set(e_t, emb.data(), 0, emb.size() * sizeof(float));
        ggml_backend_graph_compute(be, g);
        std::vector<float> hi(n_h_img), ht(n_h_txt);
        ggml_backend_tensor_get(h_i, hi.data(), 0, n_h_img * sizeof(float));
        ggml_backend_tensor_get(h_t, ht.data(), 0, n_h_txt * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return {hi, ht};
    };

    auto [hi_c, ht_c] = run(cpu);
    auto [hi_g, ht_g] = run(gpu);
    print_stats("CPU img", hi_c.data(), n_h_img);
    print_stats("GPU img", hi_g.data(), n_h_img);
    print_stats("CPU txt", ht_c.data(), n_h_txt);
    print_stats("GPU txt", ht_g.data(), n_h_txt);
    float ci = cosine(hi_c.data(), hi_g.data(), n_h_img);
    float ct = cosine(ht_c.data(), ht_g.data(), n_h_txt);
    fprintf(stdout, "  cos_h_img=%.6f cos_h_txt=%.6f %s\n", ci, ct, (ci>0.9999f&&ct>0.9999f)?"PASS":"FAIL");
    return ci > 0.9999f && ct > 0.9999f;
}

// ═══ Test: output_head ═══
static bool test_output_head(ggml_backend_t cpu, ggml_backend_t gpu) {
    int H = 1024, C = 64, it = 256;
    fprintf(stdout, "\n--- output_head H=%d C=%d ---\n", H, C); fflush(stdout);

    auto h_f     = gen_rand(H * it, 200);
    auto w_out_f = gen_rand(H * C, 201);
    auto norm_w  = gen_rand(H, 202);
    auto b1_out  = pack_b1(w_out_f.data(), C, H);

    int n_out = C * it;
    auto run = [&](ggml_backend_t be) -> std::vector<float> {
        struct ggml_init_params gp = { 256*1024*1024, NULL, true };
        ggml_context *ctx = ggml_init(gp);
        ggml_tensor *h_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, it);
        ggml_tensor *n_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor *w_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)b1_out.size(), 1);
        ggml_tensor *xn  = ggml_norm(ctx, h_t, 1e-6f);
        ggml_tensor *n_r = ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, ggml_view_1d(ctx, n_t, H, 0))), xn);
        ggml_tensor *xo  = ggml_mul(ctx, xn, n_r);
        B1LinearUserData ud = {0x31423142, H, C};
        B1Weights bw = {w_t, H, C};
        ggml_tensor *out = b1_linear(ctx, xo, bw, 4, ud);
        ggml_cgraph *g = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(g, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(h_t, h_f.data(), 0, h_f.size() * sizeof(float));
        ggml_backend_tensor_set(n_t, norm_w.data(), 0, H * sizeof(float));
        ggml_backend_tensor_set(w_t, b1_out.data(), 0, b1_out.size());
        ggml_backend_graph_compute(be, g);
        std::vector<float> r(n_out);
        ggml_backend_tensor_get(out, r.data(), 0, n_out * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return r;
    };

    auto c = run(cpu);
    auto g = run(gpu);
    print_stats("CPU", c.data(), n_out);
    print_stats("GPU", g.data(), n_out);
    float cs = cosine(c.data(), g.data(), n_out);
    fprintf(stdout, "  cos=%.6f %s\n", cs, cs>0.9999f?"PASS":"FAIL");
    return cs > 0.9999f;
}

// ═══ Test: timestep_embedding + modulation (manual ops, no composite) ═══
static bool test_timestep(ggml_backend_t cpu, ggml_backend_t gpu) {
    int H = 1024, hp = 128;
    int half = hp / 2;
    fprintf(stdout, "\n--- timestep+mod H=%d hp=%d ---\n", H, hp); fflush(stdout);

    auto te_w1 = gen_rand(2*hp * hp, 300);
    auto te_w2 = gen_rand(H * 2*hp, 301);
    auto mod_w = gen_rand(6*H * H, 302);
    float t_val = 1000.0f;

    int n_mod = 6 * H;
    auto run = [&](ggml_backend_t be) -> std::vector<float> {
        struct ggml_init_params gp = { 256*1024*1024, NULL, true };
        ggml_context *ctx = ggml_init(gp);
        ggml_tensor *t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

        // manual timestep embedding: exp(-arange(half)*log(10000)/half), cos/sin, concat
        ggml_tensor *freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, half);
        ggml_tensor *ones_v = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), 1.0f);
        // We can't easily do arange in pure ggml. Use mul_mat with a pre-computed frequency table instead.
        // Simpler: pre-compute freqs as input data and use cos/sin directly
        // Actually, let's just use individual ops: t * freqs, cos, sin, concat

        // For simplicity, we'll hardcode the timestep embedding as a weight multiplication
        // The key ops tested: mul_mat, silu, reshape
        ggml_tensor *l1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp, 2*hp);
        // te = timestep_embedding is replaced by a direct matmul with a "freqs" matrix
        // We simulate by creating a 2D input (hp, 1) filled with the timestep value * frequency pattern
        // Actually, let's use a known input that exercises mul_mat, silu, etc.

        ggml_tensor *te_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp, 1);
        ggml_tensor *te_m  = ggml_mul_mat(ctx, l1_w, te_in);
        ggml_tensor *te_s  = ggml_silu(ctx, te_m);
        ggml_tensor *l2_w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*hp, H);
        ggml_tensor *vec   = ggml_mul_mat(ctx, l2_w, te_s);
        ggml_tensor *vec_s = ggml_silu(ctx, vec);
        ggml_tensor *mod_wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 6*H);
        ggml_tensor *mod    = ggml_mul_mat(ctx, mod_wt, vec_s);
        mod = ggml_cont(ctx, ggml_reshape_2d(ctx, mod, 1, 6*H));

        ggml_cgraph *g = ggml_new_graph_custom(ctx, 512, false);
        ggml_build_forward_expand(g, mod);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

        // Fill te_in with pattern simulating timestep embedding output
        std::vector<float> te_data(hp);
        for (int i = 0; i < half; i++) {
            float freq = expf(-i * logf(10000.0f) / half);
            te_data[i] = cosf(t_val * freq);
            te_data[half + i] = sinf(t_val * freq);
        }
        ggml_backend_tensor_set(te_in, te_data.data(), 0, hp * sizeof(float));
        ggml_backend_tensor_set(l1_w, te_w1.data(), 0, te_w1.size() * sizeof(float));
        ggml_backend_tensor_set(l2_w, te_w2.data(), 0, te_w2.size() * sizeof(float));
        ggml_backend_tensor_set(mod_wt, mod_w.data(), 0, mod_w.size() * sizeof(float));
        ggml_backend_graph_compute(be, g);
        std::vector<float> r(n_mod);
        ggml_backend_tensor_get(mod, r.data(), 0, n_mod * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return r;
    };

    auto c = run(cpu);
    auto g = run(gpu);
    print_stats("CPU", c.data(), n_mod);
    print_stats("GPU", g.data(), n_mod);
    float cs = cosine(c.data(), g.data(), n_mod);
    fprintf(stdout, "  cos=%.6f %s\n", cs, cs>0.9999f?"PASS":"FAIL");
    return cs > 0.9999f;
}

// ═══ Test: single_block (reduced dims: H = hd * nh) ═══
static bool test_single_block(ggml_backend_t cpu, ggml_backend_t gpu) {
    int hd = 64, nh = 6, H = hd * nh;
    int C = 32, ctxd = 256, it = 64, tt = 32;
    int ms = H * 3, total_t = it + tt;
    fprintf(stdout, "\n--- single_block H=%d nh=%d total=%d ---\n", H, nh, total_t); fflush(stdout);

    auto combined = gen_rand(H * total_t, 400);
    auto mod_s    = gen_rand(3 * H, 401);
    auto c_cos    = gen_rand((hd/2) * total_t, 402);
    auto c_sin    = gen_rand((hd/2) * total_t, 403);
    auto nq_w     = gen_rand(hd, 404);
    auto nk_w     = gen_rand(hd, 405);
    auto w_qkv_mlp_f = gen_rand((3*H + ms*2) * H, 406);
    auto w_out_f     = gen_rand(H * (H + ms), 407);
    auto b1_qkv      = pack_b1(w_qkv_mlp_f.data(), 3*H + ms*2, H);
    auto b1_out      = pack_b1(w_out_f.data(), H, H + ms);

    auto run = [&](ggml_backend_t be) -> std::vector<float> {
        struct ggml_init_params gp = { 256*1024*1024, NULL, true };
        ggml_context *ctx = ggml_init(gp);
        ggml_tensor *comb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, total_t);
        ggml_tensor *m1d  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 3*H);
        ggml_tensor *c_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd/2, total_t);
        ggml_tensor *s_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd/2, total_t);
        ggml_tensor *n_q  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor *n_k  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor *w_qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)b1_qkv.size(), 1);
        ggml_tensor *w_o   = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)b1_out.size(), 1);

        B1LinearUserData ud1 = {0x31423142, H, 3*H+ms*2}, ud2 = {0x31423142, H+ms, H};
        B1Weights bw1 = {w_qkv, H, 3*H+ms*2}, bw2 = {w_o, H+ms, H};

        // modulation
        ggml_tensor *ones = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), 1.0f);
        ggml_tensor *m_sh = ggml_view_2d(ctx, m1d, H, 1, m1d->nb[1], 0);
        ggml_tensor *m_sc = ggml_view_2d(ctx, m1d, H, 1, m1d->nb[1], H*sizeof(float));
        ggml_tensor *m_ga = ggml_view_2d(ctx, m1d, H, 1, m1d->nb[1], 2*H*sizeof(float));
        ggml_tensor *xn = ggml_norm(ctx, comb, 1e-6f);
        xn = ggml_add(ctx, ggml_mul(ctx, xn, ggml_add(ctx,
            ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, m_sc)), xn), ones)),
            ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, m_sh)), xn));

        // qkv_mlp projection
        ggml_tensor *pa = b1_linear(ctx, xn, bw1, 4, ud1);
        int qkv_dim = 3 * H;
        ggml_tensor *qkv_t = ggml_view_2d(ctx, pa, qkv_dim, total_t, pa->nb[1], 0);
        ggml_tensor *mlp_t = ggml_view_2d(ctx, pa, ms*2, total_t, pa->nb[1], qkv_dim*sizeof(float));

        // split qkv
        auto vh = [&](ggml_tensor *t, int off) {
            return ggml_view_2d(ctx, t, H, total_t, t->nb[1], off*H*sizeof(float)); };
        ggml_tensor *qt = vh(qkv_t, 0), *kt = vh(qkv_t, 1), *vt = vh(qkv_t, 2);
        auto rh = [&](ggml_tensor *t) {
            return ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, t), hd, nh, total_t)); };
        ggml_tensor *q3 = rh(qt), *k3 = rh(kt), *v3 = rh(vt);

        q3 = rms_norm_qk(ctx, q3, n_q, hd);
        k3 = rms_norm_qk(ctx, k3, n_k, hd);

        // rope
        std::vector<Rope2DUserData> rod; rod.reserve(8);
        rod.push_back({0x524F5045, hd, nh, total_t});
        q3 = rope_2d_fwd(ctx, q3, c_t, s_t, rod.back());
        rod.push_back({0x524F5045, hd, nh, total_t});
        k3 = rope_2d_fwd(ctx, k3, c_t, s_t, rod.back());
        v3 = ggml_cont(ctx, v3);

        // attention
        ggml_tensor *qp = ggml_cont(ctx, ggml_permute(ctx, q3, 0, 2, 1, 3));
        ggml_tensor *kp = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));
        ggml_tensor *vp = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));
        float scale = 1.0f / sqrtf((float)hd);
        ggml_tensor *kf = ggml_cast(ctx, kp, GGML_TYPE_F16);
        ggml_tensor *vf = ggml_cast(ctx, vp, GGML_TYPE_F16);
        ggml_tensor *attn = ggml_flash_attn_ext(ctx, qp, kf, vf, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(ctx, attn, H, total_t);

        // mlp + output
        ggml_tensor *smlp = ggml_swiglu(ctx, mlp_t);
        ggml_tensor *scat = ggml_concat(ctx, attn, smlp, 0);
        ggml_tensor *sout = b1_linear(ctx, scat, bw2, 4, ud2);
        ggml_tensor *out  = ggml_add(ctx, comb, ggml_mul(ctx,
            ggml_repeat(ctx, column_1d(ctx, ggml_cont(ctx, m_ga)), sout), sout));

        ggml_cgraph *g = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(g, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(comb, combined.data(), 0, combined.size() * sizeof(float));
        ggml_backend_tensor_set(m1d, mod_s.data(), 0, mod_s.size() * sizeof(float));
        ggml_backend_tensor_set(c_t, c_cos.data(), 0, c_cos.size() * sizeof(float));
        ggml_backend_tensor_set(s_t, c_sin.data(), 0, c_sin.size() * sizeof(float));
        ggml_backend_tensor_set(n_q, nq_w.data(), 0, hd * sizeof(float));
        ggml_backend_tensor_set(n_k, nk_w.data(), 0, hd * sizeof(float));
        ggml_backend_tensor_set(w_qkv, b1_qkv.data(), 0, b1_qkv.size());
        ggml_backend_tensor_set(w_o, b1_out.data(), 0, b1_out.size());
        ggml_backend_graph_compute(be, g);
        std::vector<float> r(H * total_t);
        ggml_backend_tensor_get(out, r.data(), 0, r.size() * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return r;
    };

    auto c = run(cpu);
    auto g = run(gpu);
    print_stats("CPU", c.data(), H * total_t);
    print_stats("GPU", g.data(), H * total_t);
    float cs = cosine(c.data(), g.data(), H * total_t);
    fprintf(stdout, "  cos=%.6f %s\n", cs, cs>0.9999f?"PASS":"FAIL");
    return cs > 0.9999f;
}

// ═══ Test: double_block (reduced dims: H = hd * nh) ═══
static bool test_double_block(ggml_backend_t cpu, ggml_backend_t gpu) {
    int hd = 64, nh = 6, H = hd * nh;
    int C = 32, ctxd = 256, it = 64, tt = 32;
    int ms = H * 3;
    fprintf(stdout, "\n--- double_block H=%d nh=%d it=%d tt=%d ---\n", H, nh, it, tt); fflush(stdout);

    auto h_img = gen_rand(H * it, 500);
    auto h_txt = gen_rand(H * tt, 501);
    auto mod_i = gen_rand(6 * H, 502);
    auto mod_t = gen_rand(6 * H, 503);
    auto c_img = gen_rand((hd/2) * it, 504);
    auto s_img = gen_rand((hd/2) * it, 505);
    auto c_txt = gen_rand((hd/2) * tt, 506);
    auto s_txt = gen_rand((hd/2) * tt, 507);

    auto w_q  = pack_b1(gen_rand(H*H,508).data(), H, H);
    auto w_k  = pack_b1(gen_rand(H*H,509).data(), H, H);
    auto w_v  = pack_b1(gen_rand(H*H,510).data(), H, H);
    auto w_o  = pack_b1(gen_rand(H*H,511).data(), H, H);
    auto w_aq = pack_b1(gen_rand(H*H,512).data(), H, H);
    auto w_ak = pack_b1(gen_rand(H*H,513).data(), H, H);
    auto w_av = pack_b1(gen_rand(H*H,514).data(), H, H);
    auto w_ao = pack_b1(gen_rand(H*H,515).data(), H, H);
    auto w_fi = pack_b1(gen_rand(H*(ms*2),516).data(), H, ms*2);
    auto w_fo = pack_b1(gen_rand(H*ms,517).data(), ms, H);
    auto w_fci= pack_b1(gen_rand(H*(ms*2),518).data(), H, ms*2);
    auto w_fco= pack_b1(gen_rand(H*ms,519).data(), ms, H);
    auto nq_w = gen_rand(hd, 520), nk_w = gen_rand(hd, 521);
    auto naq_w= gen_rand(hd, 522), nak_w= gen_rand(hd, 523);

    auto run = [&](ggml_backend_t be) -> std::pair<std::vector<float>, std::vector<float>> {
        struct ggml_init_params gp = { 256*1024*1024, NULL, true };
        ggml_context *ctx = ggml_init(gp);
        auto t2d = [&](int n0, int n1, ggml_type ty) {
            return ggml_new_tensor_2d(ctx, ty, n0, n1); };
        ggml_tensor *hi  = t2d(H, it, GGML_TYPE_F32);
        ggml_tensor *ht  = t2d(H, tt, GGML_TYPE_F32);
        ggml_tensor *mi  = t2d(1, 6*H, GGML_TYPE_F32);
        ggml_tensor *mt  = t2d(1, 6*H, GGML_TYPE_F32);
        ggml_tensor *ci  = t2d(64, it, GGML_TYPE_F32);
        ggml_tensor *si  = t2d(64, it, GGML_TYPE_F32);
        ggml_tensor *ctt = t2d(64, tt, GGML_TYPE_F32);
        ggml_tensor *stt = t2d(64, tt, GGML_TYPE_F32);

        B1Weights bq ={t2d((int)w_q.size(),1,GGML_TYPE_I8),H,H};
        B1Weights bk ={t2d((int)w_k.size(),1,GGML_TYPE_I8),H,H};
        B1Weights bv ={t2d((int)w_v.size(),1,GGML_TYPE_I8),H,H};
        B1Weights bo ={t2d((int)w_o.size(),1,GGML_TYPE_I8),H,H};
        B1Weights baq={t2d((int)w_aq.size(),1,GGML_TYPE_I8),H,H};
        B1Weights bak={t2d((int)w_ak.size(),1,GGML_TYPE_I8),H,H};
        B1Weights bav={t2d((int)w_av.size(),1,GGML_TYPE_I8),H,H};
        B1Weights bao={t2d((int)w_ao.size(),1,GGML_TYPE_I8),H,H};
        B1Weights bfi={t2d((int)w_fi.size(),1,GGML_TYPE_I8),H,ms*2};
        B1Weights bfo={t2d((int)w_fo.size(),1,GGML_TYPE_I8),ms,H};
        B1Weights bfci={t2d((int)w_fci.size(),1,GGML_TYPE_I8),H,ms*2};
        B1Weights bfco={t2d((int)w_fco.size(),1,GGML_TYPE_I8),ms,H};
        ggml_tensor *nq =ggml_new_tensor_1d(ctx,GGML_TYPE_F32,hd);
        ggml_tensor *nk =ggml_new_tensor_1d(ctx,GGML_TYPE_F32,hd);
        ggml_tensor *naq=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,hd);
        ggml_tensor *nak=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,hd);

        std::vector<B1LinearUserData> bud; bud.reserve(32);
        std::vector<Rope2DUserData> rud; rud.reserve(16);
        auto b1=[&](ggml_tensor *a,B1Weights &w){
            bud.push_back({0x31423142,w.in_dim,w.out_dim});
            return b1_linear(ctx,a,w,4,bud.back());};

        auto mod_app=[&](ggml_tensor *x,ggml_tensor *sh,ggml_tensor *sc)->ggml_tensor*{
            ggml_tensor *xn=ggml_norm(ctx,x,1e-6f);
            ggml_tensor *ones=ggml_fill(ctx,ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1),1.0f);
            return ggml_add(ctx,ggml_mul(ctx,xn,ggml_add(ctx,
                ggml_repeat(ctx,column_1d(ctx,ggml_cont(ctx,sc)),xn),ones)),
                ggml_repeat(ctx,column_1d(ctx,ggml_cont(ctx,sh)),xn));};

        // split mod (6 x H)
        ggml_tensor *ms1[6],*ms2[6];
        for(int i=0;i<6;i++){
            ms1[i]=ggml_view_2d(ctx,mi,H,1,mi->nb[1],i*H*sizeof(float));
            ms2[i]=ggml_view_2d(ctx,mt,H,1,mt->nb[1],i*H*sizeof(float));}

        ggml_tensor *hin=mod_app(hi,ms1[0],ms1[1]);
        ggml_tensor *htn=mod_app(ht,ms2[0],ms2[1]);

        ggml_tensor *iq=b1(hin,bq),*ik=b1(hin,bk),*iv=b1(hin,bv);
        ggml_tensor *tq=b1(htn,baq),*tk=b1(htn,bak),*tv=b1(htn,bav);
        auto r3=[&](ggml_tensor *t,int sq){
            return ggml_cont(ctx,ggml_reshape_3d(ctx,t,hd,nh,sq));};
        iq=r3(iq,it);ik=r3(ik,it);iv=r3(iv,it);
        tq=r3(tq,tt);tk=r3(tk,tt);tv=r3(tv,tt);

        iq=rms_norm_qk(ctx,iq,nq,hd);ik=rms_norm_qk(ctx,ik,nk,hd);
        tq=rms_norm_qk(ctx,tq,naq,hd);tk=rms_norm_qk(ctx,tk,nak,hd);

        rud.push_back({0x524F5045,hd,nh,it}); iq=rope_2d_fwd(ctx,iq,ci,si,rud.back());
        rud.push_back({0x524F5045,hd,nh,it}); ik=rope_2d_fwd(ctx,ik,ci,si,rud.back());
        rud.push_back({0x524F5045,hd,nh,tt}); tq=rope_2d_fwd(ctx,tq,ctt,stt,rud.back());
        rud.push_back({0x524F5045,hd,nh,tt}); tk=rope_2d_fwd(ctx,tk,ctt,stt,rud.back());

        ggml_tensor *qc=ggml_concat(ctx,tq,iq,2),*kc=ggml_concat(ctx,tk,ik,2),*vc=ggml_concat(ctx,tv,iv,2);
        auto p3=[&](ggml_tensor *t){return ggml_cont(ctx,ggml_permute(ctx,t,0,2,1,3));};
        ggml_tensor *qp3=p3(qc),*kp3=p3(kc),*vp3=p3(vc);
        float s=1.0f/sqrtf((float)hd);
        ggml_tensor *kf=ggml_cast(ctx,kp3,GGML_TYPE_F16),*vf=ggml_cast(ctx,vp3,GGML_TYPE_F16);
        ggml_tensor *attn=ggml_flash_attn_ext(ctx,qp3,kf,vf,nullptr,s,0.0f,0.0f);
        ggml_flash_attn_ext_set_prec(attn,GGML_PREC_F32);
        attn=ggml_reshape_2d(ctx,attn,H,it+tt);

        ggml_tensor *iat=ggml_view_2d(ctx,attn,H,it,attn->nb[1],tt*H*sizeof(float));
        ggml_tensor *tat=ggml_view_2d(ctx,attn,H,tt,attn->nb[1],0);
        tat=b1(tat,bao); iat=b1(iat,bo);

        ggml_tensor *onev=ggml_fill(ctx,ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1),1.0f);
        auto resc=[&](ggml_tensor *x,ggml_tensor *orig,ggml_tensor *proj,ggml_tensor *gate){
            return ggml_add(ctx,orig,ggml_mul(ctx,
                ggml_repeat(ctx,column_1d(ctx,ggml_cont(ctx,gate)),proj),proj));};
        hin=resc(hi,hi,iat,ms1[2]); htn=resc(ht,ht,tat,ms2[2]);

        hin=mod_app(hin,ms1[3],ms1[4]); htn=mod_app(htn,ms2[3],ms2[4]);
        ggml_tensor *imlp=b1(hin,bfi),*tmlp=b1(htn,bfci);
        imlp=ggml_swiglu(ctx,imlp); imlp=b1(imlp,bfo);
        tmlp=ggml_swiglu(ctx,tmlp); tmlp=b1(tmlp,bfco);
        hin=resc(hin,hi,imlp,ms1[5]); htn=resc(htn,ht,tmlp,ms2[5]);

        ggml_cgraph *g=ggml_new_graph_custom(ctx,4096,false);
        ggml_build_forward_expand(g,hin); ggml_build_forward_expand(g,htn);
        ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(ctx,be);

        auto st=[&](ggml_tensor *t,void*d,size_t sz){ggml_backend_tensor_set(t,d,0,sz);};
        st(hi,h_img.data(),h_img.size()*4); st(ht,h_txt.data(),h_txt.size()*4);
        st(mi,mod_i.data(),mod_i.size()*4); st(mt,mod_t.data(),mod_t.size()*4);
        st(ci,c_img.data(),c_img.size()*4); st(si,s_img.data(),s_img.size()*4);
        st(ctt,c_txt.data(),c_txt.size()*4); st(stt,s_txt.data(),s_txt.size()*4);
        st(bq.data,w_q.data(),w_q.size()); st(bk.data,w_k.data(),w_k.size());
        st(bv.data,w_v.data(),w_v.size()); st(bo.data,w_o.data(),w_o.size());
        st(baq.data,w_aq.data(),w_aq.size()); st(bak.data,w_ak.data(),w_ak.size());
        st(bav.data,w_av.data(),w_av.size()); st(bao.data,w_ao.data(),w_ao.size());
        st(bfi.data,w_fi.data(),w_fi.size()); st(bfo.data,w_fo.data(),w_fo.size());
        st(bfci.data,w_fci.data(),w_fci.size()); st(bfco.data,w_fco.data(),w_fco.size());
        st(nq,nq_w.data(),hd*4); st(nk,nk_w.data(),hd*4);
        st(naq,naq_w.data(),hd*4); st(nak,nak_w.data(),hd*4);

        ggml_backend_graph_compute(be,g);
        std::vector<float> ri(H*it),rt(H*tt);
        ggml_backend_tensor_get(hin,ri.data(),0,ri.size()*4);
        ggml_backend_tensor_get(htn,rt.data(),0,rt.size()*4);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return {ri,rt};
    };

    auto [ic,tc] = run(cpu);
    auto [ig,tg] = run(gpu);
    print_stats("CPU img", ic.data(), H*it);
    print_stats("GPU img", ig.data(), H*it);
    print_stats("CPU txt", tc.data(), H*tt);
    print_stats("GPU txt", tg.data(), H*tt);
    float ci=cosine(ic.data(),ig.data(),H*it), ct=cosine(tc.data(),tg.data(),H*tt);
    fprintf(stdout,"  cos_img=%.6f cos_txt=%.6f %s\n",ci,ct,(ci>0.9999f&&ct>0.9999f)?"PASS":"FAIL");
    return ci>0.9999f && ct>0.9999f;
}

// ═══ Main ═══
int main() {
    fprintf(stdout,"=== WebGPU Diffuser Ops Test ===\n"); fflush(stdout);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);

    fprintf(stdout,"Init GPU...\n"); fflush(stdout);
    ggml_backend_t gpu = ggml_backend_webgpu_init();
    bool has_gpu = (gpu != nullptr);
    fprintf(stdout,"GPU: %s\n", has_gpu?"OK":"none"); fflush(stdout);

    if (!has_gpu) { fprintf(stdout,"SKIP (no GPU)\n"); ggml_backend_free(cpu); return 0; }

    bool ok = true;
    ok &= test_input_proj(cpu, gpu);
    ok &= test_timestep(cpu, gpu);
    ok &= test_output_head(cpu, gpu);
    ok &= test_single_block(cpu, gpu);
    ok &= test_double_block(cpu, gpu);

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    fprintf(stdout,"\n%s\n", ok?"ALL PASSED":"SOME FAILED");
    return ok?0:1;
}
