/**
 * WebGPU custom op tests: b1_linear + rope_2d GPU vs CPU
 * Separate contexts for CPU and GPU to avoid buffer reuse issues.
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

static float cosine(const float *a, const float *b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(fmax(na, 1e-30)) * sqrt(fmax(nb, 1e-30))));
}

static std::vector<uint8_t> pack_b1(const float *wgt, int out_dim, int in_dim) {
    int nb = in_dim / 32, rs = nb * 6;
    std::vector<uint8_t> out(out_dim * rs);
    for (int r = 0; r < out_dim; r++)
        for (int b = 0; b < nb; b++) {
            float scale = 0.0f;
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

// Create a tiny graph + run + return output
static std::vector<float> run_b1(ggml_backend_t be, const float *act, const uint8_t *b1,
                                  int batch, int in_dim, int out_dim, int n_threads) {
    struct ggml_init_params gp = { 256*1024*1024, NULL, false };
    ggml_context *ctx = ggml_init(gp);

    ggml_tensor *act_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, batch);
    ggml_tensor *wgt_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)out_dim * (in_dim/32) * 6, 1);
    ggml_set_input(act_t);
    ggml_set_input(wgt_t);
    memcpy(act_t->data, act, batch * in_dim * sizeof(float));
    memcpy(wgt_t->data, b1, out_dim * (in_dim/32) * 6);

    B1Weights w = { wgt_t, in_dim, out_dim };
    B1LinearUserData ud; ud.in_dim = ud.out_dim = 0;
    ggml_tensor *out_t = b1_linear(ctx, act_t, w, n_threads, ud);
    ggml_set_output(out_t);

    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, out_t);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    // Input data was copied to ggml context, now copy to backend buffers via tensor_set
    ggml_backend_tensor_set(act_t, act, 0, batch * in_dim * sizeof(float));
    ggml_backend_tensor_set(wgt_t, b1, 0, out_dim * (in_dim/32) * 6);

    ggml_backend_graph_compute(be, graph);

    std::vector<float> result(batch * out_dim);
    ggml_backend_tensor_get(out_t, result.data(), 0, batch * out_dim * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

static std::vector<float> run_rope(ggml_backend_t be, const float *q, const float *c, const float *s,
                                    int head_dim, int n_heads, int seq) {
    int n_el = head_dim * n_heads * seq;
    int n_cos = (head_dim / 2) * seq;

    struct ggml_init_params gp = { 256*1024*1024, NULL, false };
    ggml_context *ctx = ggml_init(gp);

    ggml_tensor *q_t   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq);
    ggml_tensor *cos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim / 2, seq);
    ggml_tensor *sin_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim / 2, seq);
    memcpy(q_t->data, q, n_el * sizeof(float));
    memcpy(cos_t->data, c, n_cos * sizeof(float));
    memcpy(sin_t->data, s, n_cos * sizeof(float));

    Rope2DUserData rud = { 0x524F5045, head_dim, n_heads, seq };
    ggml_tensor *out_t = rope_2d_fwd(ctx, q_t, cos_t, sin_t, rud);
    ggml_set_output(out_t);

    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, out_t);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    ggml_backend_tensor_set(q_t, q, 0, n_el * sizeof(float));
    ggml_backend_tensor_set(cos_t, c, 0, n_cos * sizeof(float));
    ggml_backend_tensor_set(sin_t, s, 0, n_cos * sizeof(float));

    ggml_backend_graph_compute(be, graph);

    std::vector<float> result(n_el);
    ggml_backend_tensor_get(out_t, result.data(), 0, n_el * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

int main() {
    fprintf(stdout, "=== WebGPU Custom Ops Test ===\n"); fflush(stdout);

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu_be, 4);

    fprintf(stdout, "Init GPU...\n"); fflush(stdout);
    ggml_backend_t gpu_be = ggml_backend_webgpu_init();
    bool has_gpu = (gpu_be != nullptr);
    fprintf(stdout, "GPU: %s\n", has_gpu ? "OK" : "none"); fflush(stdout);

    bool all_ok = true;

    // ═══ b1_linear ═══
    {
        int batch = 1, in_dim = 64, out_dim = 32;
        fprintf(stdout, "\n--- b1_linear b=%d in=%d out=%d ---\n", batch, in_dim, out_dim); fflush(stdout);

        std::vector<float> act(batch * in_dim), wgt(out_dim * in_dim);
        srand(42);
        for (auto &v : act) v = (float)rand() / RAND_MAX * 2 - 1;
        for (auto &v : wgt) v = (float)rand() / RAND_MAX * 2 - 1;
        auto b1 = pack_b1(wgt.data(), out_dim, in_dim);

        fprintf(stdout, "  CPU... "); fflush(stdout);
        auto cpu = run_b1(cpu_be, act.data(), b1.data(), batch, in_dim, out_dim, 4);
        fprintf(stdout, "done\n"); fflush(stdout);

        if (has_gpu) {
            fprintf(stdout, "  GPU... "); fflush(stdout);
            auto gpu = run_b1(gpu_be, act.data(), b1.data(), batch, in_dim, out_dim, 4);

            float cos = cosine(cpu.data(), gpu.data(), batch * out_dim);
            float maxd = 0;
            for (int i = 0; i < batch * out_dim; i++) maxd = fmaxf(maxd, fabsf(cpu[i] - gpu[i]));
            fprintf(stdout, "cos=%.6f maxd=%.6f %s\n", cos, maxd, cos > 0.9999f ? "PASS" : "FAIL"); fflush(stdout);
            if (cos <= 0.9999f) all_ok = false;
        } else { fprintf(stdout, "  GPU SKIP\n"); }
    }

    // ═══ rope_2d ═══
    {
        int hd = 128, nh = 4, seq = 16;
        fprintf(stdout, "\n--- rope_2d hd=%d nh=%d seq=%d ---\n", hd, nh, seq); fflush(stdout);

        int n_el = hd * nh * seq, n_cos = (hd / 2) * seq;
        std::vector<float> q(n_el), c(n_cos), s(n_cos);
        srand(123);
        for (auto &v : q) v = (float)rand() / RAND_MAX * 2 - 1;
        for (auto &v : c) v = (float)rand() / RAND_MAX * 2 - 1;
        for (auto &v : s) v = (float)rand() / RAND_MAX * 2 - 1;

        fprintf(stdout, "  CPU... "); fflush(stdout);
        auto cpu = run_rope(cpu_be, q.data(), c.data(), s.data(), hd, nh, seq);
        fprintf(stdout, "done\n"); fflush(stdout);

        if (has_gpu) {
            fprintf(stdout, "  GPU... "); fflush(stdout);
            auto gpu = run_rope(gpu_be, q.data(), c.data(), s.data(), hd, nh, seq);

            float cos = cosine(cpu.data(), gpu.data(), n_el);
            float maxd = 0;
            for (int i = 0; i < n_el; i++) maxd = fmaxf(maxd, fabsf(cpu[i] - gpu[i]));
            fprintf(stdout, "cos=%.6f maxd=%.6f %s\n", cos, maxd, cos > 0.999f ? "PASS" : "FAIL"); fflush(stdout);
            if (cos <= 0.999f) all_ok = false;
        } else { fprintf(stdout, "  GPU SKIP\n"); }
    }

    if (has_gpu) ggml_backend_free(gpu_be);
    ggml_backend_free(cpu_be);
    fprintf(stdout, "\n%s\n", all_ok ? "ALL PASSED" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
