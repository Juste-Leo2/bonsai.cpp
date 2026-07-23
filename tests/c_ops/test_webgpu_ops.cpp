/**
 * WebGPU custom op tests: b1_linear + rope_2d on GPU vs CPU reference
 *
 * Build: cmake --build build --target test_webgpu_ops
 * Run:   ./build/test_webgpu_ops
 */
#include "ggml.h"
#include "ggml-impl.h"
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
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static std::vector<uint8_t> pack_b1(const float *wgt, int out_dim, int in_dim) {
    int nb = in_dim / 32, rs = nb * 6;
    std::vector<uint8_t> out(out_dim * rs);
    for (int r = 0; r < out_dim; r++) {
        for (int b = 0; b < nb; b++) {
            float scale = 0.0f;
            for (int i = 0; i < 32; i++)
                scale += fabsf(wgt[r * in_dim + b * 32 + i]);
            scale /= 32.0f;
            uint16_t sb = ggml_fp32_to_fp16(scale);
            memcpy(&out[r * rs + b * 6], &sb, 2);
            uint32_t bits = 0;
            for (int i = 0; i < 32; i++)
                if (wgt[r * in_dim + b * 32 + i] >= 0.0f) bits |= (1u << i);
            memcpy(&out[r * rs + b * 6 + 2], &bits, 4);
        }
    }
    return out;
}

static std::vector<float> run_graph(ggml_backend_t backend, ggml_cgraph *graph,
                                     ggml_tensor *out_t, int n_out) {
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    // Upload: iterate all sources of all nodes, set data if marked as input
    for (int i = 0; i < graph->n_nodes; i++) {
        ggml_tensor *node = graph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            ggml_tensor *src = node->src[s];
            if (src && src->data && (src->flags & GGML_TENSOR_FLAG_INPUT)) {
                ggml_backend_tensor_set(src, src->data, 0, ggml_nbytes(src));
            }
        }
    }

    ggml_backend_graph_compute(backend, graph);

    std::vector<float> result(n_out);
    ggml_backend_tensor_get(out_t, result.data(), 0, n_out * sizeof(float));

    ggml_gallocr_free(galloc);
    return result;
}

int main() {
    fprintf(stdout, "=== WebGPU Custom Ops Test ===\n");

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu_be, 4);

    fprintf(stdout, "Initializing WebGPU backend...\n");
    ggml_backend_t gpu_be = ggml_backend_webgpu_init();
    bool has_gpu = (gpu_be != nullptr);
    if (has_gpu)
        fprintf(stdout, "WebGPU backend initialized.\n");
    else
        fprintf(stdout, "WebGPU not available — will skip GPU tests.\n");

    bool all_ok = true;

    // ═══════════════════════════════════════════════════════════════
    // b1_linear test
    // ═══════════════════════════════════════════════════════════════
    {
        int batch = 1, in_dim = 64, out_dim = 32;
        fprintf(stdout, "\n--- b1_linear (batch=%d in=%d out=%d) ---\n", batch, in_dim, out_dim);

        std::vector<float> act(batch * in_dim), wgt(out_dim * in_dim);
        srand(42);
        for (auto &v : act) v = (float)rand() / RAND_MAX * 2 - 1;
        for (auto &v : wgt) v = (float)rand() / RAND_MAX * 2 - 1;
        auto b1 = pack_b1(wgt.data(), out_dim, in_dim);

        struct ggml_init_params gparams = { 256 * 1024 * 1024, NULL, true };
        ggml_context *ctx = ggml_init(gparams);

        ggml_tensor *act_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, batch);
        ggml_tensor *wgt_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)b1.size(), 1);
        ggml_set_input(act_t);
        ggml_set_input(wgt_t);

        memcpy(act_t->data, act.data(), act.size() * 4);
        memcpy(wgt_t->data, b1.data(), b1.size());

        B1Weights w = { wgt_t, in_dim, out_dim };
        B1LinearUserData ud;
        ggml_tensor *out_t = b1_linear(ctx, act_t, w, 4, ud);
        ggml_set_output(out_t);

        ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, out_t);

        fprintf(stdout, "  CPU... "); fflush(stdout);
        auto cpu = run_graph(cpu_be, graph, out_t, batch * out_dim);
        fprintf(stdout, "μ=%.4f σ=%.4f\n", cpu[0], 0.0);

        if (has_gpu) {
            fprintf(stdout, "  GPU... "); fflush(stdout);
            auto gpu = run_graph(gpu_be, graph, out_t, batch * out_dim);

            float cos = cosine(cpu.data(), gpu.data(), batch * out_dim);
            float maxd = 0;
            for (int i = 0; i < batch * out_dim; i++)
                maxd = fmaxf(maxd, fabsf(cpu[i] - gpu[i]));
            fprintf(stdout, "cos=%.6f max_diff=%.6f\n", cos, maxd);
            bool ok = cos > 0.9999f;
            fprintf(stdout, "  %s\n", ok ? "PASS" : "FAIL");
            if (!ok) all_ok = false;
        } else {
            fprintf(stdout, "  SKIP (no GPU)\n");
        }
        ggml_free(ctx);
    }

    // ═══════════════════════════════════════════════════════════════
    // rope_2d test
    // ═══════════════════════════════════════════════════════════════
    {
        int head_dim = 128, n_heads = 4, seq = 16;
        fprintf(stdout, "\n--- rope_2d (head_dim=%d n_heads=%d seq=%d) ---\n",
                head_dim, n_heads, seq);

        int n_el = head_dim * n_heads * seq;
        int n_cos = (head_dim / 2) * seq;
        std::vector<float> q(n_el), cos_t(n_cos), sin_t(n_cos);
        srand(123);
        for (auto &v : q) v = (float)rand() / RAND_MAX * 2 - 1;
        for (auto &v : cos_t) v = (float)rand() / RAND_MAX * 2 - 1;
        for (auto &v : sin_t) v = (float)rand() / RAND_MAX * 2 - 1;

        struct ggml_init_params gparams = { 256 * 1024 * 1024, NULL, true };
        ggml_context *ctx = ggml_init(gparams);

        ggml_tensor *q_t   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq);
        ggml_tensor *cos_t_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim / 2, seq);
        ggml_tensor *sin_t_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim / 2, seq);
        ggml_set_input(q_t);
        ggml_set_input(cos_t_t);
        ggml_set_input(sin_t_t);

        memcpy(q_t->data, q.data(), n_el * 4);
        memcpy(cos_t_t->data, cos_t.data(), n_cos * 4);
        memcpy(sin_t_t->data, sin_t.data(), n_cos * 4);

        Rope2DUserData rud = { 0x524F5045, head_dim, n_heads, seq };
        ggml_tensor *out_t = rope_2d_fwd(ctx, q_t, cos_t_t, sin_t_t, rud);
        ggml_set_output(out_t);

        ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, out_t);

        fprintf(stdout, "  CPU... "); fflush(stdout);
        auto cpu = run_graph(cpu_be, graph, out_t, n_el);
        fprintf(stdout, "μ=%.4f\n", cpu[0]);

        if (has_gpu) {
            fprintf(stdout, "  GPU... "); fflush(stdout);
            auto gpu = run_graph(gpu_be, graph, out_t, n_el);

            float cos = cosine(cpu.data(), gpu.data(), n_el);
            float maxd = 0;
            for (int i = 0; i < n_el; i++)
                maxd = fmaxf(maxd, fabsf(cpu[i] - gpu[i]));
            fprintf(stdout, "cos=%.6f max_diff=%.6f\n", cos, maxd);
            bool ok = cos > 0.999f;
            fprintf(stdout, "  %s\n", ok ? "PASS" : "FAIL");
            if (!ok) all_ok = false;
        } else {
            fprintf(stdout, "  SKIP (no GPU)\n");
        }
        ggml_free(ctx);
    }

    if (has_gpu) ggml_backend_free(gpu_be);
    ggml_backend_free(cpu_be);

    fprintf(stdout, "\n%s\n", all_ok ? "ALL PASSED" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
