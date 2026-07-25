/**
 * WebGPU custom op tests: b1_linear + rope_2d GPU vs CPU
 * Pattern: ggml_backend_alloc_ctx_tensors (same as test-backend-ops.cpp)
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
    for (int i = 0; i < n; i++)
        dot += (double)a[i] * b[i], na += (double)a[i] * a[i], nb += (double)b[i] * b[i];
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

// Run graph on a single backend. Pattern from test-backend-ops.cpp
static std::vector<float> run_graph(ggml_backend_t backend, ggml_cgraph *graph,
                                     ggml_tensor *out_t, int n_out) {
    ggml_backend_graph_compute(backend, graph);
    std::vector<float> result(n_out);
    ggml_backend_tensor_get(out_t, result.data(), 0, n_out * sizeof(float));
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

        // --- CPU ---
        struct ggml_init_params gp = { 256*1024*1024, NULL, true };
        ggml_context *ctx = ggml_init(gp);
        ggml_tensor *act_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, batch);
        ggml_tensor *wgt_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)b1.size(), 1);
    B1Weights w = { wgt_t, in_dim, out_dim };
    B1LinearUserData *ud = (B1LinearUserData*)malloc(sizeof(B1LinearUserData));
    ud->magic = 0x31423142; ud->in_dim = ud->out_dim = 0;
    ggml_tensor *out_t = b1_linear(ctx, act_t, w, 4, *ud);
        ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, out_t);

        // Allocate + upload + compute CPU
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
        ggml_backend_tensor_set(act_t, act.data(), 0, act.size() * sizeof(float));
        ggml_backend_tensor_set(wgt_t, b1.data(), 0, b1.size());
        auto cpu = run_graph(cpu_be, graph, out_t, out_dim);
        fprintf(stdout, "  CPU μ=%.4f\n", cpu[0]); fflush(stdout);

        // --- GPU ---
        fprintf(stdout, "  GPU starting...\n"); fflush(stdout);
        if (has_gpu) {
            fprintf(stdout, "  GPU ctx...\n"); fflush(stdout);
            ggml_context *ctx2 = ggml_init(gp);
            fprintf(stdout, "  GPU tensors...\n"); fflush(stdout);
            act_t = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, in_dim, batch);
            wgt_t = ggml_new_tensor_2d(ctx2, GGML_TYPE_I8, (int64_t)b1.size(), 1);
            w = { wgt_t, in_dim, out_dim };
            fprintf(stdout, "  GPU b1_linear...\n"); fflush(stdout);
            out_t = b1_linear(ctx2, act_t, w, 4, *ud);
            fprintf(stdout, "  GPU graph...\n"); fflush(stdout);
            graph = ggml_new_graph_custom(ctx2, 256, false);
            ggml_build_forward_expand(graph, out_t);

            // Allocate + upload + compute GPU
            fprintf(stdout, "  GPU alloc...\n"); fflush(stdout);
            ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, gpu_be);
            fprintf(stdout, "  GPU set...\n"); fflush(stdout);
            ggml_backend_tensor_set(act_t, act.data(), 0, act.size() * sizeof(float));
            ggml_backend_tensor_set(wgt_t, b1.data(), 0, b1.size());
            fprintf(stdout, "  GPU compute...\n"); fflush(stdout);
            auto gpu = run_graph(gpu_be, graph, out_t, out_dim);
            fprintf(stdout, "  GPU done\n"); fflush(stdout);
        // buf freed with ctx

            float cos = cosine(cpu.data(), gpu.data(), out_dim);
            float maxd = 0;
            for (int i = 0; i < out_dim; i++) maxd = fmaxf(maxd, fabsf(cpu[i] - gpu[i]));
            fprintf(stdout, "  GPU cos=%.6f maxd=%.6f %s\n", cos, maxd,
                    cos > 0.9999f ? "PASS" : "FAIL"); fflush(stdout);
            if (cos <= 0.9999f) all_ok = false;
            // ggml_free(ctx2); (skipped for now)
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

        struct ggml_init_params gp = { 256*1024*1024, NULL, true };
        ggml_context *ctx = ggml_init(gp);
        ggml_tensor *q_t   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, seq);
        ggml_tensor *cos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd / 2, seq);
        ggml_tensor *sin_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd / 2, seq);
        Rope2DUserData rud = { 0x524F5045, hd, nh, seq };
        ggml_tensor *out_t = rope_2d_fwd(ctx, q_t, cos_t, sin_t, rud);
        ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, out_t);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
        ggml_backend_tensor_set(q_t, q.data(), 0, n_el * sizeof(float));
        ggml_backend_tensor_set(cos_t, c.data(), 0, n_cos * sizeof(float));
        ggml_backend_tensor_set(sin_t, s.data(), 0, n_cos * sizeof(float));
        auto cpu = run_graph(cpu_be, graph, out_t, n_el);
        fprintf(stdout, "  CPU μ=%.4f\n", cpu[0]); fflush(stdout);
        fprintf(stdout, "  rope_cpu_done has_gpu=%d\n", (int)has_gpu); fflush(stdout);
    // buf freed with ctx
        // ggml_free(ctx); (skipped for now)

        if (has_gpu) {
            ggml_context *ctx2 = ggml_init(gp);
            q_t   = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, hd, nh, seq);
            cos_t = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, hd / 2, seq);
            sin_t = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, hd / 2, seq);
            out_t = rope_2d_fwd(ctx2, q_t, cos_t, sin_t, rud);
            graph = ggml_new_graph_custom(ctx2, 256, false);
            ggml_build_forward_expand(graph, out_t);

            ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, gpu_be);
            ggml_backend_tensor_set(q_t, q.data(), 0, n_el * sizeof(float));
            ggml_backend_tensor_set(cos_t, c.data(), 0, n_cos * sizeof(float));
            ggml_backend_tensor_set(sin_t, s.data(), 0, n_cos * sizeof(float));
            auto gpu = run_graph(gpu_be, graph, out_t, n_el);
        // buf freed with ctx

            float cos = cosine(cpu.data(), gpu.data(), n_el);
            float maxd = 0;
            for (int i = 0; i < n_el; i++) maxd = fmaxf(maxd, fabsf(cpu[i] - gpu[i]));
            fprintf(stdout, "  GPU cos=%.6f maxd=%.6f %s\n", cos, maxd,
                    cos > 0.999f ? "PASS" : "FAIL"); fflush(stdout);
            if (cos <= 0.999f) all_ok = false;
            // ggml_free(ctx2); (skipped for now)
        } else { fprintf(stdout, "  GPU SKIP\n"); }
    }

    if (has_gpu) ggml_backend_free(gpu_be);
    ggml_backend_free(cpu_be);
    fprintf(stdout, "\n%s\n", all_ok ? "ALL PASSED" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
