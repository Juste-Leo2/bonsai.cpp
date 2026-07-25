#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "diffuser_types.h"
#include "diffuser_graph.h"
#include "b1_0_kernel.h"
#include "safetensors.h"

using namespace bonsai;

namespace bonsai {
    bool g_bonsai_debug = false;
}

struct DeferredUpload {
    ggml_tensor * tensor;
    const void * data;
    size_t size;
};

static void make_f32_weight(
    ggml_context * ctx,
    const SafeTensor & st,
    FPWeights & out,
    std::vector<DeferredUpload> & uploads)
{
    int ne0 = (st.shape.size() >= 2) ? (int)st.shape[1] : 0;
    int ne1 = (st.shape.size() >= 1) ? (int)st.shape[0] : 0;
    if (st.shape.size() == 1) {
        out.data = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne1);
    } else {
        out.data = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    }
    size_t nbytes = ne0 * ne1 * sizeof(float);
    uploads.push_back({out.data, st.data, nbytes});
}

static void make_b1_weight(
    ggml_context * ctx,
    const SafeTensor & st,
    B1Weights & out,
    std::vector<DeferredUpload> & uploads)
{
    int out_dim = (int)st.shape[0];
    int packed_cols = (int)st.shape[1];
    int in_dim = (packed_cols / B1_0_BLOCK_BYTES) * B1_0_BLOCK_SIZE;
    size_t nbytes = (size_t)out_dim * packed_cols;

    out.data = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)nbytes, 1);
    out.in_dim = in_dim;
    out.out_dim = out_dim;
    uploads.push_back({out.data, st.data, nbytes});
}

static const SafeTensor & require(const SafetensorsFile & st, const char * name) {
    auto * p = st.find(name);
    if (!p) { fprintf(stderr, "FATAL: weight '%s' not found\n", name); abort(); }
    return *p;
}

int main(int argc, char ** argv) {
    std::string embedding_path;
    std::string model_path = "models/flux2_4b_1bit.safetensors";
    int im_H = 64, im_W = 64, steps = 20, n_threads = 4;
    int max_depth = 0, max_single = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--embedding" && i + 1 < argc) embedding_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "-h" && i + 1 < argc) im_H = std::stoi(argv[++i]);
        else if (arg == "-w" && i + 1 < argc) im_W = std::stoi(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) n_threads = std::stoi(argv[++i]);
        else if (arg == "--debug") bonsai::g_bonsai_debug = true;
        else if (arg == "--dump-blocks") bonsai::g_bonsai_debug = true;
        else if (arg == "--max-depth" && i + 1 < argc) max_depth = std::stoi(argv[++i]);
        else if (arg == "--max-single" && i + 1 < argc) max_single = std::stoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s --embedding <path> [--model path] [-h H] [-w W] [--steps N] [--threads N] [--debug]\n", argv[0]);
            return 1;
        }
    }

    if (embedding_path.empty()) { fprintf(stderr, "error: --embedding is required\n"); return 1; }

    // ── 1. Open safetensors ────────────────────────────────────────
    fprintf(stdout, "Loading model from %s...\n", model_path.c_str());
    SafetensorsFile st;
    st.open(model_path);

    DiffuserParams params;
    int C = params.in_channels;
    int ctx_dim = params.context_in_dim;
    int hidden = params.hidden_size;
    int img_tokens = im_H * im_W;
    int txt_tokens = 512;

    // ── 2. Create SINGLE ggml context for weights + graph ─────────
    struct ggml_init_params gparams = { 256ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(gparams);

    std::vector<DeferredUpload> uploads;
    DiffuserWeights w;

    // Edge projections (F32)
    make_f32_weight(ctx, require(st, "x_embedder.weight"), w.img_in, uploads);
    make_f32_weight(ctx, require(st, "context_embedder.weight"), w.txt_in, uploads);

    // Time guidance (F32)
    make_f32_weight(ctx, require(st, "time_guidance_embed.timestep_embedder.linear_1.weight"), w.time_in_w1, uploads);
    w.time_in_b1.data = nullptr;
    make_f32_weight(ctx, require(st, "time_guidance_embed.timestep_embedder.linear_2.weight"), w.time_in_w2, uploads);
    w.time_in_b2.data = nullptr;

    // Modulations (F32)
    make_f32_weight(ctx, require(st, "double_stream_modulation_img.linear.weight"), w.double_mod_img, uploads);
    make_f32_weight(ctx, require(st, "double_stream_modulation_txt.linear.weight"), w.double_mod_txt, uploads);
    make_f32_weight(ctx, require(st, "single_stream_modulation.linear.weight"), w.single_mod, uploads);

    // Double blocks
    w.double_blocks.resize(params.depth);
    for (int d = 0; d < params.depth; d++) {
        auto & db = w.double_blocks[d];
        char buf[256];

#define LOAD_B1(field, fmt) \
        snprintf(buf, sizeof(buf), fmt, d); \
        make_b1_weight(ctx, require(st, buf), db.field, uploads);

#define LOAD_F32_1D(field, fmt) \
        snprintf(buf, sizeof(buf), fmt, d); \
        { \
            const auto & s = require(st, buf); \
            db.field.data = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params.head_dim); \
            uploads.push_back({db.field.data, s.data, (size_t)params.head_dim * sizeof(float)}); \
        }

        LOAD_B1(attn_to_q,   "transformer_blocks.%d.attn.to_q.weight");
        LOAD_B1(attn_to_k,   "transformer_blocks.%d.attn.to_k.weight");
        LOAD_B1(attn_to_v,   "transformer_blocks.%d.attn.to_v.weight");
        LOAD_B1(attn_to_out, "transformer_blocks.%d.attn.to_out.0.weight");
        LOAD_F32_1D(attn_norm_q, "transformer_blocks.%d.attn.norm_q.weight");
        LOAD_F32_1D(attn_norm_k, "transformer_blocks.%d.attn.norm_k.weight");
        LOAD_B1(attn_add_q,    "transformer_blocks.%d.attn.add_q_proj.weight");
        LOAD_B1(attn_add_k,    "transformer_blocks.%d.attn.add_k_proj.weight");
        LOAD_B1(attn_add_v,    "transformer_blocks.%d.attn.add_v_proj.weight");
        LOAD_B1(attn_add_out,  "transformer_blocks.%d.attn.to_add_out.weight");
        LOAD_F32_1D(attn_norm_added_q, "transformer_blocks.%d.attn.norm_added_q.weight");
        LOAD_F32_1D(attn_norm_added_k, "transformer_blocks.%d.attn.norm_added_k.weight");
        LOAD_B1(ff_linear_in,       "transformer_blocks.%d.ff.linear_in.weight");
        LOAD_B1(ff_linear_out,      "transformer_blocks.%d.ff.linear_out.weight");
        LOAD_B1(ff_ctx_linear_in,   "transformer_blocks.%d.ff_context.linear_in.weight");
        LOAD_B1(ff_ctx_linear_out,  "transformer_blocks.%d.ff_context.linear_out.weight");

#undef LOAD_B1
#undef LOAD_F32_1D
    }

    // Single blocks
    w.single_blocks.resize(params.depth_single_blocks);
    for (int s = 0; s < params.depth_single_blocks; s++) {
        auto & sb = w.single_blocks[s];
        char buf[256];

#define LOAD_B1(field, fmt) \
        snprintf(buf, sizeof(buf), fmt, s); \
        make_b1_weight(ctx, require(st, buf), sb.field, uploads);

#define LOAD_F32_1D(field, fmt) \
        snprintf(buf, sizeof(buf), fmt, s); \
        { \
            const auto & sf = require(st, buf); \
            sb.field.data = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params.head_dim); \
            uploads.push_back({sb.field.data, sf.data, (size_t)params.head_dim * sizeof(float)}); \
        }

        LOAD_B1(to_qkv_mlp_proj, "single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight");
        LOAD_B1(to_out,          "single_transformer_blocks.%d.attn.to_out.weight");
        LOAD_F32_1D(norm_q, "single_transformer_blocks.%d.attn.norm_q.weight");
        LOAD_F32_1D(norm_k, "single_transformer_blocks.%d.attn.norm_k.weight");

#undef LOAD_B1
#undef LOAD_F32_1D
    }

    // Output head (F32)
    make_f32_weight(ctx, require(st, "norm_out.linear.weight"), w.norm_out_linear, uploads);
    make_f32_weight(ctx, require(st, "proj_out.weight"), w.proj_out, uploads);

    size_t total_upload_mb = 0;
    for (auto & u : uploads) total_upload_mb += u.size;
    fprintf(stdout, "Loaded %d db + %d sb weights (%zu MB).\n",
            params.depth, params.depth_single_blocks, total_upload_mb / 1024 / 1024);

    // ── 3. Load text embeddings ────────────────────────────────────
    fprintf(stdout, "Loading embeddings from %s...\n", embedding_path.c_str());
    FILE * ef = fopen(embedding_path.c_str(), "rb");
    if (!ef) { perror("fopen embedding"); return 1; }
    fseek(ef, 0, SEEK_END);
    long emb_size = ftell(ef);
    fseek(ef, 0, SEEK_SET);
    std::vector<float> txt_emb(emb_size / sizeof(float));
    fread(txt_emb.data(), 1, emb_size, ef);
    fclose(ef);

    // ── 4. Build graph in SAME context ─────────────────────────────
    fprintf(stdout, "Building computation graph...\n");

    auto freqs_data = compute_rope_freqs_data(params.axes_dim, 4, (float)params.theta);
    ggml_tensor * freqs_table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, freqs_data.max_half, freqs_data.n_axes);

    DiffuserGraph dg = build_diffuser_graph(
        ctx, params, w, freqs_table,
        img_tokens, txt_tokens, 1, n_threads,
        max_depth, max_single);

    // ── 5. Init WebGPU backend ─────────────────────────────────────
    fprintf(stdout, "Initializing WebGPU backend...\n");
    ggml_backend_t gpu = ggml_backend_webgpu_init();
    if (!gpu) {
        fprintf(stderr, "FATAL: WebGPU backend not available\n");
        return 1;
    }

    // ── 6. Single ggml_gallocr for ALL tensors (weights + compute) ─
    fprintf(stdout, "Allocating graph (gallocr, all tensors in one buffer)...\n");
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(gpu);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    ggml_gallocr_reserve(galloc, dg.graph);
    ggml_gallocr_alloc_graph(galloc, dg.graph);
    size_t alloc_mb = ggml_gallocr_get_buffer_size(galloc, 0) / 1024 / 1024;
    fprintf(stdout, "Graph allocator reserved %zu MB total\n", alloc_mb);

    // Upload weights (skip tensors not in reduced graph, e.g. --max-depth)
    fprintf(stdout, "Uploading %zu weight tensors...\n", uploads.size());
    size_t uploaded = 0;
    for (auto & u : uploads) {
        if (u.tensor->buffer) {
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.size);
            uploaded++;
        }
    }
    fprintf(stdout, "  %zu/%zu tensors uploaded (skipped %zu not in graph)\n",
            uploaded, uploads.size(), uploads.size() - uploaded);

    // Upload rope frequencies
    ggml_backend_tensor_set(freqs_table, freqs_data.values.data(), 0,
                            freqs_data.values.size() * sizeof(float));

    // ── 7. Diffusion loop ──────────────────────────────────────────
    std::vector<float> latents(C * img_tokens);

    FILE * f_latent = fopen("tmp/x_input.bin", "rb");
    if (f_latent) {
        fprintf(stdout, "Loading initial latents from tmp/x_input.bin...\n");
        fread(latents.data(), sizeof(float), latents.size(), f_latent);
        fclose(f_latent);
    } else {
        fprintf(stdout, "tmp/x_input.bin not found, using random latents...\n");
        for (int i = 0; i < C * img_tokens; i++)
            latents[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    std::vector<float> img_ids(4 * img_tokens, 0.0f);
    for (int y = 0; y < im_H; y++) {
        for (int x = 0; x < im_W; x++) {
            int idx = y * im_W + x;
            img_ids[idx * 4 + 2] = (float)y;
            img_ids[idx * 4 + 3] = (float)x;
        }
    }
    std::vector<float> txt_ids(4 * txt_tokens, 0.0f);

    std::vector<float> timesteps(steps);
    for (int i = 0; i < steps; i++)
        timesteps[i] = 1.0f - (float)i / steps;

    std::vector<float> noise_pred(C * img_tokens);

    fprintf(stdout, "Starting diffusion with %d steps on %d x %d grid...\n", steps, im_H, im_W);

    for (int step = 0; step < steps; step++) {
        float t = timesteps[step];
        float dt = (step < steps - 1) ? t - timesteps[step + 1] : t;

        ggml_backend_tensor_set(dg.img_in, latents.data(), 0, latents.size() * sizeof(float));
        ggml_backend_tensor_set(dg.txt_in, txt_emb.data(), 0, txt_emb.size() * sizeof(float));
        float t_scaled = t * 1000.0f;
        ggml_backend_tensor_set(dg.timestep, &t_scaled, 0, sizeof(float));
        ggml_backend_tensor_set(dg.img_ids, img_ids.data(), 0, img_ids.size() * sizeof(float));
        ggml_backend_tensor_set(dg.txt_ids, txt_ids.data(), 0, txt_ids.size() * sizeof(float));

        ggml_backend_graph_compute(gpu, dg.graph);

        ggml_backend_tensor_get(dg.out, noise_pred.data(), 0, noise_pred.size() * sizeof(float));

        int nan_c = 0, inf_c = 0;
        float mn = 1e30f, mx = -1e30f, sum = 0.0f;
        for (int i = 0; i < C * img_tokens; i++) {
            float v = noise_pred[i];
            if (std::isnan(v)) nan_c++;
            if (std::isinf(v)) inf_c++;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        fprintf(stdout, "  step %3d/%d: nan=%d inf=%d min=%.4f max=%.4f mean=%.4f\n",
                step + 1, steps, nan_c, inf_c, mn, mx, sum / (C * img_tokens));

        for (int i = 0; i < C * img_tokens; i++)
            latents[i] = latents[i] + dt * noise_pred[i];
    }

    // ── 8. Save final latents ──────────────────────────────────────
    std::string out_path = "latents_final.bin";
    FILE * of = fopen(out_path.c_str(), "wb");
    if (of) { fwrite(latents.data(), sizeof(float), latents.size(), of); fclose(of); }
    fprintf(stdout, "Saved latents to %s\n", out_path.c_str());

    ggml_gallocr_free(galloc);
    ggml_backend_free(gpu);
    ggml_free(ctx);

    return 0;
}
