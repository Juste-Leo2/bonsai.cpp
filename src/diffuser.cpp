#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

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

// ── helpers ────────────────────────────────────────────────────────

static ggml_tensor * load_f32_2d(ggml_context * ctx, const SafeTensor & st, int ne0, int ne1) {
    auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    std::memcpy(t->data, st.data, ne0 * ne1 * sizeof(float));
    return t;
}

static ggml_tensor * load_f32_1d(ggml_context * ctx, const SafeTensor & st, int ne0) {
    auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    std::memcpy(t->data, st.data, ne0 * sizeof(float));
    return t;
}

static void load_b1(ggml_context * ctx, const SafeTensor & st, B1Weights & w) {
    // safetensors shape: (out_dim, in_dim/32*6)
    int out_dim = (int)st.shape[0];
    int packed_cols = (int)st.shape[1];
    int in_dim = (packed_cols / B1_0_BLOCK_BYTES) * B1_0_BLOCK_SIZE;
    size_t nbytes = (size_t)out_dim * packed_cols;

    w.data = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t)nbytes, 1);
    std::memcpy(w.data->data, st.data, nbytes);
    w.in_dim = in_dim;
    w.out_dim = out_dim;
}

static const SafeTensor & require(const SafetensorsFile & st, const char * name) {
    auto * p = st.find(name);
    if (!p) { fprintf(stderr, "FATAL: weight '%s' not found in safetensors\n", name); abort(); }
    return *p;
}

// ── main ───────────────────────────────────────────────────────────

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

    // ── 2. Load weights into ggml context ──────────────────────────
    size_t wctx_size = 2200ULL * 1024 * 1024; // 2.2 GB for all weights + metadata
    std::vector<uint8_t> wbuf(wctx_size);
    struct ggml_init_params wparams = { wctx_size, wbuf.data(), false };
    ggml_context * wctx = ggml_init(wparams);

    DiffuserWeights w;

    // Edge projections (F32, PyTorch (out,in) → ggml (in,out))
    w.img_in.data     = load_f32_2d(wctx, require(st, "x_embedder.weight"), C, hidden);
    w.txt_in.data      = load_f32_2d(wctx, require(st, "context_embedder.weight"), ctx_dim, hidden);

    // Time guidance (F32)
    w.time_in_w1.data   = load_f32_2d(wctx, require(st, "time_guidance_embed.timestep_embedder.linear_1.weight"), 256, hidden);
    w.time_in_b1.data   = nullptr; // no bias in this model
    w.time_in_w2.data   = load_f32_2d(wctx, require(st, "time_guidance_embed.timestep_embedder.linear_2.weight"), hidden, hidden);
    w.time_in_b2.data   = nullptr;

    // Modulations (F32)
    w.double_mod_img.data = load_f32_2d(wctx, require(st, "double_stream_modulation_img.linear.weight"), hidden, 6 * hidden);
    w.double_mod_txt.data = load_f32_2d(wctx, require(st, "double_stream_modulation_txt.linear.weight"), hidden, 6 * hidden);
    w.single_mod.data     = load_f32_2d(wctx, require(st, "single_stream_modulation.linear.weight"), hidden, 3 * hidden);

    // ── Double blocks ──────────────────────────────────────────────
    w.double_blocks.resize(params.depth);
    for (int d = 0; d < params.depth; d++) {
        auto & db = w.double_blocks[d];
        char buf[256];

#define LOAD_B1(field, fmt) \
        snprintf(buf, sizeof(buf), fmt, d); \
        load_b1(wctx, require(st, buf), db.field);

#define LOAD_F32_1D(field, fmt) \
        snprintf(buf, sizeof(buf), fmt, d); \
        db.field.data = load_f32_1d(wctx, require(st, buf), params.head_dim);

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

    // ── Single blocks ──────────────────────────────────────────────
    w.single_blocks.resize(params.depth_single_blocks);
    for (int s = 0; s < params.depth_single_blocks; s++) {
        auto & sb = w.single_blocks[s];
        char buf[256];

#define LOAD_B1(field, fmt) \
        snprintf(buf, sizeof(buf), fmt, s); \
        load_b1(wctx, require(st, buf), sb.field);

#define LOAD_F32_1D(field, fmt) \
        snprintf(buf, sizeof(buf), fmt, s); \
        sb.field.data = load_f32_1d(wctx, require(st, buf), params.head_dim);

        LOAD_B1(to_qkv_mlp_proj, "single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight");
        LOAD_B1(to_out,          "single_transformer_blocks.%d.attn.to_out.weight");
        LOAD_F32_1D(norm_q, "single_transformer_blocks.%d.attn.norm_q.weight");
        LOAD_F32_1D(norm_k, "single_transformer_blocks.%d.attn.norm_k.weight");

#undef LOAD_B1
#undef LOAD_F32_1D
    }

    // ── Output head (F32) ──────────────────────────────────────────
    w.norm_out_linear.data = load_f32_2d(wctx, require(st, "norm_out.linear.weight"), hidden, 2 * hidden);
    w.proj_out.data        = load_f32_2d(wctx, require(st, "proj_out.weight"), hidden, C);

    fprintf(stdout, "Loaded %d double_blocks + %d single_blocks weights.\n",
            params.depth, params.depth_single_blocks);

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

    // ── 4. Build graph ─────────────────────────────────────────────
    fprintf(stdout, "Building computation graph...\n");
    struct ggml_init_params cparams = { 256ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx_compute = ggml_init(cparams);

    auto freqs_data = compute_rope_freqs_data(params.axes_dim, 4, (float)params.theta);
    ggml_tensor * freqs_table = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, freqs_data.max_half, freqs_data.n_axes);

    DiffuserGraph dg = build_diffuser_graph(
        ctx_compute, params, w, freqs_table,
        img_tokens, txt_tokens, 1, n_threads,
        max_depth, max_single);

    // ── 5. Init backend + allocator ────────────────────────────────
    fprintf(stdout, "Initializing GGML CPU Backend and Graph Allocator...\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc, dg.graph);
    ggml_gallocr_alloc_graph(galloc, dg.graph);

    memcpy(freqs_table->data, freqs_data.values.data(), freqs_data.values.size() * sizeof(float));
    size_t alloc_size = ggml_gallocr_get_buffer_size(galloc, 0);
    fprintf(stdout, "Graph allocator reserved %zu MB for intermediate tensors\n",
            alloc_size / 1024 / 1024);

    // ── 6. Diffusion loop ──────────────────────────────────────────
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

    // Image RoPE IDs
    std::vector<float> img_ids(4 * img_tokens, 0.0f);
    for (int y = 0; y < im_H; y++) {
        for (int x = 0; x < im_W; x++) {
            int idx = y * im_W + x;
            img_ids[idx * 4 + 2] = (float)y;
            img_ids[idx * 4 + 3] = (float)x;
        }
    }
    std::vector<float> txt_ids(4 * txt_tokens, 0.0f);

    // Timesteps
    std::vector<float> timesteps(steps);
    for (int i = 0; i < steps; i++)
        timesteps[i] = 1.0f - (float)i / steps;

    fprintf(stdout, "Starting diffusion with %d steps on %d x %d grid...\n", steps, im_H, im_W);

    for (int step = 0; step < steps; step++) {
        float t = timesteps[step];
        float dt = (step < steps - 1) ? t - timesteps[step + 1] : t;

        memcpy(dg.img_in->data,     latents.data(),    latents.size() * sizeof(float));
        memcpy(dg.txt_in->data,     txt_emb.data(),    txt_emb.size() * sizeof(float));
        float t_scaled = t * 1000.0f;
        memcpy(dg.timestep->data,   &t_scaled,         sizeof(float));
        memcpy(dg.img_ids->data,    img_ids.data(),    img_ids.size() * sizeof(float));
        memcpy(dg.txt_ids->data,    txt_ids.data(),    txt_ids.size() * sizeof(float));

        ggml_backend_graph_compute(backend, dg.graph);

        // Dump debug tensors if requested
        {
            static const char * names[] = {"db0_h_img", "db0_h_txt", "sb0_combined"};
            for (int n = 0; n < (int)(sizeof(names)/sizeof(names[0])); n++) {
                ggml_tensor * t = ggml_graph_get_tensor(dg.graph, names[n]);
                if (t && t->data) {
                    char fname[256];
                    snprintf(fname, sizeof(fname), "dump_%s.bin", names[n]);
                    FILE * df = fopen(fname, "wb");
                    if (df) { fwrite(t->data, 1, ggml_nbytes(t), df); fclose(df); }
                }
            }
        }

        float * noise_pred = (float *)dg.out->data;
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

    // ── 7. Save final latents ──────────────────────────────────────
    std::string out_path = "latents_final.bin";
    FILE * of = fopen(out_path.c_str(), "wb");
    if (of) { fwrite(latents.data(), sizeof(float), latents.size(), of); fclose(of); }
    fprintf(stdout, "Saved latents to %s\n", out_path.c_str());

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx_compute);
    ggml_free(wctx);

    return 0;
}
