#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "diffuser_types.h"
#include "diffuser_graph.h"
#include "b1_0_kernel.h"

using namespace bonsai;

struct WeightLoadInfo {
    std::string name;
    enum ggml_type type;
    std::vector<int64_t> ne;
    size_t offset;
};

static uint64_t read_string(FILE * f, std::string & out) {
    uint64_t len;
    fread(&len, sizeof(len), 1, f);
    std::vector<char> buf(len);
    fread(buf.data(), 1, len, f);
    out.assign(buf.data(), len);
    return sizeof(uint64_t) + len;
}

static std::vector<WeightLoadInfo> read_gguf_tensors(const char * fname, size_t & data_offset, size_t & file_size) {
    FILE * f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", fname);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "error: invalid GGUF magic\n");
        exit(1);
    }

    uint32_t version;
    fread(&version, sizeof(version), 1, f);

    int64_t n_tensors, n_kv;
    fread(&n_tensors, sizeof(n_tensors), 1, f);
    fread(&n_kv, sizeof(n_kv), 1, f);

    for (int64_t i = 0; i < n_kv; i++) {
        std::string key;
        read_string(f, key);
        uint32_t val_type;
        fread(&val_type, sizeof(val_type), 1, f);
        switch (val_type) {
            case 0: { uint8_t v; fread(&v, sizeof(v), 1, f); break; }
            case 1: { int8_t v; fread(&v, sizeof(v), 1, f); break; }
            case 2: { uint16_t v; fread(&v, sizeof(v), 1, f); break; }
            case 3: { int16_t v; fread(&v, sizeof(v), 1, f); break; }
            case 4: { uint32_t v; fread(&v, sizeof(v), 1, f); break; }
            case 5: { int32_t v; fread(&v, sizeof(v), 1, f); break; }
            case 6: { float v; fread(&v, sizeof(v), 1, f); break; }
            case 7: { int8_t v; fread(&v, sizeof(v), 1, f); break; }
            case 8: { std::string s; read_string(f, s); break; }
            case 9: {
                uint32_t arr_type, arr_n;
                fread(&arr_type, sizeof(arr_type), 1, f);
                fread(&arr_n, sizeof(arr_n), 1, f);
                for (uint32_t j = 0; j < arr_n; j++) {
                    if (arr_type == 8) { std::string s; read_string(f, s); }
                    else { uint8_t dummy[8]; fread(dummy, 1, 8, f); }
                }
                break;
            }
            case 10: { uint64_t v; fread(&v, sizeof(v), 1, f); break; }
            case 11: { int64_t v; fread(&v, sizeof(v), 1, f); break; }
            case 12: { double v; fread(&v, sizeof(v), 1, f); break; }
            default: { uint8_t dummy[64]; fread(dummy, 1, 64, f); break; }
        }
    }

    std::vector<WeightLoadInfo> infos;
    for (int64_t i = 0; i < n_tensors; i++) {
        WeightLoadInfo info;
        read_string(f, info.name);

        uint32_t n_dims;
        fread(&n_dims, sizeof(n_dims), 1, f);

        info.ne.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            fread(&info.ne[d], sizeof(int64_t), 1, f);
        }

        uint32_t stype;
        fread(&stype, sizeof(stype), 1, f);
        info.type = (enum ggml_type)stype;

        fread(&info.offset, sizeof(size_t), 1, f);

        infos.push_back(info);
    }

    data_offset = ftell(f);
    size_t alignment = 32; // Default GGUF alignment
    size_t padding = (alignment - (data_offset % alignment)) % alignment;
    data_offset += padding;
    fclose(f);
    return infos;
}

static size_t b1_packed_size(int in_dim) {
    return (in_dim / B1_0_BLOCK_SIZE) * B1_0_BLOCK_BYTES;
}

static size_t b1_nbytes(int in_dim, int out_dim) {
    return b1_packed_size(in_dim) * out_dim;
}

static ggml_tensor * load_b1_weight(
    ggml_context * ctx,
    FILE * f,
    size_t data_offset,
    const std::string & name,
    int in_dim,
    int out_dim)
{
    size_t nbytes = b1_nbytes(in_dim, out_dim);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, nbytes, 1);
    t->data = malloc(nbytes);
    if (!t->data) { fprintf(stderr, "error: malloc failed for %s\n", name.c_str()); exit(1); }
    return t;
}

static void read_weight_data(FILE * f, ggml_tensor * t, size_t data_offset, size_t file_offset, size_t nbytes) {
    fseek(f, data_offset + file_offset, SEEK_SET);
    size_t read = fread(t->data, 1, nbytes, f);
    if (read != nbytes) {
        fprintf(stderr, "error: failed to read %zu bytes for tensor\n", nbytes);
        exit(1);
    }
}

static ggml_tensor * load_f32_weight(ggml_context * ctx, int n) {
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
}

static void find_b1_weights(
    FILE * f,
    size_t data_offset,
    const std::vector<WeightLoadInfo> & infos,
    ggml_context * ctx,
    const std::string & name,
    B1Weights & w,
    int in_dim,
    int out_dim)
{
    w.in_dim = in_dim;
    w.out_dim = out_dim;
    size_t nbytes = b1_nbytes(in_dim, out_dim);
    w.data = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, nbytes, 1);
    w.data->data = malloc(nbytes);

    for (const auto & info : infos) {
        if (info.name == name) {
            read_weight_data(f, w.data, data_offset, info.offset, nbytes);
            return;
        }
    }
    fprintf(stderr, "FATAL ERROR: required B1 weight '%s' not found in GGUF!\n", name.c_str());
    abort();
}

static void find_f32_weight(
    FILE * f,
    size_t data_offset,
    const std::vector<WeightLoadInfo> & infos,
    ggml_context * ctx,
    const std::string & name,
    FPWeights & w,
    int n)
{
    w.data = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    w.data->data = malloc(n * sizeof(float));

    for (const auto & info : infos) {
        if (info.name == name) {
            if (info.type == GGML_TYPE_F32) {
                size_t nbytes = n * sizeof(float);
                read_weight_data(f, w.data, data_offset, info.offset, nbytes);
                if (name == "transformer_blocks.0.attn.norm_q.weight") {
                    float * d = (float *)w.data->data;
                    fprintf(stderr, "DEBUG_NORM_Q GGUF LOAD F32: %f %f %f\n", d[0], d[1], d[2]);
                }
            } else if (info.type == GGML_TYPE_F16) {
                size_t nbytes = n * sizeof(uint16_t);
                std::vector<uint16_t> temp(n);
                fseek(f, data_offset + info.offset, SEEK_SET);
                size_t read = fread(temp.data(), 1, nbytes, f);
                float * out = (float *)w.data->data;
                for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(temp[i]);
            } else if (info.type == GGML_TYPE_BF16) {
                size_t nbytes = n * sizeof(uint16_t);
                std::vector<uint16_t> temp(n);
                fseek(f, data_offset + info.offset, SEEK_SET);
                size_t read = fread(temp.data(), 1, nbytes, f);
                float * out = (float *)w.data->data;
                for (int i = 0; i < n; i++) out[i] = ggml_bf16_to_fp32(ggml_bf16_t{temp[i]});
            } else {
                fprintf(stderr, "ERROR: unsupported type %d for %s\n", info.type, name.c_str());
                abort();
            }
            return;
        }
    }
    fprintf(stderr, "FATAL ERROR: required weight '%s' not found in GGUF!\n", name.c_str());
    abort();
}

static void find_f32_weight_optional(
    FILE * f,
    size_t data_offset,
    const std::vector<WeightLoadInfo> & infos,
    ggml_context * ctx,
    const std::string & name,
    FPWeights & w,
    int n)
{
    for (const auto & info : infos) {
        if (info.name == name) {
            w.data = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            w.data->data = malloc(n * sizeof(float));
            if (info.type == GGML_TYPE_F32) {
                size_t nbytes = n * sizeof(float);
                read_weight_data(f, w.data, data_offset, info.offset, nbytes);
            } else if (info.type == GGML_TYPE_F16) {
                size_t nbytes = n * sizeof(uint16_t);
                std::vector<uint16_t> temp(n);
                fseek(f, data_offset + info.offset, SEEK_SET);
                size_t read = fread(temp.data(), 1, nbytes, f);
                float * out = (float *)w.data->data;
                for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(temp[i]);
            } else if (info.type == GGML_TYPE_BF16) {
                size_t nbytes = n * sizeof(uint16_t);
                std::vector<uint16_t> temp(n);
                fseek(f, data_offset + info.offset, SEEK_SET);
                size_t read = fread(temp.data(), 1, nbytes, f);
                float * out = (float *)w.data->data;
                for (int i = 0; i < n; i++) out[i] = ggml_bf16_to_fp32(ggml_bf16_t{temp[i]});
            } else {
                fprintf(stderr, "ERROR: unsupported type %d for %s\n", info.type, name.c_str());
                abort();
            }
            return;
        }
    }
    w.data = nullptr;
}

static void load_diffuser_weights(
    FILE * f,
    size_t data_offset,
    const std::vector<WeightLoadInfo> & infos,
    ggml_context * ctx,
    const DiffuserParams & params,
    DiffuserWeights & w)
{
    int H = params.hidden_size;
    int C = params.in_channels;
    int ctx_dim = params.context_in_dim;
    int mlp_hd = params.mlp_hidden_dim;

    find_b1_weights(f, data_offset, infos, ctx, "x_embedder.weight", w.img_in, C, H);
    find_b1_weights(f, data_offset, infos, ctx, "context_embedder.weight", w.txt_in, ctx_dim, H);

    find_f32_weight(f, data_offset, infos, ctx, "time_guidance_embed.timestep_embedder.linear_1.weight", w.time_in_w1, 256 * H);
    find_f32_weight_optional(f, data_offset, infos, ctx, "time_guidance_embed.timestep_embedder.linear_1.bias", w.time_in_b1, H);
    find_f32_weight(f, data_offset, infos, ctx, "time_guidance_embed.timestep_embedder.linear_2.weight", w.time_in_w2, H * H);
    find_f32_weight_optional(f, data_offset, infos, ctx, "time_guidance_embed.timestep_embedder.linear_2.bias", w.time_in_b2, H);

    find_b1_weights(f, data_offset, infos, ctx, "double_stream_img.linear.weight", w.double_mod_img, H, 6 * H);
    find_b1_weights(f, data_offset, infos, ctx, "double_stream_txt.linear.weight", w.double_mod_txt, H, 6 * H);
    find_b1_weights(f, data_offset, infos, ctx, "single_stream_modulation.linear.weight", w.single_mod, H, 3 * H);

    w.double_blocks.resize(params.depth);
    for (int d = 0; d < params.depth; d++) {
        auto & db = w.double_blocks[d];
        std::string prefix = "transformer_blocks." + std::to_string(d);

        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.to_q.weight", db.attn_to_q, H, H);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.to_k.weight", db.attn_to_k, H, H);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.to_v.weight", db.attn_to_v, H, H);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.to_out.0.weight", db.attn_to_out, H, H);
        find_f32_weight(f, data_offset, infos, ctx, prefix + ".attn.norm_q.weight", db.attn_norm_q, params.head_dim);
        find_f32_weight(f, data_offset, infos, ctx, prefix + ".attn.norm_k.weight", db.attn_norm_k, params.head_dim);

        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.add_q_proj.weight", db.attn_add_q, H, H);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.add_k_proj.weight", db.attn_add_k, H, H);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.add_v_proj.weight", db.attn_add_v, H, H);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.to_add_out.weight", db.attn_add_out, H, H);
        find_f32_weight(f, data_offset, infos, ctx, prefix + ".attn.norm_added_q.weight", db.attn_norm_added_q, params.head_dim);
        find_f32_weight(f, data_offset, infos, ctx, prefix + ".attn.norm_added_k.weight", db.attn_norm_added_k, params.head_dim);

        find_b1_weights(f, data_offset, infos, ctx, prefix + ".ff.linear_in.weight", db.ff_linear_in, H, mlp_hd * 2);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".ff.linear_out.weight", db.ff_linear_out, mlp_hd, H);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".ff_context.linear_in.weight", db.ff_ctx_linear_in, H, mlp_hd * 2);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".ff_context.linear_out.weight", db.ff_ctx_linear_out, mlp_hd, H);
    }

    w.single_blocks.resize(params.depth_single_blocks);
    for (int s = 0; s < params.depth_single_blocks; s++) {
        auto & sb = w.single_blocks[s];
        std::string prefix = "blk." + std::to_string(s);
        int qkv_mlp_dim = 3 * H + mlp_hd * 2;

        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.to_qkv_mlp_proj.weight", sb.to_qkv_mlp_proj, H, qkv_mlp_dim);
        find_b1_weights(f, data_offset, infos, ctx, prefix + ".attn.to_out.weight", sb.to_out, H + mlp_hd, H);
        find_f32_weight(f, data_offset, infos, ctx, prefix + ".attn.norm_q.weight", sb.norm_q, params.head_dim);
        find_f32_weight(f, data_offset, infos, ctx, prefix + ".attn.norm_k.weight", sb.norm_k, params.head_dim);
    }

    find_b1_weights(f, data_offset, infos, ctx, "norm_out.linear.weight", w.norm_out_linear, H, 2 * H);
    find_b1_weights(f, data_offset, infos, ctx, "proj_out.weight", w.proj_out, H, C);
}

int main(int argc, char ** argv) {
    std::string embedding_path;
    std::string model_path = "models/flux2_4b_1bit.gguf";
    int H = 64;
    int W = 64;
    int steps = 20;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--embedding" && i + 1 < argc) embedding_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "-h" && i + 1 < argc) H = std::stoi(argv[++i]);
        else if (arg == "-w" && i + 1 < argc) W = std::stoi(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) n_threads = std::stoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s --embedding <path> [--model path] [-h H] [-w W] [--steps N] [--threads N]\n", argv[0]);
            return 1;
        }
    }

    if (embedding_path.empty()) {
        fprintf(stderr, "error: --embedding is required\n");
        return 1;
    }

    fprintf(stdout, "Loading model from %s...\n", model_path.c_str());
    size_t data_offset = 0, file_size = 0;
    std::vector<WeightLoadInfo> infos = read_gguf_tensors(model_path.c_str(), data_offset, file_size);

    FILE * f = fopen(model_path.c_str(), "rb");
    if (!f) { perror("fopen"); return 1; }

    DiffuserParams params;
    int img_tokens = H * W;
    int txt_tokens = 512;

    size_t ctx_size = 13ULL * 1024 * 1024 * 1024;
    std::vector<uint8_t> buf(ctx_size);
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(gparams);

    DiffuserWeights weights;
    load_diffuser_weights(f, data_offset, infos, ctx, params, weights);

    fprintf(stdout, "Loading embeddings from %s...\n", embedding_path.c_str());
    FILE * ef = fopen(embedding_path.c_str(), "rb");
    if (!ef) { perror("fopen embedding"); return 1; }
    fseek(ef, 0, SEEK_END);
    long emb_size = ftell(ef);
    fseek(ef, 0, SEEK_SET);
    std::vector<float> txt_emb(emb_size / sizeof(float));
    size_t emb_read = fread(txt_emb.data(), 1, emb_size, ef);
    if (emb_read != (size_t)emb_size) { fprintf(stderr, "error reading embedding\n"); return 1; }
    fclose(ef);

    fprintf(stdout, "Building computation graph...\n");
    DiffuserGraph dg = build_diffuser_graph(ctx, params, weights, img_tokens, txt_tokens, 1, n_threads);

    int C = params.in_channels;
    std::vector<float> latents(C * img_tokens);
    
    FILE * f_latent = fopen("tmp/x_input.bin", "rb");
    if (f_latent) {
        fprintf(stdout, "Loading initial latents from tmp/x_input.bin for exact parity...\n");
        size_t read_count = fread(latents.data(), sizeof(float), latents.size(), f_latent);
        if (read_count != latents.size()) {
            fprintf(stderr, "warning: expected %zu elements in x_input.bin, got %zu\n", latents.size(), read_count);
        }
        fclose(f_latent);
    } else {
        fprintf(stdout, "tmp/x_input.bin not found, using random latents...\n");
        for (int i = 0; i < C * img_tokens; i++) {
            latents[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    std::vector<float> img_ids(4 * img_tokens, 0.0f);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            img_ids[idx * 4 + 0] = 0.0f;
            img_ids[idx * 4 + 1] = 0.0f;
            img_ids[idx * 4 + 2] = (float)y;
            img_ids[idx * 4 + 3] = (float)x;
        }
    }

    std::vector<float> txt_ids(4 * txt_tokens, 0.0f);

    std::vector<float> timesteps(steps);
    for (int i = 0; i < steps; i++) {
        timesteps[i] = 1.0f - (float)i / steps;
    }

    fprintf(stdout, "Starting diffusion with %d steps on %d x %d grid...\n", steps, H, W);

    for (int step = 0; step < steps; step++) {
        float t = timesteps[step];
        float dt = (step < steps - 1) ? t - timesteps[step + 1] : t;

        memcpy(dg.img_in->data, latents.data(), latents.size() * sizeof(float));
        memcpy(dg.txt_in->data, txt_emb.data(), txt_emb.size() * sizeof(float));
        memcpy(dg.timestep->data, &t, sizeof(float));
        memcpy(dg.img_ids->data, img_ids.data(), img_ids.size() * sizeof(float));
        memcpy(dg.txt_ids->data, txt_ids.data(), txt_ids.size() * sizeof(float));

        {
            int nan_in = 0, inf_in = 0;
            for (size_t i = 0; i < latents.size(); i++) {
                if (std::isnan(latents[i])) nan_in++;
                if (std::isinf(latents[i])) inf_in++;
            }
            for (size_t i = 0; i < txt_emb.size(); i++) {
                if (std::isnan(txt_emb[i])) nan_in++;
                if (std::isinf(txt_emb[i])) inf_in++;
            }
            fprintf(stdout, "  inputs: latents nan=%d inf=%d  txt_emb nan=%d inf=%d  t=%f\n",
                nan_in, inf_in, 0, 0, t);
        }

        ggml_graph_compute_with_ctx(ctx, dg.graph, n_threads);

        {
            static const char * names[] = {
                "te_emb", "te_w1", "te_pre_silu", "te_post_silu", "vec", "vec_final"
            };
            for (int n = 0; n < (int)(sizeof(names)/sizeof(names[0])); n++) {
                ggml_tensor * t = ggml_graph_get_tensor(dg.graph, names[n]);
                if (t && t->data) {
                    const float * p = (const float *)t->data;
                    int nn = (int)ggml_nelements(t);
                    int nan_c = 0, inf_c = 0;
                    float mn = 1e30f, mx = -1e30f;
                    for (int i = 0; i < nn; i++) {
                        if (std::isnan(p[i])) nan_c++;
                        if (std::isinf(p[i])) inf_c++;
                        if (p[i] < mn) mn = p[i];
                        if (p[i] > mx) mx = p[i];
                    }
                    fprintf(stdout, "  [%s] n=%d nan=%d inf=%d min=%.4f max=%.4f\n",
                        names[n], nn, nan_c, inf_c, mn, mx);
                } else {
                    fprintf(stdout, "  [%s] not found\n", names[n]);
                }
            }
        }


        float * noise_pred = (float *)dg.out->data;

        int nan_count = 0, inf_count = 0;
        float mn = 1e30f, mx = -1e30f, sum = 0.0f;
        for (int i = 0; i < C * img_tokens; i++) {
            float v = noise_pred[i];
            if (std::isnan(v)) nan_count++;
            if (std::isinf(v)) inf_count++;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
        }
        fprintf(stdout, "  step %3d/%d: out stats: nan=%d inf=%d min=%.4f max=%.4f mean=%.4f\n",
            step + 1, steps, nan_count, inf_count, mn, mx, sum / (C * img_tokens));

        for (int i = 0; i < C * img_tokens; i++) {
            latents[i] = latents[i] + dt * noise_pred[i];
        }

        fprintf(stdout, "  step %3d/%d: t=%.4f  dt=%.4f\n", step + 1, steps, t, dt);
    }

    fprintf(stdout, "Done! Final latent shape: %d x %d x %d\n", C, H, W);

    std::string out_path = "latents_final.bin";
    FILE * of = fopen(out_path.c_str(), "wb");
    if (of) {
        fwrite(latents.data(), sizeof(float), latents.size(), of);
        fclose(of);
        fprintf(stdout, "Saved latents to %s\n", out_path.c_str());
    }

    fclose(f);
    ggml_free(ctx);

    return 0;
}
