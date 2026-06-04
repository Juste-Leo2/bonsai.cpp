#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

#include "safetensors.h"
#include "vae_graph.h"
#include "vae_types.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

void log_info(const std::string & msg) { std::cout << "[INFO] " << msg << std::endl; }
void log_err (const std::string & msg) { std::cerr << "[ERROR] " << msg << std::endl; }

// Decode a linear weight from PyTorch layout (out, in) into a ggml tensor
// stored as (ne0=in, ne1=out).
// Returns the destination ggml_tensor* (already in the weight context).
ggml_tensor * load_linear_w(ggml_context * wctx,
                            bonsai::WeightsView & view,
                            const bonsai::SafeTensor & t) {
    if (t.shape.size() != 2) {
        throw std::runtime_error("expected 2D linear weight: " + t.name);
    }
    int64_t out_dim = t.shape[0];
    int64_t in_dim  = t.shape[1];
    ggml_tensor * dst = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, in_dim, out_dim);
    bonsai::WeightsView::Linear L;
    L.dst = dst;
    L.src = reinterpret_cast<const float *>(t.data);
    L.in_dim = in_dim;
    L.out_dim = out_dim;
    view.linears.push_back(L);
    return dst;
}

// Decode a 4D conv weight from PyTorch layout (Co, Ci, kH, kW) into a ggml
// tensor stored as (ne0=kW, ne1=kH, ne2=Ci, ne3=Co).
ggml_tensor * load_conv_w(ggml_context * wctx,
                          bonsai::WeightsView & view,
                          const bonsai::SafeTensor & t) {
    if (t.shape.size() != 4) {
        throw std::runtime_error("expected 4D conv weight: " + t.name);
    }
    int64_t Co = t.shape[0];
    int64_t Ci = t.shape[1];
    int64_t kH = t.shape[2];
    int64_t kW = t.shape[3];
    ggml_tensor * dst = ggml_new_tensor_4d(wctx, GGML_TYPE_F32, kW, kH, Ci, Co);
    bonsai::WeightsView::Conv C;
    C.dst = dst;
    C.src = reinterpret_cast<const float *>(t.data);
    C.out_c = Co; C.in_c = Ci; C.kh = kH; C.kw = kW;
    view.convs.push_back(C);
    return dst;
}

// Decode a 1D weight or bias (no permutation)
ggml_tensor * load_vec(ggml_context * wctx,
                       bonsai::WeightsView & view,
                       const bonsai::SafeTensor & t) {
    if (t.shape.size() != 1) {
        throw std::runtime_error("expected 1D tensor: " + t.name);
    }
    ggml_tensor * dst = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, t.shape[0]);
    bonsai::WeightsView::Vec V;
    V.dst = dst;
    V.src = reinterpret_cast<const float *>(t.data);
    V.n = t.shape[0];
    view.vecs.push_back(V);
    return dst;
}

// Materialize all the queued permutations: copy from src to dst in ggml layout.
void materialize_views(const bonsai::WeightsView & view) {
    for (const auto & L : view.linears) {
        // src is (out, in) row-major, dst is (in, out) row-major
        // We need to transpose.
        float * dp = ggml_get_data_f32(L.dst);
        for (int64_t o = 0; o < L.out_dim; ++o) {
            for (int64_t i = 0; i < L.in_dim; ++i) {
                dp[i * L.out_dim + o] = L.src[o * L.in_dim + i];
            }
        }
    }
    for (const auto & C : view.convs) {
        // src is (Co, Ci, kH, kW) row-major
        // dst is (kW, kH, Ci, Co) row-major
        // src[co, ci, kh, kw] -> dst[kw, kh, ci, co]
        float * dp = ggml_get_data_f32(C.dst);
        for (int64_t co = 0; co < C.out_c; ++co) {
            for (int64_t ci = 0; ci < C.in_c; ++ci) {
                for (int64_t kh = 0; kh < C.kh; ++kh) {
                    for (int64_t kw = 0; kw < C.kw; ++kw) {
                        int64_t src_idx = ((co * C.in_c + ci) * C.kh + kh) * C.kw + kw;
                        int64_t dst_idx = ((kw * C.kh + kh) * C.in_c + ci) * C.out_c + co;
                        dp[dst_idx] = C.src[src_idx];
                    }
                }
            }
        }
    }
    for (const auto & V : view.vecs) {
        std::memcpy(ggml_get_data_f32(V.dst), V.src, V.n * sizeof(float));
    }
}

bonsai::Conv2d load_conv2d(ggml_context * wctx, bonsai::WeightsView & view,
                           bonsai::SafetensorsFile & st,
                           const std::string & prefix) {
    bonsai::Conv2d c;
    auto * w = st.find(prefix + ".weight");
    auto * b = st.find(prefix + ".bias");
    if (!w) throw std::runtime_error("missing weight: " + prefix + ".weight");
    if (!b) throw std::runtime_error("missing bias: " + prefix + ".bias");
    c.w = load_conv_w(wctx, view, *w);
    c.b = load_vec(wctx, view, *b);
    return c;
}

bonsai::ResNet load_resnet(ggml_context * wctx, bonsai::WeightsView & view,
                           bonsai::SafetensorsFile & st,
                           const std::string & prefix, bool has_shortcut) {
    bonsai::ResNet r;
    r.conv1 = load_conv2d(wctx, view, st, prefix + ".conv1");
    {
        auto * w = st.find(prefix + ".norm1.weight");
        auto * b = st.find(prefix + ".norm1.bias");
        if (!w || !b) throw std::runtime_error("missing norm1 in " + prefix);
        r.norm1_w = load_vec(wctx, view, *w);
        r.norm1_b = load_vec(wctx, view, *b);
    }
    r.conv2 = load_conv2d(wctx, view, st, prefix + ".conv2");
    {
        auto * w = st.find(prefix + ".norm2.weight");
        auto * b = st.find(prefix + ".norm2.bias");
        if (!w || !b) throw std::runtime_error("missing norm2 in " + prefix);
        r.norm2_w = load_vec(wctx, view, *w);
        r.norm2_b = load_vec(wctx, view, *b);
    }
    if (has_shortcut) {
        r.conv_shortcut = load_conv2d(wctx, view, st, prefix + ".conv_shortcut");
    }
    return r;
}

bonsai::Attention load_attention(ggml_context * wctx, bonsai::WeightsView & view,
                                 bonsai::SafetensorsFile & st,
                                 const std::string & prefix) {
    bonsai::Attention a;
    {
        auto * w = st.find(prefix + ".group_norm.weight");
        auto * b = st.find(prefix + ".group_norm.bias");
        if (!w || !b) throw std::runtime_error("missing group_norm in " + prefix);
        a.group_norm_w = load_vec(wctx, view, *w);
        a.group_norm_b = load_vec(wctx, view, *b);
    }
    auto load_lin = [&](const std::string & name, ggml_tensor * & W, ggml_tensor * & B) {
        auto * wt = st.find(prefix + "." + name + ".weight");
        auto * bs = st.find(prefix + "." + name + ".bias");
        if (!wt || !bs) throw std::runtime_error("missing " + name + " in " + prefix);
        W = load_linear_w(wctx, view, *wt);
        B = load_vec(wctx, view, *bs);
    };
    load_lin("to_q",    a.to_q_w,    a.to_q_b);
    load_lin("to_k",    a.to_k_w,    a.to_k_b);
    load_lin("to_v",    a.to_v_w,    a.to_v_b);
    load_lin("to_out.0", a.to_out_w,  a.to_out_b);
    return a;
}

bonsai::MidBlock load_mid_block(ggml_context * wctx, bonsai::WeightsView & view,
                                 bonsai::SafetensorsFile & st) {
    bonsai::MidBlock m;
    m.resnets.push_back(load_resnet(wctx, view, st, "decoder.mid_block.resnets.0", false));
    m.resnets.push_back(load_resnet(wctx, view, st, "decoder.mid_block.resnets.1", false));
    m.attn = load_attention(wctx, view, st, "decoder.mid_block.attentions.0");
    return m;
}

bonsai::UpBlock load_up_block(ggml_context * wctx, bonsai::WeightsView & view,
                              bonsai::SafetensorsFile & st,
                              const std::string & prefix, bool has_upsampler) {
    bonsai::UpBlock u;
    for (int i = 0; i < 3; ++i) {
        // First resnet of up_blocks.2 and .3 has conv_shortcut
        bool shortcut = (i == 0) &&
            (prefix == "decoder.up_blocks.2" || prefix == "decoder.up_blocks.3");
        u.resnets.push_back(load_resnet(wctx, view, st, prefix + ".resnets." + std::to_string(i), shortcut));
    }
    if (has_upsampler) {
        u.upsampler_conv = load_conv2d(wctx, view, st, prefix + ".upsamplers.0.conv");
        u.has_upsampler = true;
    }
    return u;
}

bonsai::VAEWeights load_vae_weights(ggml_context * wctx, bonsai::SafetensorsFile & st) {
    bonsai::WeightsView view;
    bonsai::VAEWeights W;

    log_info("loading post_quant_conv...");
    W.post_quant_conv = load_conv2d(wctx, view, st, "post_quant_conv");

    log_info("loading conv_in...");
    W.conv_in = load_conv2d(wctx, view, st, "decoder.conv_in");

    log_info("loading mid_block...");
    W.mid_block = load_mid_block(wctx, view, st);

    log_info("loading up_blocks.0...");
    W.up_blocks.push_back(load_up_block(wctx, view, st, "decoder.up_blocks.0", true));
    log_info("loading up_blocks.1...");
    W.up_blocks.push_back(load_up_block(wctx, view, st, "decoder.up_blocks.1", true));
    log_info("loading up_blocks.2...");
    W.up_blocks.push_back(load_up_block(wctx, view, st, "decoder.up_blocks.2", true));
    log_info("loading up_blocks.3...");
    W.up_blocks.push_back(load_up_block(wctx, view, st, "decoder.up_blocks.3", false));

    log_info("loading conv_norm_out...");
    {
        auto * w = st.find("decoder.conv_norm_out.weight");
        auto * b = st.find("decoder.conv_norm_out.bias");
        if (!w || !b) throw std::runtime_error("missing conv_norm_out");
        W.conv_norm_out_w = load_vec(wctx, view, *w);
        W.conv_norm_out_b = load_vec(wctx, view, *b);
    }

    log_info("loading conv_out...");
    W.conv_out = load_conv2d(wctx, view, st, "decoder.conv_out");

    log_info("materializing weight permutations...");
    materialize_views(view);

    return W;
}

void write_png(const std::string & path, int w, int h, const uint8_t * rgb) {
    int ok = stbi_write_png(path.c_str(), w, h, 3, rgb, w * 3);
    if (!ok) {
        log_err("failed to write PNG: " + path);
    } else {
        log_info("wrote " + path + " (" + std::to_string(w) + "x" + std::to_string(h) + ")");
    }
}

// Convert a CHW float tensor in [0, 1] to an HWC uint8 RGB buffer.
// Tensor layout (ggml): ne[0]=W, ne[1]=H, ne[2]=3, ne[3]=1
void chw_f32_to_hwc_u8(const float * src, uint8_t * dst, int W, int H) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int spatial = y * W + x;
            float r = src[0 * W * H + spatial];
            float g = src[1 * W * H + spatial];
            float b = src[2 * W * H + spatial];
            int idx = (y * W + x) * 3;
            dst[idx + 0] = (uint8_t) std::min(255, std::max(0, (int) (r * 255.0f)));
            dst[idx + 1] = (uint8_t) std::min(255, std::max(0, (int) (g * 255.0f)));
            dst[idx + 2] = (uint8_t) std::min(255, std::max(0, (int) (b * 255.0f)));
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path = "models/flux2-vae.safetensors";
    std::string latent_path;
    std::string output_path = "out.png";
    int latent_h = 32;
    int latent_w = 32;
    int n_threads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc)       model_path  = argv[++i];
        else if (a == "--latent" && i + 1 < argc)  latent_path = argv[++i];
        else if (a == "--output" && i + 1 < argc)  output_path = argv[++i];
        else if (a == "--H" && i + 1 < argc)       latent_h    = std::atoi(argv[++i]);
        else if (a == "--W" && i + 1 < argc)       latent_w    = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) n_threads   = std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --model <path>     path to flux2-vae.safetensors (default: models/flux2-vae.safetensors)\n"
                      << "  --latent <path>    path to input latent .bin file (BCHW F32, default: synthetic noise)\n"
                      << "  --output <path>    output PNG (default: out.png)\n"
                      << "  --H <int>          latent height (default: 32)\n"
                      << "  --W <int>          latent width  (default: 32)\n"
                      << "  --threads <int>    ggml threads (default: 4)\n";
            return 0;
        }
    }

    log_info("bonsai_vae v0.1.0 - Flux VAE decoder (ggml backend)");
    log_info("model:  " + model_path);
    log_info("latent: " + (latent_path.empty() ? std::string("<synthetic>") : latent_path) +
             " (" + std::to_string(latent_h) + "x" + std::to_string(latent_w) + ")");
    log_info("output: " + output_path);

    // --- Load safetensors ---
    log_info("opening safetensors file...");
    bonsai::SafetensorsFile st;
    try {
        st.open(model_path);
    } catch (const std::exception & e) {
        log_err(std::string("open failed: ") + e.what());
        return 1;
    }
    log_info("loaded " + std::to_string(st.tensors().size()) + " tensor entries");

    // --- Build weights context (allocates tensor storage, copies data) ---
    // Estimate weight memory: 336 MB for the full model
    size_t wctx_size = 400 * 1024 * 1024;
    ggml_init_params wparams{ wctx_size, nullptr, false };
    ggml_context * wctx = ggml_init(wparams);
    if (!wctx) {
        log_err("failed to allocate weights context");
        return 1;
    }

    bonsai::VAEWeights W;
    try {
        W = load_vae_weights(wctx, st);
    } catch (const std::exception & e) {
        log_err(std::string("loading failed: ") + e.what());
        ggml_free(wctx);
        return 1;
    }
    log_info("all weights loaded into ggml");

    // --- Load or synthesize latent ---
    const int latent_channels = 32;
    int latent_size = 1 * latent_channels * latent_h * latent_w;
    std::vector<float> latent((size_t) latent_size, 0.0f);

    if (!latent_path.empty()) {
        log_info("loading latent from " + latent_path);
        std::ifstream f(latent_path, std::ios::binary);
        if (!f) {
            log_err("failed to open latent file: " + latent_path);
            ggml_free(wctx);
            return 1;
        }
        f.read(reinterpret_cast<char *>(latent.data()), latent_size * sizeof(float));
        if (f.gcount() != (std::streamsize)(latent_size * sizeof(float))) {
            log_err("latent file too small (expected " +
                    std::to_string(latent_size * sizeof(float)) + " bytes)");
            ggml_free(wctx);
            return 1;
        }
    } else {
        log_info("synthesizing random latent (seed 42)");
        // simple LCG so output is deterministic
        uint32_t seed = 42;
        for (auto & v : latent) {
            seed = seed * 1664525u + 1013904223u;
            v = ((float)(seed & 0xFFFF) / 32768.0f) - 1.0f;  // [-1, 1]
        }
    }

    // --- Build activations context ---
    // Memory budget: ~16 GB virtual (mmap'd, so the OS only commits pages we
    // actually touch). The decoder accumulates many intermediate tensors,
    // and im2col temps for a 256x256 latent can be 600+ MB each.
    size_t actx_size = 16ULL * 1024 * 1024 * 1024;
    void * actx_buf = mmap(nullptr, actx_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (actx_buf == MAP_FAILED) {
        log_err("failed to mmap activations context buffer");
        ggml_free(wctx);
        return 1;
    }
    ggml_init_params aparams{ actx_size, actx_buf, false };
    ggml_context * actx = ggml_init(aparams);
    if (!actx) {
        log_err("failed to allocate activations context");
        ggml_free(wctx);
        return 1;
    }

    // Input latent tensor: ggml layout (W, H, C, N)
    ggml_tensor * x = ggml_new_tensor_4d(actx, GGML_TYPE_F32,
                                         (int64_t) latent_w,
                                         (int64_t) latent_h,
                                         (int64_t) latent_channels,
                                         1);
    std::memcpy(ggml_get_data_f32(x), latent.data(), latent_size * sizeof(float));

    // Build the decoder graph
    log_info("building decoder graph...");
    ggml_tensor * out = bonsai::build_decoder_graph(actx, W, x);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(actx);
    ggml_build_forward_expand(gf, out);

    log_info("graph built for decoder");
    log_info("activation memory: " +
             std::to_string(ggml_used_mem(actx) / (1024 * 1024)) + " MB");

    // --- Execute ---
    log_info("running graph on CPU (" + std::to_string(n_threads) + " threads)...");
    auto t0 = std::chrono::high_resolution_clock::now();
    ggml_status st_g = ggml_graph_compute_with_ctx(actx, gf, n_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (st_g != GGML_STATUS_SUCCESS) {
        log_err("graph compute failed with status " + std::to_string((int)st_g));
        ggml_free(actx);
        ggml_free(wctx);
        return 1;
    }
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    log_info("done in " + std::to_string((int) ms) + " ms");

    // --- Read back output ---
    int out_h = (int) out->ne[1];
    int out_w = (int) out->ne[0];
    int out_c = (int) out->ne[2];
    log_info("output: " + std::to_string(out_w) + "x" + std::to_string(out_h) +
             "x" + std::to_string(out_c));

    if (out_c != 3) {
        log_err("expected 3 output channels, got " + std::to_string(out_c));
        ggml_free(actx);
        ggml_free(wctx);
        return 1;
    }

    std::vector<float> out_data((size_t)(out_h * out_w * out_c));
    std::memcpy(out_data.data(), ggml_get_data_f32(out), out_data.size() * sizeof(float));

    // Convert to HWC uint8 and write PNG
    std::vector<uint8_t> rgb((size_t)(out_h * out_w * 3));
    chw_f32_to_hwc_u8(out_data.data(), rgb.data(), out_w, out_h);
    write_png(output_path, out_w, out_h, rgb.data());

    ggml_free(actx);
    ggml_free(wctx);
    munmap(actx_buf, actx_size);
    log_info("cleanup done. bye.");
    return 0;
}
