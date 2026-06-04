#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
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

// Load a linear weight from PyTorch layout (out, in) into a ggml tensor
// in PyTorch layout (in, out) [renamed to ggml shape (ne0=in, ne1=out)].
static ggml_tensor * load_linear_w(ggml_context * wctx,
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

// Load a 4D conv weight, stored in PyTorch (Co, Ci, kH, kW) row-major order.
// We allocate with the ggml *kernel* shape (kW, kH, Ci, Co) but the data
// is initially copied in PyTorch order. materialize_views fixes the layout
// via ggml permute + cont.
static ggml_tensor * load_conv_w(ggml_context * wctx,
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

static ggml_tensor * load_vec(ggml_context * wctx,
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

// Materialize all the queued weight permutations with manual byte copies.
//
// CRITICAL: ggml_conv_2d_direct has a non-obvious behavior. It reshapes the
// kernel (KW, KH, IC, OC) to 2D (KW*KH*IC, OC) and then does a mul_mat where
// the (KW*KH*IC) axis is the inner dim of the dot product. The mul_mat
// implementation accesses this 2D matrix in COLUMN-MAJOR (i.e. as if the
// outer axis were rows and the inner axis were columns, transposed from
// the natural row-major storage). So the EFFECTIVE layout that the conv
// reads is (OC, KW*KH*IC) row-major — NOT (KW*KH*IC, OC) as the
// reshape suggests. We must permute the PyTorch weight data accordingly.
//
// The mapping: dst byte offset (oc*knl_n + (kw*KH+kh)*IC + ic)*4 holds
// the PyTorch value [oc, ic, kh, kw].
static void materialize_views(const bonsai::WeightsView & view) {
    for (const auto & L : view.linears) {
        // For linear weights: ggml_mul_mat (in attention block) is row-major
        // with the inner dim being ne0. The weight is stored with ne=(in, out),
        // and the matmul does (in, seq) @ (in, out)^T = (out, seq). So
        // PyTorch (out, in) row-major must become ggml (in, out) row-major.
        float * dp = ggml_get_data_f32(L.dst);
        for (int64_t o = 0; o < L.out_dim; ++o) {
            for (int64_t i = 0; i < L.in_dim; ++i) {
                dp[i * L.out_dim + o] = L.src[o * L.in_dim + i];
            }
        }
    }
    for (const auto & C : view.convs) {
        // dst has ne=(KW, KH, IC, OC). The conv2d_direct reads it as a 2D
        // matrix with the inner dim being the FLATTENED (KW, KH, IC) axis
        // and the outer dim being OC, but accessed in COLUMN-MAJOR order.
        //
        // dst byte offset for the element logically at (kw, kh, ic, oc):
        //   (oc * (KW*KH*IC) + (kw*KH + kh)*IC + ic) * 4
        // = (oc * knl_n + (kw*KH+kh)*IC + ic) * 4
        //
        // PyTorch source offset for [oc, ic, kh, kw]:
        //   (oc * Ci*kH*kW + ic * kH*kW + kh * kW + kw) * 4
        float * dp = ggml_get_data_f32(C.dst);
        int64_t knl_n = C.kw * C.kh * C.in_c;
        for (int64_t oc = 0; oc < C.out_c; ++oc) {
            for (int64_t ic = 0; ic < C.in_c; ++ic) {
                for (int64_t kh = 0; kh < C.kh; ++kh) {
                    for (int64_t kw = 0; kw < C.kw; ++kw) {
                        int64_t s = ((oc * C.in_c + ic) * C.kh + kh) * C.kw + kw;
                        int64_t d = oc * knl_n + (kw * C.kh + kh) * C.in_c + ic;
                        dp[d] = C.src[s];
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

bonsai::VAEWeights load_vae_weights(ggml_context * wctx, bonsai::SafetensorsFile & st,
                                    bonsai::WeightsView & view) {
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
    int latent_h = 16;
    int latent_w = 16;
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
                      << "  --latent <path>    path to input latent .bin file (default: synthetic noise)\n"
                      << "  --output <path>    output PNG (default: out.png)\n"
                      << "  --H <int>          ENCODED latent height in (B,128,H,W) format (default: 16, output 256x256)\n"
                      << "  --W <int>          ENCODED latent width  in (B,128,H,W) format (default: 16, output 256x256)\n"
                      << "  --threads <int>    ggml threads (default: 4)\n"
                      << "\n"
                      << "Latent format: raw BCHW F32 with 128 channels (= 32 * 2*2 spatial tile).\n"
                      << "The decoder applies inv_normalize (BN) + 2x2 spatial tile + 8x decode.\n";
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

    // --- Weights context (final, contiguous) ---
    size_t wctx_size = 400 * 1024 * 1024;
    ggml_init_params wparams{ wctx_size, nullptr, false };
    ggml_context * wctx = ggml_init(wparams);
    if (!wctx) {
        log_err("failed to allocate weights context");
        return 1;
    }

    bonsai::WeightsView view;
    bonsai::VAEWeights W;
    try {
        W = load_vae_weights(wctx, st, view);
    } catch (const std::exception & e) {
        log_err(std::string("loading failed: ") + e.what());
        ggml_free(wctx);
        return 1;
    }

    log_info("materializing weight permutations...");
    materialize_views(view);
    log_info("all weights loaded into ggml");

    // --- Load BN params ---
    std::vector<float> bn_mean(128), bn_inv_std(128);
    {
        const auto * m = st.find("bn.running_mean");
        const auto * v = st.find("bn.running_var");
        if (!m || !v) {
            log_err("missing bn.running_mean / bn.running_var");
            ggml_free(wctx);
            return 1;
        }
        if (m->shape[0] != 128) {
            log_err("BN params must be 128 ch, got " + std::to_string(m->shape[0]));
            ggml_free(wctx);
            return 1;
        }
        const float * mp = reinterpret_cast<const float *>(m->data);
        const float * vp = reinterpret_cast<const float *>(v->data);
        for (int i = 0; i < 128; ++i) {
            bn_mean[i] = mp[i];
            bn_inv_std[i] = std::sqrt(vp[i] + 1e-4f);
        }
        log_info("loaded bn.running_mean / bn.running_var (128 ch)");
    }

    // --- Load or synthesize encoded latent ---
    const int encoded_channels = 128;
    int encoded_size = encoded_channels * latent_h * latent_w;
    std::vector<float> latent((size_t) encoded_size, 0.0f);

    if (!latent_path.empty()) {
        log_info("loading latent from " + latent_path);
        std::ifstream f(latent_path, std::ios::binary);
        if (!f) {
            log_err("failed to open latent file: " + latent_path);
            ggml_free(wctx);
            return 1;
        }
        f.read(reinterpret_cast<char *>(latent.data()), encoded_size * sizeof(float));
        if (f.gcount() != (std::streamsize)(encoded_size * sizeof(float))) {
            log_err("latent file too small");
            ggml_free(wctx);
            return 1;
        }
    } else {
        log_info("synthesizing random latent (seed 42)");
        uint32_t seed = 42;
        for (auto & v : latent) {
            seed = seed * 1664525u + 1013904223u;
            v = ((float)(seed & 0xFFFF) / 32768.0f) - 1.0f;
        }
    }

    // --- Preprocess: inv_normalize + 2x2 spatial tile ---
    const int post_channels = 32;
    const int post_h = 2 * latent_h;
    const int post_w = 2 * latent_w;
    int post_size = post_channels * post_h * post_w;
    std::vector<float> post((size_t) post_size, 0.0f);
    {
        const float * src = latent.data();
        for (int c = 0; c < 32; ++c) {
            for (int i = 0; i < latent_h; ++i) {
                for (int j = 0; j < latent_w; ++j) {
                    for (int di = 0; di < 2; ++di) {
                        for (int dj = 0; dj < 2; ++dj) {
                            int sc = c * 4 + di * 2 + dj;
                            int64_t sidx = (int64_t)sc * latent_h * latent_w + (int64_t)i * latent_w + j;
                            float v = src[sidx] * bn_inv_std[sc] + bn_mean[sc];
                            int64_t didx = (int64_t)c * post_h * post_w + (int64_t)(2*i + di) * post_w + (2*j + dj);
                            post[didx] = v;
                        }
                    }
                }
            }
        }
        log_info("preprocessed: (B,128," + std::to_string(latent_h) + "," +
                 std::to_string(latent_w) + ") -> (B,32," + std::to_string(post_h) +
                 "," + std::to_string(post_w) + ")");
    }

    // --- Build activations context ---
    size_t actx_size = 16ULL * 1024 * 1024 * 1024;
    void * actx_buf = mmap(nullptr, actx_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (actx_buf == MAP_FAILED) {
        log_err("failed to mmap activations context");
        ggml_free(wctx);
        return 1;
    }
    ggml_init_params aparams{ actx_size, actx_buf, false };
    ggml_context * actx = ggml_init(aparams);
    if (!actx) {
        log_err("failed to allocate activations context");
        munmap(actx_buf, actx_size);
        ggml_free(wctx);
        return 1;
    }

    // Input latent (post-BN, post-rearrange) ggml layout: (W, H, C, N)
    ggml_tensor * x = ggml_new_tensor_4d(actx, GGML_TYPE_F32,
                                         (int64_t) post_w,
                                         (int64_t) post_h,
                                         (int64_t) post_channels,
                                         1);
    std::memcpy(ggml_get_data_f32(x), post.data(), post.size() * sizeof(float));

    // Build the decoder graph
    log_info("building decoder graph...");
    ggml_tensor * out = bonsai::build_decoder_graph(actx, W, x);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(actx);
    ggml_build_forward_expand(gf, out);

    log_info("graph built for decoder");
    log_info("activation memory: " +
             std::to_string(ggml_used_mem(actx) / (1024 * 1024)) + " MB");

    log_info("running graph on CPU (" + std::to_string(n_threads) + " threads)...");
    auto t0 = std::chrono::high_resolution_clock::now();
    ggml_status st_g = ggml_graph_compute_with_ctx(actx, gf, n_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (st_g != GGML_STATUS_SUCCESS) {
        log_err("graph compute failed");
        ggml_free(actx);
        ggml_free(wctx);
        return 1;
    }
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    log_info("done in " + std::to_string((int) ms) + " ms");

    // Print stats for each intermediate that was set as output
    fprintf(stderr, "\n=== Intermediate stats (post-compute) ===\n");
    for (ggml_tensor * t = ggml_get_first_tensor(actx); t; t = ggml_get_next_tensor(actx, t)) {
        if (!t->data) continue;
        const char * name = ggml_get_name(t);
        if (!name || name[0] == '\0') continue;
        int64_t n = ggml_nelements(t);
        if (n == 0) continue;
        const float * p = (const float *) t->data;
        double sum = 0, sum2 = 0;
        float mn = 1e30f, mx = -1e30f;
        for (int64_t i = 0; i < n; ++i) {
            sum += p[i];
            sum2 += (double)p[i] * p[i];
            if (p[i] < mn) mn = p[i];
            if (p[i] > mx) mx = p[i];
        }
        double mean = sum / n, var = sum2 / n - mean * mean;
        fprintf(stderr, "  %-22s shape=[%ld,%ld,%ld,%ld] mean=%.4f stddev=%.4f min=%.4f max=%.4f\n",
                name, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                mean, std::sqrt(std::max(0.0, var)), mn, mx);
    }

    int out_h = (int) out->ne[1];
    int out_w = (int) out->ne[0];
    int out_c = (int) out->ne[2];
    log_info("output: " + std::to_string(out_w) + "x" + std::to_string(out_h) +
             "x" + std::to_string(out_c));

    if (out_c != 3) {
        log_err("expected 3 output channels");
        ggml_free(actx);
        ggml_free(wctx);
        munmap(actx_buf, actx_size);
        return 1;
    }

    std::vector<float> out_data((size_t)(out_h * out_w * out_c));
    std::memcpy(out_data.data(), ggml_get_data_f32(out), out_data.size() * sizeof(float));

    // Per-channel stats (raw, before clamp)
    int sp = out_h * out_w;
    for (int c = 0; c < out_c; ++c) {
        float sum = 0, mn = 1e30f, mx = -1e30f;
        for (int i = 0; i < sp; ++i) {
            float v = out_data[c * sp + i];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        sum /= sp;
        log_info("ch " + std::to_string(c) + ": mean=" + std::to_string(sum) +
                 " min=" + std::to_string(mn) + " max=" + std::to_string(mx));
    }

    std::vector<uint8_t> rgb((size_t)(out_h * out_w * 3));
    chw_f32_to_hwc_u8(out_data.data(), rgb.data(), out_w, out_h);
    write_png(output_path, out_w, out_h, rgb.data());

    ggml_free(actx);
    ggml_free(wctx);
    munmap(actx_buf, actx_size);
    log_info("cleanup done. bye.");
    return 0;
}
