// Quick test: just post_quant_conv + conv_in + conv_out (no resnet, no attention, no up_blocks).
// Compare to PyTorch to find where values diverge.
#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <vector>

#include "safetensors.h"
#include "vae_graph.h"
#include "vae_types.h"

using namespace bonsai;

static void log_info(const std::string & m) { std::cout << "[INFO] " << m << "\n"; }
static void log_err (const std::string & m) { std::cerr << "[ERROR] " << m << "\n"; }

static ggml_tensor * load_conv_w(ggml_context * wctx, WeightsView & view, const SafeTensor & t) {
    int64_t Co = t.shape[0], Ci = t.shape[1], kH = t.shape[2], kW = t.shape[3];
    ggml_tensor * dst = ggml_new_tensor_4d(wctx, GGML_TYPE_F32, kW, kH, Ci, Co);
    WeightsView::Conv C; C.dst = dst; C.src = (const float*)t.data;
    C.out_c = Co; C.in_c = Ci; C.kh = kH; C.kw = kW;
    view.convs.push_back(C);
    return dst;
}
static ggml_tensor * load_vec(ggml_context * wctx, WeightsView & view, const SafeTensor & t) {
    ggml_tensor * dst = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, t.shape[0]);
    WeightsView::Vec V; V.dst = dst; V.src = (const float*)t.data; V.n = t.shape[0];
    view.vecs.push_back(V);
    return dst;
}
static void materialize_views(const WeightsView & view) {
    for (auto & C : view.convs) {
        float * dp = ggml_get_data_f32(C.dst);
        for (int64_t co = 0; co < C.out_c; ++co)
            for (int64_t ci = 0; ci < C.in_c; ++ci)
                for (int64_t kh = 0; kh < C.kh; ++kh)
                    for (int64_t kw = 0; kw < C.kw; ++kw) {
                        int64_t s = ((co * C.in_c + ci) * C.kh + kh) * C.kw + kw;
                        int64_t d = ((kw * C.kh + kh) * C.in_c + ci) * C.out_c + co;
                        dp[d] = C.src[s];
                    }
    }
    for (auto & V : view.vecs) {
        std::memcpy(ggml_get_data_f32(V.dst), V.src, V.n * sizeof(float));
    }
}

static ggml_tensor * bias_chw(ggml_context * ctx, ggml_tensor * b) {
    ggml_tensor * v = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    return ggml_cont(ctx, v);
}

int main(int argc, char** argv) {
    std::string model_path = "/home/leo/bonsai.cpp/models/flux2-vae.safetensors";
    std::string latent_path = "/tmp/vae_latent_zero.bin";

    SafetensorsFile st;
    st.open(model_path);
    log_info("loaded safetensors");

    size_t wctx_size = 50 * 1024 * 1024;
    ggml_context * wctx = ggml_init({ wctx_size, nullptr, false });

    WeightsView view;
    Conv2d pq_conv, conv_in, conv_out;
    auto * w = st.find("post_quant_conv.weight"); auto * b = st.find("post_quant_conv.bias");
    pq_conv.w = load_conv_w(wctx, view, *w); pq_conv.b = load_vec(wctx, view, *b);
    w = st.find("decoder.conv_in.weight"); b = st.find("decoder.conv_in.bias");
    conv_in.w = load_conv_w(wctx, view, *w); conv_in.b = load_vec(wctx, view, *b);
    w = st.find("decoder.conv_out.weight"); b = st.find("decoder.conv_out.bias");
    conv_out.w = load_conv_w(wctx, view, *w); conv_out.b = load_vec(wctx, view, *b);
    log_info("loaded 3 convs");
    materialize_views(view);
    log_info("materialized");

    // Print first few values of permuted conv_in weight to verify against PyTorch
    {
        float * dp = ggml_get_data_f32(conv_in.w);
        int Co = 512, Ci = 32, kH = 3, kW = 3;
        log_info("conv_in.w (permuted):");
        // Print in a clean format
        // dst[0,0,0,0] = PyTorch[0,0,0,0] expected -0.0086
        // dst[0,0,0,1] = PyTorch[1,0,0,0] expected -0.0016  at offset 4*Co=2048
        // dst[1,0,0,0] = PyTorch[0,0,0,1] expected 0.0070  at offset 1
        // dst[0,1,0,0] = PyTorch[0,0,1,0] expected -0.0091  at offset 4*kW=12
        // dst[0,0,1,0] = PyTorch[0,1,0,0] expected 0.0220  at offset 4*kW*kH=36
        // dst[2,2,31,511] = PyTorch[511,31,2,2] expected 0.0107  at offset 147455
        log_info("  mem[0]    dst[0,0,0,0]: " + std::to_string(dp[0]) + " (expect -0.0086)");
        log_info("  mem[1]    dst[1,0,0,0]: " + std::to_string(dp[1]) + " (expect 0.0070)");
        log_info("  mem[12]   dst[0,1,0,0]: " + std::to_string(dp[12]) + " (expect -0.0091)");
        log_info("  mem[36]   dst[0,0,1,0]: " + std::to_string(dp[36]) + " (expect 0.0220)");
        log_info("  mem[2048] dst[0,0,0,1]: " + std::to_string(dp[2048]) + " (expect -0.0016)");
        log_info("  mem[147455] dst[2,2,31,511]: " + std::to_string(dp[147455]) + " (expect 0.0107)");
    }

    // activations context (mmap)
    size_t actx_size = 2ULL * 1024 * 1024 * 1024;
    void * actx_buf = mmap(nullptr, actx_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    ggml_context * actx = ggml_init({ actx_size, actx_buf, false });

    // Load latent
    int H = 32, W = 32, C = 32;
    std::vector<float> lat((size_t)H*W*C, 0.0f);
    if (!latent_path.empty()) {
        std::ifstream f(latent_path, std::ios::binary);
        f.read((char*)lat.data(), lat.size() * sizeof(float));
    }

    ggml_tensor * x = ggml_new_tensor_4d(actx, GGML_TYPE_F32, W, H, C, 1);
    std::memcpy(ggml_get_data_f32(x), lat.data(), lat.size() * sizeof(float));

    // post_quant_conv
    ggml_tensor * h = ggml_conv_2d(actx, pq_conv.w, x, 1, 1, 0, 0, 1, 1);
    h = ggml_add(actx, h, bias_chw(actx, pq_conv.b));
    log_info("post_quant_conv: ne=[W,H,C,N] = [" +
             std::to_string(h->ne[0]) + "," + std::to_string(h->ne[1]) + "," +
             std::to_string(h->ne[2]) + "," + std::to_string(h->ne[3]) + "]");

    // conv_in
    h = ggml_conv_2d(actx, conv_in.w, h, 1, 1, 1, 1, 1, 1);
    h = ggml_add(actx, h, bias_chw(actx, conv_in.b));
    log_info("conv_in: ne=[" +
             std::to_string(h->ne[0]) + "," + std::to_string(h->ne[1]) + "," +
             std::to_string(h->ne[2]) + "," + std::to_string(h->ne[3]) + "]");

    // skip all the rest, just go straight to conv_out for a sanity check
    // Actually let's just print conv_in output stats
    ggml_set_output(h);
    ggml_cgraph * gf = ggml_new_graph(actx);
    ggml_build_forward_expand(gf, h);

    log_info("running graph...");
    auto t0 = std::chrono::high_resolution_clock::now();
    ggml_graph_compute_with_ctx(actx, gf, 4);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    log_info("done in " + std::to_string((int)ms) + " ms");

    // Read back conv_in output
    int64_t n = h->ne[0] * h->ne[1] * h->ne[2] * h->ne[3];
    std::vector<float> out(n);
    std::memcpy(out.data(), ggml_get_data_f32(h), n * sizeof(float));

    // stats per channel
    int64_t sp = h->ne[0] * h->ne[1];  // spatial
    int n_ch = (int)h->ne[2];
    std::vector<double> sums(n_ch, 0.0);
    for (int c = 0; c < n_ch; ++c) {
        for (int64_t i = 0; i < sp; ++i) sums[c] += out[c * sp + i];
        sums[c] /= sp;
    }
    log_info("conv_in per-channel mean (first 10):");
    for (int c = 0; c < 10 && c < n_ch; ++c) {
        log_info("  ch " + std::to_string(c) + ": " + std::to_string(sums[c]));
    }
    log_info("conv_in global mean: " + std::to_string(std::accumulate(sums.begin(), sums.end(), 0.0) / n_ch));

    // Save conv_in output as raw for comparison
    std::ofstream raw("/tmp/vae_convin_cpp.bin", std::ios::binary);
    raw.write((char*)out.data(), n * sizeof(float));
    log_info("wrote /tmp/vae_convin_cpp.bin");

    ggml_free(actx);
    ggml_free(wctx);
    munmap(actx_buf, actx_size);
    return 0;
}
