// Trimmed test: load post-BN-rearrange input, run conv_in, compare to PyTorch
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
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
static void materialize_views(const WeightsView & view, ggml_context * mctx) {
    for (auto & L : view.linears) {
        ggml_tensor * src = ggml_new_tensor_2d(mctx, GGML_TYPE_F32, L.in_dim, L.out_dim);
        std::memcpy(ggml_get_data_f32(src), L.src, L.in_dim * L.out_dim * sizeof(float));
        ggml_tensor * t1 = ggml_permute(mctx, src, 1, 0, 2, 3);
        ggml_tensor * t2 = ggml_cont(mctx, t1);
        ggml_tensor * t3 = ggml_cont(mctx, ggml_transpose(mctx, t2));
        std::memcpy(ggml_get_data_f32(L.dst), ggml_get_data_f32(t3),
                    L.in_dim * L.out_dim * sizeof(float));
    }
    for (auto & C : view.convs) {
        ggml_tensor * src = ggml_new_tensor_4d(mctx, GGML_TYPE_F32, C.kw, C.kh, C.in_c, C.out_c);
        std::memcpy(ggml_get_data_f32(src), C.src,
                    C.kw * C.kh * C.in_c * C.out_c * sizeof(float));
        ggml_tensor * t1 = ggml_cont(mctx, ggml_permute(mctx, src, 3, 2, 1, 0));
        ggml_tensor * t2 = ggml_cont(mctx, ggml_permute(mctx, t1, 3, 2, 1, 0));
        std::memcpy(ggml_get_data_f32(C.dst), ggml_get_data_f32(t2),
                    C.kw * C.kh * C.in_c * C.out_c * sizeof(float));
    }
    for (auto & V : view.vecs) {
        std::memcpy(ggml_get_data_f32(V.dst), V.src, V.n * sizeof(float));
    }
}

static ggml_tensor * channel_1d(ggml_context * ctx, ggml_tensor * b) {
    ggml_tensor * v = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    return ggml_cont(ctx, v);
}

int main(int argc, char** argv) {
    std::string model_path = "/home/leo/bonsai.cpp/models/flux2-vae.safetensors";
    std::string input_path = "/tmp/vae_post_bchw.bin";
    std::string ref_path = "/tmp/vae_ref_convin.bin";

    SafetensorsFile st;
    st.open(model_path);
    log_info("loaded safetensors");

    size_t wctx_size = 50 * 1024 * 1024;
    ggml_context * wctx = ggml_init({ wctx_size, nullptr, false });

    WeightsView view;
    auto * w = st.find("post_quant_conv.weight"); auto * b = st.find("post_quant_conv.bias");
    ggml_tensor * pq_w = load_conv_w(wctx, view, *w);
    ggml_tensor * pq_b = load_vec(wctx, view, *b);
    w = st.find("decoder.conv_in.weight"); b = st.find("decoder.conv_in.bias");
    ggml_tensor * ci_w = load_conv_w(wctx, view, *w);
    ggml_tensor * ci_b = load_vec(wctx, view, *b);
    size_t mctx_size = 2ULL * 1024 * 1024 * 1024;
    void * mctx_buf = mmap(nullptr, mctx_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    ggml_context * mctx = ggml_init({ mctx_size, mctx_buf, false });
    materialize_views(view, mctx);
    ggml_free(mctx);
    munmap(mctx_buf, mctx_size);

    // Activations context
    size_t actx_size = 2ULL * 1024 * 1024 * 1024;
    void * actx_buf = mmap(nullptr, actx_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    ggml_context * actx = ggml_init({ actx_size, actx_buf, false });

    // Load input: (1, 32, 32, 32) BCHW F32
    int H = 32, W = 32, C = 32;
    std::vector<float> in((size_t)H*W*C, 0.0f);
    {
        std::ifstream f(input_path, std::ios::binary);
        if (!f) { log_err("cannot open " + input_path); return 1; }
        f.read((char*)in.data(), in.size() * sizeof(float));
    }
    log_info("loaded input: " + std::to_string(in.size()) + " floats");

    ggml_tensor * x = ggml_new_tensor_4d(actx, GGML_TYPE_F32, W, H, C, 1);
    std::memcpy(ggml_get_data_f32(x), in.data(), in.size() * sizeof(float));

    // post_quant_conv (1x1, 32->32)
    ggml_tensor * h = ggml_conv_2d_direct(actx, pq_w, x, 1, 1, 0, 0, 1, 1);
    h = ggml_add(actx, h, channel_1d(actx, pq_b));

    // conv_in (3x3, 32->512, padding 1)
    h = ggml_conv_2d_direct(actx, ci_w, h, 1, 1, 1, 1, 1, 1);
    h = ggml_add(actx, h, channel_1d(actx, ci_b));

    ggml_set_output(h);
    ggml_cgraph * gf = ggml_new_graph(actx);
    ggml_build_forward_expand(gf, h);

    log_info("running conv_in graph...");
    ggml_status st_g = ggml_graph_compute_with_ctx(actx, gf, 4);
    if (st_g != GGML_STATUS_SUCCESS) { log_err("graph failed"); return 1; }

    int64_t n = h->ne[0] * h->ne[1] * h->ne[2] * h->ne[3];
    std::vector<float> out(n);
    std::memcpy(out.data(), ggml_get_data_f32(h), n * sizeof(float));

    // Compare with reference
    std::vector<float> ref(n);
    {
        std::ifstream f(ref_path, std::ios::binary);
        if (!f) { log_err("cannot open " + ref_path); return 1; }
        f.read((char*)ref.data(), n * sizeof(float));
    }
    log_info("C++ conv_in: min=" + std::to_string(*std::min_element(out.begin(), out.end())) +
             " max=" + std::to_string(*std::max_element(out.begin(), out.end())) +
             " mean=" + std::to_string(std::accumulate(out.begin(), out.end(), 0.0) / n));
    log_info("REF conv_in: min=" + std::to_string(*std::min_element(ref.begin(), ref.end())) +
             " max=" + std::to_string(*std::max_element(ref.begin(), ref.end())) +
             " mean=" + std::to_string(std::accumulate(ref.begin(), ref.end(), 0.0) / n));

    // Per-channel comparison (first 5 channels)
    int64_t sp = W * H;
    int n_ch = 512;
    for (int c = 0; c < 5; ++c) {
        double sum_cpp = 0, sum_ref = 0;
        for (int64_t i = 0; i < sp; ++i) {
            sum_cpp += out[c * sp + i];
            sum_ref += ref[c * sp + i];
        }
        sum_cpp /= sp;
        sum_ref /= sp;
        log_info("ch " + std::to_string(c) + ": C++=" + std::to_string(sum_cpp) + " REF=" + std::to_string(sum_ref) +
                 " diff=" + std::to_string(sum_cpp - sum_ref));
    }

    // Max abs diff
    double max_diff = 0;
    for (int64_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs((double)out[i] - (double)ref[i]));
    }
    log_info("max abs diff: " + std::to_string(max_diff));

    ggml_free(actx);
    ggml_free(wctx);
    munmap(actx_buf, actx_size);
    return 0;
}
