#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void f32_write(const char *path, const ggml_tensor *t) {
    FILE *fp = fopen(path, "wb");
    fwrite(t->data, 1, ggml_nbytes(t), fp);
    fclose(fp);
}

int main(int argc, char **argv) {
    int H = 3072, n_threads = 4;
    const char *t_path = "t_scaled.bin", *w1_path = "w1.bin", *w2_path = "w2.bin";
    const char *w_mod_img_path = "w_mod_img.bin", *w_mod_txt_path = "w_mod_txt.bin";
    const char *w_single_path = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--t" && i+1<argc) t_path = argv[++i];
        else if (arg == "--w1" && i+1<argc) w1_path = argv[++i];
        else if (arg == "--w2" && i+1<argc) w2_path = argv[++i];
        else if (arg == "--w-mod-img" && i+1<argc) w_mod_img_path = argv[++i];
        else if (arg == "--w-mod-txt" && i+1<argc) w_mod_txt_path = argv[++i];
        else if (arg == "--w-single" && i+1<argc) w_single_path = argv[++i];
        else if (arg == "--threads" && i+1<argc) n_threads = std::stoi(argv[++i]);
    }

    fprintf(stderr, "Init context...\n");
    static std::vector<uint8_t> buf(1024ULL*1024*1024);
    struct ggml_init_params cparams = { buf.size(), buf.data(), false };
    ggml_context *ctx = ggml_init(cparams);

    auto load_f32 = [&](const char *path, int ne0, int ne1) {
        ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        t->data = malloc(ne0 * ne1 * sizeof(float));
        FILE *f = fopen(path, "rb"); if (!f) abort();
        fread(t->data, sizeof(float), ne0*ne1, f);
        fclose(f);
        return t;
    };

    ggml_tensor *timestep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(timestep);
    { FILE *f=fopen(t_path,"rb"); fread(timestep->data,sizeof(float),1,f); fclose(f); }

    ggml_tensor *w1 = load_f32(w1_path, 256, H);
    ggml_tensor *w2 = load_f32(w2_path, H, H);
    ggml_tensor *w_mod_img = load_f32(w_mod_img_path, H, 6*H);
    ggml_tensor *w_mod_txt = load_f32(w_mod_txt_path, H, 6*H);

    ggml_tensor *w_single = nullptr;
    if (w_single_path) w_single = load_f32(w_single_path, H, 3*H);

    // ── Build graph (matches HF: flip sin/cos + LayerNorm) ──
    ggml_tensor *te_raw = ggml_timestep_embedding(ctx, timestep, 256, 10000);
    // HF Timesteps(flip_sin_to_cos=True) → [sin, cos]; ggml → [cos, sin]. Swap.
    ggml_tensor *te_cos = ggml_view_2d(ctx, te_raw, 128, 1, te_raw->nb[1], 0);
    ggml_tensor *te_sin = ggml_view_2d(ctx, te_raw, 128, 1, te_raw->nb[1], 128*sizeof(float));
    ggml_tensor *te_emb = ggml_concat(ctx, te_sin, te_cos, 0);
    ggml_set_name(te_emb, "te_emb");

    ggml_tensor *w1_2d = ggml_reshape_2d(ctx, w1, 256, H);
    ggml_tensor *te_pre_silu = ggml_mul_mat(ctx, w1_2d, te_emb);
    ggml_set_name(te_pre_silu, "te_pre_silu");

    ggml_tensor *te_post_silu = ggml_silu(ctx, te_pre_silu);
    ggml_set_name(te_post_silu, "te_post_silu");

    ggml_tensor *w2_2d = ggml_reshape_2d(ctx, w2, H, H);
    ggml_tensor *vec = ggml_mul_mat(ctx, w2_2d, te_post_silu);
    ggml_set_name(vec, "vec");

    // HF: Flux2Modulation = SiLU(vec) → linear
    ggml_tensor * vec_silu = ggml_silu(ctx, vec);
    ggml_set_name(vec_silu, "vec_silu");

    ggml_tensor *w_mod_img_2d = ggml_reshape_2d(ctx, w_mod_img, H, 6*H);
    ggml_tensor *mod_img = ggml_mul_mat(ctx, w_mod_img_2d, vec_silu);
    ggml_set_name(mod_img, "mod_img");

    ggml_tensor *w_mod_txt_2d = ggml_reshape_2d(ctx, w_mod_txt, H, 6*H);
    ggml_tensor *mod_txt = ggml_mul_mat(ctx, w_mod_txt_2d, vec_silu);
    ggml_set_name(mod_txt, "mod_txt");

    ggml_tensor *mod_single = nullptr;
    if (w_single) {
        ggml_tensor *w_single_2d = ggml_reshape_2d(ctx, w_single, H, 3*H);
        mod_single = ggml_mul_mat(ctx, w_single_2d, vec_silu);
        ggml_set_name(mod_single, "mod_single");
    }

    ggml_set_output(te_emb); ggml_set_output(te_pre_silu); ggml_set_output(te_post_silu);
    ggml_set_output(vec); ggml_set_output(vec_silu);
    ggml_set_output(mod_img); ggml_set_output(mod_txt);
    if (mod_single) ggml_set_output(mod_single);

    fprintf(stderr, "Building graph...\n");
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 65536, false);
    ggml_build_forward_expand(graph, te_emb);
    ggml_build_forward_expand(graph, te_pre_silu);
    ggml_build_forward_expand(graph, te_post_silu);
    ggml_build_forward_expand(graph, vec);
    ggml_build_forward_expand(graph, vec_silu);
    ggml_build_forward_expand(graph, mod_img);
    ggml_build_forward_expand(graph, mod_txt);
    if (mod_single) ggml_build_forward_expand(graph, mod_single);

    fprintf(stderr, "Init backend...\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    fprintf(stderr, "Computing...\n");
    ggml_backend_graph_compute(backend, graph);
    fprintf(stderr, "Done!\n");

    f32_write("te_emb.bin", te_emb);
    f32_write("te_pre_silu.bin", te_pre_silu);
    f32_write("te_post_silu.bin", te_post_silu);
    f32_write("vec.bin", vec);
    f32_write("vec_silu.bin", vec_silu);
    f32_write("mod_img.bin", mod_img);
    f32_write("mod_txt.bin", mod_txt);
    if (mod_single) f32_write("mod_single.bin", mod_single);

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
