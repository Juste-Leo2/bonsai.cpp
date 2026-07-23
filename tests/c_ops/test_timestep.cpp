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

static ggml_tensor * column_1d(ggml_context * ctx, ggml_tensor * b) {
    ggml_tensor * v = ggml_reshape_4d(ctx, b, b->ne[0], 1, 1, 1);
    return ggml_cont(ctx, v);
}

int main(int argc, char **argv) {
    int H = 3072;
    int n_threads = 4;

    const char *t_path = "t_scaled.bin";
    const char *w1_path = "w1.bin";
    const char *w2_path = "w2.bin";
    const char *w_mod_img_path = "w_mod_img.bin";
    const char *w_mod_txt_path = "w_mod_txt.bin";
    bool has_b1 = false;
    bool has_b2 = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--t" && i+1<argc) t_path = argv[++i];
        else if (arg == "--w1" && i+1<argc) w1_path = argv[++i];
        else if (arg == "--w2" && i+1<argc) w2_path = argv[++i];
        else if (arg == "--w-mod-img" && i+1<argc) w_mod_img_path = argv[++i];
        else if (arg == "--w-mod-txt" && i+1<argc) w_mod_txt_path = argv[++i];
        else if (arg == "--threads" && i+1<argc) n_threads = std::stoi(argv[++i]);
    }

    // Check if bias files exist
    FILE *fb1 = fopen("b1.bin", "rb");
    if (fb1) { has_b1 = true; fclose(fb1); }
    FILE *fb2 = fopen("b2.bin", "rb");
    if (fb2) { has_b2 = true; fclose(fb2); }

    fprintf(stderr, "Init context...\n");
    static std::vector<uint8_t> meta_buf(1024ULL*1024*1024);
    struct ggml_init_params cparams = { meta_buf.size(), meta_buf.data(), false };
    ggml_context *ctx = ggml_init(cparams);

    ggml_tensor *timestep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(timestep);

    ggml_tensor *w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, H);
    ggml_tensor *b1 = nullptr;
    ggml_tensor *w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, H);
    ggml_tensor *b2 = nullptr;
    ggml_tensor *w_mod_img = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 6*H);
    ggml_tensor *w_mod_txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 6*H);

    w1->data = malloc(256*H*sizeof(float));
    w2->data = malloc(H*H*sizeof(float));
    w_mod_img->data = malloc(H*6*H*sizeof(float));
    w_mod_txt->data = malloc(H*6*H*sizeof(float));

    if (has_b1) { b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H); b1->data = malloc(H*sizeof(float)); }
    if (has_b2) { b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H); b2->data = malloc(H*sizeof(float)); }

    {
        FILE *f = fopen(t_path, "rb");
        fread(timestep->data, sizeof(float), 1, f);
        fclose(f);
    }
    {
        FILE *f = fopen(w1_path, "rb");
        fread(w1->data, sizeof(float), 256*H, f);
        fclose(f);
    }
    if (has_b1) { FILE *f = fopen("b1.bin", "rb"); fread(b1->data, sizeof(float), H, f); fclose(f); }
    {
        FILE *f = fopen(w2_path, "rb");
        fread(w2->data, sizeof(float), H*H, f);
        fclose(f);
    }
    if (has_b2) { FILE *f = fopen("b2.bin", "rb"); fread(b2->data, sizeof(float), H, f); fclose(f); }
    {
        FILE *f = fopen(w_mod_img_path, "rb");
        fread(w_mod_img->data, sizeof(float), H*6*H, f);
        fclose(f);
    }
    {
        FILE *f = fopen(w_mod_txt_path, "rb");
        fread(w_mod_txt->data, sizeof(float), H*6*H, f);
        fclose(f);
    }

    // --- Build graph ---
    ggml_tensor *te_emb = ggml_timestep_embedding(ctx, timestep, 256, 10000);
    ggml_set_name(te_emb, "te_emb");

    ggml_tensor *w1_2d = ggml_reshape_2d(ctx, w1, 256, H);
    ggml_tensor *te_pre_silu = ggml_mul_mat(ctx, w1_2d, te_emb);
    if (b1) te_pre_silu = ggml_add(ctx, te_pre_silu, column_1d(ctx, b1));
    ggml_set_name(te_pre_silu, "te_pre_silu");

    ggml_tensor *te_post_silu = ggml_silu(ctx, te_pre_silu);
    ggml_set_name(te_post_silu, "te_post_silu");

    ggml_tensor *w2_2d = ggml_reshape_2d(ctx, w2, H, H);
    ggml_tensor *vec = ggml_mul_mat(ctx, w2_2d, te_post_silu);
    if (b2) vec = ggml_add(ctx, vec, column_1d(ctx, b2));
    ggml_set_name(vec, "vec");

    ggml_tensor *vec_silu = ggml_silu(ctx, vec);
    ggml_set_name(vec_silu, "vec_silu");

    ggml_tensor *w_mod_img_2d = ggml_reshape_2d(ctx, w_mod_img, H, 6*H);
    ggml_tensor *mod_img = ggml_mul_mat(ctx, w_mod_img_2d, vec_silu);
    ggml_set_name(mod_img, "mod_img");

    ggml_tensor *w_mod_txt_2d = ggml_reshape_2d(ctx, w_mod_txt, H, 6*H);
    ggml_tensor *mod_txt = ggml_mul_mat(ctx, w_mod_txt_2d, vec_silu);
    ggml_set_name(mod_txt, "mod_txt");

    ggml_set_output(te_emb);
    ggml_set_output(te_pre_silu);
    ggml_set_output(te_post_silu);
    ggml_set_output(vec);
    ggml_set_output(vec_silu);
    ggml_set_output(mod_img);
    ggml_set_output(mod_txt);

    fprintf(stderr, "Building graph...\n");
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 65536, false);

    ggml_build_forward_expand(graph, te_emb);
    ggml_build_forward_expand(graph, te_pre_silu);
    ggml_build_forward_expand(graph, te_post_silu);
    ggml_build_forward_expand(graph, vec);
    ggml_build_forward_expand(graph, vec_silu);
    ggml_build_forward_expand(graph, mod_img);
    ggml_build_forward_expand(graph, mod_txt);

    fprintf(stderr, "Init backend...\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    fprintf(stderr, "Init galloc...\n");
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    fprintf(stderr, "Reserve...\n");
    ggml_gallocr_reserve(galloc, graph);
    fprintf(stderr, "Alloc...\n");
    ggml_gallocr_alloc_graph(galloc, graph);
    fprintf(stderr, "Computing...\n");
    ggml_backend_graph_compute(backend, graph);
    fprintf(stderr, "Done!\n");

    f32_write("te_emb.bin", ggml_graph_get_tensor(graph, "te_emb"));
    f32_write("te_pre_silu.bin", ggml_graph_get_tensor(graph, "te_pre_silu"));
    f32_write("te_post_silu.bin", ggml_graph_get_tensor(graph, "te_post_silu"));
    f32_write("vec.bin", ggml_graph_get_tensor(graph, "vec"));
    f32_write("vec_silu.bin", ggml_graph_get_tensor(graph, "vec_silu"));
    f32_write("mod_img.bin", ggml_graph_get_tensor(graph, "mod_img"));
    f32_write("mod_txt.bin", ggml_graph_get_tensor(graph, "mod_txt"));

    fprintf(stderr, "Saved all outputs\n");

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
