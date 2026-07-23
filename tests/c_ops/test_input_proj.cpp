#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static void f32_write(const char *path, const ggml_tensor *t) {
    FILE *fp = fopen(path, "wb");
    fwrite(t->data, 1, ggml_nbytes(t), fp);
    fclose(fp);
}

int main(int argc, char **argv) {
    int H = 3072;
    int C = 128;
    int ctx_dim = 7680;
    int img_tokens = 4096;
    int txt_tokens = 512;
    int n_threads = 4;

    const char *act_path = "act.bin";
    const char *emb_path = "emb.bin";
    const char *w_img_path = "w_img.bin";
    const char *w_txt_path = "w_txt.bin";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--act" && i+1<argc) act_path = argv[++i];
        else if (arg == "--emb" && i+1<argc) emb_path = argv[++i];
        else if (arg == "--w-img" && i+1<argc) w_img_path = argv[++i];
        else if (arg == "--w-txt" && i+1<argc) w_txt_path = argv[++i];
        else if (arg == "--threads" && i+1<argc) n_threads = std::stoi(argv[++i]);
    }

    struct ggml_init_params cparams = { 256ULL*1024*1024, NULL, true };
    ggml_context *ctx = ggml_init(cparams);

    ggml_tensor *img_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, img_tokens);
    ggml_tensor *txt_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ctx_dim, txt_tokens);

    ggml_set_input(img_in);
    ggml_set_input(txt_in);

    ggml_tensor *w_img = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, H);
    ggml_tensor *w_txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ctx_dim, H);

    w_img->data = malloc(C * H * sizeof(float));
    w_txt->data = malloc(ctx_dim * H * sizeof(float));

    {
        FILE *fp = fopen(w_img_path, "rb");
        fread(w_img->data, 1, C*H*sizeof(float), fp);
        fclose(fp);
        fprintf(stderr, "Loaded w_img (%d,%d) from %s\n", H, C, w_img_path);
    }
    {
        FILE *fp = fopen(w_txt_path, "rb");
        fread(w_txt->data, 1, ctx_dim*H*sizeof(float), fp);
        fclose(fp);
        fprintf(stderr, "Loaded w_txt (%d,%d) from %s\n", H, ctx_dim, w_txt_path);
    }

    ggml_tensor *h_img = ggml_mul_mat(ctx, w_img, img_in);
    ggml_tensor *h_txt = ggml_mul_mat(ctx, w_txt, txt_in);
    ggml_set_output(h_img);
    ggml_set_output(h_txt);

    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, h_img);
    ggml_build_forward_expand(graph, h_txt);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    {
        FILE *fp = fopen(act_path, "rb");
        fread(img_in->data, 1, C*img_tokens*sizeof(float), fp);
        fclose(fp);
        fprintf(stderr, "Loaded act (%d,%d) from %s\n", img_tokens, C, act_path);
    }
    {
        FILE *fp = fopen(emb_path, "rb");
        fread(txt_in->data, 1, ctx_dim*txt_tokens*sizeof(float), fp);
        fclose(fp);
        fprintf(stderr, "Loaded emb (%d,%d) from %s\n", txt_tokens, ctx_dim, emb_path);
    }

    fprintf(stderr, "Computing...\n");
    ggml_backend_graph_compute(backend, graph);
    fprintf(stderr, "Done!\n");

    f32_write("h_img.bin", h_img);
    f32_write("h_txt.bin", h_txt);
    fprintf(stderr, "Saved h_img [%lld,%lld] and h_txt [%lld,%lld]\n",
        (long long)h_img->ne[0], (long long)h_img->ne[1],
        (long long)h_txt->ne[0], (long long)h_txt->ne[1]);

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
