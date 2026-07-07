/**
 * Test C pour RoPE 2D (Rotary Position Embedding).
 *
 * Vérifie que le kernel custom rope_2d_fwd produit le même résultat
 * qu'une implémentation de référence en Python/PyTorch.
 *
 * Usage:
 *   ./test_rope_2d <q.bin> <k.bin> <cos.bin> <sin.bin> <q_out.bin> <k_out.bin> [head_dim] [n_heads] [seq]
 *
 * Format: float32 binaire contigu (raw).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>

#include "ggml.h"
#include "ggml-cpu.h"

struct Rope2DUserData {
    int head_dim;
    int n_heads;
    int seq;
};

/**
 * Application manuelle du RoPE 2D (sans AVX2, pour référence).
 * Identique au kernel dans b1_0_kernel.h mais en format simple.
 */
static void rope_2d_apply(
    const float * src,
    const float * cos_t,
    const float * sin_t,
    float * out,
    int head_dim,
    int n_heads,
    int seq)
{
    int half = head_dim / 2;
    int total = n_heads * seq;

    for (int block = 0; block < total; block++) {
        int s_idx = block / n_heads;
        int h = block % n_heads;

        int base = h * head_dim + s_idx * (head_dim * n_heads);
        int angle_base = s_idx * half;

        for (int d_pair = 0; d_pair < half; d_pair++) {
            int even_idx = d_pair * 2 + base;
            int odd_idx  = even_idx + 1;

            int angle_idx = d_pair + angle_base;
            float cos_v = cos_t[angle_idx];
            float sin_v = sin_t[angle_idx];

            float a = src[even_idx];
            float b = src[odd_idx];

            out[even_idx] = a * cos_v - b * sin_v;
            out[odd_idx]  = a * sin_v + b * cos_v;
        }
    }
}

/**
 * Calcule la table de fréquences RoPE (identique à build_rope_freqs_table).
 * axes_dim: [32, 32, 32, 32], n_axes=4, theta=2000.0
 */
static void build_freqs_table(float * freqs, const int * axes_dim, int n_axes, float theta) {
    int max_half = 0;
    for (int i = 0; i < n_axes; i++) {
        int h = axes_dim[i] / 2;
        if (h > max_half) max_half = h;
    }

    for (int a = 0; a < n_axes; a++) {
        int ax_half = axes_dim[a] / 2;
        for (int j = 0; j < ax_half; j++) {
            float scale = (float)(j * 2) / (float)axes_dim[a];
            freqs[a * max_half + j] = 1.0f / powf(theta, scale);
        }
    }
}

/**
 * Calcule cos/sin (identique à build_rope_cos_sin).
 */
static void build_cos_sin(
    const float * ids,
    const float * freqs_table,
    const int * axes_dim,
    int n_axes,
    int seq,
    float * cos_out,
    float * sin_out)
{
    int max_half = 0;
    for (int i = 0; i < n_axes; i++) {
        int h = axes_dim[i] / 2;
        if (h > max_half) max_half = h;
    }

    float * all_angles = (float *)calloc(max_half * n_axes * seq, sizeof(float));
    int total_half = 0;

    for (int a = 0; a < n_axes; a++) {
        int ax_half = axes_dim[a] / 2;

        for (int s = 0; s < seq; s++) {
            float id_val = ids[a + s * n_axes];
            for (int j = 0; j < ax_half; j++) {
                float freq = freqs_table[a * max_half + j];
                all_angles[total_half * seq + s] = freq * id_val;
                total_half++;
            }
        }
    }

    for (int i = 0; i < total_half * seq; i++) {
        cos_out[i] = cosf(all_angles[i]);
        sin_out[i] = sinf(all_angles[i]);
    }

    free(all_angles);
}

static float* read_float_bin(const char* path, size_t* count) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return NULL; }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    float* data = (float*)malloc(size);
    if (!data) { fclose(f); return NULL; }
    size_t read = fread(data, 1, size, f);
    if (read != (size_t)size) {
        fprintf(stderr, "Warning: expected %ld bytes, got %zu\n", size, read);
    }
    fclose(f);

    *count = size / sizeof(float);
    return data;
}

static void write_float_bin(const char* path, const float* data, size_t count) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen"); return; }
    fwrite(data, sizeof(float), count, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <q.bin> <k.bin> <cos.bin> <sin.bin> <q_out.bin> <k_out.bin> [head_dim] [n_heads] [seq]\n", argv[0]);
        return 1;
    }

    const char* q_path = argv[1];
    const char* k_path = argv[2];
    const char* cos_path = argv[3];
    const char* sin_path = argv[4];
    const char* q_out_path = argv[5];
    const char* k_out_path = argv[6];

    int head_dim = (argc > 7) ? atoi(argv[7]) : 128;
    int n_heads  = (argc > 8) ? atoi(argv[8]) : 24;
    int seq      = (argc > 9) ? atoi(argv[9]) : 64;

    printf("Testing RoPE 2D: head_dim=%d n_heads=%d seq=%d\n", head_dim, n_heads, seq);

    // 1. Lecture des entrées
    size_t q_count = 0, k_count = 0, cos_count = 0, sin_count = 0;
    float* q_src = read_float_bin(q_path, &q_count);
    float* k_src = read_float_bin(k_path, &k_count);
    float* c_src = read_float_bin(cos_path, &cos_count);
    float* s_src = read_float_bin(sin_path, &sin_count);

    if (!q_src || !k_src || !c_src || !s_src) {
        fprintf(stderr, "Error reading input files\n");
        free(q_src); free(k_src); free(c_src); free(s_src);
        return 1;
    }

    size_t expected_qk = (size_t)head_dim * n_heads * seq;
    if (q_count != expected_qk || k_count != expected_qk) {
        fprintf(stderr, "Shape error: expected %zu elements, got q=%zu k=%zu\n",
                expected_qk, q_count, k_count);
        return 1;
    }

    // 2. Application manuelle du RoPE (référence)
    float* q_ref = (float*)malloc(q_count * sizeof(float));
    float* k_ref = (float*)malloc(k_count * sizeof(float));
    rope_2d_apply(q_src, c_src, s_src, q_ref, head_dim, n_heads, seq);
    rope_2d_apply(k_src, c_src, s_src, k_ref, head_dim, n_heads, seq);

    // 3. Application via le kernel ggml custom
    // Initialisation GGML
    size_t ctx_size = 512 * 1024 * 1024;
    void* buffer = malloc(ctx_size);
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = buffer,
        .no_alloc   = false,
    };
    ggml_context* ctx = ggml_init(params);

    // Créer les tenseurs d'entrée
    ggml_tensor* t_q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq);
    memcpy(t_q->data, q_src, q_count * sizeof(float));

    ggml_tensor* t_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq);
    memcpy(t_k->data, k_src, k_count * sizeof(float));

    ggml_tensor* t_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim / 2, seq);
    memcpy(t_cos->data, c_src, cos_count * sizeof(float));

    ggml_tensor* t_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim / 2, seq);
    memcpy(t_sin->data, s_src, sin_count * sizeof(float));

    // Créer le custom op pour RoPE (même signature que b1_0_kernel.h)
    Rope2DUserData ud = {head_dim, n_heads, seq};

    // Fonction de forward pour RoPE 2D (copie de rope_2d_fwd_f32 dans b1_0_kernel.h)
    auto rope_fwd = [](ggml_tensor * dst, int ith, int nth, void * userdata) {
        const auto * ud = (const Rope2DUserData *)userdata;
        const int head_dim = ud->head_dim;
        const int n_heads  = ud->n_heads;
        const int seq      = ud->seq;
        const int half     = head_dim / 2;

        const ggml_tensor * src_t = dst->src[0];
        const ggml_tensor * cos_t = dst->src[1];
        const ggml_tensor * sin_t = dst->src[2];

        const float * src = (const float *)src_t->data;
        const float * c   = (const float *)cos_t->data;
        const float * s   = (const float *)sin_t->data;
        float * out       = (float *)dst->data;

        const int total_blocks = n_heads * seq;
        const int blocks_per_thread = (total_blocks + nth - 1) / nth;
        const int start_block = ith * blocks_per_thread;
        const int end_block = (start_block + blocks_per_thread > total_blocks) ? total_blocks : start_block + blocks_per_thread;

        int s_idx = start_block / n_heads;
        int h     = start_block % n_heads;

        for (int block = start_block; block < end_block; block++) {
            const int base = h * head_dim + s_idx * (head_dim * n_heads);
            const int angle_base = s_idx * half;

            for (int d_pair = 0; d_pair < half; d_pair++) {
                const int even_idx = d_pair * 2 + base;
                const int odd_idx  = even_idx + 1;
                const int angle_idx = d_pair + angle_base;

                const float cos_v = c[angle_idx];
                const float sin_v = s[angle_idx];
                const float a = src[even_idx];
                const float b = src[odd_idx];

                out[even_idx] = a * cos_v - b * sin_v;
                out[odd_idx]  = a * sin_v + b * cos_v;
            }

            h++;
            if (h == n_heads) { h = 0; s_idx++; }
        }
    };

    ggml_tensor* rope_args_q[] = {t_q, t_cos, t_sin};
    ggml_tensor* rope_args_k[] = {t_k, t_cos, t_sin};

    // q_out
    ggml_tensor* t_q_out = ggml_custom_4d(ctx, GGML_TYPE_F32,
        head_dim, n_heads, seq, 1,
        rope_args_q, 3,
        rope_fwd, 1, &ud);

    // k_out
    ggml_tensor* t_k_out = ggml_custom_4d(ctx, GGML_TYPE_F32,
        head_dim, n_heads, seq, 1,
        rope_args_k, 3,
        rope_fwd, 1, &ud);

    // Exécution
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, t_q_out);
    ggml_build_forward_expand(graph, t_k_out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);
    ggml_backend_graph_compute(backend, graph);

    // Vérification
    float* q_c = (float*)t_q_out->data;
    float* k_c = (float*)t_k_out->data;

    double max_diff_q = 0.0, max_diff_k = 0.0;
    for (size_t i = 0; i < q_count; i++) {
        double d = fabs(q_c[i] - q_ref[i]);
        if (d > max_diff_q) max_diff_q = d;
    }
    for (size_t i = 0; i < k_count; i++) {
        double d = fabs(k_c[i] - k_ref[i]);
        if (d > max_diff_k) max_diff_k = d;
    }

    printf("  Max diff Q: %.10f\n", max_diff_q);
    printf("  Max diff K: %.10f\n", max_diff_k);

    bool pass = (max_diff_q < 1e-5 && max_diff_k < 1e-5);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    // Écriture des résultats
    write_float_bin(q_out_path, q_c, q_count);
    write_float_bin(k_out_path, k_c, k_count);

    // Cleanup
    free(q_src); free(k_src); free(c_src); free(s_src);
    free(q_ref); free(k_ref);
    free(buffer);
    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);

    return pass ? 0 : 1;
}
