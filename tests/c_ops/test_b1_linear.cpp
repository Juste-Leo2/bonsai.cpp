/**
 * Test C pour b1_linear: valide la quantification B1_0 et le kernel de multiplication.
 *
 * Usage:
 *   ./test_b1_linear <act.bin> <weight.bin> <out.bin>
 *
 * Où:
 *   act.bin    : float32 [batch, in_dim] (row-major)
 *   weight.bin : float32 [out_dim, in_dim] (row-major, sera quantifié en B1_0)
 *   out.bin    : float32 [batch, out_dim] (résultat de b1_linear)
 *
 * Format binaire simple: juste les données float32 contiguës.
 * Le test quantifie les poids en B1_0 et exécute b1_linear.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

/* Copie de b1_0_kernel.h pour être autonome dans le test */
#ifndef B1_0_KERNEL_H
#define B1_0_KERNEL_H

#include "ggml.h"
#include <cstdint>
#include <cstring>
#include <cmath>

static constexpr int B1_0_BLOCK_SIZE = 32;
static constexpr int B1_0_BLOCK_BYTES = 6;

struct B1LinearUserData {
    int in_dim;
    int out_dim;
};

/* Version simplifiée de la fonction b1_linear (sans AVX2 pour ce test) */
static void b1_linear_fwd(
    ggml_tensor * dst,
    int ith,
    int nth,
    void * userdata)
{
    const auto * ud = (const B1LinearUserData *)userdata;
    const int in_dim = ud->in_dim;
    const int out_dim = ud->out_dim;

    const ggml_tensor * act_t = dst->src[0];
    const ggml_tensor * w_t = dst->src[1];

    const float * act = (const float *)act_t->data;
    const uint8_t * w = (const uint8_t *)w_t->data;
    float * out = (float *)dst->data;

    const int batch = (int)act_t->ne[1];
    const int num_blocks = in_dim / B1_0_BLOCK_SIZE;
    const int row_stride = num_blocks * B1_0_BLOCK_BYTES;

    const int rows_per_thread = (out_dim + nth - 1) / nth;
    const int row_start = ith * rows_per_thread;
    const int row_end = (row_start + rows_per_thread > out_dim) ? out_dim : row_start + rows_per_thread;

    const int B_TILE = 4;
    for (int b0 = 0; b0 < batch; b0 += B_TILE) {
        int b_count = (b0 + B_TILE <= batch) ? B_TILE : (batch - b0);
        for (int r = row_start; r < row_end; r++) {
            for (int b = b0; b < b0 + b_count; b++) {
                float sum = 0.0f;
                const float* act_b = act + b * in_dim;

                for (int blk = 0; blk < num_blocks; blk++) {
                    const int byte_off = r * row_stride + blk * B1_0_BLOCK_BYTES;

                    uint16_t scale_bits;
                    memcpy(&scale_bits, &w[byte_off], 2);
                    float scale = ggml_fp16_to_fp32(scale_bits);

                    uint32_t bits;
                    memcpy(&bits, &w[byte_off + 2], 4);

                    float block_sum = 0.0f;
                    for (int i = 0; i < B1_0_BLOCK_SIZE; i++) {
                        int i_dim = blk * B1_0_BLOCK_SIZE + i;
                        float a = act_b[i_dim];
                        if ((bits >> i) & 1) {
                            block_sum += a;
                        } else {
                            block_sum -= a;
                        }
                    }
                    sum += scale * block_sum;
                }

                if (std::isnan(sum) || std::isinf(sum)) {
                    fprintf(stderr, "B1_FATAL_OUT: NaN generated in b1_linear! out_dim=%d r=%d b=%d\n", out_dim, r, b);
                    abort();
                }
                out[b * out_dim + r] = sum;
            }
        }
    }
}

static inline ggml_tensor * b1_linear(
    ggml_context * ctx,
    ggml_tensor * act,
    ggml_tensor * w_data,
    int in_dim,
    int out_dim,
    int n_threads)
{
    B1LinearUserData * ud = (B1LinearUserData *)malloc(sizeof(B1LinearUserData));
    ud->in_dim = in_dim;
    ud->out_dim = out_dim;

    ggml_tensor * args[] = { act, w_data };
    return ggml_custom_4d(ctx, GGML_TYPE_F32,
        out_dim, act->ne[1], 1, 1,
        args, 2,
        b1_linear_fwd,
        n_threads, ud);
}

/* Quantification B1_0: convertit un tableau float en format compressé B1_0 */
static uint8_t * quantize_b1_0(const float* weights, int in_dim, int out_dim) {
    int num_blocks = in_dim / B1_0_BLOCK_SIZE;
    size_t nbytes = out_dim * num_blocks * B1_0_BLOCK_BYTES;
    uint8_t * out = (uint8_t *)malloc(nbytes);
    if (!out) { return NULL; }

    for (int r = 0; r < out_dim; r++) {
        for (int blk = 0; blk < num_blocks; blk++) {
            // Calculer la scale (mean abs)
            float scale = 0.0f;
            for (int i = 0; i < B1_0_BLOCK_SIZE; i++) {
                int idx = r * in_dim + blk * B1_0_BLOCK_SIZE + i;
                scale += fabsf(weights[idx]);
            }
            scale /= B1_0_BLOCK_SIZE;

            // Pack les signes
            uint32_t bits = 0;
            for (int i = 0; i < B1_0_BLOCK_SIZE; i++) {
                int idx = r * in_dim + blk * B1_0_BLOCK_SIZE + i;
                if (weights[idx] >= 0) {
                    bits |= (1u << i);
                }
            }

            // Écrire le bloc
            int byte_off = (r * num_blocks + blk) * B1_0_BLOCK_BYTES;
            uint16_t scale_bits = ggml_fp32_to_fp16(scale);
            memcpy(&out[byte_off], &scale_bits, 2);
            memcpy(&out[byte_off + 2], &bits, 4);
        }
    }

    return out;
}

#endif /* B1_0_KERNEL_H */

/* Fonction utilitaire pour lire un fichier binaire de poids float32 */
static float* read_float_bin(const char* path, size_t* count) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return NULL; }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    float* data = (float*)malloc(size);
    size_t read = fread(data, 1, size, f);
    if (read != (size_t)size) {
        fprintf(stderr, "Warning: expected %ld bytes, got %zu\n", size, read);
    }
    fclose(f);

    *count = size / sizeof(float);
    return data;
}

static void write_float_bin(const char* path, float* data, size_t count) {
    FILE* f = fopen(path, "wb");
    fwrite(data, sizeof(float), count, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <act.bin> <weight.bin> <out.bin> [batch] [in_dim] [out_dim]\n", argv[0]);
        fprintf(stderr, "  act.bin    : float32 [batch * in_dim]\n");
        fprintf(stderr, "  weight.bin : float32 [out_dim * in_dim]\n");
        fprintf(stderr, "  out.bin    : float32 [batch * out_dim]\n");
        return 1;
    }

    const char* act_path = argv[1];
    const char* weight_path = argv[2];
    const char* out_path = argv[3];

    int batch = (argc > 4) ? atoi(argv[4]) : 1;
    int in_dim = (argc > 5) ? atoi(argv[5]) : 64;
    int out_dim = (argc > 6) ? atoi(argv[6]) : 32;

    printf("Testing b1_linear: batch=%d in_dim=%d out_dim=%d\n", batch, in_dim, out_dim);

    /* 1. Lecture des inputs */
    size_t act_count = 0, weight_count = 0;
    float* act_data = read_float_bin(act_path, &act_count);
    float* weight_data = read_float_bin(weight_path, &weight_count);

    if (!act_data || !weight_data) {
        fprintf(stderr, "Error reading input files\n");
        return 1;
    }

    printf("  Read %zu floats from %s\n", act_count, act_path);
    printf("  Read %zu floats from %s\n", weight_count, weight_path);

    /* 2. Vérification des dimensions */
    if (act_count != (size_t)batch * in_dim) {
        fprintf(stderr, "Error: expected act_count=%d, got %zu\n", batch * in_dim, act_count);
        return 1;
    }
    if (weight_count != (size_t)out_dim * in_dim) {
        fprintf(stderr, "Error: expected weight_count=%d, got %zu\n", out_dim * in_dim, weight_count);
        return 1;
    }

    /* 3. Quantification B1_0 des poids */
    printf("  Quantizing weights to B1_0...\n");
    uint8_t* b1_weights = quantize_b1_0(weight_data, in_dim, out_dim);
    if (!b1_weights) {
        fprintf(stderr, "Error quantizing weights\n");
        return 1;
    }

    /* 4. Allocation et initialisation GGML */
    size_t ctx_size = 1024 * 1024 * 1024; // 1GB
    void* buffer = malloc(ctx_size);
    if (!buffer) {
        fprintf(stderr, "Error allocating memory\n");
        return 1;
    }

    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = buffer,
        .no_alloc   = false,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Error initializing GGML context\n");
        return 1;
    }

    /* 5. Création des tenseurs */
    ggml_tensor* t_act = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, batch);
    memcpy(t_act->data, act_data, act_count * sizeof(float));

    // Les poids quantifiés B1_0 sont stockés comme des bytes raw
    int num_blocks_total = out_dim * (in_dim / B1_0_BLOCK_SIZE) * B1_0_BLOCK_BYTES;
    ggml_tensor* t_w = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, num_blocks_total);
    memcpy(t_w->data, b1_weights, num_blocks_total);

    /* 6. Exécution de b1_linear */
    printf("  Running b1_linear...\n");
    ggml_tensor* t_out = b1_linear(ctx, t_act, t_w, in_dim, out_dim, 4);

    // Construire le graphe et allouer
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, t_out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    // Exécuter
    ggml_backend_graph_compute(backend, graph);

    /* 7. Récupération du résultat */
    size_t out_count = batch * out_dim;
    float* out_data = (float*)malloc(out_count * sizeof(float));
    memcpy(out_data, t_out->data, out_count * sizeof(float));

    /* 8. Écriture du résultat */
    write_float_bin(out_path, out_data, out_count);
    printf("  Written %zu floats to %s\n", out_count, out_path);

    /* 9. Cleanup */
    free(act_data);
    free(weight_data);
    free(b1_weights);
    free(out_data);
    free(buffer);
    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);

    printf("  Done!\n");
    return 0;
}
