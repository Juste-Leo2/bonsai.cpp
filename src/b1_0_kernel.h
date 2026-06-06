#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>

namespace bonsai {

inline float fp16_to_float(const uint16_t v) {
    if (v == 0) return 0.0f;
    uint32_t sign = (v & 0x8000) << 16;
    uint32_t exp = (v & 0x7C00) >> 10;
    uint32_t mant = v & 0x03FF;
    if (exp == 0) {
        exp = 0;
        mant = mant;
    } else if (exp == 31) {
        exp = 0x1F;
        mant = mant;
    } else {
        exp += 112;
    }
    uint32_t f = sign | (exp << 23) | (mant << 13);
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

inline int popcount64(uint64_t x) {
#ifdef __POPCOUNT__
    return __builtin_popcountll(x);
#else
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
#endif
}

struct B1LinearUserData {
    int in_dim;
    int out_dim;
};

inline void b1_linear_f32_f32(ggml_tensor * dst, int ith, int nth, void * userdata) {
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

    for (int r = row_start; r < row_end; r++) {
        for (int b = 0; b < batch; b++) {
            float sum = 0.0f;

            for (int blk = 0; blk < num_blocks; blk++) {
                const int byte_off = r * row_stride + blk * B1_0_BLOCK_BYTES;

                uint16_t scale_bits;
                memcpy(&scale_bits, &w[byte_off], 2);
                float scale = fp16_to_float(scale_bits);

                uint32_t bits;
                memcpy(&bits, &w[byte_off + 2], 4);

                float block_sum = 0.0f;
                for (int i = 0; i < B1_0_BLOCK_SIZE; i++) {
                    float a = act[(blk * B1_0_BLOCK_SIZE + i) * batch + b];
                    if ((bits >> i) & 1) {
                        block_sum += a;
                    } else {
                        block_sum -= a;
                    }
                }
                sum += scale * block_sum;
            }

            out[r * batch + b] = sum;
        }
    }
}

inline ggml_tensor * b1_linear(ggml_context * ctx, ggml_tensor * act, const B1Weights & w, int n_threads, B1LinearUserData & ud) {
    ud.in_dim = w.in_dim;
    ud.out_dim = w.out_dim;

    ggml_tensor * args[] = { act, w.data };
    return ggml_custom_4d(ctx, GGML_TYPE_F32,
        w.out_dim, act->ne[1], 1, 1,
        args, 2,
        b1_linear_f32_f32,
        n_threads, &ud);
}

} // namespace bonsai
