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

struct Rope2DUserData {
    int head_dim;
    int n_heads;
    int seq;
};

inline void rope_2d_fwd_f32(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * ud = (const Rope2DUserData *)userdata;
    const int head_dim = ud->head_dim;
    const int n_heads  = ud->n_heads;
    const int seq      = ud->seq;
    const int half     = head_dim / 2;

    const ggml_tensor * src_t  = dst->src[0];
    const ggml_tensor * cos_t  = dst->src[1];
    const ggml_tensor * sin_t  = dst->src[2];

    const float * src = (const float *)src_t->data;
    const float * c   = (const float *)cos_t->data;
    const float * s   = (const float *)sin_t->data;
    float * out       = (float *)dst->data;

    const int head_stride = n_heads * seq;
    const int total       = half * n_heads * seq;
    const int per_thread  = (total + nth - 1) / nth;
    const int start       = ith * per_thread;
    const int end         = (start + per_thread > total) ? total : start + per_thread;

    for (int idx = start; idx < end; idx++) {
        const int d_pair  = idx / (n_heads * seq);
        const int rem     = idx - d_pair * (n_heads * seq);
        const int h       = rem / seq;
        const int s_idx   = rem - h * seq;

        const int d_even = d_pair * 2;
        const int d_odd  = d_even + 1;

        // GGML storage for [head_dim, n_heads, seq]: x[d + h*D + s*D*H]
        const int base    = h * head_dim + s_idx * (head_dim * n_heads);
        const int even_idx = d_even + base;
        const int odd_idx  = d_odd  + base;

        const int angle_idx = d_pair * seq + s_idx;
        const float cos_v   = c[angle_idx];
        const float sin_v   = s[angle_idx];

        const float a = src[even_idx];
        const float b = src[odd_idx];

        out[even_idx] = a * cos_v - b * sin_v;
        out[odd_idx]  = a * sin_v + b * cos_v;
    }
    (void)head_stride;
}

inline ggml_tensor * rope_2d_fwd(ggml_context * ctx, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_t, Rope2DUserData & ud) {
    ggml_tensor * args[] = { x, cos_t, sin_t };
    return ggml_custom_4d(ctx, GGML_TYPE_F32,
        (int64_t)ud.head_dim, (int64_t)ud.n_heads, (int64_t)ud.seq, 1,
        args, 3,
        rope_2d_fwd_f32,
        GGML_N_TASKS_MAX, &ud);
}

} // namespace bonsai
