#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <cmath>

#ifdef __AVX2__
#include <immintrin.h>
#endif

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

    if (ith == 0 && !w_t->data) {
        fprintf(stderr, "B1_SEGFAULT: w_t->data is null! in_dim=%d out_dim=%d act_ne=[%lld,%lld,%lld,%lld] w_t_ne=[%lld,%lld,%lld,%lld] act_data=%p\n",
            in_dim, out_dim,
            (long long)act_t->ne[0], (long long)act_t->ne[1], (long long)act_t->ne[2], (long long)act_t->ne[3],
            (long long)w_t->ne[0], (long long)w_t->ne[1], (long long)w_t->ne[2], (long long)w_t->ne[3],
            act_t->data);
    }

    const float * act = (const float *)act_t->data;
    const uint8_t * w = (const uint8_t *)w_t->data;
    float * out = (float *)dst->data;

    if (ith == 0) {
        fprintf(stderr, "B1_DBG: in_dim=%d out_dim=%d num_blocks=%d row_stride=%d nbytes=%lld act_data=%p w_data=%p out_data=%p\n",
            in_dim, out_dim, in_dim/32, (in_dim/32)*6,
            (long long)w_t->ne[0], (void*)act, (void*)w, (void*)out);

        // Check for NaN in input (only thread 0 checks first batch to save time)
        for (int i = 0; i < in_dim; i++) {
            if (std::isnan(act[i])) {
                fprintf(stderr, "B1_FATAL: act_data contains NaN in b1_linear! in_dim=%d out_dim=%d\n", in_dim, out_dim);
                abort();
            }
        }
    }

    const int batch = (int)act_t->ne[1];
    const int num_blocks = in_dim / B1_0_BLOCK_SIZE;
    const int row_stride = num_blocks * B1_0_BLOCK_BYTES;

    const int rows_per_thread = (out_dim + nth - 1) / nth;
    const int row_start = ith * rows_per_thread;
    const int row_end = (row_start + rows_per_thread > out_dim) ? out_dim : row_start + rows_per_thread;

#ifdef __AVX2__
    if (ith == 0 && act_t->ne[1] > 0) { // On s'assure de ne l'afficher qu'une fois
        static bool printed = false;
        if (!printed) {
            fprintf(stdout, "\n BONSAI INFO: b1_linear uses the AVX2-optimized path + cache tiling !\n\n");
            printed = true;
        }
    }
    const __m256i msb_only = _mm256_set1_epi32(0x80000000);
    const __m256i shift0 = _mm256_set_epi32(31-7, 31-6, 31-5, 31-4, 31-3, 31-2, 31-1, 31-0);
    const __m256i shift1 = _mm256_set_epi32(31-15, 31-14, 31-13, 31-12, 31-11, 31-10, 31-9, 31-8);
    const __m256i shift2 = _mm256_set_epi32(31-23, 31-22, 31-21, 31-20, 31-19, 31-18, 31-17, 31-16);
    const __m256i shift3 = _mm256_set_epi32(31-31, 31-30, 31-29, 31-28, 31-27, 31-26, 31-25, 31-24);

    const int B_TILE = 32;
    const int R_TILE = 4;

    for (int b0 = 0; b0 < batch; b0 += B_TILE) {
        int b_count = (b0 + B_TILE <= batch) ? B_TILE : (batch - b0);
        
        for (int r0 = row_start; r0 < row_end; r0 += R_TILE) {
            int r_count = (r0 + R_TILE <= row_end) ? R_TILE : (row_end - r0);
            
            for (int b = b0; b < b0 + b_count; b++) {
                const float* act_b = act + b * in_dim;
                
                if (r_count == 4) {
                    __m256 sum0 = _mm256_setzero_ps();
                    __m256 sum1 = _mm256_setzero_ps();
                    __m256 sum2 = _mm256_setzero_ps();
                    __m256 sum3 = _mm256_setzero_ps();

                    const uint8_t* w0 = w + (r0 + 0) * row_stride;
                    const uint8_t* w1 = w + (r0 + 1) * row_stride;
                    const uint8_t* w2 = w + (r0 + 2) * row_stride;
                    const uint8_t* w3 = w + (r0 + 3) * row_stride;

                    for (int blk = 0; blk < num_blocks; blk++) {
                        __m256 a0 = _mm256_loadu_ps(act_b + blk * 32 + 0);
                        __m256 a1 = _mm256_loadu_ps(act_b + blk * 32 + 8);
                        __m256 a2 = _mm256_loadu_ps(act_b + blk * 32 + 16);
                        __m256 a3 = _mm256_loadu_ps(act_b + blk * 32 + 24);

                        // Row 0
                        {
                            uint16_t s_bits; memcpy(&s_bits, w0, 2);
                            __m256 vscale = _mm256_set1_ps(fp16_to_float(s_bits));
                            uint32_t bits; memcpy(&bits, w0 + 2, 4); w0 += 6;
                            __m256i v_inv = _mm256_set1_epi32(~bits);
                            __m256 m0 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift0), msb_only));
                            __m256 m1 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift1), msb_only));
                            __m256 m2 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift2), msb_only));
                            __m256 m3 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift3), msb_only));
                            __m256 term = _mm256_add_ps(_mm256_add_ps(_mm256_xor_ps(a0, m0), _mm256_xor_ps(a1, m1)),
                                                        _mm256_add_ps(_mm256_xor_ps(a2, m2), _mm256_xor_ps(a3, m3)));
                            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(term, vscale));
                        }
                        // Row 1
                        {
                            uint16_t s_bits; memcpy(&s_bits, w1, 2);
                            __m256 vscale = _mm256_set1_ps(fp16_to_float(s_bits));
                            uint32_t bits; memcpy(&bits, w1 + 2, 4); w1 += 6;
                            __m256i v_inv = _mm256_set1_epi32(~bits);
                            __m256 m0 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift0), msb_only));
                            __m256 m1 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift1), msb_only));
                            __m256 m2 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift2), msb_only));
                            __m256 m3 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift3), msb_only));
                            __m256 term = _mm256_add_ps(_mm256_add_ps(_mm256_xor_ps(a0, m0), _mm256_xor_ps(a1, m1)),
                                                        _mm256_add_ps(_mm256_xor_ps(a2, m2), _mm256_xor_ps(a3, m3)));
                            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(term, vscale));
                        }
                        // Row 2
                        {
                            uint16_t s_bits; memcpy(&s_bits, w2, 2);
                            __m256 vscale = _mm256_set1_ps(fp16_to_float(s_bits));
                            uint32_t bits; memcpy(&bits, w2 + 2, 4); w2 += 6;
                            __m256i v_inv = _mm256_set1_epi32(~bits);
                            __m256 m0 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift0), msb_only));
                            __m256 m1 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift1), msb_only));
                            __m256 m2 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift2), msb_only));
                            __m256 m3 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift3), msb_only));
                            __m256 term = _mm256_add_ps(_mm256_add_ps(_mm256_xor_ps(a0, m0), _mm256_xor_ps(a1, m1)),
                                                        _mm256_add_ps(_mm256_xor_ps(a2, m2), _mm256_xor_ps(a3, m3)));
                            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(term, vscale));
                        }
                        // Row 3
                        {
                            uint16_t s_bits; memcpy(&s_bits, w3, 2);
                            __m256 vscale = _mm256_set1_ps(fp16_to_float(s_bits));
                            uint32_t bits; memcpy(&bits, w3 + 2, 4); w3 += 6;
                            __m256i v_inv = _mm256_set1_epi32(~bits);
                            __m256 m0 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift0), msb_only));
                            __m256 m1 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift1), msb_only));
                            __m256 m2 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift2), msb_only));
                            __m256 m3 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(v_inv, shift3), msb_only));
                            __m256 term = _mm256_add_ps(_mm256_add_ps(_mm256_xor_ps(a0, m0), _mm256_xor_ps(a1, m1)),
                                                        _mm256_add_ps(_mm256_xor_ps(a2, m2), _mm256_xor_ps(a3, m3)));
                            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(term, vscale));
                        }
                    }

                    auto hsum256_ps = [](__m256 v) -> float {
                        __m128 vlow = _mm256_castps256_ps128(v);
                        __m128 vhigh = _mm256_extractf128_ps(v, 1);
                        vlow = _mm_add_ps(vlow, vhigh);
                        vlow = _mm_hadd_ps(vlow, vlow);
                        vlow = _mm_hadd_ps(vlow, vlow);
                        return _mm_cvtss_f32(vlow);
                    };

                    float s0 = hsum256_ps(sum0);
                    float s1 = hsum256_ps(sum1);
                    float s2 = hsum256_ps(sum2);
                    float s3 = hsum256_ps(sum3);

                    if (std::isnan(s0) || std::isinf(s0) || std::isnan(s1) || std::isinf(s1) || 
                        std::isnan(s2) || std::isinf(s2) || std::isnan(s3) || std::isinf(s3)) {
                        fprintf(stderr, "B1_FATAL_OUT: NaN generated in AVX2 b1_linear! r=%d b=%d\n", r0, b);
                        abort();
                    }

                    out[b * out_dim + r0 + 0] = s0;
                    out[b * out_dim + r0 + 1] = s1;
                    out[b * out_dim + r0 + 2] = s2;
                    out[b * out_dim + r0 + 3] = s3;
                } else {
                    for (int rr = 0; rr < r_count; rr++) {
                        int r = r0 + rr;
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
                            fprintf(stderr, "B1_FATAL_OUT: NaN generated in b1_linear! r=%d b=%d\n", r, b);
                            abort();
                        }
                        out[b * out_dim + r] = sum;
                    }
                }
            }
        }
    }
#else
    if (ith == 0 && act_t->ne[1] > 0) {
        static bool printed = false;
        if (!printed) {
            fprintf(stdout, "\n BONSAI WARNING: b1_linear uses the SLOW path (scalar). Compile with -mavx2 !\n\n");
            printed = true;
        }
    }
    const int B_TILE = 32;
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
                    float scale = fp16_to_float(scale_bits);

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
                    fprintf(stderr, "B1_FATAL_OUT: NaN generated in b1_linear! in_dim=%d out_dim=%d r=%d b=%d\n", in_dim, out_dim, r, b);
                    abort();
                }
                out[b * out_dim + r] = sum;
            }
        }
    }
#endif
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

    const int total_blocks = n_heads * seq;
    const int blocks_per_thread = (total_blocks + nth - 1) / nth;
    const int start_block = ith * blocks_per_thread;
    const int end_block = (start_block + blocks_per_thread > total_blocks) ? total_blocks : start_block + blocks_per_thread;

    int s_idx = start_block / n_heads;
    int h     = start_block % n_heads;

    for (int block = start_block; block < end_block; block++) {
        // Calculate the base pointer offsets once per sequence-head pair
        const int base = h * head_dim + s_idx * (head_dim * n_heads);
        const int angle_base = s_idx * half;

        // Innermost loop processes half the head_dim sequentially
        for (int d_pair = 0; d_pair < half; d_pair++) {
            const int even_idx = d_pair * 2 + base;
            const int odd_idx  = even_idx + 1;

            const int angle_idx = d_pair + angle_base;
            const float cos_v   = c[angle_idx];
            const float sin_v   = s[angle_idx];

            const float a = src[even_idx];
            const float b = src[odd_idx];

            out[even_idx] = a * cos_v - b * sin_v;
            out[odd_idx]  = a * sin_v + b * cos_v;
        }

        // Advance sequential tracking variables (0 integer divisions per element)
        h++;
        if (h == n_heads) {
            h = 0;
            s_idx++;
        }
    }
}

inline ggml_tensor * rope_2d_fwd(ggml_context * ctx, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_t, Rope2DUserData & ud) {
    ggml_tensor * args[] = { x, cos_t, sin_t };
    return ggml_custom_4d(ctx, GGML_TYPE_F32,
        (int64_t)ud.head_dim, (int64_t)ud.n_heads, (int64_t)ud.seq, 1,
        args, 3,
        rope_2d_fwd_f32,
        GGML_N_TASKS_MAX, &ud);
}

struct DebugCheckUserData {
    const char * label;
    int * counter;
};

inline void debug_check_fwd_f32(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * ud = (const DebugCheckUserData *)userdata;
    const ggml_tensor * src_t = dst->src[0];
    const float * src = (const float *)src_t->data;
    float * out = (float *)dst->data;
    const int n = (int)ggml_nelements(src_t);
    int nan_c = 0, inf_c = 0;
    float mn = 1e30f, mx = -1e30f;
    for (int i = 0; i < n; i++) {
        float v = src[i];
        if (std::isnan(v)) nan_c++;
        if (std::isinf(v)) inf_c++;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    if (ith == 0) {
        fprintf(stderr, "[CHK %s] n=%d nan=%d inf=%d min=%.4f max=%.4f\n",
            ud->label, n, nan_c, inf_c, mn, mx);
    }
    if (ud->counter) (*ud->counter)++;
    if (dst->data != src_t->data) {
        memcpy(out, src, ggml_nbytes(src_t));
    }
}

inline ggml_tensor * debug_check(ggml_context * ctx, ggml_tensor * x, const char * label, int * counter) {
    DebugCheckUserData * ud = (DebugCheckUserData *)malloc(sizeof(DebugCheckUserData));
    ud->label = label;
    ud->counter = counter;
    ggml_tensor * args[] = { x };
    return ggml_custom_4d(ctx, x->type, x->ne[0], x->ne[1], x->ne[2], x->ne[3],
        args, 1, debug_check_fwd_f32, 1, ud);
}

} // namespace bonsai
