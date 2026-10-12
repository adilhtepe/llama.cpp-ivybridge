// Checks the CPU flash_attn_ext kernels (chunked decode, tiled prefill and the per-row
// fallback) against attention built from plain ggml ops: K/V cast to F32, mul_mat,
// soft_max_ext, mul_mat. Covers F16 and quantized K/V, several head sizes and KV lengths
// around the chunked decode threshold. Also checks that q8_0/q4_0 row dequantization
// matches the scalar formula bit for bit and that soft_max matches a scalar expf reference.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <initializer_list>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// failed checks; main exits non-zero on any failures
static int n_failed = 0;

static float frand(float lo, float hi) {
    return lo + (hi - lo) * ((float) rand() / (float) RAND_MAX);
}

// The CPU-specific control API is resolved through the backend registry: with
// GGML_BACKEND_DL the CPU backend is a runtime-loaded module and its symbols
// are not available at link time
static void cpu_set_use_ref(ggml_backend_t backend, bool use_ref) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_use_ref)(ggml_backend_t, bool) =
        (void (*)(ggml_backend_t, bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
    if (!set_use_ref) {
        fprintf(stderr, "ggml_backend_cpu_set_use_ref not available\n");
        exit(1);
    }
    set_use_ref(backend, use_ref);
}

static void cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_n_threads)(ggml_backend_t, int) =
        (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (!set_n_threads) {
        fprintf(stderr, "ggml_backend_set_n_threads not available\n");
        exit(1);
    }
    set_n_threads(backend, n_threads);
}

// fill t from a flat f32 source of rows x cols; quantized types go through ggml_quantize_chunk
static void fill_tensor(struct ggml_tensor * t, const float * src, int64_t rows, int64_t cols) {
    GGML_ASSERT(t->ne[0] == cols);
    GGML_ASSERT(rows * cols == ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
        return;
    }
    void * q = malloc(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) q, rows * cols);
    } else {
        ggml_quantize_chunk(t->type, src, q, 0, rows, cols, NULL);
    }
    ggml_backend_tensor_set(t, q, 0, ggml_nbytes(t));
    free(q);
}

// normalized mean squared error, the metric used by test-backend-ops
static double nmse(const float * ref, const float * out, int64_t n, float * max_err) {
    double num = 0.0;
    double den = 0.0;
    *max_err = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double) ref[i] - (double) out[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
        *max_err = fmaxf(*max_err, fabsf((float) d));
    }
    return den > 0.0 ? num / den : num;
}

enum mask_kind {
    MASK_NONE,
    MASK_CAUSAL,
    MASK_BLOCKS, // two of every three 32-position blocks fully masked: whole KV blocks skipped mid-sequence
    MASK_TAIL,   // everything but the last 40 positions masked: blocks and decode chunks skipped before any max is set
};

static const char * mask_kind_name(enum mask_kind kind) {
    switch (kind) {
        case MASK_CAUSAL: return "causal";
        case MASK_BLOCKS: return "blocks";
        case MASK_TAIL:   return "tail";
        default:          return "none";
    }
}

static bool mask_is_masked(enum mask_kind kind, int64_t n_kv, int64_t n_q, int64_t iq, int64_t ik) {
    switch (kind) {
        case MASK_CAUSAL: return ik > n_kv - n_q + iq;
        case MASK_BLOCKS: return (ik / 32) % 3 != 2;
        case MASK_TAIL:   return ik < n_kv - 40;
        default:          return false;
    }
}

// q: [DK, n_q, n_head], k: [DK, n_kv, n_head_kv], v: [DV, n_kv, n_head_kv]
static void test_flash_attn(ggml_backend_t backend, int64_t DK, int64_t DV, int64_t n_head, int64_t n_head_kv,
                            int64_t n_kv, int64_t n_q, ggml_type type_K, ggml_type type_V,
                            enum mask_kind mask_kind, float max_bias, float q_range) {
    srand(0xBEEF);

    const float scale = 1.0f / sqrtf((float) DK);

    float * q_data = (float *) malloc(DK * n_q  * n_head    * sizeof(float));
    float * k_data = (float *) malloc(DK * n_kv * n_head_kv * sizeof(float));
    float * v_data = (float *) malloc(DV * n_kv * n_head_kv * sizeof(float));
    for (int64_t i = 0; i < DK * n_q  * n_head;    ++i) q_data[i] = frand(-q_range, q_range);
    for (int64_t i = 0; i < DK * n_kv * n_head_kv; ++i) k_data[i] = frand(-1.0f, 1.0f);
    for (int64_t i = 0; i < DV * n_kv * n_head_kv; ++i) v_data[i] = frand(-1.0f, 1.0f);

    struct ggml_init_params ip = { 256*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DK, n_q,  n_head);
    struct ggml_tensor * k = ggml_new_tensor_3d(ctx, type_K,        DK, n_kv, n_head_kv);
    struct ggml_tensor * v = ggml_new_tensor_3d(ctx, type_V,        DV, n_kv, n_head_kv);

    struct ggml_tensor * mask = NULL;
    if (mask_kind != MASK_NONE || max_bias > 0.0f) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_q);
    }

    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    // reference: K/V in F32, scores with mul_mat, softmax, values with mul_mat
    struct ggml_tensor * k32 = ggml_cast(ctx, k, GGML_TYPE_F32);
    struct ggml_tensor * v32 = ggml_cast(ctx, v, GGML_TYPE_F32);
    struct ggml_tensor * kq  = ggml_mul_mat(ctx, k32, q);                        // [n_kv, n_q, n_head]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, max_bias);
    struct ggml_tensor * vt  = ggml_cont(ctx, ggml_transpose(ctx, v32));         // [n_kv, DV, n_head_kv]
    struct ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);                        // [DV, n_q, n_head]
    struct ggml_tensor * ref = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [DV, n_head, n_q]

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, ref);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    fill_tensor(q, q_data, n_q  * n_head,    DK);
    fill_tensor(k, k_data, n_kv * n_head_kv, DK);
    fill_tensor(v, v_data, n_kv * n_head_kv, DV);

    if (mask) {
        ggml_fp16_t * m = (ggml_fp16_t *) malloc(n_kv * n_q * sizeof(ggml_fp16_t));
        for (int64_t iq = 0; iq < n_q; ++iq) {
            for (int64_t ik = 0; ik < n_kv; ++ik) {
                const bool masked = mask_is_masked(mask_kind, n_kv, n_q, iq, ik);
                m[iq * n_kv + ik] = ggml_fp32_to_fp16(masked ? -INFINITY : 0.0f);
            }
        }
        ggml_backend_tensor_set(mask, m, 0, ggml_nbytes(mask));
        free(m);
    }

    const int64_t n_out = ggml_nelements(out);
    float * out_fast = (float *) malloc(n_out * sizeof(float));
    float * out_ref  = (float *) malloc(n_out * sizeof(float));
    float * ref_data = (float *) malloc(n_out * sizeof(float));

    // fast paths (chunked decode / tiled prefill / per-row), then the vec-only reference path
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, out_fast, 0, ggml_nbytes(out));
    ggml_backend_tensor_get(ref, ref_data, 0, ggml_nbytes(ref));
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, out_ref, 0, ggml_nbytes(out));
    cpu_set_use_ref(backend, false);

    const double tol = 5e-4;

    float max_err_fast, max_err_ref;
    const double err_fast = nmse(ref_data, out_fast, n_out, &max_err_fast);
    const double err_ref  = nmse(ref_data, out_ref,  n_out, &max_err_ref);
    const bool ok = err_fast <= tol && err_ref <= tol;

    const char * path = (n_q == 1 && n_kv >= 512) ? "chunked" : (n_q >= 64 ? "tiled" : "rows");
    printf("TEST fattn DK=%lld DV=%lld heads=%lld/%lld n_kv=%lld n_q=%lld K=%s V=%s mask=%s max_bias=%.1f q_range=%.0f (%s): %s "
           "(fast nmse: %.2e max_err: %.2e, vec nmse: %.2e max_err: %.2e, tol: %.0e)\n",
           (long long) DK, (long long) DV, (long long) n_head, (long long) n_head_kv, (long long) n_kv, (long long) n_q,
           ggml_type_name(type_K), ggml_type_name(type_V), mask_kind_name(mask_kind), max_bias, q_range,
           path, ok ? "PASS" : "FAIL", err_fast, max_err_fast, err_ref, max_err_ref, tol);

    if (!ok) {
        ++n_failed;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(q_data); free(k_data); free(v_data);
    free(out_fast); free(out_ref); free(ref_data);
}

// a NaN query row must give a NaN output row and leave the other rows finite. With a NaN Q every
// score of every KV block is NaN, which the per-row kernel's running max would otherwise drop
static void test_nan_propagation(ggml_backend_t backend, int64_t DK, int64_t n_kv, int64_t n_q, ggml_type type_KV) {
    srand(0xBEEF);

    const int64_t n_head = 2;
    const int64_t nan_iq = 1; // query row that is set to NaN, in head 0

    float * q_data = (float *) malloc(DK * n_q  * n_head * sizeof(float));
    float * k_data = (float *) malloc(DK * n_kv * n_head * sizeof(float));
    float * v_data = (float *) malloc(DK * n_kv * n_head * sizeof(float));
    for (int64_t i = 0; i < DK * n_q  * n_head; ++i) q_data[i] = frand(-1.0f, 1.0f);
    for (int64_t i = 0; i < DK * n_kv * n_head; ++i) k_data[i] = frand(-1.0f, 1.0f);
    for (int64_t i = 0; i < DK * n_kv * n_head; ++i) v_data[i] = frand(-1.0f, 1.0f);
    for (int64_t d = 0; d < DK; ++d) q_data[nan_iq * DK + d] = NAN;

    struct ggml_init_params ip = { 16*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DK, n_q,  n_head);
    struct ggml_tensor * k = ggml_new_tensor_3d(ctx, type_KV,       DK, n_kv, n_head);
    struct ggml_tensor * v = ggml_new_tensor_3d(ctx, type_KV,       DK, n_kv, n_head);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_q);

    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / sqrtf((float) DK), 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    fill_tensor(q, q_data, n_q  * n_head, DK);
    fill_tensor(k, k_data, n_kv * n_head, DK);
    fill_tensor(v, v_data, n_kv * n_head, DK);

    ggml_fp16_t * m = (ggml_fp16_t *) malloc(n_kv * n_q * sizeof(ggml_fp16_t));
    for (int64_t iq = 0; iq < n_q; ++iq) {
        for (int64_t ik = 0; ik < n_kv; ++ik) {
            m[iq * n_kv + ik] = ggml_fp32_to_fp16(mask_is_masked(MASK_CAUSAL, n_kv, n_q, iq, ik) ? -INFINITY : 0.0f);
        }
    }
    ggml_backend_tensor_set(mask, m, 0, ggml_nbytes(mask));
    free(m);

    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);

    // out: [DV, n_head, n_q]
    float * o = (float *) malloc(ggml_nbytes(out));
    ggml_backend_tensor_get(out, o, 0, ggml_nbytes(out));

    int64_t n_nan_expected = 0, n_nan_unexpected = 0;
    for (int64_t iq = 0; iq < n_q; ++iq) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t d = 0; d < DK; ++d) {
                const bool is_nan = isnan(o[(iq * n_head + h) * DK + d]);
                if (iq == nan_iq && h == 0) {
                    n_nan_expected += is_nan;
                } else {
                    n_nan_unexpected += is_nan;
                }
            }
        }
    }
    const bool ok = n_nan_expected == DK && n_nan_unexpected == 0;

    printf("TEST nan_propagation DK=%lld n_kv=%lld n_q=%lld KV=%s: %s (NaN in the NaN row: %lld/%lld, elsewhere: %lld)\n",
           (long long) DK, (long long) n_kv, (long long) n_q, ggml_type_name(type_KV), ok ? "PASS" : "FAIL",
           (long long) n_nan_expected, (long long) DK, (long long) n_nan_unexpected);

    if (!ok) {
        ++n_failed;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(q_data); free(k_data); free(v_data); free(o);
}

// to_float of a q8_0/q4_0 row must give exactly (int) q * d, the scalar reference formula
static void test_dequant_exact(ggml_type type) {
    srand(0xBEEF);

    const int64_t n = 32 * 37;
    float * src = (float *) malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; ++i) src[i] = frand(-8.0f, 8.0f);

    const size_t row_size = ggml_row_size(type, n);
    uint8_t * q = (uint8_t *) malloc(row_size);
    ggml_quantize_chunk(type, src, q, 0, 1, n, NULL);

    float * out = (float *) malloc(n * sizeof(float));
    ggml_get_type_traits(type)->to_float(q, out, n);

    float * ref = (float *) malloc(n * sizeof(float));
    const int64_t nb = n / 32;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const uint8_t * blk = q + ib * ggml_type_size(type);
        ggml_fp16_t d16;
        memcpy(&d16, blk, sizeof(d16));
        const float d = ggml_fp16_to_fp32(d16);
        if (type == GGML_TYPE_Q8_0) {
            const int8_t * qs = (const int8_t *) (blk + 2);
            for (int j = 0; j < 32; ++j) {
                ref[ib*32 + j] = qs[j]*d;
            }
        } else {
            const uint8_t * qs = blk + 2;
            for (int j = 0; j < 16; ++j) {
                const int x0 = (qs[j] & 0x0F) - 8;
                const int x1 = (qs[j] >>   4) - 8;
                ref[ib*32 + j +  0] = x0*d;
                ref[ib*32 + j + 16] = x1*d;
            }
        }
    }

    const bool ok = memcmp(ref, out, n * sizeof(float)) == 0;
    printf("TEST dequant %s bit-exact: %s\n", ggml_type_name(type), ok ? "PASS" : "FAIL");
    if (!ok) {
        ++n_failed;
    }

    free(src); free(q); free(out); free(ref);
}

// soft_max_ext against a scalar expf reference, with masked (-inf) entries
static void test_soft_max(ggml_backend_t backend, int64_t n_cols, int64_t n_rows) {
    srand(0xBEEF);

    float * x = (float *) malloc(n_cols * n_rows * sizeof(float));
    ggml_fp16_t * m = (ggml_fp16_t *) malloc(n_cols * n_rows * sizeof(ggml_fp16_t));
    for (int64_t i = 0; i < n_cols * n_rows; ++i) {
        x[i] = frand(-30.0f, 30.0f);
        m[i] = ggml_fp32_to_fp16((rand() % 4 == 0) ? -INFINITY : 0.0f);
    }

    struct ggml_init_params ip = { 16*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * src  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_rows);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_cols, n_rows);
    struct ggml_tensor * out  = ggml_soft_max_ext(ctx, src, mask, 0.5f, 0.0f);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(src,  x, 0, ggml_nbytes(src));
    ggml_backend_tensor_set(mask, m, 0, ggml_nbytes(mask));

    ggml_backend_graph_compute(backend, gf);

    float * y = (float *) malloc(n_cols * n_rows * sizeof(float));
    ggml_backend_tensor_get(out, y, 0, ggml_nbytes(out));

    float max_err = 0.0f;
    for (int64_t r = 0; r < n_rows; ++r) {
        float mx = -INFINITY;
        for (int64_t c = 0; c < n_cols; ++c) {
            const float v = 0.5f*x[r*n_cols + c] + ggml_fp16_to_fp32(m[r*n_cols + c]);
            mx = fmaxf(mx, v);
        }
        double sum = 0.0;
        for (int64_t c = 0; c < n_cols; ++c) {
            const float v = 0.5f*x[r*n_cols + c] + ggml_fp16_to_fp32(m[r*n_cols + c]);
            sum += expf(v - mx);
        }
        for (int64_t c = 0; c < n_cols; ++c) {
            const float v = 0.5f*x[r*n_cols + c] + ggml_fp16_to_fp32(m[r*n_cols + c]);
            const float ref = (float) (expf(v - mx) / sum);
            max_err = fmaxf(max_err, fabsf(ref - y[r*n_cols + c]) / fmaxf(ref, 1e-6f));
        }
    }

    const float tol = 1e-5f;
    const bool ok = max_err <= tol;
    printf("TEST soft_max %lldx%lld: %s (max rel err: %.2e, tol: %.0e)\n",
           (long long) n_cols, (long long) n_rows, ok ? "PASS" : "FAIL", max_err, tol);
    if (!ok) {
        ++n_failed;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(x); free(m); free(y);
}

int main(void) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        fprintf(stderr, "failed to initialize the CPU backend\n");
        return 1;
    }
    // more than one thread so the chunked decode path splits the KV range
    cpu_set_n_threads(backend, 4);

    test_dequant_exact(GGML_TYPE_Q8_0);
    test_dequant_exact(GGML_TYPE_Q4_0);

    test_soft_max(backend, 64,   16);
    test_soft_max(backend, 513,  8);
    test_soft_max(backend, 1027, 3);

    const ggml_type kv_types[] = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F32 };

    for (ggml_type type : kv_types) {
        for (int64_t hs : { 64, 128, 256 }) {
            // decode: n_q = 1, KV lengths around the chunked path threshold
            for (int64_t n_kv : { 511, 512, 513, 1030 }) {
                test_flash_attn(backend, hs, hs, 4, 2, n_kv, 1, type, type, MASK_CAUSAL, 0.0f, 1.0f);
            }
            // prefill: n_q >= 64 takes the tiled path, with ragged KV and Q tiles
            test_flash_attn(backend, hs, hs, 4, 2, 513, 64, type, type, MASK_CAUSAL, 0.0f, 1.0f);
            test_flash_attn(backend, hs, hs, 4, 2, 511, 75, type, type, MASK_CAUSAL, 0.0f, 1.0f);
            // small batch: per-row path
            test_flash_attn(backend, hs, hs, 4, 2, 512, 3, type, type, MASK_CAUSAL, 0.0f, 1.0f);
        }
    }

    // no mask, ALiBi, large logits (frequent rescaling of the running max), mixed K/V types, uneven head sizes
    test_flash_attn(backend, 128, 128, 4, 4, 512,  1, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_NONE,   0.0f, 1.0f);
    test_flash_attn(backend, 128, 128, 4, 4, 513, 64, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, MASK_NONE,   0.0f, 1.0f);
    test_flash_attn(backend, 128, 128, 8, 2, 512, 64, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 8.0f, 1.0f);
    test_flash_attn(backend, 128, 128, 8, 2, 513,  1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, MASK_CAUSAL, 8.0f, 1.0f);
    test_flash_attn(backend,  64,  64, 4, 2, 1030, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 20.0f);
    test_flash_attn(backend,  64,  64, 4, 2, 1030, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, MASK_CAUSAL, 0.0f, 10.0f);
    test_flash_attn(backend,  64,  64, 4, 2, 513, 64, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, MASK_CAUSAL, 0.0f, 20.0f);
    test_flash_attn(backend,  64,  64, 4, 2, 512,  1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend,  64,  64, 4, 2, 512, 64, GGML_TYPE_Q4_0, GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend,  72,  72, 4, 2, 513, 64, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    // per-row path with F16 V and DV not a multiple of 32/8: covers every tail of the f16->f32 mad
    test_flash_attn(backend,  64,  76, 4, 2, 512,  3, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend,  64,  40, 4, 2, 511,  1, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend, 192, 128, 4, 2, 513, 64, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend, 192, 128, 4, 2, 513,  1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, MASK_CAUSAL, 0.0f, 1.0f);

    // whole KV blocks masked: mid-sequence (blocks) and before any running max exists (tail), on the
    // per-row path, the chunked decode path (with fully masked chunks) and the tiled path
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        for (enum mask_kind kind : { MASK_BLOCKS, MASK_TAIL }) {
            test_flash_attn(backend, 64, 64, 4, 2,  200,  3, type, type, kind, 0.0f, 1.0f);
            test_flash_attn(backend, 64, 64, 4, 2, 1030,  1, type, type, kind, 0.0f, 1.0f);
            test_flash_attn(backend, 64, 64, 4, 2,  513, 64, type, type, kind, 0.0f, 1.0f);
        }
    }

    // NaN scores must not be dropped by the blocked running max of the per-row kernel
    // (F16/F32 K only: Q quantized to q8_0 for a quantized K does not keep NaN in a defined way)
    test_nan_propagation(backend, 64, 100, 3, GGML_TYPE_F16);
    test_nan_propagation(backend, 64, 100, 3, GGML_TYPE_F32);

    ggml_backend_free(backend);

    if (n_failed) {
        printf("%d test(s) failed\n", n_failed);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
