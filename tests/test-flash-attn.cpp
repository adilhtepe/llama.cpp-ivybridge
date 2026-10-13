// Checks the CPU flash_attn_ext kernels (chunked decode with and without grouped q heads, tiled
// prefill and the per-row fallback) against a scalar double precision reference computed straight
// from the definition: K/V dequantized to F32, scores, mask, ALiBi, logit softcap, sinks, softmax.
// Covers F16 and quantized K/V, several head sizes and KV lengths around the chunked decode
// threshold, every mask shape the kernels distinguish, batched (ne3 > 1) inputs and different
// thread counts. Also checks that q8_0/q4_0 row dequantization matches the scalar formula bit for
// bit and that soft_max matches a scalar expf reference.

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

// thread count of the backend outside the cases that set their own
static const int DEFAULT_N_THREADS = 4;

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

// fill t from a flat f32 source of rows x cols; quantized types go through ggml_quantize_chunk.
// Returns the values the kernels see (the source after the round trip through t's type), malloc'd
static float * fill_tensor(struct ggml_tensor * t, const float * src, int64_t rows, int64_t cols) {
    GGML_ASSERT(t->ne[0] == cols);
    GGML_ASSERT(rows * cols == ggml_nelements(t));
    float * seen = (float *) malloc(rows * cols * sizeof(float));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
        memcpy(seen, src, rows * cols * sizeof(float));
        return seen;
    }
    void * q = malloc(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) q, rows * cols);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) q, seen, rows * cols);
    } else {
        ggml_quantize_chunk(t->type, src, q, 0, rows, cols, NULL);
        ggml_get_type_traits(t->type)->to_float(q, seen, rows * cols);
    }
    ggml_backend_tensor_set(t, q, 0, ggml_nbytes(t));
    free(q);
    return seen;
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
    MASK_SPARSE, // causal plus every fifth position (shifted per mask head): partially masked blocks on every path
};

static const char * mask_kind_name(enum mask_kind kind) {
    switch (kind) {
        case MASK_CAUSAL: return "causal";
        case MASK_BLOCKS: return "blocks";
        case MASK_TAIL:   return "tail";
        case MASK_SPARSE: return "sparse";
        default:          return "none";
    }
}

// mh: index of the mask head (0 when one mask is shared by all heads)
static bool mask_is_masked(enum mask_kind kind, int64_t n_kv, int64_t n_q, int64_t iq, int64_t ik, int64_t mh) {
    switch (kind) {
        case MASK_CAUSAL: return ik > n_kv - n_q + iq;
        case MASK_BLOCKS: return (ik / 32) % 3 != 2;
        case MASK_TAIL:   return ik < n_kv - 40;
        case MASK_SPARSE: return ik > n_kv - n_q + iq || (ik*7 + mh*3) % 5 == 0;
        default:          return false;
    }
}

struct fa_case {
    int64_t DK, DV;
    int64_t n_head, n_head_kv; // q heads, K heads
    int64_t n_head_v;          // V heads
    int64_t n_kv, n_q;
    int64_t n_batch;           // ne3 of q, k and v
    ggml_type type_K, type_V;
    enum mask_kind mask_kind;
    int64_t mask_heads;        // mask->ne[2]: 1 = one mask for all heads
    float max_bias;
    float logit_softcap;
    bool sinks;
    float q_range;
    float scale;               // 0: 1/sqrt(DK)
    int n_threads;             // 0: DEFAULT_N_THREADS
};

static fa_case fa(int64_t DK, int64_t DV, int64_t n_head, int64_t n_head_kv, int64_t n_kv, int64_t n_q,
                  ggml_type type_K, ggml_type type_V, enum mask_kind mask_kind, float max_bias, float q_range) {
    fa_case c;
    c.DK = DK; c.DV = DV;
    c.n_head = n_head; c.n_head_kv = n_head_kv; c.n_head_v = n_head_kv;
    c.n_kv = n_kv; c.n_q = n_q; c.n_batch = 1;
    c.type_K = type_K; c.type_V = type_V;
    c.mask_kind = mask_kind; c.mask_heads = 1;
    c.max_bias = max_bias; c.logit_softcap = 0.0f; c.sinks = false;
    c.q_range = q_range; c.scale = 0.0f; c.n_threads = 0;
    return c;
}

// ALiBi slope of head h, as computed by the kernels
static double alibi_slope(const fa_case & c, int64_t h) {
    if (c.max_bias <= 0.0f) {
        return 1.0;
    }
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2((double) c.n_head));
    const float m0 = powf(2.0f, -(c.max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(c.max_bias / 2.0f) / n_head_log2);
    return (uint32_t) h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1);
}

// the mask row of query iq for mask head mh: -inf where masked; with ALiBi the distance to the
// query position elsewhere (as the KV cache mask holds), 0 otherwise
static void mask_row(const fa_case & c, int64_t iq, int64_t mh, ggml_fp16_t * row) {
    for (int64_t ik = 0; ik < c.n_kv; ++ik) {
        float v = 0.0f;
        if (mask_is_masked(c.mask_kind, c.n_kv, c.n_q, iq, ik, mh)) {
            v = -INFINITY;
        } else if (c.max_bias > 0.0f) {
            v = -(float) llabs((c.n_kv - c.n_q + iq) - ik);
        }
        row[ik] = ggml_fp32_to_fp16(v);
    }
}

// Scalar reference in double precision, straight from the definition.
// q: [n_batch][n_head][n_q][DK], k: [n_batch][n_head_kv][n_kv][DK], v: [n_batch][n_head_v][n_kv][DV],
// mask: [mask_heads][n_q][n_kv] (NULL without mask), sinks: [n_head] (NULL without sinks),
// out: [n_batch][n_q][n_head][DV], the layout of the flash_attn_ext result
static void reference_attn(const fa_case & c, const float * q, const float * k, const float * v,
                           const ggml_fp16_t * mask, const float * sinks, float scale, float * out) {
    double * s = (double *) malloc(c.n_kv * sizeof(double));
    const int64_t rk = c.n_head / c.n_head_kv;
    const int64_t rv = c.n_head / c.n_head_v;

    for (int64_t ib = 0; ib < c.n_batch; ++ib) {
        for (int64_t h = 0; h < c.n_head; ++h) {
            const double slope = alibi_slope(c, h);
            const float * kh = k + ((ib*c.n_head_kv + h/rk)*c.n_kv)*c.DK;
            const float * vh = v + ((ib*c.n_head_v  + h/rv)*c.n_kv)*c.DV;
            const ggml_fp16_t * mh = mask ? mask + ((h % c.mask_heads)*c.n_q)*c.n_kv : NULL;

            for (int64_t iq = 0; iq < c.n_q; ++iq) {
                const float * qr = q + ((ib*c.n_head + h)*c.n_q + iq)*c.DK;

                double M = -INFINITY;
                for (int64_t ik = 0; ik < c.n_kv; ++ik) {
                    double dot = 0.0;
                    for (int64_t d = 0; d < c.DK; ++d) {
                        dot += (double) qr[d] * kh[ik*c.DK + d];
                    }
                    double sc = dot * scale;
                    if (c.logit_softcap != 0.0f) {
                        sc = c.logit_softcap * tanh(sc / c.logit_softcap);
                    }
                    if (mh) {
                        sc += slope * ggml_fp16_to_fp32(mh[iq*c.n_kv + ik]);
                    }
                    s[ik] = sc;
                    M = fmax(M, sc);
                }
                if (sinks) {
                    M = fmax(M, sinks[h]);
                }

                double sum = sinks ? exp(sinks[h] - M) : 0.0;
                float * o = out + ((ib*c.n_q + iq)*c.n_head + h)*c.DV;
                double * acc = (double *) calloc(c.DV, sizeof(double));
                for (int64_t ik = 0; ik < c.n_kv; ++ik) {
                    const double p = exp(s[ik] - M);
                    if (p == 0.0) {
                        continue;
                    }
                    sum += p;
                    for (int64_t d = 0; d < c.DV; ++d) {
                        acc[d] += p * vh[ik*c.DV + d];
                    }
                }
                for (int64_t d = 0; d < c.DV; ++d) {
                    o[d] = sum > 0.0 ? (float) (acc[d] / sum) : 0.0f;
                }
                free(acc);
            }
        }
    }
    free(s);
}

static const char * fa_path(const fa_case & c) {
    if (c.n_q == 1 && c.n_batch == 1 && c.n_kv >= 512) {
        return "chunked";
    }
    return c.n_q >= 64 ? "tiled" : "rows";
}

static void run_flash_attn(ggml_backend_t backend, const fa_case & c) {
    srand(0xBEEF);

    const float scale = c.scale != 0.0f ? c.scale : 1.0f / sqrtf((float) c.DK);

    const int64_t nq = c.DK * c.n_q  * c.n_head    * c.n_batch;
    const int64_t nk = c.DK * c.n_kv * c.n_head_kv * c.n_batch;
    const int64_t nv = c.DV * c.n_kv * c.n_head_v  * c.n_batch;
    float * q_data = (float *) malloc(nq * sizeof(float));
    float * k_data = (float *) malloc(nk * sizeof(float));
    float * v_data = (float *) malloc(nv * sizeof(float));
    for (int64_t i = 0; i < nq; ++i) q_data[i] = frand(-c.q_range, c.q_range);
    for (int64_t i = 0; i < nk; ++i) k_data[i] = frand(-1.0f, 1.0f);
    for (int64_t i = 0; i < nv; ++i) v_data[i] = frand(-1.0f, 1.0f);

    struct ggml_init_params ip = { 256*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.DK, c.n_q,  c.n_head,    c.n_batch);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, c.type_K,      c.DK, c.n_kv, c.n_head_kv, c.n_batch);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, c.type_V,      c.DV, c.n_kv, c.n_head_v,  c.n_batch);

    struct ggml_tensor * mask = NULL;
    if (c.mask_kind != MASK_NONE || c.max_bias > 0.0f) {
        mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.n_kv, c.n_q, c.mask_heads);
    }
    struct ggml_tensor * sinks = c.sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.n_head) : NULL;

    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, c.max_bias, c.logit_softcap);
    ggml_flash_attn_ext_add_sinks(out, sinks);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    float * q_seen = fill_tensor(q, q_data, c.n_q  * c.n_head    * c.n_batch, c.DK);
    float * k_seen = fill_tensor(k, k_data, c.n_kv * c.n_head_kv * c.n_batch, c.DK);
    float * v_seen = fill_tensor(v, v_data, c.n_kv * c.n_head_v  * c.n_batch, c.DV);

    ggml_fp16_t * m = NULL;
    if (mask) {
        m = (ggml_fp16_t *) malloc(c.mask_heads * c.n_q * c.n_kv * sizeof(ggml_fp16_t));
        for (int64_t mh = 0; mh < c.mask_heads; ++mh) {
            for (int64_t iq = 0; iq < c.n_q; ++iq) {
                mask_row(c, iq, mh, m + (mh*c.n_q + iq)*c.n_kv);
            }
        }
        ggml_backend_tensor_set(mask, m, 0, ggml_nbytes(mask));
    }

    // sink logits well above the scores so the sink takes a visible share of the softmax
    float * sk = NULL;
    if (sinks) {
        sk = (float *) malloc(c.n_head * sizeof(float));
        for (int64_t h = 0; h < c.n_head; ++h) {
            sk[h] = 6.0f + frand(0.0f, 3.0f);
        }
        ggml_backend_tensor_set(sinks, sk, 0, ggml_nbytes(sinks));
    }

    const int64_t n_out = ggml_nelements(out);
    float * out_fast = (float *) malloc(n_out * sizeof(float));
    float * out_ref  = (float *) malloc(n_out * sizeof(float));
    float * ref_data = (float *) malloc(n_out * sizeof(float));

    reference_attn(c, q_seen, k_seen, v_seen, m, sk, scale, ref_data);

    const int n_threads = c.n_threads > 0 ? c.n_threads : DEFAULT_N_THREADS;
    cpu_set_n_threads(backend, n_threads);

    // fast paths (chunked decode / tiled prefill / per-row), then the vec-only reference path
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, out_fast, 0, ggml_nbytes(out));
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, out_ref, 0, ggml_nbytes(out));
    cpu_set_use_ref(backend, false);

    cpu_set_n_threads(backend, DEFAULT_N_THREADS);

    const double tol = 5e-4;

    float max_err_fast, max_err_ref;
    const double err_fast = nmse(ref_data, out_fast, n_out, &max_err_fast);
    const double err_ref  = nmse(ref_data, out_ref,  n_out, &max_err_ref);
    const bool ok = err_fast <= tol && err_ref <= tol;

    printf("TEST fattn DK=%lld DV=%lld heads=%lld/%lld/%lld batch=%lld n_kv=%lld n_q=%lld K=%s V=%s mask=%s/%lld "
           "max_bias=%.1f softcap=%.0f sinks=%d q_range=%.0f nth=%d (%s): %s "
           "(fast nmse: %.2e max_err: %.2e, vec nmse: %.2e max_err: %.2e, tol: %.0e)\n",
           (long long) c.DK, (long long) c.DV, (long long) c.n_head, (long long) c.n_head_kv, (long long) c.n_head_v,
           (long long) c.n_batch, (long long) c.n_kv, (long long) c.n_q, ggml_type_name(c.type_K), ggml_type_name(c.type_V),
           mask_kind_name(c.mask_kind), (long long) c.mask_heads, c.max_bias, c.logit_softcap, (int) c.sinks, c.q_range,
           n_threads, fa_path(c), ok ? "PASS" : "FAIL", err_fast, max_err_fast, err_ref, max_err_ref, tol);

    if (!ok) {
        ++n_failed;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(q_data); free(k_data); free(v_data);
    free(q_seen); free(k_seen); free(v_seen);
    free(m); free(sk);
    free(out_fast); free(out_ref); free(ref_data);
}

// q: [DK, n_q, n_head], k: [DK, n_kv, n_head_kv], v: [DV, n_kv, n_head_kv]
static void test_flash_attn(ggml_backend_t backend, int64_t DK, int64_t DV, int64_t n_head, int64_t n_head_kv,
                            int64_t n_kv, int64_t n_q, ggml_type type_K, ggml_type type_V,
                            enum mask_kind mask_kind, float max_bias, float q_range) {
    run_flash_attn(backend, fa(DK, DV, n_head, n_head_kv, n_kv, n_q, type_K, type_V, mask_kind, max_bias, q_range));
}

// a NaN query row must give a NaN output row and leave the other rows finite, including the other
// q heads of the same kv head. With a NaN Q every score of every KV block is NaN, which the
// per-row kernel's running max would otherwise drop
static void test_nan_propagation(ggml_backend_t backend, int64_t DK, int64_t n_kv, int64_t n_q, ggml_type type_KV,
                                 int64_t n_head, int64_t n_head_kv, int64_t nan_head) {
    srand(0xBEEF);

    const int64_t nan_iq = n_q > 1 ? 1 : 0; // query row that is set to NaN, in head nan_head

    float * q_data = (float *) malloc(DK * n_q  * n_head    * sizeof(float));
    float * k_data = (float *) malloc(DK * n_kv * n_head_kv * sizeof(float));
    float * v_data = (float *) malloc(DK * n_kv * n_head_kv * sizeof(float));
    for (int64_t i = 0; i < DK * n_q  * n_head;    ++i) q_data[i] = frand(-1.0f, 1.0f);
    for (int64_t i = 0; i < DK * n_kv * n_head_kv; ++i) k_data[i] = frand(-1.0f, 1.0f);
    for (int64_t i = 0; i < DK * n_kv * n_head_kv; ++i) v_data[i] = frand(-1.0f, 1.0f);
    for (int64_t d = 0; d < DK; ++d) q_data[(nan_head * n_q + nan_iq) * DK + d] = NAN;

    struct ggml_init_params ip = { 16*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DK, n_q,  n_head);
    struct ggml_tensor * k = ggml_new_tensor_3d(ctx, type_KV,       DK, n_kv, n_head_kv);
    struct ggml_tensor * v = ggml_new_tensor_3d(ctx, type_KV,       DK, n_kv, n_head_kv);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_q);

    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / sqrtf((float) DK), 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    free(fill_tensor(q, q_data, n_q  * n_head,    DK));
    free(fill_tensor(k, k_data, n_kv * n_head_kv, DK));
    free(fill_tensor(v, v_data, n_kv * n_head_kv, DK));

    ggml_fp16_t * m = (ggml_fp16_t *) malloc(n_kv * n_q * sizeof(ggml_fp16_t));
    for (int64_t iq = 0; iq < n_q; ++iq) {
        for (int64_t ik = 0; ik < n_kv; ++ik) {
            m[iq * n_kv + ik] = ggml_fp32_to_fp16(mask_is_masked(MASK_CAUSAL, n_kv, n_q, iq, ik, 0) ? -INFINITY : 0.0f);
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
                if (iq == nan_iq && h == nan_head) {
                    n_nan_expected += is_nan;
                } else {
                    n_nan_unexpected += is_nan;
                }
            }
        }
    }
    const bool ok = n_nan_expected == DK && n_nan_unexpected == 0;

    printf("TEST nan_propagation DK=%lld heads=%lld/%lld n_kv=%lld n_q=%lld KV=%s nan_head=%lld: %s "
           "(NaN in the NaN row: %lld/%lld, elsewhere: %lld)\n",
           (long long) DK, (long long) n_head, (long long) n_head_kv, (long long) n_kv, (long long) n_q,
           ggml_type_name(type_KV), (long long) nan_head, ok ? "PASS" : "FAIL",
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
    cpu_set_n_threads(backend, DEFAULT_N_THREADS);

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

    // chunked decode with the q heads of one kv head grouped: group widths 1, 2, 4, 8 (Qwen3-30B-A3B),
    // 16 (two groups per kv head) and 12 (uneven split), odd head sizes, every KV type and mask kind
    for (ggml_type type : kv_types) {
        for (int64_t gqa : { 1, 2, 4, 8, 12, 16 }) {
            test_flash_attn(backend, 128, 128, 2*gqa, 2, 1030, 1, type, type, MASK_CAUSAL, 0.0f, 1.0f);
        }
        test_flash_attn(backend, 128, 128, 16, 2, 4101, 1, type, type, MASK_CAUSAL, 0.0f, 1.0f);
        test_flash_attn(backend, 128, 128, 16, 2,  520, 1, type, type, MASK_NONE,   0.0f, 1.0f);
        test_flash_attn(backend, 128, 128, 16, 2, 1030, 1, type, type, MASK_BLOCKS, 0.0f, 1.0f);
        test_flash_attn(backend, 128, 128, 16, 2, 1030, 1, type, type, MASK_TAIL,   0.0f, 1.0f);
        test_flash_attn(backend, 128, 128, 16, 2, 1030, 1, type, type, MASK_CAUSAL, 0.0f, 20.0f);
        test_flash_attn(backend, 128, 128, 16, 2, 1030, 1, type, type, MASK_CAUSAL, 8.0f, 1.0f);
    }
    test_flash_attn(backend,  72,  72, 16, 2, 1030, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend,  64,  40, 16, 2, 1030, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend,  76,  64, 16, 2, 1030, 1, GGML_TYPE_F32,  GGML_TYPE_F32,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend, 192, 128, 16, 2, 1030, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend, 128, 128, 16, 2, 1030, 1, GGML_TYPE_Q8_0, GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    // DK and DV both off the multiple of 8 on the grouped decode path
    test_flash_attn(backend,  40,  40, 16, 2, 1030, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend,  76,  76, 16, 2, 1030, 1, GGML_TYPE_F16,  GGML_TYPE_F16,  MASK_CAUSAL, 0.0f, 1.0f);
    test_flash_attn(backend,  76,  76, 16, 2, 1030, 1, GGML_TYPE_F32,  GGML_TYPE_F32,  MASK_CAUSAL, 0.0f, 1.0f);

    // a causal mask masks nothing for a single query: partially masked blocks on the decode paths
    // come from the sparse mask, with the group widths 1, 4, 8 and the 8+4 split of 12
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        for (int64_t gqa : { 1, 4, 8, 12 }) {
            test_flash_attn(backend, 128, 128, 2*gqa, 2, 1030, 1, type, type, MASK_SPARSE, 0.0f, 1.0f);
        }
        test_flash_attn(backend, 64, 64, 4, 2, 513, 64, type, type, MASK_SPARSE, 0.0f, 1.0f);
        test_flash_attn(backend, 64, 64, 4, 2, 512,  3, type, type, MASK_SPARSE, 0.0f, 1.0f);
    }

    // features the grouped decode kernel hands back to the per-row kernel: a mask per head (one for
    // every head and one shared by groups of heads), logit softcap, K and V with different head
    // counts, and sinks on every path
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        for (int64_t n_q : { 1, 64, 3 }) {
            const int64_t n_kv = n_q == 64 ? 513 : 1030;

            fa_case c = fa(128, 128, 16, 2, n_kv, n_q, type, type, MASK_SPARSE, 0.0f, 1.0f);
            c.mask_heads = 16;
            run_flash_attn(backend, c);
            c.mask_heads = 4;
            run_flash_attn(backend, c);

            c = fa(128, 128, 16, 2, n_kv, n_q, type, type, MASK_CAUSAL, 0.0f, 3.0f);
            c.logit_softcap = 10.0f;
            run_flash_attn(backend, c);

            c = fa(128, 128, 16, 2, n_kv, n_q, type, type, MASK_CAUSAL, 0.0f, 1.0f);
            c.n_head_v = 4;
            run_flash_attn(backend, c);

            c = fa(128, 128, 16, 2, n_kv, n_q, type, type, MASK_CAUSAL, 0.0f, 1.0f);
            c.sinks = true;
            run_flash_attn(backend, c);
        }
    }
    {
        // sinks on the chunked decode path without grouping (softcap), with ALiBi and without a mask
        fa_case c = fa(64, 64, 8, 2, 1030, 1, GGML_TYPE_F16, GGML_TYPE_F16, MASK_CAUSAL, 0.0f, 3.0f);
        c.sinks = true;
        c.logit_softcap = 10.0f;
        run_flash_attn(backend, c);
        c = fa(64, 64, 8, 2, 1030, 1, GGML_TYPE_F16, GGML_TYPE_F16, MASK_CAUSAL, 8.0f, 1.0f);
        c.sinks = true;
        run_flash_attn(backend, c);
        c = fa(64, 64, 8, 2, 1030, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, MASK_NONE, 0.0f, 1.0f);
        c.sinks = true;
        run_flash_attn(backend, c);
    }

    // thread counts: one thread (one KV chunk, one unit per thread), more threads than KV chunks
    // with work (n_kv=529 over 24 threads leaves the last chunk empty) and more threads than
    // query tiles (4 heads x 2 tiles of 64 rows); n_q >= 128 covers several tiles per head
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        fa_case c = fa(128, 128, 16, 2, 1030, 1, type, type, MASK_SPARSE, 0.0f, 1.0f);
        c.n_threads = 1;
        run_flash_attn(backend, c);
        c = fa(128, 128, 16, 2, 529, 1, type, type, MASK_SPARSE, 0.0f, 1.0f);
        c.n_threads = 24;
        run_flash_attn(backend, c);
        c = fa(64, 64, 4, 2, 513, 128, type, type, MASK_CAUSAL, 0.0f, 1.0f);
        c.n_threads = 1;
        run_flash_attn(backend, c);
        c.n_threads = 24;
        run_flash_attn(backend, c);
        c = fa(64, 64, 4, 2, 300, 150, type, type, MASK_CAUSAL, 0.0f, 1.0f);
        run_flash_attn(backend, c);
    }

    // batched inputs (ne3 > 1): tiled, per-row and single-query (per-row, as the chunked path takes one batch)
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        for (int64_t n_q : { 128, 3, 1 }) {
            fa_case c = fa(64, 64, 4, 2, 513, n_q, type, type, MASK_CAUSAL, 0.0f, 1.0f);
            c.n_batch = 2;
            run_flash_attn(backend, c);
        }
        fa_case c = fa(64, 64, 4, 2, 300, 128, type, type, MASK_SPARSE, 0.0f, 1.0f);
        c.n_batch = 3;
        c.n_threads = 24;
        run_flash_attn(backend, c);
    }

    // NaN scores must not be dropped by the blocked running max of the per-row kernel
    // (F16/F32 K only: Q quantized to q8_0 for a quantized K does not keep NaN in a defined way)
    test_nan_propagation(backend, 64, 100, 3, GGML_TYPE_F16, 2, 2, 0);
    test_nan_propagation(backend, 64, 100, 3, GGML_TYPE_F32, 2, 2, 0);
    // the same on the chunked decode path, without and with grouped q heads: a NaN head must not
    // reach the other heads of its group (groups of 4, of 8, and the 8+4 split of 12)
    test_nan_propagation(backend, 64, 1030, 1, GGML_TYPE_F16,  2, 2, 0);
    test_nan_propagation(backend, 64, 1030, 1, GGML_TYPE_F32,  2, 2, 0);
    test_nan_propagation(backend, 64, 1030, 1, GGML_TYPE_F16,  8, 2, 1);
    test_nan_propagation(backend, 64, 1030, 1, GGML_TYPE_F32, 16, 2, 5);
    test_nan_propagation(backend, 64, 1030, 1, GGML_TYPE_F16, 24, 2, 9);

    ggml_backend_free(backend);

    if (n_failed) {
        printf("%d test(s) failed\n", n_failed);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
