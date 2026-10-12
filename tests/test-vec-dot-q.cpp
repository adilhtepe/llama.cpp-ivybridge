// Checks the optimized vec_dot kernels against the generic implementation and
// against exact integer arithmetic for the block types used by the common
// K-quant and legacy quantization formats.

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

extern "C" {
void ggml_vec_dot_q4_0_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
}

static const char * RESULT_STR[] = {"ok", "FAILED"};

// relative to the sum of the absolute products of the dequantized values
constexpr double MAX_REL_ERROR = 1e-5;

static uint32_t rng_state = 12345;

static uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

// uniform integer in [lo, hi]
static int rng_int(int lo, int hi) {
    return lo + (int) (rng_next() % (uint32_t) (hi - lo + 1));
}

static float rng_float(void) {
    return (float) rng_int(-1000, 1000) / 250.0f;
}

struct vec_dot_case {
    ggml_type       type;
    ggml_vec_dot_t  generic;
};

// block layouts, kept local so the test does not depend on the ggml internals headers
constexpr int QK_K_SIZE  = 256;
constexpr int QK8_0_SIZE = 32;

static uint16_t fp16_one(void) {
    return ggml_fp32_to_fp16(1.0f);
}

// q8_0: fp16 d + 32 int8
static void fill_q8_0_unit(uint8_t * blk, int lo, int hi) {
    const uint16_t d = fp16_one();
    memcpy(blk, &d, 2);
    for (int i = 0; i < QK8_0_SIZE; i++) {
        blk[2 + i] = (uint8_t) (int8_t) rng_int(lo, hi);
    }
}

// q4_0: fp16 d + 16 nibble bytes
static void fill_q4_0_unit(uint8_t * blk) {
    const uint16_t d = fp16_one();
    memcpy(blk, &d, 2);
    for (int i = 0; i < QK8_0_SIZE/2; i++) {
        blk[2 + i] = (uint8_t) rng_int(0, 255);
    }
}

// q8_K: float d + 256 int8 + 16 int16 bsums
static void fill_q8_K_unit(uint8_t * blk, int lo, int hi) {
    const float d = 1.0f;
    memcpy(blk, &d, 4);
    int8_t * qs = (int8_t *) (blk + 4);
    int16_t * bsums = (int16_t *) (blk + 4 + QK_K_SIZE);
    for (int i = 0; i < QK_K_SIZE; i++) {
        qs[i] = (int8_t) rng_int(lo, hi);
    }
    for (int j = 0; j < QK_K_SIZE/16; j++) {
        int s = 0;
        for (int l = 0; l < 16; l++) {
            s += qs[16*j + l];
        }
        bsums[j] = (int16_t) s;
    }
}

// q4_K: fp16 d, fp16 dmin, 12 scale bytes, 128 nibble bytes
static void fill_q4_K_unit(uint8_t * blk) {
    const uint16_t one = fp16_one();
    memcpy(blk + 0, &one, 2);
    memcpy(blk + 2, &one, 2);
    for (int i = 0; i < 12 + QK_K_SIZE/2; i++) {
        blk[4 + i] = (uint8_t) rng_int(0, 255);
    }
}

// q6_K: 128 ql bytes, 64 qh bytes, 16 int8 scales, fp16 d
static void fill_q6_K_unit(uint8_t * blk) {
    for (int i = 0; i < QK_K_SIZE/2 + QK_K_SIZE/4 + QK_K_SIZE/16; i++) {
        blk[i] = (uint8_t) rng_int(0, 255);
    }
    const uint16_t one = fp16_one();
    memcpy(blk + QK_K_SIZE/2 + QK_K_SIZE/4 + QK_K_SIZE/16, &one, 2);
}

static void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// exact integer dot products of the unit-scale blocks
static int64_t exact_q4_0(const uint8_t * x, const uint8_t * y, int nb) {
    int64_t sum = 0;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t * qx = x + ib*(2 + QK8_0_SIZE/2) + 2;
        const int8_t  * qy = (const int8_t *) (y + ib*(2 + QK8_0_SIZE) + 2);
        for (int j = 0; j < QK8_0_SIZE/2; j++) {
            sum += ((qx[j] & 0x0F) - 8) * qy[j];
            sum += ((qx[j] >>   4) - 8) * qy[j + QK8_0_SIZE/2];
        }
    }
    return sum;
}

static int64_t exact_q8_0(const uint8_t * x, const uint8_t * y, int nb) {
    int64_t sum = 0;
    for (int ib = 0; ib < nb; ib++) {
        const int8_t * qx = (const int8_t *) (x + ib*(2 + QK8_0_SIZE) + 2);
        const int8_t * qy = (const int8_t *) (y + ib*(2 + QK8_0_SIZE) + 2);
        for (int j = 0; j < QK8_0_SIZE; j++) {
            sum += qx[j] * qy[j];
        }
    }
    return sum;
}

static int64_t exact_q4_K(const uint8_t * x, const uint8_t * y, int nb) {
    const size_t bx = 4 + 12 + QK_K_SIZE/2;
    const size_t by = 4 + QK_K_SIZE + 2*QK_K_SIZE/16;
    int64_t sum = 0;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t * sc = x + ib*bx + 4;
        const uint8_t * q4 = x + ib*bx + 16;
        const int8_t  * q8 = (const int8_t *) (y + ib*by + 4);
        for (int j = 0; j < QK_K_SIZE/32; j++) {
            uint8_t d, m;
            get_scale_min_k4(j, sc, &d, &m);
            const uint8_t * qb = q4 + (j/2)*32;
            for (int l = 0; l < 32; l++) {
                const int q = (j % 2 == 0) ? (qb[l] & 0xF) : (qb[l] >> 4);
                const int yv = q8[32*j + l];
                sum += (int64_t) (d*q - m) * yv;
            }
        }
    }
    return sum;
}

static int64_t exact_q6_K(const uint8_t * x, const uint8_t * y, int nb) {
    const size_t bx = QK_K_SIZE/2 + QK_K_SIZE/4 + QK_K_SIZE/16 + 2;
    const size_t by = 4 + QK_K_SIZE + 2*QK_K_SIZE/16;
    int64_t sum = 0;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t * ql = x + ib*bx;
        const uint8_t * qh = x + ib*bx + QK_K_SIZE/2;
        const int8_t  * sc = (const int8_t *) (x + ib*bx + QK_K_SIZE/2 + QK_K_SIZE/4);
        const int8_t  * q8 = (const int8_t *) (y + ib*by + 4);
        for (int n = 0; n < QK_K_SIZE; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l/16;
                const int q1 = (int8_t) ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int8_t) ((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int8_t) ((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += (int64_t) sc[is + 0] * q1 * q8[l +  0];
                sum += (int64_t) sc[is + 2] * q2 * q8[l + 32];
                sum += (int64_t) sc[is + 4] * q3 * q8[l + 64];
                sum += (int64_t) sc[is + 6] * q4 * q8[l + 96];
            }
            ql += 64; qh += 32; sc += 8; q8 += 128;
        }
    }
    return sum;
}

// Unit scales turn the dot product into an integer; the q8 range is chosen so that every
// partial sum stays below 2^24 and the float result must match the integer exactly.
static int test_exact(const vec_dot_case & c, int n, int y_lo, int y_hi, bool verbose) {
    const auto * tr     = ggml_get_type_traits_cpu(c.type);
    const auto * tr_ref = ggml_get_type_traits(c.type);
    const int blck = tr_ref->blck_size;
    assert(n % blck == 0);
    const int nb = n / blck;

    std::vector<uint8_t> x(ggml_row_size(c.type, n) + 64);
    std::vector<uint8_t> y(ggml_row_size(tr->vec_dot_type, n) + 64);

    int64_t expected = 0;
    switch (c.type) {
        case GGML_TYPE_Q4_0:
            for (int ib = 0; ib < nb; ib++) {
                fill_q4_0_unit(x.data() + ib*(2 + QK8_0_SIZE/2));
                fill_q8_0_unit(y.data() + ib*(2 + QK8_0_SIZE), y_lo, y_hi);
            }
            expected = exact_q4_0(x.data(), y.data(), nb);
            break;
        case GGML_TYPE_Q8_0:
            for (int ib = 0; ib < nb; ib++) {
                fill_q8_0_unit(x.data() + ib*(2 + QK8_0_SIZE), -127, 127);
                fill_q8_0_unit(y.data() + ib*(2 + QK8_0_SIZE), y_lo, y_hi);
            }
            expected = exact_q8_0(x.data(), y.data(), nb);
            break;
        case GGML_TYPE_Q4_K:
            for (int ib = 0; ib < nb; ib++) {
                fill_q4_K_unit(x.data() + ib*(4 + 12 + QK_K_SIZE/2));
                fill_q8_K_unit(y.data() + ib*(4 + QK_K_SIZE + 2*QK_K_SIZE/16), y_lo, y_hi);
            }
            expected = exact_q4_K(x.data(), y.data(), nb);
            break;
        case GGML_TYPE_Q6_K:
            for (int ib = 0; ib < nb; ib++) {
                fill_q6_K_unit(x.data() + ib*(QK_K_SIZE/2 + QK_K_SIZE/4 + QK_K_SIZE/16 + 2));
                fill_q8_K_unit(y.data() + ib*(4 + QK_K_SIZE + 2*QK_K_SIZE/16), y_lo, y_hi);
            }
            expected = exact_q6_K(x.data(), y.data(), nb);
            break;
        default:
            assert(false);
    }

    float got = INFINITY;
    float got_ref = INFINITY;
    tr->vec_dot(n, &got, 0, x.data(), 0, y.data(), 0, 1);
    c.generic(n, &got_ref, 0, x.data(), 0, y.data(), 0, 1);

    const bool failed = got != (float) expected || got_ref != (float) expected;
    if (failed || verbose) {
        printf("  %-5s exact   n=%5d y in [%4d,%3d]: %s (expected=%lld got=%.1f generic=%.1f)\n",
               ggml_type_name(c.type), n, y_lo, y_hi, RESULT_STR[failed],
               (long long) expected, got, got_ref);
    }
    return failed;
}

// Quantized random floats: the optimized kernel, the generic kernel and a double
// precision sum over the dequantized values must agree to float rounding.
static int test_random(const vec_dot_case & c, int n, bool verbose) {
    const auto * tr     = ggml_get_type_traits_cpu(c.type);
    const auto * tr_ref = ggml_get_type_traits(c.type);
    const auto * tr_dot = ggml_get_type_traits_cpu(tr->vec_dot_type);
    const auto * tr_dot_ref = ggml_get_type_traits(tr->vec_dot_type);

    std::vector<float> xf(n);
    std::vector<float> yf(n);
    for (int i = 0; i < n; i++) {
        xf[i] = rng_float();
        yf[i] = rng_float();
    }

    std::vector<uint8_t> x(ggml_row_size(c.type, n) + 64);
    std::vector<uint8_t> y(ggml_row_size(tr->vec_dot_type, n) + 64);
    tr->from_float(xf.data(), x.data(), n);
    tr_dot->from_float(yf.data(), y.data(), n);

    std::vector<float> xd(n);
    std::vector<float> yd(n);
    tr_ref->to_float(x.data(), xd.data(), n);
    if (tr->vec_dot_type == GGML_TYPE_Q8_K) {
        // q8_K is an intermediate type without a dequantizer
        for (int ib = 0; ib < n/QK_K_SIZE; ib++) {
            const uint8_t * blk = y.data() + ib*(4 + QK_K_SIZE + 2*QK_K_SIZE/16);
            float d;
            memcpy(&d, blk, 4);
            for (int l = 0; l < QK_K_SIZE; l++) {
                yd[ib*QK_K_SIZE + l] = d * (int8_t) blk[4 + l];
            }
        }
    } else {
        tr_dot_ref->to_float(y.data(), yd.data(), n);
    }

    double ref = 0;
    double mag = 0;
    for (int i = 0; i < n; i++) {
        ref += (double) xd[i] * yd[i];
        mag += fabs((double) xd[i] * yd[i]);
    }

    float got = INFINITY;
    float got_ref = INFINITY;
    tr->vec_dot(n, &got, 0, x.data(), 0, y.data(), 0, 1);
    c.generic(n, &got_ref, 0, x.data(), 0, y.data(), 0, 1);

    const double err     = fabs(got - ref) / mag;
    const double err_ref = fabs(got_ref - ref) / mag;
    const bool failed = !(err < MAX_REL_ERROR) || !(err_ref < MAX_REL_ERROR);
    if (failed || verbose) {
        printf("  %-5s random  n=%5d: %s (ref=%.4f got=%.4f generic=%.4f rel_err=%.2e rel_err_generic=%.2e)\n",
               ggml_type_name(c.type), n, RESULT_STR[failed], ref, got, got_ref, err, err_ref);
    }
    return failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }

    ggml_cpu_init();

    const vec_dot_case cases[] = {
        { GGML_TYPE_Q4_0, ggml_vec_dot_q4_0_q8_0_generic },
        { GGML_TYPE_Q8_0, ggml_vec_dot_q8_0_q8_0_generic },
        { GGML_TYPE_Q4_K, ggml_vec_dot_q4_K_q8_K_generic },
        { GGML_TYPE_Q6_K, ggml_vec_dot_q6_K_q8_K_generic },
    };

    int num_failed = 0;

    for (const auto & c : cases) {
        ggml_quantize_init(c.type);
        printf("Testing %s\n", ggml_type_name(c.type));

        const int blck = ggml_get_type_traits(c.type)->blck_size;
        // block counts of 1..3 modulo 4 cover the two block loop and the scalar tail of the
        // legacy kernels, both on their own and after the four block loop
        const int sizes[] = { blck, 2*blck, 3*blck, 4*blck, 5*blck, 6*blck, 7*blck, 8*blck, 2048, 4096 };

        for (int rep = 0; rep < 8; rep++) {
            for (int n : sizes) {
                // quantize_row_q8_* never produces -128 and the kernels rely on that;
                // the narrower K-quant ranges keep all partial sums exactly representable in float
                int y_lo = -127, y_hi = 127;
                if (c.type == GGML_TYPE_Q4_K) { y_lo =  -32; y_hi =  31; }
                if (c.type == GGML_TYPE_Q6_K) { y_lo = n > blck ? -2 : -8; y_hi = n > blck ? 1 : 7; }
                num_failed += test_exact(c, n, y_lo, y_hi, verbose && rep == 0);
                num_failed += test_random(c, n, verbose && rep == 0);
            }
        }
    }

    if (num_failed) {
        printf("%d tests failed\n", num_failed);
    } else {
        printf("all tests passed\n");
    }

    return num_failed > 0;
}
