// Compares CPU backend kernels that have SIMD paths against scalar references:
// RMS_NORM (sum of squares accumulated in double) and ACC (ggml_vec_add_f32).

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void graph_compute(ggml_context * ctx, ggml_tensor * t, int n_threads) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
}

static int test_rms_norm(ggml_context * ctx, std::mt19937 & rng, int64_t ne0, int64_t ne1, float amp, int n_threads) {
    const float eps = 1e-6f;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    float * xd = (float *) x->data;

    std::normal_distribution<float> nrm(0.0f, amp);
    for (int64_t i = 0; i < ne0*ne1; i++) {
        xd[i] = nrm(rng);
    }

    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    graph_compute(ctx, y, n_threads);

    const float * yd = (const float *) y->data;

    // reference: sequential accumulation of the f32 squares in double
    double max_rel = 0.0;
    for (int64_t i1 = 0; i1 < ne1; i1++) {
        const float * xr = xd + i1*ne0;
        const float * yr = yd + i1*ne0;

        double sum = 0.0;
        for (int64_t i0 = 0; i0 < ne0; i0++) {
            sum += (double)(xr[i0]*xr[i0]);
        }
        const float mean  = sum/ne0;
        const float scale = 1.0f/sqrtf(mean + eps);

        for (int64_t i0 = 0; i0 < ne0; i0++) {
            const float ref = xr[i0]*scale;
            const double rel = fabs((double) yr[i0] - (double) ref) / std::max(fabs((double) ref), 1e-30);
            max_rel = std::max(max_rel, rel);
        }
    }

    // the vector path sums in a different order, which can move the f32 mean by an ulp
    const bool ok = max_rel <= 1e-6;
    printf("  RMS_NORM ne0=%5lld ne1=%lld amp=%g: max rel err %.3e %s\n", (long long) ne0, (long long) ne1, amp, max_rel, ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}

static int test_acc(ggml_context * ctx, std::mt19937 & rng, int64_t ne0a, int64_t ne1a, int64_t ne0b, int64_t ne1b, int64_t off0, int64_t off1, int n_threads) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0a, ne1a);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0b, ne1b);

    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    float * ad = (float *) a->data;
    float * bd = (float *) b->data;
    for (int64_t i = 0; i < ne0a*ne1a; i++) {
        ad[i] = uni(rng);
    }
    for (int64_t i = 0; i < ne0b*ne1b; i++) {
        bd[i] = uni(rng);
    }

    const size_t offset = (off1*ne0a + off0)*sizeof(float);
    ggml_tensor * c = ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset);
    graph_compute(ctx, c, n_threads);

    std::vector<float> ref(ad, ad + ne0a*ne1a);
    for (int64_t i1 = 0; i1 < ne1b; i1++) {
        for (int64_t i0 = 0; i0 < ne0b; i0++) {
            ref[(off1 + i1)*ne0a + off0 + i0] += bd[i1*ne0b + i0];
        }
    }

    const bool ok = memcmp(ref.data(), c->data, ne0a*ne1a*sizeof(float)) == 0;
    printf("  ACC a=[%lld,%lld] b=[%lld,%lld] off=[%lld,%lld]: %s\n", (long long) ne0a, (long long) ne1a, (long long) ne0b, (long long) ne1b, (long long) off0, (long long) off1, ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}

int main(int /*argc*/, const char ** /*argv*/) {
    ggml_cpu_init();

    struct ggml_init_params params = {
        /* .mem_size   = */ 64*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    std::mt19937 rng(42);

    int n_failed = 0;

    for (int n_threads : { 1, 3 }) {
        for (float amp : { 1.0f, 1e-3f, 1e+3f }) {
            for (int64_t ne0 : { 1, 3, 8, 15, 16, 17, 31, 32, 33, 100, 1000, 2048, 4099 }) {
                ggml_context * ctx = ggml_init(params);
                n_failed += test_rms_norm(ctx, rng, ne0, 5, amp, n_threads);
                ggml_free(ctx);
            }
        }

        for (int64_t ne0b : { 1, 7, 8, 9, 16, 29, 64, 1021 }) {
            ggml_context * ctx = ggml_init(params);
            n_failed += test_acc(ctx, rng, ne0b + 11, 6, ne0b, 4, 3, 1, n_threads);
            ggml_free(ctx);
        }
    }

    printf("%s\n", n_failed == 0 ? "all tests passed" : "some tests failed");

    return n_failed > 0 ? 1 : 0;
}
