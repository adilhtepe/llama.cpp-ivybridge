// Checks that the CPU backend quantize_row_q8_K is bit-exact with quantize_row_q8_K_ref.
// q8_K has no to_float, so test-quantize-fns does not cover it.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static ggml_from_float_t q8_K_from_float = nullptr;

static int check(const std::string & name, const std::vector<float> & x) {
    const int64_t k  = x.size();
    const int64_t nb = k / QK_K;

    std::vector<block_q8_K> y_ref(nb);
    std::vector<block_q8_K> y_cpu(nb);

    // the reference does not touch bsums of all-zero blocks, so start from identical contents
    memset(y_ref.data(), 0x5a, nb*sizeof(block_q8_K));
    memset(y_cpu.data(), 0x5a, nb*sizeof(block_q8_K));

    quantize_row_q8_K_ref(x.data(), y_ref.data(), k);
    q8_K_from_float(x.data(), y_cpu.data(), k);

    if (memcmp(y_ref.data(), y_cpu.data(), nb*sizeof(block_q8_K)) == 0) {
        return 0;
    }

    for (int64_t i = 0; i < nb; i++) {
        if (memcmp(&y_ref[i], &y_cpu[i], sizeof(block_q8_K)) == 0) {
            continue;
        }
        printf("FAILED %s: block %" PRId64 " differs (d ref=%g cpu=%g)\n", name.c_str(), i, y_ref[i].d, y_cpu[i].d);
        for (int j = 0; j < QK_K; j++) {
            if (y_ref[i].qs[j] != y_cpu[i].qs[j]) {
                printf("    qs[%d]: ref=%d cpu=%d x=%g\n", j, y_ref[i].qs[j], y_cpu[i].qs[j], x[i*QK_K + j]);
                break;
            }
        }
        for (int j = 0; j < QK_K/16; j++) {
            if (y_ref[i].bsums[j] != y_cpu[i].bsums[j]) {
                printf("    bsums[%d]: ref=%d cpu=%d\n", j, y_ref[i].bsums[j], y_cpu[i].bsums[j]);
                break;
            }
        }
        break;
    }
    return 1;
}

int main(int /*argc*/, const char ** /*argv*/) {
    ggml_cpu_init();

    q8_K_from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    int n_failed = 0;
    int n_tests  = 0;

    // random blocks over a wide range of magnitudes, several blocks per call
    const float amplitudes[] = { 1.0f, 1e-3f, 1e-20f, 1e-36f, 1e+20f, 3e+38f, 127.0f, 0.5f };
    for (float amp : amplitudes) {
        for (int rep = 0; rep < 64; rep++) {
            const int64_t nb = 1 + rep % 7;
            std::vector<float> x(nb*QK_K);
            for (auto & v : x) {
                v = (rep % 2 == 0 ? uni(rng) : nrm(rng)) * amp;
            }
            n_tests++;
            n_failed += check("random amp=" + std::to_string(amp), x);
        }
    }

    // sparse blocks: mostly zeros, with a few nonzero values
    for (int rep = 0; rep < 256; rep++) {
        std::vector<float> x(3*QK_K, 0.0f);
        const int nnz = 1 + rep % 5;
        for (int n = 0; n < nnz; n++) {
            x[rng() % x.size()] = nrm(rng);
        }
        n_tests++;
        n_failed += check("sparse", x);
    }

    // rounding ties: iscale is exactly -1 and every other element lands on a .5
    {
        std::vector<float> x(QK_K);
        for (int j = 1; j < QK_K - 1; j++) {
            x[j] = (j - 128.0f) + 0.5f;
        }
        x[0]      =  127.0f;
        x[QK_K-1] = -127.0f;
        n_tests++;
        n_failed += check("ties", x);
        for (auto & v : x) {
            v = -v;
        }
        n_tests++;
        n_failed += check("ties negated", x);
    }

    // the sign of the scale follows the first element reaching the maximum magnitude
    for (int p0 = 0; p0 < QK_K; p0 += 37) {
        for (int p1 = 0; p1 < QK_K; p1 += 41) {
            if (p0 == p1) {
                continue;
            }
            std::vector<float> x(QK_K);
            for (auto & v : x) {
                v = 0.5f*uni(rng);
            }
            x[p0] =  1.0f;
            x[p1] = -1.0f;
            n_tests++;
            n_failed += check("sign tie", x);
        }
    }

    // single nonzero element, zeros and negative zeros
    for (int p : { 0, 1, 7, 8, 15, 16, 100, 254, 255 }) {
        for (float v : { 1.0f, -1.0f, 1e-30f, -3e38f }) {
            std::vector<float> x(2*QK_K, 0.0f);
            for (int j = 0; j < QK_K; j += 3) {
                x[QK_K + j] = -0.0f;
            }
            x[p] = v;
            x[QK_K + (QK_K - 1 - p)] = v;
            n_tests++;
            n_failed += check("single nonzero", x);
        }
    }
    {
        std::vector<float> x(4*QK_K, 0.0f);
        for (int j = 0; j < QK_K; j++) {
            x[QK_K + j] = -0.0f;
            x[2*QK_K + j] = (j % 2) ? -0.0f : 0.0f;
        }
        n_tests++;
        n_failed += check("zeros", x);
    }

#ifdef NDEBUG
    // the reference asserts on these in debug builds: -127/max overflows, or the block contains inf/nan
    for (float amp : { 1e-38f, 1e-41f, FLT_TRUE_MIN }) {
        for (int rep = 0; rep < 16; rep++) {
            std::vector<float> x(QK_K);
            for (auto & v : x) {
                v = uni(rng) * amp;
            }
            n_tests++;
            n_failed += check("overflowing scale amp=" + std::to_string(amp), x);
        }
    }
    for (float special : { INFINITY, -INFINITY, NAN }) {
        for (int p : { 0, 5, 255 }) {
            std::vector<float> x(QK_K);
            for (auto & v : x) {
                v = nrm(rng);
            }
            x[p] = special;
            n_tests++;
            n_failed += check("special value", x);
        }
    }
#endif

    printf("%d/%d q8_K blocks sets bit-exact with the reference\n", n_tests - n_failed, n_tests);

    return n_failed > 0 ? 1 : 0;
}
