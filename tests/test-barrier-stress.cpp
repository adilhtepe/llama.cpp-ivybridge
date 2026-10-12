// Model-free stress test for the CPU threadpool: barrier, static and work-stealing chunking.
// Random graphs are evaluated once with a single thread and then repeatedly with varying
// thread counts; every node must be bit-identical, since each output row is produced by
// exactly one thread regardless of how the work was split.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static std::mt19937 g_rng;

static int rnd_int(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(g_rng);
}

template <typename T>
static T pick(const std::vector<T> & v) {
    return v[rnd_int(0, (int) v.size() - 1)];
}

static void fill_f32(struct ggml_tensor * t) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        p[i] = d(g_rng);
    }
}

// random f32 data quantized/converted into whatever type the tensor has
static void fill_any(struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        fill_f32(t);
        return;
    }

    std::vector<float> tmp(ggml_nelements(t));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto & x : tmp) {
        x = d(g_rng);
    }

    const int64_t n_per_row = t->ne[0];
    const int64_t nrows     = ggml_nelements(t) / n_per_row;

    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t *) t->data, ggml_nelements(t));
    } else {
        ggml_quantize_chunk(t->type, tmp.data(), t->data, 0, nrows, n_per_row, nullptr);
    }
}

static struct ggml_tensor * new_weight(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2 = 1) {
    struct ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    fill_any(t);
    return t;
}

// a random weight type whose block size divides the row length
static enum ggml_type random_weight_type(int64_t ne0) {
    const std::vector<enum ggml_type> types = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
    };
    while (true) {
        const enum ggml_type type = pick(types);
        if (ne0 % ggml_blck_size(type) == 0) {
            return type;
        }
    }
}

// a chain of random ops on a [K, N] activation; mul_mat sizes are chosen so that both the
// static per-thread split and the work-stealing chunk loop get used
static struct ggml_cgraph * build_random_graph(struct ggml_context * ctx, int n_ops) {
    int64_t K = pick<int64_t>({256, 512, 1024});
    int64_t N = pick<int64_t>({1, 5, 32, 97, 128});

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    fill_f32(x);

    for (int i = 0; i < n_ops; ++i) {
        switch (rnd_int(0, 9)) {
            case 0:
            case 1:
            case 2:
                {
                    const int64_t M = pick<int64_t>({64, 192, 512, 1024});
                    struct ggml_tensor * w = new_weight(ctx, random_weight_type(K), K, M);
                    x = ggml_mul_mat(ctx, w, x);
                    K = M;
                } break;
            case 3:
                {
                    const int64_t M      = pick<int64_t>({64, 256, 768});
                    const int64_t n_as   = pick<int64_t>({4, 8, 16});
                    const int64_t n_used = pick<int64_t>({1, 2, 4});

                    struct ggml_tensor * w   = new_weight(ctx, random_weight_type(K), K, M, n_as);
                    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, N);
                    for (int64_t j = 0; j < ggml_nelements(ids); ++j) {
                        ((int32_t *) ids->data)[j] = rnd_int(0, (int) n_as - 1);
                    }

                    struct ggml_tensor * b   = ggml_reshape_3d(ctx, x, K, 1, N);
                    struct ggml_tensor * out = ggml_mul_mat_id(ctx, w, b, ids); // [M, n_used, N]

                    // keep the first selected expert of every token -> [M, N]
                    x = ggml_cont(ctx, ggml_view_2d(ctx, out, M, N, out->nb[2], 0));
                    K = M;
                } break;
            case 4:
                {
                    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
                    fill_f32(b);
                    x = rnd_int(0, 1) ? ggml_add(ctx, x, b) : ggml_mul(ctx, x, b);
                } break;
            case 5: x = ggml_scale(ctx, x, 0.5f); break;
            case 6: x = rnd_int(0, 1) ? ggml_silu(ctx, x) : ggml_gelu(ctx, x); break;
            case 7: x = ggml_rms_norm(ctx, x, 1e-5f); break;
            case 8: x = ggml_soft_max(ctx, x); break;
            case 9:
                {
                    struct ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
                    x = ggml_cpy(ctx, ggml_cpy(ctx, x, h), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N));
                } break;
        }
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// many tiny nodes: the barrier dominates and ops with few rows leave most threads idle
static struct ggml_cgraph * build_tiny_graph(struct ggml_context * ctx, int n_nodes) {
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3);
    fill_f32(x);

    struct ggml_tensor * w = new_weight(ctx, GGML_TYPE_Q4_0, 64, 64);

    for (int i = 0; i < n_nodes; ++i) {
        switch (i % 4) {
            case 0: x = ggml_mul_mat(ctx, w, x); break;
            case 1: x = ggml_scale(ctx, x, 0.25f); break;
            case 2: x = ggml_rms_norm(ctx, x, 1e-5f); break;
            case 3: x = ggml_silu(ctx, x); break;
        }
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4*n_nodes + 16, false);
    ggml_build_forward_expand(gf, x);
    return gf;
}

static void compute(struct ggml_cgraph * gf, int n_threads, struct ggml_threadpool * tp) {
    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, tp);

    std::vector<uint8_t> work_data(cplan.work_size + 1);
    cplan.work_data = work_data.data();

    const enum ggml_status st = ggml_graph_compute(gf, &cplan);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_graph_compute failed: %d\n", (int) st);
        exit(1);
    }
}

static std::vector<std::vector<uint8_t>> snapshot(struct ggml_cgraph * gf) {
    std::vector<std::vector<uint8_t>> out(ggml_graph_n_nodes(gf));
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        struct ggml_tensor * t = ggml_graph_node(gf, i);
        out[i].assign((const uint8_t *) t->data, (const uint8_t *) t->data + ggml_nbytes(t));
    }
    return out;
}

static bool check(struct ggml_cgraph * gf, const std::vector<std::vector<uint8_t>> & ref, int n_threads, const char * what) {
    bool ok = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        struct ggml_tensor * t = ggml_graph_node(gf, i);
        if (ref[i].size() != ggml_nbytes(t) || memcmp(ref[i].data(), t->data, ref[i].size()) != 0) {
            fprintf(stderr, "  MISMATCH (%s, %d threads): node %d %s %s [%ld, %ld, %ld]\n",
                what, n_threads, i, ggml_op_name(t->op), ggml_type_name(t->type),
                (long) t->ne[0], (long) t->ne[1], (long) t->ne[2]);
            ok = false;
        }
    }
    return ok;
}

static std::vector<int> thread_counts(int n_max) {
    std::vector<int> v = {2, 3, n_max/2, n_max - 1, n_max, rnd_int(1, n_max), rnd_int(1, n_max)};
    v.erase(std::remove_if(v.begin(), v.end(), [n_max](int n) { return n < 1 || n > n_max; }), v.end());
    return v;
}

// one graph: reference with 1 thread, then every thread count, alternating between the
// persistent pool and a disposable one
static int run_graph(struct ggml_cgraph * gf, int n_threads, struct ggml_threadpool * tp, int n_repeat, const char * what) {
    compute(gf, 1, tp);
    const auto ref = snapshot(gf);

    int n_fail = 0;
    int round  = 0;

    for (int r = 0; r < n_repeat; ++r) {
        for (int n : thread_counts(n_threads)) {
            compute(gf, n, (round++ % 4 == 3) ? nullptr : tp);
            if (!check(gf, ref, n, what)) {
                n_fail++;
            }
        }
    }

    return n_fail;
}

int main(int argc, char ** argv) {
    int n_threads = std::max(1, std::min(8, (int) std::thread::hardware_concurrency()));
    int n_iters   = 8;

    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--numa" && i + 1 < argc) {
            const std::string s = argv[++i];
            enum ggml_numa_strategy strategy = GGML_NUMA_STRATEGY_DISABLED;
            if      (s == "distribute") { strategy = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (s == "isolate")    { strategy = GGML_NUMA_STRATEGY_ISOLATE;    }
            else if (s == "numactl")    { strategy = GGML_NUMA_STRATEGY_NUMACTL;    }
            else {
                fprintf(stderr, "unknown numa strategy '%s'\n", s.c_str());
                return 1;
            }
            ggml_numa_init(strategy);
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() > 0) { n_threads = std::atoi(positional[0].c_str()); }
    if (positional.size() > 1) { n_iters   = std::atoi(positional[1].c_str()); }

    if (n_threads < 1) {
        fprintf(stderr, "invalid thread count\n");
        return 1;
    }

    fprintf(stderr, "test-barrier-stress: n_threads %d, n_iters %d, numa %d\n", n_threads, n_iters, (int) ggml_is_numa());

    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    if (!tp) {
        fprintf(stderr, "threadpool create failed\n");
        return 1;
    }

    int n_fail = 0;

    for (int it = 0; it < n_iters; ++it) {
        g_rng.seed(1234 + it);

        struct ggml_init_params params = {
            /* .mem_size   = */ 256u*1024*1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_cgraph * gf = build_random_graph(ctx, rnd_int(8, 24));
        fprintf(stderr, "  iter %d: random graph, %d nodes\n", it, ggml_graph_n_nodes(gf));
        n_fail += run_graph(gf, n_threads, tp, 3, "random");

        ggml_free(ctx);
    }

    {
        g_rng.seed(4321);

        struct ggml_init_params params = {
            /* .mem_size   = */ 64u*1024*1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_cgraph * gf = build_tiny_graph(ctx, 2000);
        fprintf(stderr, "  tiny graph, %d nodes\n", ggml_graph_n_nodes(gf));
        n_fail += run_graph(gf, n_threads, tp, 4, "tiny");

        ggml_free(ctx);
    }

    ggml_threadpool_free(tp);

    if (n_fail) {
        fprintf(stderr, "FAILED: %d mismatching runs\n", n_fail);
        return 1;
    }

    fprintf(stderr, "OK\n");
    return 0;
}
