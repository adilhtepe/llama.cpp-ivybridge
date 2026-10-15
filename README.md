# llama.cpp

> ## Ivy Bridge fork notes
>
> This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at tag `b11574`, adapted for **AVX-only x86 CPUs** (AVX, F16C, SSE4.2; no AVX2, FMA or BMI), for example Xeon E5 v2 (Ivy Bridge-EP). The work is on the `ivybridge` branch. It is not an upstream contribution.
>
> ### What changed
> - **Matmul:** tiled `mul_mat` / `mul_mat_id` (including MoE experts) for `q4_0`, `q5_0`, `q8_0`, `iq4_nl` and `mxfp4`, using a 128-bit SSSE3 microtile. Enabled only when AVX2 is not available.
> - **Quantization and vector ops:** AVX `quantize_row_q8_K` (bit-exact with the reference), vectorized `rms_norm` sum of squares, 256-bit `vec_add_f32`, 256-bit `expf` / `silu` without FMA.
> - **`vec_dot`:** F16C block-scale conversion for `q4_0` / `q8_0`, tuned `q4_K` / `q6_K` kernels.
> - **Flash attention:** blocked KV loop, `q8_0` / `q4_0` K and V in the fast paths, AVX row dequantization, a decode kernel that processes the query heads sharing one KV head together (grouped-query attention), and tile-aligned work splitting in the prompt path.
> - **Threading:** atomic barrier under OpenMP with yield when oversubscribed, 128-byte padding of hot counters.
> - **Tests:** `test-tiled-mulmat` extended, new `test-quantize-q8-k`, `test-cpu-ops`, `test-vec-dot-q`, `test-flash-attn` (checked against a scalar double-precision reference), `test-barrier-stress`.
>
> ### Results
> Machine: 2x Xeon E5-2697 v2, DDR3-1600, 8x16 GB. Model: a community Q4_K_M GGUF of Qwen3-Coder-30B-A3B-Instruct (abliterated). Both binaries were built with the same flags (`-march=ivybridge`, AVX2/FMA/BMI2 off).
>
> **Speed** (`llama-bench -t 12 -p 128 -n 32 -d 0,4096 -r 3`, pinned to one NUMA node with local memory, page cache dropped before every run, upstream and fork run alternately three times each, median [min..max], tokens/s):
>
> | Test | upstream `b11574` | this fork | change |
> |---|---|---|---|
> | prompt, empty context | 48.00 [47.61..48.34] | 50.41 [50.21..50.76] | +5.0 % |
> | generation, empty context | 18.30 [18.22..18.32] | 18.52 [18.28..18.60] | +1.2 % (within noise) |
> | prompt, 4096 tokens deep | 19.26 [19.22..19.28] | 29.30 [29.10..29.31] | +52 % |
> | generation, 4096 tokens deep | 8.09 [8.08..8.12] | 12.36 [12.23..12.38] | +53 % |
>
> The gain is in long contexts. Generation on an empty context is not measurably faster.
>
> **Quality** (`llama-perplexity`, wikitext-2 test set, `-c 512 -fa on`, same weights and settings for both binaries):
>
> | Path | upstream `b11574` | this fork | std. error |
> |---|---|---|---|
> | prompt processing, 40 chunks | 8.6446 | 8.6427 | 0.258 |
> | one token at a time (`-b 1 -ub 1`), 8 chunks | 9.7359 | 9.7230 | 0.655 |
>
> The differences are far below the standard error. The one-token path is the one that exercises the grouped-query decode kernel. Results are not bit-identical to upstream (summation order and `expf` evaluation differ in a few kernels).
>
> ### Known limitations
> - Measured end to end on one machine and one model. The other changes (tiled matmul for the 32-element formats, `vec_dot` tuning, threading) are covered by correctness tests and micro-measurements, not by an end-to-end comparison.
> - The grouped decode kernel is not gated by architecture. Without AVX it falls back to scalar code and may be slower than upstream. It was not benchmarked on AVX2 or AVX-512 hardware.
> - If a V row contains `inf` or `NaN` and its softmax weight underflows to zero, the grouped decode kernel can propagate `NaN` to the whole head, where upstream would not. Weights of real models are finite.
> - `test-backend-ops` skips the CPU backend when it is the only backend, so the new CPU code is verified by the dedicated tests above.
>
> ### Build
> ```sh
> cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
>   -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_F16C=ON \
>   -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_BMI2=OFF -DGGML_AVX512=OFF \
>   -DCMAKE_C_FLAGS="-march=ivybridge -mtune=ivybridge" \
>   -DCMAKE_CXX_FLAGS="-march=ivybridge -mtune=ivybridge"
> cmake --build build -j
> ```
> `GGML_AVX2`, `GGML_FMA` and `GGML_BMI2` default to ON when `GGML_NATIVE=OFF`, so they have to be switched off explicitly.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
