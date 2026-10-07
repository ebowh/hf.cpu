# hf.cpu

A CPU-only LLM inference engine in C99, aimed at small, memory-limited
machines (AVX2 laptops). It reads GGUF files made for llama.cpp and
ik_llama.cpp, talks over stdin/stdout/stderr only, and is meant to be driven
from a pipeline or from Emacs. See `docs/DESIGN-SPACE.md` for the research and
design decisions and `docs/PROTOCOL.md` for the wire protocol.

**Status: phase 1a.** Dense transformers (Qwen2, Qwen3, Llama layouts) run
end to end on one thread. What exists and is tested:

* `--op generate`: tokenizer, forward pass with a paged f16 KV cache, greedy and
  sampled decoding, per-token top-N logprobs, UTF-8-safe streamed text, resident
  model between requests. Checked against an independent double-precision Perl
  implementation on tiny random models of every layout and weight type, and
  (on your machines) against llama.cpp via `tests/check-golden.pl`
* Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K, F32, F16 weights with AVX2 kernels (Q8_0/Q8_K
  activation quantization); every other type works through the slower
  dequantizing path until its kernel is written

Phase 0 foundations, also tested:

* checked allocation with fault injection (no allocation is assumed to succeed)
* platform layer for Linux and macOS (both confirmed running on the target machines), FreeBSD (written, untested)
* CPU feature, cache, memory and strict-overcommit probing and a measured
  machine profile (`--op doctor`)
* one option vocabulary for command line, settings file and per-request lines
* the stdin request loop (`prompts`, `args`, `none` modes) and event protocol
* a hardened, fuzzed GGUF v2/v3 reader (`--op inspect`)
* a byte-level BPE tokenizer with a hand-written Qwen2 pre-tokenizer (no regex
  engine), checked against all 46 of llama.cpp's Qwen2 vocabulary test cases and
  against token ids recorded from llama.cpp for real prompts (`--op tokenize`)
* the ggml type table and scalar reference dequantizers for 18 types,
  verified bit-for-bit against upstream `ggml-quants.c`
* Perl tools: `tools/gguf-inspect.pl` (independent reader), `tools/mkgguf.pl`
  (synthetic GGUF generator), `tools/golden.pl` (records reference tokens and
  logprobs from a running llama-server), `tools/gen-unicode.pl`

## Build and test

    make            # builds ./hfcpu (POSIX make, cc, no dependencies)
    make test       # C unit tests + end-to-end tests (needs perl)
    make asan       # everything again under AddressSanitizer + UBSan
    make diff-dequant REF=/path/to/llama.cpp/ggml/src   # compare with upstream

## Try it

    ./hfcpu --stdin-mode none --op doctor
    ./hfcpu --stdin-mode none --op inspect --model model.gguf
    printf 'hello\x1eworld\x1e' | ./hfcpu --op echo --system 'Be terse.'

## Layout

    src/    engine sources (hfc.h common types, pal.* platform, probe.* CPU and machine,
            opts.* options, proto.* stdin/stdout, gguf.* + ggtype.* model files, ops.* requests)
    tests/  C tests, end-to-end tests (e2e.pl), upstream differential test
    tools/  Perl utilities
    docs/   design space exploration and protocol
