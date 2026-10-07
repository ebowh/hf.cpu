# hf.cpu

A CPU-only LLM inference engine in C99, aimed at small, memory-limited
machines (AVX2 laptops). It reads GGUF files made for llama.cpp and
ik_llama.cpp, talks over stdin/stdout/stderr only, and is meant to be driven
from a pipeline or from Emacs. See `docs/DESIGN-SPACE.md` for the research and
design decisions and `docs/PROTOCOL.md` for the wire protocol.

**Status: phase 0 (foundations).** There is no inference yet. What exists and
is tested:

* checked allocation with fault injection (no allocation is assumed to succeed)
* platform layer for Linux and macOS (both confirmed running on the target machines), FreeBSD (written, untested)
* CPU feature, cache, memory and strict-overcommit probing and a measured
  machine profile (`--op doctor`)
* one option vocabulary for command line, settings file and per-request lines
* the stdin request loop (`prompts`, `args`, `none` modes) and event protocol
* a hardened, fuzzed GGUF v2/v3 reader (`--op inspect`)
* the ggml type table and scalar reference dequantizers for 18 types,
  verified bit-for-bit against upstream `ggml-quants.c`
* Perl tools: `tools/gguf-inspect.pl` (independent reader), `tools/mkgguf.pl`
  (synthetic GGUF generator)

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
