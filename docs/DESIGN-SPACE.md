# hf.cpu: design-space exploration

Status: research notes, no code. Facts verified by web search are marked **[V]**. Everything else is from memory or derived by me, and is marked **[M]** (memory) or **[D]** (derived estimate). Treat [M] and [D] as hypotheses until checked against sources or measurements.

Targets: HP EliteBook 850 G6 (8th-gen U-series, likely 4C/8T, 32 GB) and MacBook Pro early 2015 (Broadwell 2C/4T, 16 GB LPDDR3-1866, 4 MB L3). Both are AVX2+FMA+F16C, with no AVX-512 and no VNNI.

---

## 0. Thesis

1. **Decode is bandwidth-bound and prefill is compute-bound.** The engine is mostly two different programs sharing one weight format. Almost every trick below targets one of the two.
2. **On 2-4 weak cores, the compute/bandwidth balance is tight.** Q4 decode is only about 2x away from compute-bound, so ALU efficiency of the dequant+dot kernels matters even for decode. See §1.
3. **Disk is cheaper than compute.** Loading a saved KV prefix takes seconds, while recomputing it takes minutes. Content-addressed persistent prefix caches are the single biggest user-visible win for a resume-days-later, pipelined workload.
4. **Overcommit-off is easier than it looks if the memory design is right.** Read-only file-backed mappings are not charged to the commit limit. Reserving `PROT_NONE` address space is not charged either. Block-structured KV makes growth a non-event. See §5.
5. **Per-model C code does not scale. Block library + declarative model spec + numerical-conformance harness does.** See §9.
6. **Autotune from probes, with analytical priors.** Measure the machine once, cache the result keyed by CPU signature, and keep learning online. See §8.

---

## 1. Roofline for the two machines [D, verify by probing]

Rough numbers only, to show the shape of the problem.

| | MBP 2015 (5557U-class) | EliteBook 850 G6 |
|---|---|---|
| Cores/threads | 2/4 | probably 4/8 |
| Peak DRAM BW | 29.8 GB/s (LPDDR3-1866, dual ch.) | 38.4 GB/s if 2 DIMMs in dual channel, **half** if single channel |
| Realistic read BW | ~18-22 GB/s | ~22-30 GB/s |
| Cache | L2 256 KB/core, L3 4 MB, **128 MB eDRAM L4** (Iris 6100, probe it) | L2 256 KB/core, L3 6-8 MB |
| int8 MAC throughput | ~150 GOPS effective with Q4 dequant overhead | ~2x |

**Decode ceiling** is BW / bytes per token. Include the KV read and the lm_head.
- 1B at Q4: ~0.7 GB gives 25-30 t/s.
- 4B at Q4: ~2.5 GB gives ~7-8 t/s.
- 9B at Q4: ~5.3 GB gives ~3.5-4 t/s.

**Prefill ceiling** is ops / (2 x params). On the MBP that is roughly 75 t/s for 1B, 19 t/s for 4B and 8 t/s for 9B, and real kernels reach 50-70% of that. The EliteBook is about 2x on 4 cores.

**Consequences:**
- A 4k-token prompt to a 9B model costs about 8 minutes of prefill on the MBP. That is the entire argument for persistent prefix caching.
- The "free batch" of speculative decoding is small. Q4 does about 4 ops per weight byte per token, and the compute/bandwidth ridge sits near n = 2-3 tokens on these machines. Verifying 8 draft tokens is not free here. It becomes compute-bound at roughly 3, so speculation helps (§4.3) but with modest `k`. The calibrator must measure the ridge rather than assume it.
- KV traffic grows with context. Qwen3-4B (36 layers, 8 KV heads, head_dim 128 [M]) has 147 KB/token of f16 KV. At 8k context that is ~1.2 GB read per token, half of the weight traffic. Hybrid models with few attention layers (§2) and quantized KV change this a lot.
- The lm_head is large. A vocab of 150k x hidden 2560 at Q6 is ~300 MB, over 10% of per-token traffic for a 4B model. Tricks in §4.5.
- Laptops throttle. 15 W-class parts under sustained AVX2 fall well below burst clocks. The calibrator must measure sustained throughput (30-60 s runs), not just burst.

---

## 2. Your model list is architecturally diverse

The block library must cover the union of these.

| Model | Architecture | Source |
|---|---|---|
| MiniCPM5-1B/2B | Plain `LlamaForCausalLM`, dense GQA (2B: 42 layers, 16Q/2KV heads, 128k ctx) | [V] |
| Qwen3-4B | Dense GQA with QK-norm | [M] |
| Qwen3.5 (0.8B, 2B, 4B, 9B) | Hybrid: Gated DeltaNet (linear attention) and full attention at 3:1. 0.8B: 18 GDN + 6 attn layers. 9B: 24 GDN + 8 attn. Natively multimodal. Context 262k | [V] |
| LFM2.5 (350M, 1.2B, 2.6B...) | Gated short convolutions interleaved with GQA. A VL line (450M/1.6B/3B) and an 8B-A1B MoE variant also exist | [V] |
| Ling-3.0-tiny | MoE 7.9B total, 1.3B active. 24 layers, 3x Kimi Delta Attention (KDA) then 1x Multi-head Latent Attention (MLA). 8-of-128 routed experts plus 1 shared. 256k ctx | [V] |
| Ornith-1.5-9B | **Unknown to me.** Inspect its GGUF | - |

**Findings:**
- **MiniCPM5 being plain Llama means the day-one target is easy.** Start there.
- **Recurrent state is not a KV cache.** Qwen3.5 (GDN), Ling (KDA) and LFM2 (conv state) carry fixed-size state per layer. That state cannot be truncated back to an arbitrary prefix, so prompt-cache design must handle "checkpoint at specific token positions" (§6).
- **Hybrids are a gift on this hardware.** Qwen3.5-9B has 8 attention layers instead of 36, so KV traffic is cut about 4x. Long context becomes affordable. Hybrids are probably the best models for this project.
- **MoE is great for decode bandwidth.** Ling reads about 1.3B active params per token, so decode speed is comparable to a 1.3B model. It needs ~4.5 GB at Q4 resident. It breaks "verify k tokens for the price of one": the union of experts across k tokens approaches all of them.
- **MLA and KDA are exotic.** llama.cpp support for these may lag, so check GGUF availability before committing to them.
- **Vision (LFM2.5-VL, Qwen3.5):** a ViT encoder plus projector, usually shipped as a separate `mmproj` GGUF in llama.cpp. It needs image decode and preprocessing in C, and ViT prefill is expensive. Cache image embeddings on disk, content-addressed by image hash (§6).

First deliverable of any tool: `inspect <gguf>`. It prints architecture, hyperparameters, layer pattern, tensor types and which blocks are missing from our library.

---

## 3. Prior art catalog

### 3.1 llama.cpp [M]
- GGUF, ggml graph executor, runtime CPU dispatch, many quant types. Legacy: Q4_0/1, Q5_0/1, Q8_0. K-quants: Q2_K to Q6_K. i-quants: IQ1/2/3/4. Also MXFP4 and ternary TQ1_0/TQ2_0, plus BF16.
- **Runtime repacking** to interleaved layouts (Q4_0 4x8/8x8, Q4_K 8x8 etc.) for AVX2/NEON/AVX512 GEMM and GEMV. The repacked copy lives in anonymous memory, so it breaks mmap sharing and doubles RSS during load.
- Flash-attention CPU path. KV quantization (q8_0, q4_0...). `llamafile` sgemm integration for prefill.
- Prompt caching: `--prompt-cache` file, server slot save/restore, `--cache-reuse` (KV shifting for partial prefix reuse), context checkpoints for recurrent/SWA models.
- Speculative decoding: draft model, n-gram cache / lookup, EAGLE-3. `-ot` tensor overrides, `--n-cpu-moe`. GBNF grammars, JSON-schema to grammar, imatrix, context shift, SWA, NUMA options, `--mlock`/`--no-mmap`.
- Its model support is hand-written C++ per architecture with shared helper builders (`build_norm`, `build_attn`, `build_ffn`, `build_moe_ffn`). That is already a block library. GGUF arch-specific metadata keys, tensor names and the conversion script's weight transformations are the **semantic contract** we must match.

### 3.2 ik_llama.cpp [M]
CPU-performance fork by Iwan Kawrakow.
- IQK quants (iq2_k, iq3_k, iq4_k, iq5_k, iq4_ks, trellis "KT" quants) giving better quality per bit.
- **Row-interleaved `_R4`/`_R8` types** (q4_0_r8, q8_0_r8, iq4_xs_r8, q4_k_r4...) with run-time repack (`-rtr`).
- Fused ops (rms_norm+mul, fused up/gate for MoE, fused QKV), `-fmoe`, smart expert reduction (`-ser`), FlashMLA variants, attention max-batch (`-amb`) to bound scratch, Hadamard-transformed K cache, and a much faster CPU flash attention with quantized KV.
- Its GGUFs use extra ggml type ids that must be handled (they diverge from mainline numbering; verify exact ranges).
- Its `llama-quantize --custom-q` regexes show the value of per-tensor mixed quantization.

### 3.3 llamafile (Mozilla / J. Tunney) [M]
- **tinyBLAS / `llamafile_sgemm`**: register-tiled small-matrix kernels that gave 2-5x prompt-eval speedups on ordinary CPUs. Mostly folded back into ggml.
- Cosmopolitan APE single-file portability, weights embedded in a zip and mmapped, runtime CPU feature dispatch.
- Lesson: **prefill GEMM quality matters more than anything else on weak CPUs.**

### 3.4 DS4 / DwarfStar 4 (antirez) [V]
A small, self-contained C inference engine **specific to DeepSeek V4 Flash**, not a generic GGUF runner. Does not link GGML. Credits llama.cpp's kernels, quant formats and GGUF ecosystem. Its stated scope includes model-specific loading, prompt rendering, tool calling and **KV state handling in RAM and on disk**. Backends are Metal, CUDA and ROCm, so it has no CPU/AVX2 path. Worth reading for the on-disk KV design and for how a "narrow engine" stays small. I have not read its code, so everything about its internals beyond that is unknown to me. It is the opposite design point from ours: one model, many backends, versus many models, one CPU backend.

### 3.5 Others to mine
- **llama2.c / llm.c (Karpathy)**: minimal-C readability baseline.
- **bitnet.cpp and T-MAC (Microsoft)**: LUT-based low-bit mpGEMM. Weights are decomposed into bit-planes and activations precompute tables of partial sums, indexed via `vpshufb` on AVX2. Strong at 1-3 bit, uncertain at 4 [M].
- **CTranslate2, OpenVINO, Intel IPEX-LLM / neural-speed, ONNX Runtime GenAI**: CPU int8 inference with oneDNN-style blocked kernels.
- **vLLM**: PagedAttention (fixed blocks, free list), automatic prefix caching (block hash chains), chunked prefill, lazily grown KV.
- **SGLang**: RadixAttention (token radix tree over cached prefixes), compressed-FSM constrained decoding.
- **llguidance / XGrammar**: fast token-mask computation for grammars.
- **LMCache / Mooncake**: KV stored outside GPU memory (disk, remote), reused across requests.
- **Speculation**: prompt-lookup decoding, SuffixDecoding, lookahead decoding, Medusa, EAGLE 1-3, LayerSkip / self-speculative, MTP heads, FR-Spec (reduced-vocab draft).
- **KV management**: StreamingLLM attention sinks, H2O / SnapKV eviction, KIVI (K per-channel, V per-token quant), cross-layer KV sharing, sliding-window layers, MLA.
- **Quantization research**: AWQ, GPTQ, QuIP#, AQLM, QTIP, HQQ, QuaRot / SpinQuant (rotation before quantizing).
- **BLAS design**: GotoBLAS / BLIS (panel packing, microkernel, analytical cache-blocking model by Low et al. 2016), ATLAS / FFTW (autotune plus saved "wisdom").
- **HF**: `transformers` "modular" files (a model written as a diff vs another model), the `kernels` hub, `optimum` export paths, `gguf-py`.
- **MLX-LM**: small per-model files (~200 lines) built from shared layers. A compact, readable reference for architecture semantics.

---

## 4. Compute-side knobs and tricks

### 4.1 Weight layout and kernels
- **Offline repack** into interleaved 4- or 8-row tiles sized to the SIMD width and dot-product instruction. Fewer concurrent streams helps the hardware prefetcher, and activation loads are shared across rows. Store it in a sidecar file (§7) and do not repack at load time.
- **Type descriptor table**: for each GGUF type record block size, bytes, dequant function, activation partner type, available repack variants. Adding an ik type is one table row plus kernels.
- **Priority types for these machines**: Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, IQ4_XS / IQ4_NL (a 16-entry table, a natural fit for `vpshufb`), BF16/F16 for small embeddings, then IQ2/IQ3 and ik types as needed. MXFP4 only if a target model ships it.
- **AVX2 int8 dot**: `vpmaddubsw` (u8 x s8) followed by `vpmaddwd` against ones, then `vpaddd`. Use the abs/sign trick for s8 x s8. Clamp activations to ±127 to avoid saturation [M]. Quantize activations **once per input vector** and reuse them across all weight rows sharing the input (Q/K/V, gate/up).
- **Three kernel regimes by token count `n`**:
  1. `n = 1` GEMV: stream packed weights, dot directly.
  2. `n = 2..~8`: multi-vector dot. Unpack weights in registers once and apply to `n` activation vectors.
  3. `n >= ~16`: GotoBLAS-style GEMM. Unpack a weight panel to int8 in L2, block in L1, run a register-tile microkernel (AVX2 has 16 ymm registers).
- **Numerical conformance**: partition work over output rows only, never over the K dimension. Every dot product then accumulates in the same order regardless of thread count, so results are **bit-identical across thread counts**. That makes gates reproducible and cache validity checkable.
- **Fusions**: rmsnorm+quantize, fused QKV, RoPE fused into the KV write, flash-style attention (online softmax, no O(ctx) scratch), SwiGLU gate*up fused, residual-add fused into the down projection, final norm + lm_head.
- **Cheap wins**: set MXCSR FTZ/DAZ (denormals are very slow on x86, and decaying recurrent states and softmax tails produce them). Use polynomial `exp`/`tanh` instead of 64k-entry f16 tables, because the tables cost half of a 256 KB L2.
- **Dispatch**: separate translation units compiled with `-mavx2 -mfma -mf16c` (or `__attribute__((target))`), a function-pointer table selected from CPUID, and a scalar C reference for every kernel (used in tests and as the fallback). Keep the ISA layer open to AVX-VNNI (Alder-Lake-N "puny PCs" have it) and NEON+dotprod (Raspberry Pi 5).
- **Language**: C99 core (restrict, stdint, inline, mixed declarations). Intrinsics are a compiler extension in any case, so strict C89 buys nothing. Keep SIMD isolated in a few files.

### 4.2 Threading
- One thread per **physical** core for decode (HT rarely helps a bandwidth-bound loop). Probe whether all hardware threads help prefill, usually only ~10-20%.
- Pin threads (`sched_setaffinity` on Linux. macOS only offers affinity hints, so measure whether it matters there).
- **Static plan-time scheduling.** The graph is fixed per model, so precompute op lists and per-thread row ranges aligned to tile and cache-line boundaries. No atomic work queue and no false sharing. Use spin-then-sleep barriers (`pause` loop, then futex). Fusion cuts barrier count per layer to about 4-6.
- **Good-citizen mode**: run at lower priority (`nice`, `SCHED_IDLE`, QoS background on macOS) so Emacs stays responsive. Offer it as a knob.
- Software prefetch distance and non-temporal (`prefetchnta`) hints on weight streams. Hardware prefetchers mostly cover this, so treat both as autotune parameters and not as defaults.

### 4.3 Speculative decoding
- **n-gram / prompt lookup / suffix automaton over prompt+history**: zero model cost, high acceptance on edit / summarize / extract / rewrite stages, which are common in pipelines. Do this first.
- **Draft model**: same tokenizer needed. Qwen3.5-0.8B drafting for 9B costs ~7% of the verifier's bytes per draft token. Check that LFM2.5-350M shares a vocab with the larger LFM2.5 models.
- **MTP heads**: if a GGUF retains multi-token-prediction tensors, they are a free draft model.
- **Adaptive `k`**: tune online from observed acceptance and the measured ridge point (§1). Sizes beyond 3-4 are probably wasted here. Disable on MoE or measure, because the union of experts erodes the benefit.
- Tree verification costs compute, so it is probably not worth it on 2 cores.

### 4.4 Batched fan-out (an exception to "one inference at a time")
Parallel sampling of `n` continuations (best-of-n, self-consistency, retries after a failed gate) shares the prefix, and decode of `n` sequences together costs far less than `n` x one sequence because weights are read once. The multi-vector kernel from §4.1 gives it nearly for free. Worth exposing as `n=` in a request. **Question for you in §12.**

### 4.5 lm_head and sampling
- Fuse top-k / argmax into the lm_head loop, so the full vocab logits are not materialized for greedy decoding.
- **Constrained decoding (grammar/JSON schema/regex)**: compute logits only for allowed tokens (sparse lm_head rows). That makes constrained output cheaper than unconstrained, and also cuts gate failures.
- Skip logits for all non-final prefill tokens.
- Samplers: temperature, top-k, top-p, min-p, typical, mirostat, repetition / presence / frequency penalties, DRY, XTC, logit bias. Seeded RNG for reproducibility.
- **Thinking budget** (Qwen3/3.5-style reasoning): after N thinking tokens, inject the closing tag and force the answer (budget forcing). Essential at 4-8 t/s.
- **Judge-by-logits mode**: one prefill, then read logits for a small label set (yes/no, A/B/C, score 1-5), with no generation. This is the cheapest possible quality gate. Also expose per-token logprobs, mean logprob and entropy as gate signals, and a `score` mode (perplexity of given text).

### 4.6 Context and attention
- **Flash-style streaming attention** over KV blocks. Scratch is O(block), not O(ctx x heads).
- **KV quantization**: f16, q8_0, q4_0. K is more sensitive than V. K stored in the same block format as quantized queries gives an int8 QK dot. A Hadamard/rotation before quantizing (ik, QuaRot-style) helps at low bits [M].
- **Tiered precision**: recent tokens f16/q8, older sealed blocks q4. Progressive down-quantization of cold blocks.
- **RoPE position handling**: K stored post-RoPE (cheaper attention, shifting needs re-rotation) or pre-RoPE (rotate on the fly, enables position-shifted reuse and sliding windows without recompute). Probably post-RoPE with a re-rotate pass for shifts, as llama.cpp does.
- **Sliding-window layers, attention sinks, MLA latent cache** as layer types.
- **Context shifting / eviction** (sink + window) as a configurable overflow policy, since the pipeline can also summarize externally.

### 4.7 MoE
- Experts are contiguous per-layer slabs in the sidecar. Only routed experts' pages are touched, so the page cache naturally holds the hot set.
- After the router runs, `madvise(WILLNEED)` the selected experts' pages. `mincore()` shows what is resident.
- Smart expert reduction (drop low-weight experts) is a quality-for-speed trade, exposed as an option.

### 4.8 Vision
- Decode PNG/JPEG in-tree (a stb-style decoder), preprocess/tile, run ViT + projector, inject embeddings.
- Cache embeddings on disk by `hash(image bytes, preprocessing params, mmproj id)`.
- Prefer passing images as file paths or fd offsets in the protocol, not inline over stdin.

---

## 5. Memory management, overcommit-off, and context growth

**Linux strict overcommit (`vm.overcommit_memory=2`) facts** [M, verify on the target kernels]:
- Private writable anonymous mappings are charged to `Committed_AS` at `mmap` time. Failure is `ENOMEM` at the call, which is predictable and good.
- `PROT_NONE` private mappings are **not** charged until made writable (`mprotect`).
- Read-only `MAP_SHARED` file mappings are **not** charged. Writable `MAP_PRIVATE` file mappings **are**. `MAP_NORESERVE` is ignored in mode 2.
- Therefore: **mmap weights read-only**, and let the kernel's page cache be the elasticity.
- macOS has no strict commit accounting. Memory pressure shows up as compression and swap, so the engine must watch free + inactive pages instead.

**Allocation policy:**
- Every allocation goes through a wrapper that returns an error. No `malloc` in the hot path.
- Compute a **static memory plan** at load time: weights (file-backed), persistent state (KV blocks, recurrent state), per-token scratch (bump arena, reset each token), per-ubatch scratch. Fail early with a precise message if the plan does not fit.
- Overflow-check every size computation. Treat GGUF as hostile input (§10).
- Budget inputs: `MemAvailable`, `CommitLimit` vs `Committed_AS`, cgroup `memory.max`, `RLIMIT_AS`/`RLIMIT_MEMLOCK` on Linux. `hw.memsize` and `vm_stat` on macOS.
- Degradation ladder when memory is short: shrink ubatch, quantize KV, drop to a lower KV tier, shrink context, spill sealed KV blocks to disk, then refuse with a clear error. Set `oom_score_adj` so we die before the user's other processes do.

**vLLM-style incremental context:**
- KV is a pool of fixed-size blocks (e.g. 32-128 tokens of all layers). Context growth allocates blocks on demand, with **no copy and no realloc peak**. Parameters: `ctx-init`, `ctx-step`, `ctx-max`. The model's rope config (YaRN/NTK) is separate from how much is allocated.
- Alternative for contiguous layouts: reserve address space with `PROT_NONE` up to `ctx-max`, then `mprotect` more in increments. This costs no commit until used and works in strict mode. Keep it as an implementation choice behind the block abstraction.
- **Sealed (full) blocks are immutable.** They can be flushed to the cache store and re-mapped **read-only from the file**. Cold KV then costs no commit and is evictable by the kernel. That gives graceful long-context degradation on a 16 GB machine. Prefetch the next layer's blocks with `madvise(WILLNEED)` while computing the current layer.

**Page-cache hygiene:**
- Detect thrashing from `ru_majflt` per token and warn on stderr with the model's required RSS.
- `mlock` weights when `RLIMIT_MEMLOCK` allows.
- Write cache files with `O_DIRECT` or `sync_file_range` + `posix_fadvise(DONTNEED)` so writes do not evict model pages.
- In a pipeline, `posix_fadvise(DONTNEED)` or `MADV_PAGEOUT` a finished model, or **leave it if the next stage reuses it** (a retry loop). If RAM allows (16 GB: a 5.5 GB 9B plus a 2.5 GB 4B fits), prefetch the next stage's weights while the current stage finishes.
- THP (`MADV_HUGEPAGE`) for anonymous arenas (scratch, KV tail). File-backed weights stay 4 KB pages, and sequential streaming makes TLB cost small.

**Cold start:** a 5 GB model at 500 MB/s-1 GB/s is 5-10 s. Read in execution order (layer 0 first) from a prefetch thread so first-layer compute overlaps the rest of the load.

---

## 6. Persistent prompt / state cache

**What gets stored:** token ids, model identity hash (weights + tokenizer + template + KV dtype + rope config + format version), KV blocks, recurrent-state checkpoints, optionally sampler state.

**Key insights:**
1. **Content-addressed prefix store.** Hash chain of (parent hash, block tokens, model id). Lookup by longest-matching prefix, vLLM APC style. Or a token radix tree (SGLang). Emacs can then **resend the whole transcript every turn** and the engine finds the shared prefix automatically, with no session ids to manage. A system prompt shared by pipeline stages is deduplicated.
2. **Granularity matters here.** At 10-20 t/s prefill, losing up to a 255-token chunk costs ~15 s. For attention-only models any prefix of a stored sequence is valid (causal), so store token-ordered linear segments and **reuse at exact token length**. For hybrids reuse is only valid at checkpoint positions.
3. **Frozen prefix blocks mmapped read-only from the cache file, tail blocks anonymous.** Block-level copy-on-write in userland avoids the commit charge of `MAP_PRIVATE` writable mappings.
4. **Recurrent state checkpoints** (GDN, KDA, conv state): fixed-size per checkpoint, but you need them at the right positions. Policy: end of system prompt, end of each turn, and every N tokens by a cost model. This is the same idea as llama.cpp's context checkpoints.
5. **Tokenization boundary and template pitfalls.** Re-tokenizing a transcript can differ from the generated tokens at turn boundaries, and templates such as Qwen's drop previous-turn reasoning, which invalidates the cache after the assistant message. The cache must compare **token ids**. Offer a token-exact API (the client sends ids or the engine returns them) and a canonical-form option.
6. **Fork and rollback.** A gate failure retries from the post-prefix checkpoint with a different seed or temperature, and costs no prefill. This is trivial for append-only block KV (truncate pointers) and needs state snapshots for recurrent layers.
7. **Cache warming.** A `warm` request prefills and stores each stage's static preamble ahead of time (at install or overnight).
8. **Load vs recompute decision**: load if `bytes / measured_disk_bw < tokens / measured_prefill_rate`. A 1.2 GB KV loads in ~1-2 s, versus ~9 min of prefill for 8k tokens on a 9B model [D].

**Store hygiene:** atomic publish (write temp, `fsync`, `rename`), per-block checksums (xxh3 / crc32c) verified lazily, `flock` for single writer, versioned headers, LRU with a byte quota and pinned entries, graceful `ENOSPC` (continue without cache), SSD wear awareness (store at checkpoints, not every turn). Compression (lz4) is probably not worth it for f16 KV, so test q8/q4 KV on disk first. Hybrids store far less (fewer attention layers).

**Disk cost** is ~150 KB/token f16 for Qwen3-4B-class dense models, so 100k tokens is ~15 GB. Quota management and quantized storage are not optional.

---

## 7. Layout so things can be dumped and mmapped back

- **Offsets, not pointers.** Every persisted structure is position-independent, FlatBuffers-style, little-endian only (refuse otherwise), with a magic, version and a header checksum.
- **Alignment:** 64 B (cache line, AVX), 4 KB (page-granular madvise per tensor), optionally 2 MB for anonymous huge-page arenas. Pad rows to cache-line multiples.
- **Sidecar "prepared model" file** keyed by hash(GGUF) + kernel layout version + ISA: the weights already repacked into the interleaved tile layout, tensors in execution order, page aligned. mmapped read-only, so it is shareable, evictable, free of commit charge, and loads in milliseconds. This removes the llama.cpp/ik `-rtr` cost (seconds-minutes, doubled RAM). Both your machines are AVX2, so one layout serves both.
- **Prepared tokenizer sidecar**: vocab, merges, prefix trie or hash table, in mmappable form. Parsing 150k tokens from GGUF arrays on every start is wasted time.
- **KV block layout**: uniform fixed-size blocks, each holding all layers, K then V per layer, head-major inside a block so one head's `QK^T` scans contiguous memory. A uniform pool gives a trivial free-list allocator, and eviction, spill and persistence all use the same unit. Block size is a tradeoff: small blocks give fine reuse and cheap growth, but large dense models have ~150 KB/token, so 32 tokens is ~4.7 MB per block.
- **Model registry file** (plain key=value, easy for C and Emacs Lisp): name, path, sidecar, chosen draft model, template, default sampler, stop tokens, thinking format, per-machine tuned settings.

---

## 8. Self-tuning: how far can it go

**Probe at runtime** (cache result in `~/.cache/hfcpu/<cpu-signature>.prof`, revalidate by hashing CPUID + memory size + kernel version):
- Features and topology: CPUID leaves 1/7/4/0xB/0x16, sysfs `cache/index*`, `sysconf`, macOS `sysctl hw.l2cachesize`, `hw.l3cachesize` etc.
- **Read bandwidth vs thread count** (STREAM-like) gives the decode thread count directly, and detects single-channel DIMMs.
- **Latency curve** (pointer chasing) gives cache sizes, associativity, TLB reach, and shows whether the Broadwell eDRAM L4 exists.
- Peak int8/FMA throughput at 1..N threads, and **sustained** throughput over 30-60 s (thermal throttling).
- Hyperthreading benefit, barrier latency, `mmap`/page-fault cost, sequential and random disk read (cold cache), `memcpy` bandwidth.

**Derive, don't ask the user:**
- GEMM blocking (MC, KC, NC, MR, NR) from an **analytical model** (Low et al. 2016) that lands within a few percent. A short local search then closes the gap. Cache the winner like FFTW "wisdom", keyed by (CPU signature, kernel version, shape class, quant type).
- Prefill ubatch size from L2/L3 fit of activations plus the weight panel.
- Decode thread count from the BW scaling curve. Prefill thread count from the compute scaling curve.
- KV block size from L1/L2. Prefetch distance and NTA on/off by microbenchmark.
- Speculation `k` from the measured ridge and online acceptance.
- Cache load-vs-recompute threshold from measured rates.
- Memory budgets from current availability.

**Learn over time:** an append-only log of (config, phase, tokens/s, thermal state). Pipelines repeat stages, which is ideal for online hill-climbing or a bandit over discrete knobs. Explore only on safe, low-stakes runs, use paired trials and medians, and always keep a known-good fallback. A roofline predictor (`max(bytes/BW, ops/peak)` x fitted efficiency) flags anomalies such as swapping, throttling or a bad kernel choice.

**Cannot be derived:** quality tradeoffs (quant choice, expert reduction, KV precision), sampling settings, and acceptance rates before seeing text.

---

## 9. The "building blocks" idea (HF-style model spec)

**Short answer: yes, but target a declarative spec plus a validation harness, not auto-generated C.**

**Why it is feasible:**
- HF `transformers` models are compositions of a small vocabulary: norm, attention, MLP, MoE, conv, SSM/linear-attention, embedding, head. Newer HF models are literally written as **diffs against other models** (`modular_*.py`), which is the structure we need. MiniCPM5 is plain Llama.
- llama.cpp already did this for C++ (`build_*` helpers). GGUF carries `general.architecture` and per-arch hyperparameters.

**Layers of a spec** (a data file, not code):
1. **Layer pattern**: per-layer block type (Qwen3.5 3x GDN + 1x attn, Ling 3x KDA + 1x MLA, LFM conv/attn).
2. **Block parameters**: norm (RMS/LN, eps, +1 offset), attention (GQA, QK-norm, RoPE variant interleaved vs split-half, partial rotary, YaRN/mrope, sliding window, sinks, output gate, softcap), MLP (SwiGLU/GeGLU), MoE (router softmax/sigmoid, top-k, normalized top-k, shared experts, expert bias, groups), recurrent (GDN/KDA/Mamba/conv), residual layout (pre/post/sandwich/parallel), embedding scaling, tied lm_head, muP scalings (MiniCPM family), MTP heads.
3. **Tensor-name mapping** and weight transforms (Q/K permutation, expert stacking, norm folding, conv reshape), copied semantically from llama.cpp's converter.
4. **Tokenizer and template descriptor**: pre-tokenizer type, special tokens, stop tokens, thinking tags, chat template.

**Where each input comes from:** `config.json` (architecture, dims, `layer_types`), `generation_config.json` (sampler defaults), `tokenizer_config.json` (chat template), the GGUF metadata (already contains most of this), and the HF `modeling_*.py` (structural ground truth).

**Automation tiers:**
- **Tier 0:** the GGUF's architecture string maps to a spec in our library, so it runs. This covers every llama.cpp-supported model.
- **Tier 1 (semi-automatic, the realistic goal):** read `modeling_*.py` / `modular_*.py` plus `config.json` and emit a spec draft. An LLM-assisted generator (me) can do this well, because the output is a small structured file and not arbitrary code. A human reviews it.
- **Tier 2:** trace the HF model with `torch.export` / `torch.fx` using a tiny dummy config to get an op graph, then pattern-match onto fused blocks. Unmatched ops fall back to a small portable set of reference primitives (elementwise, reductions, matmul, softmax, gather, conv1d, scan), so any model **runs slowly but correctly** while known blocks get fast kernels. This is the escape hatch for novel architectures. It is fragile for `trust_remote_code` models and dynamic control flow.
- **Not feasible:** fully automatic from arbitrary Python or custom CUDA kernels.

**The validation harness is what makes semi-automation safe.** Run the HF model on a tiny input, dump per-layer activations via hooks, run the engine, and bisect to the first diverging op. Same harness against llama.cpp output serves as the oracle for GGUF models. Without it, generated specs are untrustworthy.

**Interpreted spec vs generated C:** interpret the spec. Per-layer graph dispatch costs microseconds against millisecond tokens. Shape specialization (head_dim 64/80/96/128/256) can be macro-instantiated inside kernels. Generated C is only needed for a genuinely **new block type**, which is the rare case and does need human or LLM-written kernel code.

**Offline tooling in Python** (spec generator, harness, converter) seems unavoidable because the source of truth is Python. The runtime stays pure C. **Question for you in §12.**

**Minimal block set for your list:** RMSNorm, GQA with QK-norm, RoPE variants, SwiGLU, tied embeddings, muP scalings, short gated conv (LFM), gated delta rule (Qwen3.5), KDA + MLA (Ling), MoE with shared experts and sigmoid router, ViT + projector (VL). The first few cover MiniCPM5 and Qwen3-4B. The rest arrive roughly in this order.

---

## 10. Reliability and testing

- **GGUF is untrusted input.** The format has had parser memory-safety bugs in the wild [M]. Bound-check every count, offset and dimension against file size, cap allocations, reject overflow. Fuzz the parser.
- **mmap hazards**: a file truncated under a mapping gives `SIGBUS`. Handle it or verify with `fstat`, and consider a private snapshot of the sidecar.
- **I/O**: handle `EINTR`, short writes, `ENOSPC`, `EPIPE` (stdout closed: save cache, exit cleanly), `SIGINT` (checked between ops; abort the turn, keep the cache), and ignore `SIGPIPE`.
- **Test oracles**: scalar reference for every kernel, llama.cpp/ik logits as a conformance corpus (tolerance-based), HF activations for new specs, golden-token tests per quant type.
- **Fault injection**: allocator that fails the Nth call, `ulimit -v`, cgroup limits, a VM with `vm.overcommit_memory=2`. Run ASan/UBSan/Valgrind builds in CI, and long soak runs for thermal and leak behavior.
- **Observability**: per-op timers aggregated per token, optional Chrome-trace output, `perf_event_open` counters where available, a `doctor` request that prints the machine profile and recommended settings.

---

## 11. Process model and protocol

- **stdin/stdout/stderr only.** Suggested framing: length-prefixed header + body requests. Streaming events on stdout (token text, logprobs, done + stats). Structured `key=value` log lines on stderr with levels. Emacs can read these with its native S-expression or JSON parsers. Pass images and big blobs as file paths.
- **One-shot vs resident.** With the sidecar mmapped and the page cache warm, **process-per-stage startup is nearly free** (target ~100 ms). That gives perfect memory reclamation, crash containment and per-stage rlimits. A persistent stream mode (multiple requests, explicit `load`/`reset`) is still useful for chat. Supporting both costs little.
- **Cancellation**: poll stdin non-blocking between tokens for a control message, plus signal handling.
- **Request types**: `generate`, `tokenize`, `score` (perplexity/logprobs), `classify` (label logits), `embed` (if wanted), `warm`, `inspect`, `doctor`, `load`/`unload`, `cache gc`.
- **Chat templates**: GGUF embeds Jinja. A full Jinja engine in C is heavy. Options: let Emacs/Lisp apply templates, with the engine taking raw text and parsing special tokens, or ship a tiny Jinja subset interpreter, or store per-model templates in the registry. I would start with the first.

---

## 12. Open questions for you

1. **Python for offline tooling** (spec generation, validation harness, quant/sidecar helpers): acceptable, with the runtime pure C?
2. **Batched fan-out** (`n` continuations of one prefix in one request): in scope? It is the biggest cheap throughput win and shares kernels with speculation.
3. **Quality gates**: do you want `classify` / `score` / logprob outputs as first-class request types?
4. **Platforms**: macOS on the 2015 MBP is a hard constraint (no `O_DIRECT`, no strict overcommit, different `madvise` semantics, no thread affinity). Linux on the EliteBook? Is macOS the one that must be bit-for-bit identical to Linux?
5. **Vision** (LFM2.5-VL, Qwen3.5 multimodal): needed early, or after text is solid?
6. **Quantizing / converting**: is it fine to rely on llama.cpp's `llama-quantize` and the HF converter offline, with our engine only consuming GGUFs and building sidecars?
7. **Disk budget** for caches on each machine, and the disk type (NVMe vs SATA) on the EliteBook.
8. **Ornith-1.5-9B**: where is it published and in what architecture? I could not identify it.

---

## 13. Suggested phasing (for discussion)

0. **Foundations**: GGUF reader (hardened), type table, scalar reference kernels, tokenizer (BPE/SentencePiece), machine probe and profile file.
1. **MiniCPM5 / Qwen3 dense path**: Q4_K/Q8_0 AVX2 kernels, GEMV + GEMM regimes, flash attention, f16/q8 KV, block pool, stdin/stdout protocol, conformance against llama.cpp.
2. **Memory robustness**: static plan, fault injection, strict-overcommit CI, incremental context growth.
3. **Persistent prefix cache** and sidecar prepared models.
4. **Hybrid blocks**: short conv (LFM), GDN (Qwen3.5), recurrent checkpoints.
5. **Spec decoding** (n-gram first, then draft), constrained decoding, thinking budget, classify/score modes.
6. **MoE and Ling** (KDA, MLA), then vision.
7. **Spec generator + validation harness** for new architectures.
8. **Online tuning** from the run log.

---

Sources checked [V]:
- [DwarfStar 4 roadmap summary](https://pasqualepillitteri.it/de/news/2256/ds4-antirez-deepseek-v4-flash-inferenz-engine), [ds4 forks](https://github.com/24601/ds4)
- [Qwen3.5 small models (GDN 3:1, layer counts)](https://trilogyai.substack.com/p/deep-dive-qwen-35-brings-native-multimodality), [ApX Qwen3.5-0.8B](https://apxml.com/models/qwen35-08b)
- [LFM2.5 docs (vLLM recipes)](https://docs.vllm.ai/projects/recipes/en/latest/LiquidAI/LFM2.5.html), [LFM2.5 architecture overview](https://lilting.ch/en/articles/lfm-hybrid-architecture)
- [Ling-3.0-tiny](https://recipes.vllm.ai/inclusionAI/Ling-3.0-tiny)
- [MiniCPM5-2B](https://huggingface.co/OpenBMB/MiniCPM5-2B)
