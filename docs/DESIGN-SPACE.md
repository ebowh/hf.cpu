# hf.cpu: design-space exploration

Status: research notes, no code. Facts verified by web search are marked **[V]**. Everything else is from memory or derived by me, and is marked **[M]** (memory) or **[D]** (derived estimate). Treat [M] and [D] as hypotheses until checked against sources or measurements. Updated after your decisions on tooling (Perl), fan-out, resident protocol, platforms and disk.

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
| CPU | i7-5557U-class Broadwell [M] | **i7-8665U** Whiskey Lake [V]: 4C/8T, 14 nm, 15 W TDP (cTDP 10-25 W), 1.9 GHz base, up to 4.8 GHz single-core boost, 8 MB L3, AVX2 (no AVX-512) |
| Cores/threads | 2/4 | 4/8 |
| Peak DRAM BW | 29.8 GB/s (LPDDR3-1866, dual ch.) | 2 x 16 GB DIMMs, so dual-channel DDR4-2400 = 38.4 GB/s peak (the CPU supports DDR4-2400) [V] |
| Realistic read BW | ~18-22 GB/s | ~25-30 GB/s |
| Cache | L2 256 KB/core, L3 4 MB, **128 MB eDRAM L4** (Iris 6100, probe it) | L2 256 KB/core, L3 8 MB |
| int8 MAC throughput | ~150 GOPS effective with Q4 dequant overhead | ~2x, **but only at the power limit the laptop sustains**. A 15 W part will not hold all-core AVX2 at boost clocks; expect 2.5-3 GHz sustained [D] |

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
| Ornith-1.5-9B | Dense 9B, MIT license, 32k context, coding/agentic tuned. Described as "building upon Qwen3.5 and Gemma4 architectures", so probably a Qwen3.5-style GDN hybrid. GGUFs exist. The HF page is blocked from my sandbox, so I could not read `config.json`. **Confirm the layer pattern from the file** | [V] third-party listing only |
| Bonsai 2 (PrismML) | **Ternary** (-1/0/+1 weights, ~1.7 bits/weight) post-training quantization of Qwen3.8. The only member I could confirm is **Ternary-Bonsai-2-27B**: 27B params in ~5.9 GB (GGUF type `PTQ1_0`, ~5.93 GB) or ~7.2 GB (`PQ2_0`, a 2-bit packing), Apache 2.0, 262k ctx, reasoning, tool calling and image understanding, claimed 98.2% retention vs full precision (vendor claim). The collection page is blocked from my sandbox, so I do not know what other sizes exist or whether any are small enough to be a fast model for you. These are not mainline ggml types, so they need a fork's format spec (see §4.9). Architecture of Qwen3.8 itself is unconfirmed, probably a Qwen3.5-style GDN hybrid with a vision tower | [V] for 27B facts, rest [D] |
| LightOnOCR-2-1B | Document OCR VLM, ~1B total. Pixtral-style native-resolution ViT (initialized from the Mistral-Small-3.1 vision encoder), 2-layer GELU MLP projector with 2x2 spatial merge (4x fewer visual tokens), Qwen3-based text decoder. Image in, markdown/LaTeX out | [V] |
| Gemma 3 270M / 1B / 4B | Sandwich norms (pre and post), GeGLU, QK-norm, 5:1 local:global attention with sliding windows (270M: 15 of 18 layers local), dual RoPE bases (local vs global), tied embeddings. **262k vocab**: in the 270M model ~170M of ~270M params are embeddings. 4B adds a SigLIP vision tower | [V] 270M and vocab, rest [M] |
| Gemma 4 E2B / E4B (and Gemma 3n E2B/E4B) | "Effective" parameter counts via **Per-Layer Embeddings (PLE)**: 2.3B/4.5B effective out of 5B/8B total. Sliding window 512, local:global 4:1 (E2B) / 5:1 (E4B), last layer always global, GQA, 128k ctx. **KV sharing**: later layers reuse earlier layers' KV (E2B: 35 layers but 15 KV-producing; E4B: 42 layers, 24 KV-producing). Multimodal | [V] |

**Findings:**
- **MiniCPM5 being plain Llama means the day-one target is easy.** Start there.
- **Recurrent state is not a KV cache.** Qwen3.5 (GDN), Ling (KDA) and LFM2 (conv state) carry fixed-size state per layer. That state cannot be truncated back to an arbitrary prefix, so prompt-cache design must handle "checkpoint at specific token positions" (§6).
- **Hybrids are a gift on this hardware.** Qwen3.5-9B has 8 attention layers instead of 36, so KV traffic is cut about 4x. Long context becomes affordable. Hybrids are probably the best models for this project.
- **MoE is great for decode bandwidth.** Ling reads about 1.3B active params per token, so decode speed is comparable to a 1.3B model. It needs ~4.5 GB at Q4 resident. It breaks "verify k tokens for the price of one": the union of experts across k tokens approaches all of them.
- **MLA and KDA are exotic.** llama.cpp support for these may lag, so check GGUF availability before committing to them.
- **Gemma changes four engine decisions:**
  1. **Huge vocab means lm_head dominates small models.** 262k x hidden is a large fraction of every token's bytes for the 270M-1B models. Sparse lm_head (constrained decoding), fused top-k, and reduced-vocab drafting (FR-Spec) matter far more than for Qwen.
  2. **PLE tables are lookup tables, not matmul weights.** Only one row per token per layer is read. Keep them in a separate cold region of the sidecar, `MADV_RANDOM`, excluded from the resident-set budget, and let the page cache decide. This is the perfect mmap use case, and it is why E2B/E4B fit comfortably on 16 GB.
  3. **Sliding-window layers need ring-buffer KV** bounded at the window (512-1024), so their blocks are recycled as the window slides. Persistent-cache reuse then has the same "valid only at certain positions" caveat as recurrent state unless the old local KV is kept (§6).
  4. **Cross-layer KV sharing** means the layer-to-KV-slot map is part of the spec, and the KV block holds only the KV-producing layers.
- **Vision (LFM2.5-VL, Qwen3.5, Gemma):** a ViT encoder plus projector, usually shipped as a separate `mmproj` GGUF in llama.cpp. It needs image decode and preprocessing in C, and ViT prefill is expensive. Cache image embeddings on disk, content-addressed by image hash (§6).

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

### 4.4 Batched fan-out (in scope, decided)
Parallel sampling of `n` continuations (best-of-n, self-consistency, retries after a failed gate) shares the prefix, and decode of `n` sequences together costs far less than `n` x one sequence because weights are read once. The multi-vector kernel from §4.1 gives it nearly for free.
- Request parameter `--n N` (and later per-branch parameters such as seed or temperature lists). One prefill, then fork the KV at the shared prefix: block-pointer sharing with copy-on-write of the tail block, plus a state snapshot copy for recurrent layers.
- The same machinery serves **N different prompts** in one request (map over chunks with the same model), where shared-prefix detection dedupes whatever overlaps.
- Throughput grows with `n` until the compute ridge (§1), around n = 3-4 on the MacBook and a bit higher on the EliteBook. Beyond that it is linear. The planner caps the effective decode batch at the measured ridge and runs the rest in waves. Cap `n` by memory too (KV per branch).
- Output is demultiplexed with a branch index in each streamed event (§11).
- MoE caveat: the expert union across `n` branches grows toward all experts, so the gain is smaller for Ling.
- Interaction with speculation: both consume the same spare-compute budget. A batch of `n` branches with draft `k` costs `n x (k+1)` rows, so the planner picks one or the other (or splits the budget), not both at full size.

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
- **Vision is now core, not optional.** LFM2.5-VL, Gemma 3/4, Qwen3.5/3.8 (Bonsai 2, probably Ornith too) and LightOnOCR all need it. The encoder families differ (SigLIP-style, Pixtral-style with 2D RoPE and native resolution, Qwen-VL-style with window attention and a patch merger, Gemma's own), so the block library gains: patch embedding (a conv), ViT blocks with learned or 2D-RoPE positions, windowed/block-diagonal attention, pixel-shuffle or spatial-merge projectors, and MLP projectors.
- **The ViT is the expensive part.** A native-resolution page image at 16 px patches is thousands of patches, and a ~400M-param ViT is on the order of a TFLOP per thousand patches, plus attention that is quadratic in patch count [D, very rough]. On the MacBook that means tens of seconds to minutes per page image. So: cache embeddings (above), expose a resolution cap, and treat ViT prefill as a first-class cost in the planner and in the progress output.
- PDF rasterization and image resizing are pipeline-side jobs (`pdftoppm`, Perl, ImageMagick), not engine jobs. The engine decodes PNG/JPEG only.
- **OCR workloads (LightOnOCR) are decode-bound and long:** a dense page is 1-3k output tokens. The 1B decoder at Q8/Q4 gets maybe 25-30 t/s on the MacBook [D], so ~1-2 min per page plus the ViT. **Fan-out helps here directly**: N pages through the same model decode as a batch, which shares weight reads and raises throughput until the compute ridge (n of about 3-4). Prompt lookup does not help (the output is not in the prompt), but a repetition-loop detector is needed, since OCR models can degenerate into loops, and that is a natural quality gate.

### 4.9 Ternary models (Bonsai 2)
- **What ternary buys:** bytes. 27B at ~1.7 bits/weight is ~5.9 GB, which fits both machines, and decode reads ~5.9 GB per token, so ~3-5 tokens/s is the bandwidth ceiling [D]. **What it does not buy:** prefill compute. A 27B dense (if it is dense) model still performs ~54 GFLOP-equivalents per token, so prefill is roughly 3x slower than the 9B models: on the order of 2-3 tokens/s on the MacBook and 5-7 on the EliteBook [D]. A 1000-token prompt is minutes. So Bonsai 2 27B is a **prompt-cache-mandatory, latency-tolerant pipeline stage**, a good fit for "think hard once" gates, not interactive chat.
- **Formats:** mainline ggml has `TQ1_0` (~1.69 bpw, base-3 packing of 5 trits per byte) and `TQ2_0` (~2.06 bpw). `PTQ1_0` (~1.7 bpw) and `PQ2_0` (~2 bpw) look like same-family variants in PrismML's fork, but I cannot see their spec. **Need the format definition** (block size, scale type and layout, which tensors stay higher precision such as embeddings, lm_head, norms) from the fork's source or the GGUF itself. Perl can dump the tensor-type table and block bytes from a downloaded file, so this is recoverable without the web page.
- **AVX2 kernel idea:** the multiply-free trick is to keep weights as unsigned {0,1,2}, use `vpmaddubsw` (u8 x s8) exactly as for Q4, and subtract the precomputed activation block sum (`sum(a*(w-1)) = sum(a*w) - sum(a)`). Decoding 5-trits-per-byte with multiply-and-shift is cheap in SIMD. This puts ternary at roughly Q4's ALU cost per weight, so ternary is **not** faster for prefill. `vpsignb` is an alternative (sign-apply) but needs a signed weight byte. LUT/bit-serial methods (T-MAC, bitnet.cpp) are the research path, and worth a measured trial only after the baseline kernels exist.
- **Activation quantization** is per-token int8 with block scales (BitNet-style models are trained for this, post-training ternarized ones usually tolerate a finer block scale). Whatever PrismML's fork does is the numerical reference.
- **Memory:** 5.9 GB of weights plus KV for a 262k-capable hybrid fits 16 GB comfortably if the model is a GDN hybrid with few attention layers. Use incremental context (§5).

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
- **Rewrite rules for existing GGUFs ("piggyback")**: the sidecar builder loads a published GGUF and writes our layout. Default is a **lossless re-layout**: every transform must be bit-exact with respect to what llama.cpp would compute, so the existing quantizations (Unsloth, bartowski, vendor GGUFs) are reused as-is. Allowed lossless transforms: interleave rows into SIMD tiles, fuse QKV and gate/up into single tensors, pre-transpose, drop tensors the run does not need (vision tower for text-only runs, MTP heads), precompute derived constants (e.g. exp of a stored log-decay), reorder tensors into execution order, split cold lookup tables (PLE, embeddings) into their own region. **Lossy changes are a separate, explicit step** (re-quantize with `llama-quantize` plus imatrix on the rented machine, or convert between types), never silent. The sidecar records the source GGUF hash and the list of transforms applied.

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
- **Tier 2 (restored, run on a rented machine):** trace the HF model with `torch.export` / `torch.fx` on a tiny random-init config of the same architecture, dump the op graph and module tree as JSON, and pattern-match it onto our fused blocks (done in Perl, offline). Unmatched ops fall back to a small set of reference primitives in the engine (elementwise, reductions, matmul, softmax, gather, conv1d, scan), so a model with a novel composition **runs slowly but correctly** while known blocks get fast kernels. Fragile for `trust_remote_code` models and data-dependent control flow, so it is an accelerator for writing specs, never the only path.
- **Not feasible:** fully automatic from arbitrary Python or custom CUDA kernels.

**Where I push back on "Perl only":** Python is the right tool for exactly one job, the **capture kit**, because the ground truth lives in PyTorch. It runs rarely, on a rented machine, and its outputs are plain files (JSON plus raw little-endian float arrays) that Perl and C consume. Perl stays the daily driver for everything else (inspect, lint, orchestrate, diff, cache management). Nothing in the engine, its build, or day-to-day workflow ever needs Python. I will write the capture scripts, and you only run them. If you would rather not touch Python at all, the llama.cpp oracle (§10) covers all models it supports, at lower fidelity for new architectures.

**Capture kit (one run per model, on a rented box with enough RAM, ideally a GPU for speed):**
1. Module tree + `config.json` + tensor name/shape/dtype table (to build the spec's tensor map).
2. `torch.export` / FX op graph from a tiny random-init instance.
3. Per-module activation fixtures for a few fixed prompts on the real weights (fp32 or bf16): embeddings, per-layer inputs/outputs, attention and recurrent state, final logits and top-k.
4. Greedy generation traces (e.g. 256 tokens with logprobs), including a long-context one, as end-to-end conformance.
5. Tokenizer vectors: `encode`/`decode` for a large corpus of edge-case strings (Unicode, whitespace, special tokens), so our hand-written pre-tokenizer is tested against HF's.
6. **Chat-template golden vectors, plus the Jinja AST** (`jinja2.Environment().parse`) exported as JSON. Our engine then needs a tiny interpreter for the handful of node types real templates use, validated against the vectors, instead of a full Jinja. This is a better answer than the client-side templating I suggested earlier.
7. Vision: preprocessed pixel tensors, per-block ViT outputs, projector outputs for a few images.
8. Recurrent layers: states at chosen positions, to validate GDN/KDA/conv checkpoints.

**Other rented-machine jobs, same pattern:**
- **Quantization sweeps** with `llama-quantize` plus an importance matrix, from the original BF16 weights: candidate quants per model, then KL-divergence and perplexity against BF16. That picks the best file per model per machine by quality per byte. ik_llama's own quant types (IQ4_KS and friends) can only be produced this way, since few are published as GGUFs.
- **Speculation studies**: acceptance rates for candidate draft/target pairs (Qwen3.5-0.8B for 9B, LFM2.5-350M for larger LFMs) on text resembling your stages.
- **Oracle logits** from llama.cpp / ik_llama.cpp for every shipped GGUF, as test fixtures.
- Any experiment too heavy for the target laptops (layer-skip profiles, KV-quant quality curves).

**What Perl tooling can and cannot do** (core modules only: `JSON::PP`, `Digest::SHA`, `Time::HiRes`, `File::Temp`, `Getopt::Long`, `pack`/`unpack`; avoid CPAN):
- **Can:** parse `config.json`, `generation_config.json`, `tokenizer_config.json`, the safetensors header (JSON after an 8-byte length), and GGUF metadata/tensor tables. Emit and lint spec files. Print the `inspect` coverage report (which blocks a model needs vs what the engine has). Build sidecars (repack is a byte-shuffling job, but it is heavy for Perl at multi-GB scale, so **the repack tool should be a C mode of the engine itself**, with Perl orchestrating). Drive test runs, diff logits files, bisect divergences, manage the cache directory, and generate the machine-profile report.
- **Cannot:** parse arbitrary Python source reliably, or run HF models. Drafting a spec from `modeling_*.py` is done by me reading the source as text, helped by the Tier 2 op-graph dump, and the spec is reviewed by hand. Perl lints it.
- **Ground truth for validation:** (1) llama.cpp/ik_llama.cpp binaries as the oracle for every architecture they support; (2) our scalar reference kernels as the oracle for the SIMD kernels; (3) the capture-kit fixtures from the rented machine for HF-level truth. All three are plain files or binaries, so Perl and C consume them without PyTorch.

**The validation harness is what makes semi-automation safe.** Feed a tiny prompt, dump per-layer activations from the engine (a `--dump-layers` debug mode), diff against the oracle, and bisect to the first diverging op. Without it, hand- or LLM-written specs are untrustworthy.

**Interpreted spec vs generated C:** interpret the spec. Per-layer graph dispatch costs microseconds against millisecond tokens. Shape specialization (head_dim 64/80/96/128/256) can be macro-instantiated inside kernels. Generated C is only needed for a genuinely **new block type**, which is the rare case and does need human or LLM-written kernel code.

**Offline tooling is Perl** (decided), runtime and heavy byte-crunching tools are C. See the capability split above.

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

**Decided: one resident process that loads once and serves many requests for a long time.** stdin/stdout/stderr only. Two input modes, chosen at startup:

**Mode A: argv-lines.** Each stdin line is one request, written as command-line parameters (`--n 4 --temp 0.7 --prompt-file /tmp/p.txt`).
- Parsed with POSIX-shell-style word splitting only: single quotes, double quotes with `\"` and `\\`, backslash. **No expansion** (no `$`, globs, backticks), so request lines are safe to build from untrusted text.
- The same option parser serves the real command line and each request line. Startup options are the defaults, and request options override them per request. This keeps one flag vocabulary for the whole engine.
- Prompt text should normally come via `--prompt-file PATH` (or `--prompt-fd`, or `--prompt-inline-hex`), because a newline-terminated line cannot carry arbitrary prompts. A short `--prompt 'text'` is allowed, with `\n` escapes.
- Fits shell scripts and `perl` pipelines well: `print $fh "--model qwen3 --n 3 --prompt-file $f\n"`.

**Mode B: record stream.** stdin carries request bodies (prompts) separated by a single `RECORD_SEPARATOR` byte.
- Suggested value: ASCII 0x1E (RS), configurable with `--record-sep BYTE`. Prompt text containing that byte is an error (or escapable via a configurable escape byte). Using a length-prefixed variant is an alternative for fully binary-safe bodies.
- Per-request parameters come from the startup command line (applied to every record). To override per record without a second channel, an optional **header**: a record may start with an argv-style parameter line terminated by ASCII 0x1F (UNIT_SEPARATOR), then the body. No 0x1F means the whole record is the body and all params are defaults.
- Fits Emacs well: it concatenates buffer text + `\x1e` and `process-send-string`.

**Common to both modes**
- **Output** mirrors input framing. In record mode, each response ends with the same RS byte. In argv mode, each response ends with a line `\n`-terminated trailer event, and a request id may be passed with `--id`. Events on stdout: token text deltas, logprobs, branch index (for `--n`), `done` with stats (tokens, prefill and decode rates, cache hit length, KV bytes, timings), or `error` with a code. Log and progress lines go to stderr as `level key=value ...` (prefill progress, cache hit/miss, memory plan, thrash warnings). Emacs can parse either stream with native readers. A machine-readable stats line per request lets pipeline controllers evaluate gates and tuning.
- **Requests are processed strictly one at a time, in order.** EOF on stdin ends the session cleanly (flush, save cache, exit 0).
- **Cancellation**: between tokens (and between layers during long prefill) the engine polls stdin non-blocking for a control line (`!cancel`, `!id N`) and checks a signal flag (`SIGINT` cancels the current request but keeps the process alive; `SIGTERM` exits after saving). In record mode control lines use a reserved first byte so they cannot be confused with prompts. This is an open detail, listed in §12.
- **Model switching inside the process**: `--model NAME` per request. The engine keeps an LRU of resident (mmapped) models under a memory budget. An unneeded model's pages are released with `MADV_DONTNEED`/`fadvise` (§5), so a pipeline that alternates 4B and 9B models costs only page-cache residency, not reload parsing, thanks to sidecars. `--model` may name a registry entry or a path.
- **Long-running hygiene**: no heap growth per request (arena reset per request), a leak counter reported in `doctor`, periodic cache GC, re-stat of model files so a replaced GGUF is detected and the sidecar rebuilt, `SIGHUP` re-reads the registry and settings, and a per-request watchdog (`--timeout`). Crash recovery is the supervisor's job: the persistent cache is crash-safe (atomic publishes), so a restart loses nothing but process warmth.
- **Process-per-stage still works** for free (stdin EOF after one request) and gives hard isolation when wanted.
- **Request types**: `generate`, `tokenize`, `score` (perplexity/logprobs), `classify` (label logits), `embed` (if wanted), `warm`, `inspect`, `doctor`, `load`/`unload`, `cache gc`. Selected with `--op NAME`, default `generate`.
- **Chat templates**: GGUF embeds Jinja and a full Jinja engine in C is heavy. Plan: the capture kit exports each model's Jinja AST plus golden rendering vectors (§9), and the engine ships a small interpreter for the few node types real templates use, validated against the vectors. Until then, the client applies templates and the engine takes raw text with special-token parsing.

---

## 12. Decisions so far and remaining questions

**Decided (by you):**
- Runtime and byte-heavy tools are C. Everyday offline tooling is Perl. Your Perl preference is a preference, not a ban: Python is acceptable for the rarely run capture kit on rented machines (§9).
- **Batched fan-out** is in scope (§4.4).
- Resident, load-once, many-requests process with two stdin modes: argv-lines and RS-separated records (§11).
- Platforms: Linux on the EliteBook, macOS on the MacBook, FreeBSD should also work (§14).
- Hardware: EliteBook i7-8665U (4C/8T, 8 MB L3), 2 x 16 GB DDR4 (dual channel), ~512 GB NVMe. MacBook 256 GB SSD. Cache budget is a setting (§15).
- `llama-quantize` is fine for offline quantizing. Reuse existing GGUFs and rewrite them into our layout (§7). Rented machines are available for one-off jobs.

**Decided (by me, say if you disagree):**
- macOS, Linux and FreeBSD behave **identically in results** (bit-identical logits on the same ISA) and differ only in the platform layer (§14).
- Sidecar building is a C mode of the engine, orchestrated by Perl, and defaults to **lossless re-layout** only (§7).
- Vision is **core**, no longer a later phase: five of your models need it (§4.8).
- Chat templates: capture the Jinja AST on the rented machine and interpret a small subset in the engine, validated against golden vectors (§9, §11).

**Still open:**
1. **Bonsai 2 members**: I could only confirm the 27B. Which sizes are in the collection? Paste the list (HF and prismml.com are blocked from my sandbox). The 27B is ~5.9 GB but compute-heavy, so a slow, cache-mandatory stage, not a fast one (§4.9).
2. **PrismML format spec** for `PTQ1_0` / `PQ2_0`: the fork's source or a GGUF header dump (I can write the Perl dump tool first).
3. **Ornith-1.5-9B `config.json`** (layer pattern) and **Qwen3.8 `config.json`**.
4. **Quality-gate request types**: are `classify` / `score` / logprob outputs wanted first-class (my assumption: yes)?
5. **Control channel in record mode**: reserved leading byte for `!cancel`, or a second file descriptor (`--control-fd 3`)? The fd is cleaner but less portable to simple pipelines.
6. **Which Gemma sizes**: 270M, 1B, 4B, Gemma 3n/4 E2B/E4B, all?
7. **MacBook details**: exact CPU (I assume i7-5557U from "dual-core i7, 4 MB L3") and macOS version (Monterey 12 is the last official release for a 2015 MBP, as I recall).
8. **Rented machine**: when you are ready, I will write the capture kit and a runbook with the exact commands. What GPU or RAM class is easy for you to rent? (The 27B needs ~54 GB in bf16 to trace, the others far less.)

---

## 14. Portability: Linux, macOS, FreeBSD

**Policy: identical results, platform-specific mechanisms.** All numerical code is the same C on every OS. Hot paths use our own `exp`/`tanh`/`erf` polynomials (no libm differences), fixed reduction order (§4.1), and no data-dependent threading order, so logits are **bit-identical across OSes and thread counts on the same ISA**. Sidecar and cache files are little-endian, offset-based and portable between the machines as long as the ISA layout tag matches (AVX2 on both). That also lets you warm a cache on one machine and copy it to the other.

**A thin platform abstraction layer (PAL)** is the only place `#ifdef` lives:

| Concern | Linux | macOS | FreeBSD |
|---|---|---|---|
| CPU features, cache sizes | CPUID first (`cpuid.h`), sysfs `cache/index*` as cross-check | CPUID, `sysctl hw.l2cachesize` / `hw.l3cachesize` / `hw.cachelinesize` | CPUID, `sysctl hw.*` (limited) |
| Memory availability | `/proc/meminfo` (`MemAvailable`, `CommitLimit`), cgroups | `host_statistics64` / `vm_stat` (free + inactive), no strict commit | `sysctl vm.stats.vm.*`, `vm.overcommit` flags, `RLIMIT_*` |
| Overcommit semantics | strict mode 2 possible, rules in §5 | no strict accounting, compression and swap instead | optional accounting via `vm.overcommit`, swap-backed. **Verify** the actual semantics when porting |
| mmap + advice | `madvise` (`WILLNEED`, `DONTNEED`, `RANDOM`, `HUGEPAGE`, `COLD`/`PAGEOUT`), `MAP_POPULATE` | `madvise` subset, no `MAP_POPULATE`; superpages via `VM_FLAGS_SUPERPAGE_SIZE_2MB` | `madvise`, `MAP_ALIGNED_SUPER` |
| File cache control | `posix_fadvise`, `O_DIRECT`, `sync_file_range` | `fcntl(F_NOCACHE)`, `F_RDADVISE`, no `posix_fadvise` | `posix_fadvise`, `O_DIRECT` |
| Durability | `fdatasync` | **`fcntl(F_FULLFSYNC)`** (plain `fsync` does not flush the drive cache) | `fsync` |
| Preallocation | `fallocate` / `posix_fallocate` | `fcntl(F_PREALLOCATE)` + `ftruncate` | `posix_fallocate` |
| Thread pinning | `sched_setaffinity` | affinity tags are only hints. Measure whether it matters; likely skip | `cpuset_setaffinity` |
| Low-priority mode | `nice`, `SCHED_IDLE` | QoS class background (`pthread_set_qos_class_self_np`) | `nice`, `rtprio` idle |
| Locks | `flock` / `fcntl` locks | `flock` | `flock` |
| Futex / sleeping barriers | `futex` | `__ulock_wait` or pthread cond (Apple's private API, avoid) | `_umtx_op` or pthread cond |

**Everything above has a portable fallback** (plain `pread`, `pthread` cond, no advice) so a new platform is a stub PAL first and an optimized one later. The fallback must be correct, never silently wrong.

**Build:** plain C99, POSIX `make` subset (BSD make and GNU make both work, so no GNU-isms), `cc` = clang or gcc, per-ISA files compiled with explicit `-m` flags selected by a tiny `make` variable or script. The build must **not** need Perl. Only the offline tools do (Perl ships with macOS, and on FreeBSD is `pkg install perl5`). No third-party libraries at all.

**Per-platform notes:**
- **macOS on the 2015 MBP:** Intel x86_64. Last officially supported macOS is, as I recall, Monterey, so keep the toolchain compatible with Xcode 14-era clang and avoid newer SDK-only APIs. Thermal throttling is severe, and macOS gives less control, so the calibrator's sustained-throughput test and the "good-citizen" mode matter more here. Apple's compressed memory can make `RSS` misleading.
- **FreeBSD:** similar to Linux for mmap and fadvise, no sysfs, so rely on CPUID plus `sysctl`. ZFS is common on FreeBSD, and ZFS's ARC **double-caches** mmapped files and does not play well with `mmap` + page cache. Prefer UFS (or accept a larger memory reserve) for the weights/cache directory. Not an issue for your two machines, but a thing to note when it is tested.
- **Linux (EliteBook):** strict overcommit testing happens here (VM or cgroup), plus perf counters. The same tests run on macOS where the mechanisms exist.
- **CI matrix:** Linux x86_64 (primary), macOS x86_64, FreeBSD x86_64 in a VM, and a scalar-only build to prove the fallback.

---

## 15. Disk budget for the persistent cache

**Sizing (rough):** a dense 4B at f16 KV is ~150 KB/token, so 100k cached tokens is ~15 GB, or ~4-8 GB at q8/q4 KV. Hybrids (Qwen3.5, Gemma with sliding windows and KV sharing, LFM2.5) store several times less. A 512 GB NVMe or 256 GB SSD can hold tens to hundreds of sessions. On the MacBook, model files and sidecars take 15-40 GB already, so its cache budget is the tighter one.

**Parameters:** `--cache-dir PATH`, `--cache-quota SIZE` (e.g. `80G`), `--cache-min-free SIZE` (never let the filesystem drop below this, checked with `statvfs` before every write), per-model sub-quotas, a TTL, and pinned entries (e.g. system-prompt caches for pipeline stages). The same settings also come from the registry/config file. On exceeding quota: evict least-recently-used unpinned blocks first, always as whole atomic units.

**Dedicated space (optional, yours to decide):**
- **Linux:** a separate partition, an LVM volume, or a loopback file formatted as ext4/XFS. A dedicated filesystem gives a **hard quota**, keeps cache churn from fragmenting your main data, and lets you mount with `noatime`. XFS project quotas or btrfs subvolume quotas enforce a limit without a partition.
- **macOS:** APFS lets you add a volume with a quota (`diskutil apfs addVolume ... -quota`) without repartitioning, which is the macOS equivalent of "create a partition". Disable Spotlight and Time Machine indexing for that volume.
- **FreeBSD:** a UFS slice, or a ZFS dataset with `quota`/`reservation` (but see the ARC note above).
- The engine just needs a directory and a quota number. The partition or volume is an operations choice and never required.

**Behavior:** large sequential writes in block units (SSD-friendly), content-addressed dedupe (§6), write at checkpoints, not every turn, `fallocate` where it exists, and avoiding `atime` updates. NVMe wear is a non-issue at this write volume. Expose `cache stat` / `cache gc` / `cache pin` operations (Perl tools plus the engine's `--op` requests).

---

## 16. Suggested phasing (for discussion)

0. **Foundations**: platform abstraction layer (Linux + macOS + FreeBSD stubs), argv-line and record-stream protocol, GGUF reader (hardened), type table, scalar reference kernels, tokenizer (BPE/SentencePiece), machine probe and profile file.
1. **MiniCPM5 / Qwen3 dense path**: Q4_K/Q8_0 AVX2 kernels, GEMV + GEMM regimes, flash attention, f16/q8 KV, block pool, stdin/stdout protocol, conformance against llama.cpp.
2. **Memory robustness**: static plan, fault injection, strict-overcommit CI, incremental context growth.
3. **Persistent prefix cache** and sidecar prepared models.
4. **Hybrid blocks**: short conv (LFM), GDN (Qwen3.5), recurrent checkpoints.
5. **Spec decoding** (n-gram first, then draft), constrained decoding, thinking budget, classify/score modes.
6. **MoE and Ling** (KDA, MLA), then vision.
7. **Vision** (ViT blocks, image decode, embedding cache) moves up next to step 4, because five of your models need it. LightOnOCR is a good first VLM (1B decoder, fan-out over pages).
8. **Spec linter, `inspect` coverage report and validation harness** (Perl) for new architectures. Gemma (sandwich norms, SWA ring KV, PLE tables, KV sharing, 262k vocab) is a good test of the spec vocabulary.
9. **Online tuning** from the run log.

---

Sources checked [V]:
- [DwarfStar 4 roadmap summary](https://pasqualepillitteri.it/de/news/2256/ds4-antirez-deepseek-v4-flash-inferenz-engine), [ds4 forks](https://github.com/24601/ds4)
- [Qwen3.5 small models (GDN 3:1, layer counts)](https://trilogyai.substack.com/p/deep-dive-qwen-35-brings-native-multimodality), [ApX Qwen3.5-0.8B](https://apxml.com/models/qwen35-08b)
- [LFM2.5 docs (vLLM recipes)](https://docs.vllm.ai/projects/recipes/en/latest/LiquidAI/LFM2.5.html), [LFM2.5 architecture overview](https://lilting.ch/en/articles/lfm-hybrid-architecture)
- [Ling-3.0-tiny](https://recipes.vllm.ai/inclusionAI/Ling-3.0-tiny)
- [MiniCPM5-2B](https://huggingface.co/OpenBMB/MiniCPM5-2B)
- [Ornith 1.5 9B (third-party listing)](https://featherless.ai/models/ornith-ai/Ornith-1.5-9B)
- [Bonsai 2 27B coverage](https://www.mindstudio.ai/blog/ternary-bonsai-2-27b-2bit-model), [PrismML Ternary Bonsai 2 27B listing](https://openrouter.ai/prism-ml/ternary-bonsai-2-27b)
- [LightOnOCR paper](https://arxiv.org/html/2601.14251)
- [Core i7-8665U specs](https://en.wikichip.org/wiki/intel/core_i7/i7-8665u)
- [Gemma 4 deep dive](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4), [KV sharing and PLE notes](https://sebastianraschka.com/llm-architecture-gallery/kv-sharing/), [Gemma 3 270M](https://en.immers.cloud/ai/google/gemma-3-270m/)
