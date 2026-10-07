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
| **Qwen2.5 (0.5B, 1.5B, 3B, 7B, plus 14B/32B; Coder, Math, DeepSeek-R1-Distill-Qwen)** | **Top priority, per you.** Plain dense Qwen2 architecture (`qwen2` in llama.cpp): RMSNorm, GQA with **bias on Q/K/V** (the one difference from Llama), SwiGLU, RoPE theta 1e6, **no QK-norm** (Qwen3 added it), 32k context (more with YaRN on larger sizes), vocab ~152k. Tied embeddings on 0.5B/1.5B/3B, untied on 7B+. From memory the shapes are: 0.5B 24 layers hidden 896 (14Q/2KV, head_dim 64); 1.5B 28 layers hidden 1536 (12Q/2KV); 3B 36 layers hidden 2048 (16Q/2KV); 7B 28 layers hidden 3584 (28Q/4KV). Aggressive GQA means **a tiny KV cache** (3B ~36 KB/token, 7B ~57 KB/token). The 0.5B/1.5B models are the classic draft models for the 3B/7B (same tokenizer). A huge fine-tune ecosystem (Coder, Math, R1-distills) all share this architecture | [M], confirm with `inspect` |
| Qwen3-4B | Dense GQA with QK-norm | [M] |
| Qwen3.5 (0.8B, 2B, 4B, 9B) | Hybrid: Gated DeltaNet (linear attention) and full attention at 3:1. 0.8B: 18 GDN + 6 attn layers. 9B: 24 GDN + 8 attn. Natively multimodal. Context 262k | [V] |
| LFM2.5 (350M, 1.2B, 2.6B...) | Gated short convolutions interleaved with GQA. A VL line (450M/1.6B/3B) and an 8B-A1B MoE variant also exist | [V] |
| Ling-3.0-tiny | MoE 7.9B total, 1.3B active. 24 layers, 3x Kimi Delta Attention (KDA) then 1x Multi-head Latent Attention (MLA). 8-of-128 routed experts plus 1 shared. 256k ctx | [V] |
| Ornith-1.5-9B | **Confirmed from your `config.json`: it is a Qwen3.5-9B-class model** (`Qwen3_5ForConditionalGeneration`). 32 layers = 24 Gated-DeltaNet (linear) + 8 full attention (every 4th). Hidden 4096, MLP 12288 (SwiGLU), full attention 16 Q heads / **4 KV heads**, head_dim 256, **output gate** on attention, **partial RoPE 25%** (64 of 256 dims), interleaved **mRoPE** sections [11,11,10], theta 1e7. Linear layers: conv kernel 4, 16 key heads and 32 value heads, both dim 128, fp32 state. Vocab 248,320, **untied** lm_head, 1 **MTP** layer, vision tower (27 layers, hidden 1152, patch 16, 2x2 merge, temporal patch 2). Config says 262k positions, the model card says 32k. Same spec as Qwen3.5-9B, so one spec file covers both | [V] config |
| Bonsai 2 (PrismML), **27B** | Ternary post-training quantization of Qwen3.8, **`qwen3_5` family** per your `config.json` (`model_type: prism_hadamard_qwen35`). 64 layers = 48 Gated-DeltaNet + 16 full attention, hidden 5120, MLP 17408, attention 24 Q / 4 KV heads, head_dim 256, output gate (`swish`), 48 value heads / 16 key heads for the linear layers, no MTP, vocab 248,320, untied. **Hadamard-rotated**: the config lists every projection and the lm_head with a Hadamard `block: 1024`. Apache 2.0, 262k ctx, image understanding. The only confirmed size. Formats (read from the fork's source, see §4.9): **`PQ2_0` (group 128, fork-private ggml id 142, ~2.125 bpw, ~7.2 GB, the fork's default download), `PTQ1_0` (group 128, id 143, 1.75 bpw, ~5.9 GB), and `Q2_0` (group 64, id 42, 2.25 bpw), which loads in upstream llama.cpp** (the model card points upstream users to `Ternary-Bonsai-27B-Q2_g64.gguf`). Needs the fork `PrismML-Eng/llama.cpp`, branch `prism`, for the g128 types | [V] config + fork source |
| LightOnOCR-2-1B | **Confirmed from your `config.json`.** Text decoder is plain **Qwen3-0.6B-shaped**: 28 layers, hidden 1024, MLP 3072, 16 Q / 8 KV heads, head_dim 128, QK-norm, tied embeddings, vocab 151,936, theta 1e6, 16k positions. Vision is **Pixtral**: 24 layers, hidden 1024, MLP 4096 (gated SiLU), 16 heads x head_dim 64, patch 14, **image_size 1540** (so up to 110x110 = **12,100 patches per page**), 2D RoPE theta 1e4, projector with 2x2 spatial merge (so up to 3,025 image tokens). Image in, markdown/LaTeX out | [V] config |
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

### 2.1 Worked numbers from the two configs you supplied [D, derived from config.json, verify by measurement]

**Ornith-1.5-9B / Qwen3.5-9B (24 GDN + 8 attention layers):**
- **Parameters**: embeddings and lm_head are ~1.0B each (248,320 x 4096), so **~2B of the 9B are the two vocab tables**. The MLPs are ~4.8B, linear-attention layers ~1.6B, full-attention layers ~0.5B.
- **Decode bytes per token at Q4_K-ish**: ~4.0 GB of layer weights plus ~0.8 GB for the Q6 lm_head, so **~4.8 GB**. Bandwidth ceiling: **~4 t/s on the MacBook, ~5-6 t/s on the EliteBook**. The lm_head alone is ~15% of the traffic, so the sparse lm_head and fused top-k tricks (§4.5) pay off here too.
- **Prefill**: ~16 GFLOP per token (embedding lookup is free, lm_head only on the last token). Ideal ~11 t/s on the MacBook and ~20 on the EliteBook, real maybe 60-70% of that.
- **KV cache is tiny**: only 8 layers x 2 x 4 KV heads x 256 dims x 2 B = **32 KB/token** (a dense Qwen3-4B is 147 KB/token). 32k tokens is ~1 GB at f16, and even the full 262k is ~8.6 GB, or ~4.3 GB at q8.
- **Recurrent state per checkpoint**: 24 layers x 32 value heads x 128 x 128 x 4 B (fp32) = **50 MB**, plus ~2.4 MB of conv state, so **~52 MB per checkpoint**. That is the KV of ~1,600 tokens, so checkpoint policy is a real cost trade (§6): a checkpoint every ~1-2k tokens is reasonable, not every turn of a short chat.
- **GDN does not need the fancy chunked algorithm here.** The per-token recurrence costs ~75 MFLOP over all linear layers, against ~16 GFLOP of matmul per token. The recurrence is parallel across the 32 value heads (that is the threading axis) and the 2 MB per-layer state stays in L3. **Implement the simple sequential recurrence first**; the chunked/WY formulation is a GPU optimization that buys little on 2-4 CPU cores.
- **MTP head**: the config has one multi-token-prediction layer, which can serve as a free draft head (§4.3). Its draft step still pays the 0.8 GB lm_head, so the net gain is modest, to be measured.
- **One spec file serves Qwen3.5 (0.8B-9B), Ornith-1.5-9B, and likely Qwen3.8/Bonsai 2** (same `qwen3_5` family), which makes this the most valuable architecture to get right.

**Qwen2.5 family (top priority) [D, shapes from memory]:**
- **Decode bytes per token at Q4_K-ish** and bandwidth ceilings (MacBook ~20 GB/s, EliteBook ~27 GB/s): 0.5B ~0.4 GB (compute and overhead bound, 40+ t/s), 1.5B ~1.0 GB (~20 / ~27 t/s), 3B ~1.9 GB (~10 / ~14 t/s), 7B ~4.7 GB (~4 / ~5.5 t/s), 14B ~9 GB (~2 / ~3 t/s; fits both machines but is tight on the 16 GB MacBook next to macOS).
- **KV is tiny** thanks to 2-4 KV heads: ~29 KB/token (1.5B), ~37 KB (3B), ~57 KB (7B). 32k tokens of 7B is only ~1.9 GB at f16.
- **Why it is the right first target:** one plain, widely used architecture, the only novelty versus Llama is Q/K/V bias, no recurrent state, and the 0.5B/1.5B models give a ready-made **draft-model pair** for the 3B/7B so speculation can be tested early (same tokenizer). Qwen3, MiniCPM5 and the Qwen2.5-derived fine-tunes then come almost free.

**Bonsai 2 27B (from your config) [D]:**
- ~24.3B parameters in the layers (MLPs 17.1B, 48 linear-attention layers 5.5B, 16 full-attention layers 1.7B) plus ~2.5B in the two vocab tables.
- **KV**: 16 full-attention layers x 2 x 4 KV heads x 256 x 2 B = **65 KB/token**, so 32k tokens is ~2.1 GB.
- **Recurrent state per checkpoint**: 48 layers x 48 value heads x 128 x 128 x 4 B = **~151 MB** (about the KV of 2,300 tokens), so checkpoint spacing should be wider than for the 9B.
- **Speed**: ~49 GOP per token. At ~150 GOPS (MacBook) that is ~3 t/s decode, about the same as the 5.9 GB bandwidth limit, so both limits bind. EliteBook ~5-6 t/s. Prefill is the same order or slower, ~3 t/s MacBook and ~6 t/s EliteBook. Strictly a cache-mandatory, latency-tolerant stage.

**LightOnOCR-2-1B (Qwen3-0.6B-shaped decoder + Pixtral ViT):**
- **Decoder**: ~0.44B non-embedding + 0.16B tied embedding. At Q8 that is ~0.6 GB per token, so **~25-30 t/s on the MacBook, ~35-40 on the EliteBook**. KV is 28 x 2 x 8 x 128 x 2 B = 114 KB/token (standard dense), so a page of ~3k image tokens + ~2k output is ~0.6 GB, fine.
- **The ViT dominates the cost.** A full-size page is up to 12,100 patches. Linear layers: 2 x 0.4B x 12,100 = ~10 TFLOP. Attention (flash-style, 24 layers, quadratic): 4 x 12,100^2 x 1024 x 24 = ~14 TFLOP. **Total ~24 TFLOP per full-resolution page.** At ~100 GFLOPS sustained fp32 on the MacBook that is **~4 minutes per page**, and ~2 minutes on the EliteBook, before the ~3k-token LM prefill (~2.7 TFLOP, ~25 s) and 1-3k tokens of decode (~1-2 min).
- **Levers**: render pages at lower resolution (1024 px longest side is ~5,300 patches, about **4-5x cheaper** because attention falls quadratically), tile pages, cache embeddings by image hash (retries and re-gates cost nothing), run pages as a fan-out batch for the decoder, and make the ViT attention flash-style so the 12k x 12k score matrix is never materialized. Accuracy versus resolution is for you to measure on your documents.
- Vision is deferred past v1, and LightOnOCR is the natural first VLM when it returns.


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
- **Small ternary models change the picture.** The earlier Bonsai generation (1.7B, 4B, 8B) at ~1.75 bits/weight is ~0.4 / 0.9 / 1.7 GB. For the 8B, bandwidth would allow ~12-15 t/s, but compute is ~16 GFLOP/token, which at ~150 GOPS is **~9 t/s, so decode becomes compute-bound**, not bandwidth-bound. Ternary is therefore where **ALU-efficient kernels pay off most**, and the T-MAC / LUT approach (fewer operations per weight, not just fewer bytes) is worth a real trial against the `vpmaddubsw` baseline once the baseline exists. Those small models are also potential speculative-draft candidates for the 27B if they share a tokenizer.

- **Formats (read from the fork's `ggml-common.h` / `ggml-quants.c` on branch `prism`; the fetch tool summarizes pages, so confirm the bit layouts against the source or a GGUF dump before writing kernels):**
  - **`Q2_0`, group 64, ggml id 42** (upstream-compatible): `{ fp16 d; uint8 qs[16]; }` = 18 bytes per 64 weights = **2.25 bpw**. Code `q` in 0..3 maps to **`(q - 1) * d`**, so levels are {-1, 0, +1, +2} x d. Quantizer: `round(w/d) + 1`, clamped.
  - **`PQ2_0`, group 128, id 142** (fork-private): the same codec with one fp16 scale per 128 weights, `{ fp16 d; uint8 qs[32]; }` = 34 bytes = **2.125 bpw**, ~6% smaller than g64.
  - **`PTQ1_0`, group 128, id 143** (fork-private ternary): `{ uint8 qs[24]; uint8 qh[2]; fp16 d; }` = 28 bytes = **1.75 bpw**. Base-3 packing, five trits per byte (a `TQ1_0` relative, which is 1.6875 bpw), value `(trit - 1) * d`.
  - Mainline `TQ1_0` / `TQ2_0` exist too, but the Bonsai files use the types above.
  - **These are simple codecs.** The 2-bit one is a plain unpack with one multiply by `d`, and it is the same `u8 x s8` `vpmaddubsw` pattern as Q4. So **`Q2_0` g64 is an easy first ternary kernel**, with stock llama.cpp as the oracle, and `PQ2_0` / `PTQ1_0` are small variations.
- **Hadamard rotation (`prism_hadamard_qwen35`), now read from the included `llama-graph.cpp`:** the weights were quantized in a rotated basis, and the graph compensates at run time. In `build_lora_mm` (and the MoE variant), for every weight tensor `w` that has an entry in `hadamard_rotations`, the **input activation is transformed before the matmul**: (1) optionally permuted from tiled `[hd, nk, rep]` to grouped `[hd, rep, nk]` feature order, (2) optionally multiplied elementwise by a `signs` tensor, (3) multiplied by the rotation `rot` through `llama_mul_mat_hadamard`. The result is **memoized per (activation, rot)** so Q/K/V (or gate/up) sharing an input rotate once. The embedding table is also stored rotated: after `get_rows`, the row is multiplied by `rot` and then by `signs` (`h = s * (H z)`) to restore the original basis. The config gives block 1024 for every projection and the lm_head.
  - **Still missing from that file** (so Bonsai 2 stays deferred, as you said): the definition of `llama_mul_mat_hadamard` (does `rot` hold a block Hadamard matrix or is it a hint for a fast transform, and is it normalized by 1/sqrt(n)?), the **loader** that fills `hadamard_rotations`/`hadamard_inverses` (GGUF tensor names for `rot`, `signs`, permutation parameters), and which weights get which rotation. Those live in the fork's `llama-model.cpp`/loader and a `hadamard` source file. Asking for those two or three files would be enough to implement it.
  - **Cost is small** if the transform is a fast Walsh-Hadamard (about 10 adds per element, under 1% of the matmul) and about 2x the matmul work for a 1024x1024 dense multiply if it is not. Either way it is exactly specifiable once the definition is known, and it needs bit-exact semantics.
  - The same file also shows **mainline llama.cpp's KV-cache rotation**: when KV quantization is on, `q` and `k` are multiplied by `self_k_rot` and `v` by `self_v_rot`, and the attention output is multiplied by `self_v_rot` again. The rotation is orthogonal, so attention is unchanged, but it spreads outliers and makes quantized K/V much more accurate. **Adopt this for our q8/q4 KV cache** (§4.6); it is the "Hadamard-transformed K cache" trick I listed from ik_llama, now confirmed upstream.
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

**Capture kit (one run per model, on the EliteBook for models up to ~12B, on a rented box only if needed (§17)):**
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

**Decided: one resident process that loads once and serves many requests for a long time.** stdin/stdout/stderr only.

**Everything is an option, and every option can be given on the real command line.** The command line (plus an optional settings file) sets the session: model, threads, cache dir and quota, context policy, sampler, **system prompt** (`--system TEXT` or `--system-file PATH`), `--n`, stop rules, and so on. **`stdin` only augments that.** What stdin carries depends on `--stdin-mode`, and a request's effective options are layered:

`built-in defaults  <  settings file  <  real command line  <  per-request options`

**`--stdin-mode prompts` (record stream, the Emacs and pipeline default).** stdin carries **only the prompts**, one per record, separated by a single `RECORD_SEPARATOR` byte.
- Suggested separator ASCII 0x1E (RS), configurable with `--record-sep BYTE`. A prompt containing that byte is an error, or escapable with a configurable escape byte. A length-prefixed variant is the fully binary-safe alternative.
- The system prompt stays on the command line and is therefore the shared, cached prefix of every request. Nothing else is needed per record.
- **Optional per-record header** for the rare override: a record may start with an argv-style option line terminated by ASCII 0x1F (UNIT_SEPARATOR), then the prompt. No 0x1F means the whole record is the prompt.
- With no separator before EOF the whole of stdin is one prompt, so a one-shot run is just `engine --model m.gguf --system "Be terse." < prompt.txt`.

**`--stdin-mode args` (argv-lines).** Each stdin line is one request, written as command-line options that override the session options for that request (`--n 4 --temp 0.7 --prompt-file /tmp/p.txt`, or a short `--prompt 'text'`).
- Parsed with POSIX-shell-style word splitting only: single quotes, double quotes with `\"` and `\\`, backslash. **No expansion** (no `$`, globs, backticks), so request lines are safe to build from untrusted text.
- **The same option parser serves the real command line and each request line**, which keeps one flag vocabulary and one set of validation errors for the whole engine.
- A newline-terminated line cannot carry an arbitrary prompt, so prompts normally come via `--prompt-file PATH` (or `--prompt-fd`, or `--prompt-hex`). Fits shell and Perl pipelines well: `print $fh "--model qwen2.5-7b --n 3 --prompt-file $f\n"`.

**`--prompt TEXT` / `--prompt-file PATH` on the real command line with `--stdin-mode none`** runs a single request and exits, for scripts that want no stdin at all.

Because the system prompt and most options are fixed for a session, **the persistent prefix cache (§6) hits from the first token** of every request after the first, which is why `--system` belongs on the command line and not in each request.

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
- Runtime and byte-heavy tools are C. Everyday offline tooling is Perl. Python is acceptable for the rarely run capture kit (§9).
- **Batched fan-out** is in scope (§4.4).
- Resident, load-once, many-requests process with two stdin modes: argv-lines and RS-separated records (§11).
- Platforms: Linux on the EliteBook (i7-8665U, 2 x 16 GB DDR4, ~512 GB NVMe), macOS **12.7.6** on the MacBook (256 GB SSD), FreeBSD should also work (§14). Cache budget is a setting (§15).
- `llama-quantize` is fine. Reuse existing GGUFs and rewrite them into our layout (§7). Rental is acceptable but optional (§17).
- **Vision can be deferred.** v1 is text-only.
- **Qwen2.5 matters more than Bonsai 2.** Qwen2.5 becomes the first implementation target; Bonsai 2 is late.
- **You will try many sizes** of Gemma and the others. One spec per architecture family, not per size, and a `bench` request for any GGUF (§9).
- For Bonsai 2 you want the **PQ2_0** (group-128) 27B, which needs the PrismML fork. **Skip Bonsai 2 for now** if the fork's Hadamard definition is not available.
- **Qwen2.5 default list approved**: 0.5B, 1.5B, 3B, 7B Instruct, plus Coder 7B and R1-Distill 7B as fine-tune checks.
- **Options model:** every option may be given on the real command line, stdin only augments it (prompts, or per-request option lines). System prompt stays a command-line option (§11).

**Decided (by me, say if you disagree):**
- OS-identical results (bit-identical logits on the same ISA); only the platform layer differs (§14).
- Sidecar building is a C mode of the engine, lossless re-layout by default (§7).
- Vision deferred in implementation, reserved in the design.
- Quality-gate request types are first-class: `--logprobs N` (default off), `classify`, `score`, and a default repetition-loop stop for generation.
- Cancellation: `SIGINT` cancels the current request, `!` control lines in argv mode, optional `--control-fd N` in record mode.
- Gemma order: 270M, 1B, then 4 E2B/E4B.
- **Bonsai 2 plan:** I **corrected my earlier guess** about the formats: `PQ2_0` is the fork-private **group-128** type (id 142), and upstream llama.cpp's type is **`Q2_0` group 64** (id 42). Both use the same 2-bit codec, `(q - 1) * d`. I would implement **`Q2_0` g64 first** (stock llama.cpp is the oracle, no fork needed), then `PQ2_0` (a one-line change in block size) and `PTQ1_0`. The Hadamard rotation is the real work (§4.9).

**Still open (none blocks the start):**
1. **Bonsai 2 is deferred.** The included `llama-graph.cpp` shows the structure of the Hadamard rotation (§4.9) but not its definition or loader. If you want it later, add the fork's Hadamard source file(s) and the model loader (`llama-model.cpp` or wherever `hadamard_rotations` is filled).
2. **MacBook CPU**: assumed i7-5557U, the probe will confirm.

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
- **macOS 12.7.6 on the 2015 MBP (confirmed Monterey, the last release this model gets):** Intel x86_64, Darwin 21.x. The newest Xcode that runs on Monterey is 14.2 [M], so the toolchain is Apple clang 14-era: C99 is fine, intrinsics for AVX2/FMA/F16C work, set `-mmacosx-version-min=12.0`, and avoid anything in newer SDKs. Avoid private APIs (`__ulock_*`), and use pthread condition variables plus short spin loops for barriers. `clock_gettime` and `pthread_set_qos_class_self_np` are available. Do not rely on `MAP_POPULATE`, `posix_fadvise`, or `O_DIRECT` (use `F_NOCACHE`, `F_RDADVISE`, `madvise`). Thermal throttling is severe on this machine and macOS exposes little control, so the calibrator's sustained-throughput test and the "good-citizen" low-priority mode matter most here. Apple's memory compressor makes `RSS` misleading, so budget from `host_statistics64` (free + inactive) and watch for compressor growth. Dev workflow: build natively on the MacBook, and test the strict-overcommit and cgroup paths on the Linux machine.
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
1. **Qwen2.5 dense path first (0.5B-7B), then MiniCPM5 / Qwen3**: Q4_K/Q8_0 AVX2 kernels, GEMV + GEMM regimes, flash attention, f16/q8 KV, block pool, stdin/stdout protocol, conformance against llama.cpp.
2. **Memory robustness**: static plan, fault injection, strict-overcommit CI, incremental context growth.
3. **Persistent prefix cache** and sidecar prepared models.
4. **Hybrid blocks**: short conv (LFM), GDN (Qwen3.5), recurrent checkpoints.
5. **Spec decoding** (n-gram first, then draft), constrained decoding, thinking budget, classify/score modes.
6. **Gemma 3 270M/1B, then Gemma 4 E2B/E4B** (PLE region, KV sharing, SWA ring KV), **MoE and Ling** (KDA, MLA).
7. **Vision (deferred past v1)**: ViT blocks, image decode, embedding cache. The spec vocabulary, protocol (`--image`) and cache keys reserve room from phase 0. LightOnOCR is a good first VLM (1B decoder, fan-out over pages).
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
- [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (branch `prism`; `ggml-common.h`, `ggml-quants.c`, `ggml.h`, `llama-graph.cpp` read via a summarizing fetch tool), [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo/)
- [LightOnOCR paper](https://arxiv.org/html/2601.14251)
- [Core i7-8665U specs](https://en.wikichip.org/wiki/intel/core_i7/i7-8665u)
- [Gemma 4 deep dive](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4), [KV sharing and PLE notes](https://sebastianraschka.com/llm-architecture-gallery/kv-sharing/), [Gemma 3 270M](https://en.immers.cloud/ai/google/gemma-3-270m/)

---

## 17. One-off heavy jobs: do you need to rent anything?

**Short answer: probably not for v1, and a few dollars if you do.** The capture kit and quantization jobs scale with model size, and nearly all your models are small.

**What each job needs**

| Job | Resource | Does the EliteBook (32 GB) do it? |
|---|---|---|
| Op-graph trace (`torch.export` on a tiny random-init config) | trivial | yes, seconds |
| Activation/logit fixtures on real weights (HF, bf16) | 2 bytes per parameter | **yes up to ~12B**: Qwen3.5-9B / Ornith (18 GB), Qwen3-4B, MiniCPM5, Gemma 4 E4B (~8B raw, 16 GB), LFM2.5, LightOnOCR. Slow on CPU, but fixtures are a few short prompts |
| Same for **Bonsai 2 27B** (54 GB bf16) | 64+ GB | no, but **skip it**: use PrismML's fork as the oracle for the ternary GGUF, since the HF-level truth for a post-training ternarized model is the fork's behavior anyway |
| `llama-quantize` from bf16 | RAM about the model size | yes up to ~12B |
| imatrix calibration | forward passes over ~100k tokens | yes for 9B (hours on CPU, run overnight), faster on a rented GPU |
| KL-divergence/perplexity sweeps over many quants | repeated forward passes | possible but slow; this is where a GPU saves days |
| Speculation acceptance studies | forward passes on sample text | yes, small models |

**Where an hour of GPU is actually worth it:** imatrix and KL sweeps for the 9B-class models, and fixture capture for anything above ~12B. Both are one-time.

**Options (prices from search results, which disagree with each other in places, so check live):**
- **Hourly GPU marketplaces (Vast.ai, RunPod, Lambda, and similar):** an RTX 4090 (24 GB) is roughly **$0.3-0.5/hour**, an A100 80 GB roughly **$1.2-1.5/hour**, an H100 roughly **$2/hour**. A 24 GB card fits any bf16 model up to ~10B, enough for every imatrix and fixture job except the 27B. An evening of work is single-digit to low double-digit dollars. This is the best fit for "one-time cost, no commitment".
- **Hetzner dedicated GPU servers:** GEX44 (RTX 4000 SFF Ada 20 GB, 64 GB RAM) about **EUR 184/month plus EUR 79 setup**; GEX130 (RTX 6000 Ada 48 GB, 128 GB RAM) about **EUR 838/month plus EUR 79 setup** [V, search results]. They are billed monthly, so a one-time job costs a month. Good if you want a standing box, poor for a few hours.
- **Hetzner Cloud CPU instances** (hourly, up to ~48 vCPU / 192 GB RAM, from memory [M]): cheap enough for a 64 GB+ CPU-only bf16 run of the 27B (slow, but for a handful of short prompts it works), with no GPU. This is the "tough to rent 64 GB" answer if you do want HF-level fixtures for the 27B. Hetzner's dedicated **server auction** also has 64-128 GB RAM CPU boxes at modest monthly prices [M], again a month minimum.
- **Skip entirely:** use existing GGUFs plus llama.cpp / ik_llama.cpp / the PrismML fork as oracles. That covers correctness for everything llama.cpp-supported. What you lose is HF-level fixtures for architectures llama.cpp lacks, and custom imatrix quants.

**My recommendation:** do **no** rental until phase 4 (hybrids) or later, when the first quantization sweeps matter. Then rent a 24 GB GPU by the hour for a day. Revisit the 27B only if you decide Bonsai 2 27B is worth running at ~3 t/s.
