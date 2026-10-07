# hfcpu protocol (phase 0)

`hfcpu` is one resident process. It reads requests from **stdin**, writes
responses to **stdout**, and logs to **stderr**. Nothing else is a request
channel.

## Options

Every option can be given on the real command line, in a settings file
(`--settings FILE`, `key = value` lines, `#` comments), and, unless marked
*startup only*, in a per-request option line. Effective options for a request:

    defaults  <  settings file  <  command line  <  per-request options

`hfcpu --help` lists them. Booleans accept `--name`, `--no-name`,
`--name=false`. Values: `--name value` or `--name=value`. Sizes take `K M G T`
(1024-based) suffixes.

## stdin modes (`--stdin-mode`)

| mode | stdin carries | per-request options |
|---|---|---|
| `prompts` (default) | prompts only, separated by the record separator byte | optional header (below) |
| `args` | one request per line, written as command-line options | the line itself |
| `none` | nothing; `--prompt` / `--prompt-file` on the command line, one request, then exit | n/a |

### `prompts`

* Records are separated by `--record-sep` (default RS, 0x1e; accepts `RS`,
  `US`, `LF`, `NUL`, a number such as `30` or `0x1e`, or a single character).
* A final record without a terminator is a request; an empty stdin is no
  request; a stream with no separator at all is one request.
* A prompt cannot contain the separator byte.
* A record may start with an **option line terminated by 0x1f (US)**, then the
  prompt: `--n 3 --id r7<US>the prompt<RS>`. To send a prompt that itself contains
  0x1f, start the record with an empty header (`<US>prompt...`).
* The system prompt is a command-line option (`--system` / `--system-file`).
* A record larger than `--max-record` (default 64M) is skipped, answered with an
  `ERANGE` error, and the stream stays in sync.

### `args`

* Each line is parsed with POSIX-shell-style word splitting and **no
  expansion**: single quotes are literal; double quotes allow `\"`, `\\`, `\n`,
  `\t`; outside quotes a backslash escapes the next character.
* Blank lines and lines starting with `#` are ignored.
* Control lines start with `!`: `!quit`, `!stats`, `!cancel` (no request is
  running when a line is read, so it only reports that).
* Long prompts go through `--prompt-file`.

## Responses

A response is a sequence of event lines on stdout:

    @name key=value key="value with spaces" ...

Values are quoted when empty or when they contain space, `"`, `\`, `=` or
control characters; inside quotes `\"`, `\\`, `\n`, `\r`, `\t` and `\xHH` are
escapes. A **payload event** carries `len=N`, then a newline, exactly N raw
bytes, and a newline:

    @prompt len=5
    hello

Every request produces `@begin id=... op=...` first and `@done status=ok|error`
last. On error, `@done` has `code` (`EINVAL`, `EIO`, `EFORMAT`, `ERANGE`,
`ENOMEM`, `ENOTSUP`) and `msg`. In `prompts` mode the separator byte follows
every `@done`, so a client can split the stream on it.

## Operations (`--op`)

| op | does |
|---|---|
| `echo` | reports the effective options and the resolved system and prompt text (protocol testing) |
| `inspect` | `--model FILE.gguf`: header, metadata, per-type tensor statistics; `--list-tensors` for every tensor |
| `doctor` | CPU features, caches, memory and commit limits, measured read bandwidth by thread count, FMA throughput (1 thread, all cores, all threads) and latency curve; `--probe-sustained S` adds a per-second all-core trace to expose throttling; `--probe-reps N` repeats the bandwidth sweep and reports median and range; caches the profile in `--cache-dir` (default `~/.cache/hfcpu`) |
| `generate` | `--model FILE.gguf` plus a prompt (stdin record, `--prompt`, `--prompt-file`, or `--prompt-ids "1 2 3"` to skip the tokenizer). Events: `@prompt`, `@prefill` (tokens, cached, ms, tok_per_s, kv_bytes; `cached` prompt tokens were reused from the previous request, see `--prefix-cache`), one `@token pos= id=` per generated token (with `logprob=` and `top="id:lp ..."` when `--logprobs N`), `@text` payloads (UTF-8 safe), and `@gen` (tokens, ms, tok_per_s, stop=eos/length/ctx/stop/timeout/cancel). `--chat raw|auto|chatml|llama3` wraps the prompt as a user turn (after an optional `--system` turn) and opens the assistant turn; `--system` alone implies `auto`, which picks by the special tokens in the vocabulary. Markers are special tokens, user text never is. Qwen2 models get their default system prompt when none is given. The default `raw` tokenizes the prompt as is (and `--prompt-ids` bypasses all of this). The model stays loaded between requests |
| `bench` | `--model FILE.gguf`: prefill and decode speed for each thread count in `--bench-threads` (default 1 up to the physical core count, plus all hardware threads), over `--bench-prompt` and `--bench-gen` tokens, median of `--probe-reps` runs: `@bench threads= prefill_tok_s= decode_tok_s=`. Use it to choose `--threads` for a machine |
| `tokenize` | `--model FILE.gguf` (needs tokenizer metadata): token ids of the prompt as `@tokens count=N ids="..."`; `--no-parse-special` treats `<|im_start|>` and friends as plain text |

## Signals and exit codes

`SIGINT`/`SIGTERM` stop the loop between requests (exit 130). `SIGPIPE` is
ignored; a closed stdout ends the process cleanly (exit 1). Startup option
errors exit 2. In `--stdin-mode none`, a failed request exits 1.

## Prefix cache

The resident model keeps one inference context. After a `generate` request it holds the KV state of the prompt plus the tokens generated (all but the last sampled one). The next request reuses the longest common token prefix, always re-evaluating at least its last prompt token, so repeated system prompts and multi-turn conversations only pay for what is new. Results are bit-identical with and without reuse (`--no-prefix-cache` turns it off). The cache is dropped when another model is loaded, `--threads`/`--ctx-max`/`--batch` change, or a request fails midway. It lives in memory only; a persistent on-disk cache is a later step.

## Profiling

`generate --profile` adds two `@profile` events (`stage=prefill` and `stage=decode`) with the wall-clock milliseconds spent in each part of the forward pass (embed, glue = norms/rope/KV writes/residuals, qkv, attn, wo, gate_up, silu, down, head = the final projection to logits, quant = activation quantization) plus `sample`. The phases of the decode stage add up to roughly the decode time; matmul phases include their thread-pool synchronization.

## Fan-out (`--n N`, 2..64)

`generate --n N` decodes N continuations of one prompt together: the prompt is prefilled once, each sequence gets a fork of the KV state (full 64-token blocks are shared read-only, the partial block is copied) and every decode step reads the weights once for all active sequences, which is where the speedup comes from on memory-bound machines. Sequence i samples with seed `--seed + i`, and its tokens are bit-identical to a single run with that seed. With `--n` above 1 the `@token`, `@text` and `@gen` events carry `seq=i`, and a final `@genall sequences= tokens= ms= tok_per_s=` reports the aggregate. N is limited by `--batch`. The prefix cache afterwards holds the prompt only.

## Stopping a generation

* `--stop TEXT` ends a sequence just before the first occurrence of TEXT (`stop=stop`). Escapes `\n \t \r \\ \xHH` are understood and several strings are separated with `\x1f` (up to 8). Matching works on the output bytes and across token boundaries; text that could still turn into a stop string is held back until it cannot, so the stop text is never emitted.
* `--timeout SECONDS` limits a request. Expiry during prefill is an error (`code=ECANCEL`); expiry while generating ends the sequences with `stop=timeout` and keeps what was produced.
* SIGINT ends the running request the same way (`stop=cancel`, status ok) and the process keeps serving; SIGINT while idle, and SIGTERM at any time, ends the process.
