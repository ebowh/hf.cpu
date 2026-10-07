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
| `tokenize` | `--model FILE.gguf` (needs tokenizer metadata): token ids of the prompt as `@tokens count=N ids="..."`; `--no-parse-special` treats `<|im_start|>` and friends as plain text |
| `generate` | not implemented yet (phase 1); answers `ENOTSUP` |

## Signals and exit codes

`SIGINT`/`SIGTERM` stop the loop between requests (exit 130). `SIGPIPE` is
ignored; a closed stdout ends the process cleanly (exit 1). Startup option
errors exit 2. In `--stdin-mode none`, a failed request exits 1.
