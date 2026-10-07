/* opts.c - table-driven option parsing. */
#include "opts.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { K_STR, K_INT, K_U64, K_SIZE, K_DBL, K_BOOL, K_ENUM, K_BYTE };
enum { S_SESSION, S_REQUEST };

typedef struct {
    const char  *name;
    int          kind;
    size_t       off;
    int          scope;
    double       lo, hi;           /* numeric range (K_INT, K_DBL, K_U64) */
    const char **choices;          /* K_ENUM, NULL-terminated */
    const char  *help;
} opt_def;

static const char *stdin_modes[]  = { "prompts", "args", "none", NULL };
static const char *log_levels[]   = { "error", "warn", "info", "debug", NULL };
static const char *chat_modes[]   = { "raw", "auto", "chatml", "llama3", NULL };

#define OFF(f) offsetof(hfc_opts, f)
#define NOLIM  0, 0

static const opt_def defs[] = {
    { "settings",       K_STR,  OFF(settings),       S_SESSION, NOLIM, NULL, "settings file (key = value lines)" },
    { "stdin-mode",     K_ENUM, OFF(stdin_mode),     S_SESSION, NOLIM, stdin_modes, "prompts | args | none" },
    { "record-sep",     K_BYTE, OFF(record_sep),     S_SESSION, NOLIM, NULL, "record separator byte (RS, 0x1e, 30, or a char)" },
    { "control-fd",     K_INT,  OFF(control_fd),     S_SESSION, -1, 1023, NULL, "extra fd carrying control lines, -1 = none" },
    { "cache-dir",      K_STR,  OFF(cache_dir),      S_SESSION, NOLIM, NULL, "directory for profiles and persistent caches" },
    { "cache-quota",    K_SIZE, OFF(cache_quota),    S_SESSION, NOLIM, NULL, "persistent cache size limit, 0 = disabled" },
    { "cache-min-free", K_SIZE, OFF(cache_min_free), S_SESSION, NOLIM, NULL, "free space always left on the cache filesystem" },
    { "threads",        K_INT,  OFF(threads),        S_SESSION, 0, 256, NULL, "worker threads, 0 = auto" },
    { "ctx-init",       K_SIZE, OFF(ctx_init),       S_SESSION, NOLIM, NULL, "initial context tokens" },
    { "ctx-step",       K_SIZE, OFF(ctx_step),       S_SESSION, NOLIM, NULL, "context growth increment in tokens" },
    { "ctx-max",        K_SIZE, OFF(ctx_max),        S_SESSION, NOLIM, NULL, "maximum context tokens" },
    { "max-record",     K_SIZE, OFF(max_record),     S_SESSION, NOLIM, NULL, "largest accepted prompt or request line" },
    { "probe-force",    K_BOOL, OFF(probe_force),    S_SESSION, NOLIM, NULL, "ignore the cached machine profile" },
    { "batch",          K_INT,  OFF(batch),          S_SESSION, 1, 4096, NULL, "prefill tokens per step" },
    { "log-level",      K_ENUM, OFF(log_level),      S_SESSION, NOLIM, log_levels, "error | warn | info | debug" },

    { "op",             K_STR,  OFF(op),             S_REQUEST, NOLIM, NULL, "generate | inspect | doctor | echo" },
    { "model",          K_STR,  OFF(model),          S_REQUEST, NOLIM, NULL, "GGUF file" },
    { "id",             K_STR,  OFF(id),             S_REQUEST, NOLIM, NULL, "request id echoed in the response" },
    { "chat",           K_ENUM, OFF(chat),           S_REQUEST, NOLIM, chat_modes, "raw | auto | chatml | llama3 (wrap the prompt as a user turn; --system implies auto)" },
    { "system",         K_STR,  OFF(system),         S_REQUEST, NOLIM, NULL, "system prompt text" },
    { "system-file",    K_STR,  OFF(system_file),    S_REQUEST, NOLIM, NULL, "system prompt from a file" },
    { "prompt",         K_STR,  OFF(prompt),         S_REQUEST, NOLIM, NULL, "prompt text (short prompts)" },
    { "prompt-file",    K_STR,  OFF(prompt_file),    S_REQUEST, NOLIM, NULL, "prompt from a file" },
    { "prompt-ids",     K_STR,  OFF(prompt_ids),     S_REQUEST, NOLIM, NULL, "prompt as token ids, space or comma separated" },
    { "n",              K_INT,  OFF(n),              S_REQUEST, 1, 64, NULL, "generate: completions decoded together from one prompt (seeds seed, seed+1, ...)" },
    { "temp",           K_DBL,  OFF(temp),           S_REQUEST, 0, 5, NULL, "sampling temperature" },
    { "top-k",          K_INT,  OFF(top_k),          S_REQUEST, 0, 1000000, NULL, "top-k, 0 = off" },
    { "top-p",          K_DBL,  OFF(top_p),          S_REQUEST, 0, 1, NULL, "top-p" },
    { "min-p",          K_DBL,  OFF(min_p),          S_REQUEST, 0, 1, NULL, "min-p" },
    { "seed",           K_U64,  OFF(seed),           S_REQUEST, NOLIM, NULL, "RNG seed (deterministic)" },
    { "max-tokens",     K_INT,  OFF(max_tokens),     S_REQUEST, 1, 1 << 24, NULL, "maximum new tokens" },
    { "logprobs",       K_INT,  OFF(logprobs),       S_REQUEST, 0, 100, NULL, "top-N logprobs per token, 0 = off" },
    { "stop-on-repeat", K_BOOL, OFF(stop_on_repeat), S_REQUEST, NOLIM, NULL, "stop on repetition loops" },
    { "timeout",        K_DBL,  OFF(timeout),        S_REQUEST, 0, 1e7, NULL, "per-request time limit in seconds, 0 = none" },
    { "list-tensors",   K_BOOL, OFF(list_tensors),   S_REQUEST, NOLIM, NULL, "inspect: list every tensor" },
    { "prefix-cache",   K_BOOL, OFF(prefix_cache),   S_REQUEST, NOLIM, NULL, "generate: reuse the KV state of the longest shared token prefix of the previous request (default on)" },
    { "parse-special",  K_BOOL, OFF(parse_special),  S_REQUEST, NOLIM, NULL, "tokenize: recognise special tokens in the text (default on)" },
    { "bench-prompt",   K_INT,  OFF(bench_prompt),   S_REQUEST, 1, 32768, NULL, "bench: prompt tokens to prefill" },
    { "bench-gen",      K_INT,  OFF(bench_gen),      S_REQUEST, 1, 4096, NULL, "bench: tokens to decode" },
    { "bench-threads",  K_STR,  OFF(bench_threads),  S_REQUEST, NOLIM, NULL, "bench: thread counts to try, e.g. 1,2,4 (default 1..physical cores)" },
    { "probe-sustained", K_INT, OFF(probe_sustained), S_REQUEST, 0, 600, NULL, "doctor: seconds of all-core load to measure throttling, 0 = skip" },
    { "probe-reps",     K_INT,  OFF(probe_reps),     S_REQUEST, 1, 32, NULL, "doctor: repeat the bandwidth sweep N times, report median and range" },
};
#define NDEFS (sizeof defs / sizeof defs[0])

static char **str_slot(hfc_opts *o, const opt_def *d) { return (char **)((char *)o + d->off); }
static void  *slot(hfc_opts *o, const opt_def *d)     { return (void *)((char *)o + d->off); }

/* ---- defaults, copy, free ---------------------------------------------------- */

void hfc_opts_defaults(hfc_opts *o)
{
    memset(o, 0, sizeof *o);
    o->stdin_mode = HFC_STDIN_PROMPTS;
    o->record_sep = 0x1e;
    o->control_fd = -1;
    o->cache_min_free = (uint64_t)10 << 30;
    o->ctx_init = 2048;
    o->ctx_step = 1024;
    o->ctx_max = 32768;
    o->max_record = (uint64_t)64 << 20;
    o->log_level = HFC_LOG_INFO;
    o->n = 1;
    o->probe_reps = 1;
    o->batch = 64;
    o->parse_special = 1;
    o->prefix_cache = 1;
    o->n = 1;
    o->bench_prompt = 256;
    o->bench_gen = 32;
    o->temp = 0.8;
    o->top_k = 40;
    o->top_p = 0.95;
    o->min_p = 0.05;
    o->max_tokens = 512;
    o->stop_on_repeat = 1;
    /* string defaults are set lazily so defaults never allocate */
}

void hfc_opts_free(hfc_opts *o)
{
    size_t i;
    for (i = 0; i < NDEFS; i++)
        if (defs[i].kind == K_STR) { hfc_free(*str_slot(o, &defs[i])); *str_slot(o, &defs[i]) = NULL; }
}

hfc_status hfc_opts_copy(hfc_opts *dst, const hfc_opts *src)
{
    hfc_opts tmp = *src;
    size_t i;
    for (i = 0; i < NDEFS; i++) {
        if (defs[i].kind != K_STR) continue;
        {
            const char *s = *(char * const *)((const char *)src + defs[i].off);
            char **t = str_slot(&tmp, &defs[i]);
            *t = NULL;
            if (s && !(*t = hfc_strdup(s))) {
                hfc_opts_free(&tmp);
                return HFC_ENOMEM;
            }
        }
    }
    *dst = tmp;
    return HFC_OK;
}

/* ---- value parsing ----------------------------------------------------------- */

hfc_status hfc_parse_size(const char *s, uint64_t *out)
{
    char *end;
    unsigned long long v;
    uint64_t mult = 1;

    if (!s || !isdigit((unsigned char)*s)) return HFC_EINVAL;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno == ERANGE) return HFC_ERANGE;
    if (*end) {
        const char *rest = end + 1;
        switch (toupper((unsigned char)*end)) {
        case 'K': mult = (uint64_t)1 << 10; break;
        case 'M': mult = (uint64_t)1 << 20; break;
        case 'G': mult = (uint64_t)1 << 30; break;
        case 'T': mult = (uint64_t)1 << 40; break;
        default:  return HFC_EINVAL;
        }
        if (*rest && strcmp(rest, "B") && strcmp(rest, "b") && strcmp(rest, "iB") && strcmp(rest, "ib"))
            return HFC_EINVAL;
    }
    if (!hfc_mul_u64((uint64_t)v, mult, out)) return HFC_ERANGE;
    return HFC_OK;
}

static int parse_bool(const char *s, int *out)
{
    if (!strcmp(s, "1") || !strcmp(s, "true") || !strcmp(s, "yes") || !strcmp(s, "on")) { *out = 1; return 1; }
    if (!strcmp(s, "0") || !strcmp(s, "false") || !strcmp(s, "no") || !strcmp(s, "off")) { *out = 0; return 1; }
    return 0;
}

static int parse_byte(const char *s, int *out)
{
    static const struct { const char *n; int v; } names[] = {
        { "NUL", 0 }, { "EOT", 4 }, { "LF", 10 }, { "FF", 12 }, { "FS", 28 }, { "GS", 29 },
        { "RS", 30 }, { "US", 31 }
    };
    size_t i;
    char *end;
    long v;
    for (i = 0; i < sizeof names / sizeof names[0]; i++)
        if (!strcmp(s, names[i].n)) { *out = names[i].v; return 1; }
    if (s[0] && !s[1] && !isdigit((unsigned char)s[0])) { *out = (unsigned char)s[0]; return 1; }
    errno = 0;
    v = strtol(s, &end, 0);
    if (errno || end == s || *end || v < 0 || v > 255) return 0;
    *out = (int)v;
    return 1;
}

static const opt_def *find_def(const char *name, size_t len)
{
    size_t i;
    for (i = 0; i < NDEFS; i++)
        if (strlen(defs[i].name) == len && memcmp(defs[i].name, name, len) == 0) return &defs[i];
    return NULL;
}

static hfc_status set_value(hfc_opts *o, const opt_def *d, const char *v, char *err, size_t cap)
{
#define FAIL(...) do { snprintf(err, cap, __VA_ARGS__); return HFC_EINVAL; } while (0)
    switch (d->kind) {
    case K_STR: {
        char *c = hfc_strdup(v);
        if (!c) { snprintf(err, cap, "out of memory"); return HFC_ENOMEM; }
        hfc_free(*str_slot(o, d));
        *str_slot(o, d) = c;
        return HFC_OK;
    }
    case K_INT: {
        char *end;
        long x;
        errno = 0;
        x = strtol(v, &end, 10);
        if (errno || end == v || *end) FAIL("--%s: '%s' is not an integer", d->name, v);
        if ((double)x < d->lo || (double)x > d->hi) FAIL("--%s: %s out of range [%g, %g]", d->name, v, d->lo, d->hi);
        *(int *)slot(o, d) = (int)x;
        return HFC_OK;
    }
    case K_U64: {
        char *end;
        unsigned long long x;
        if (!isdigit((unsigned char)v[0])) FAIL("--%s: '%s' is not a non-negative integer", d->name, v);
        errno = 0;
        x = strtoull(v, &end, 10);
        if (errno || end == v || *end) FAIL("--%s: '%s' is not a valid integer", d->name, v);
        *(uint64_t *)slot(o, d) = (uint64_t)x;
        return HFC_OK;
    }
    case K_SIZE: {
        uint64_t x;
        if (hfc_parse_size(v, &x) != HFC_OK) FAIL("--%s: '%s' is not a size (e.g. 4096, 512M, 80G)", d->name, v);
        *(uint64_t *)slot(o, d) = x;
        return HFC_OK;
    }
    case K_DBL: {
        char *end;
        double x;
        errno = 0;
        x = strtod(v, &end);
        if (errno || end == v || *end || !isfinite(x)) FAIL("--%s: '%s' is not a number", d->name, v);
        if (x < d->lo || x > d->hi) FAIL("--%s: %s out of range [%g, %g]", d->name, v, d->lo, d->hi);
        *(double *)slot(o, d) = x;
        return HFC_OK;
    }
    case K_BOOL: {
        int b;
        if (!parse_bool(v, &b)) FAIL("--%s: '%s' is not a boolean", d->name, v);
        *(int *)slot(o, d) = b;
        return HFC_OK;
    }
    case K_ENUM: {
        int i;
        for (i = 0; d->choices[i]; i++)
            if (!strcmp(d->choices[i], v)) { *(int *)slot(o, d) = i; return HFC_OK; }
        FAIL("--%s: '%s' is not one of the allowed values", d->name, v);
    }
    case K_BYTE: {
        int b;
        if (!parse_byte(v, &b)) FAIL("--%s: '%s' is not a byte value", d->name, v);
        *(int *)slot(o, d) = b;
        return HFC_OK;
    }
    }
    FAIL("internal error");
#undef FAIL
}

hfc_status hfc_opts_apply(hfc_opts *o, int argc, char **argv, hfc_opts_ctx ctx, char *err, size_t cap)
{
    int i;
    for (i = 0; i < argc; i++) {
        const char *a = argv[i];
        const char *eq, *val = NULL;
        const opt_def *d;
        size_t nlen;
        int negate = 0;
        hfc_status rc;

        if (a[0] != '-' || a[1] != '-' || a[2] == '\0') {
            snprintf(err, cap, "unexpected argument '%s' (options start with --)", a);
            return HFC_EINVAL;
        }
        a += 2;
        eq = strchr(a, '=');
        nlen = eq ? (size_t)(eq - a) : strlen(a);
        if (eq) val = eq + 1;
        d = find_def(a, nlen);
        if (!d && nlen > 3 && !strncmp(a, "no-", 3)) {
            d = find_def(a + 3, nlen - 3);
            if (d && d->kind == K_BOOL && !eq) negate = 1; else d = NULL;
        }
        if (!d) { snprintf(err, cap, "unknown option '--%.*s'", (int)nlen, a); return HFC_EINVAL; }
        if (ctx == HFC_CTX_REQUEST && d->scope == S_SESSION) {
            snprintf(err, cap, "option '--%s' can only be set at startup, not per request", d->name);
            return HFC_EINVAL;
        }
        if (d->kind == K_BOOL && !val) {
            val = negate ? "0" : "1";
        } else if (!val) {
            if (i + 1 >= argc) { snprintf(err, cap, "option '--%s' needs a value", d->name); return HFC_EINVAL; }
            val = argv[++i];
        }
        rc = set_value(o, d, val, err, cap);
        if (rc != HFC_OK) return rc;
    }
    return HFC_OK;
}

hfc_status hfc_opts_load_settings(hfc_opts *o, const char *path, char *err, size_t cap)
{
    FILE *f = fopen(path, "r");
    char line[4096];
    int lineno = 0;
    if (!f) { snprintf(err, cap, "cannot open settings file '%s'", path); return HFC_EIO; }
    while (fgets(line, sizeof line, f)) {
        char *p = line, *key, *val, *end;
        const opt_def *d;
        hfc_status rc;
        lineno++;
        if (!strchr(line, '\n') && !feof(f)) {
            fclose(f);
            snprintf(err, cap, "%s:%d: line too long", path, lineno);
            return HFC_EINVAL;
        }
        while (isspace((unsigned char)*p)) p++;
        if (*p == '#' || *p == '\0') continue;
        key = p;
        while (*p && *p != '=' && !isspace((unsigned char)*p)) p++;
        end = p;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '=') p++;
        while (isspace((unsigned char)*p)) p++;
        val = p;
        *end = '\0';
        p = val + strlen(val);
        while (p > val && isspace((unsigned char)p[-1])) *--p = '\0';
        d = find_def(key, strlen(key));
        if (!d || !strcmp(d->name, "settings")) {
            fclose(f);
            snprintf(err, cap, "%s:%d: unknown or disallowed setting '%s'", path, lineno, key);
            return HFC_EINVAL;
        }
        if (d->kind == K_BOOL && *val == '\0') val = "1";
        if (*val == '\0') {
            fclose(f);
            snprintf(err, cap, "%s:%d: setting '%s' has no value", path, lineno, key);
            return HFC_EINVAL;
        }
        rc = set_value(o, d, val, err, cap);
        if (rc != HFC_OK) { fclose(f); return rc; }
    }
    fclose(f);
    return HFC_OK;
}

/* ---- shell-style splitting ------------------------------------------------------ */

#define MAX_ARGS 4096

void hfc_argv_free(char **argv, int argc)
{
    int i;
    if (!argv) return;
    for (i = 0; i < argc; i++) hfc_free(argv[i]);
    hfc_free(argv);
}

hfc_status hfc_shell_split(const char *s, size_t len, char ***out_argv, int *out_argc, char *err, size_t cap)
{
    char **argv = NULL;
    char *cur = NULL;
    size_t curlen = 0, curcap = 0, i = 0;
    int argc = 0, have = 0;      /* have: a word is in progress (even if empty) */
    char quote = 0;
    hfc_status rc = HFC_ENOMEM;

    *out_argv = NULL;
    *out_argc = 0;
    argv = (char **)hfc_calloc(MAX_ARGS + 1, sizeof *argv);
    if (!argv) { snprintf(err, cap, "out of memory"); return HFC_ENOMEM; }

#define PUSHC(c) do { \
        if (curlen + 1 >= curcap) { \
            size_t nc_ = curcap ? curcap * 2 : 64; \
            char *p_ = (char *)hfc_realloc(cur, nc_); \
            if (!p_) goto oom; \
            cur = p_; curcap = nc_; \
        } \
        cur[curlen++] = (char)(c); have = 1; } while (0)
#define ENDWORD() do { \
        if (have) { \
            if (argc >= MAX_ARGS) { snprintf(err, cap, "too many arguments"); rc = HFC_EINVAL; goto fail; } \
            if (!cur) { cur = hfc_malloc(1); if (!cur) goto oom; } \
            cur[curlen] = '\0'; \
            argv[argc++] = cur; cur = NULL; curlen = curcap = 0; have = 0; \
        } } while (0)

    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\0') { snprintf(err, cap, "NUL byte in request line"); rc = HFC_EINVAL; goto fail; }
        if (quote == '\'') {
            if (c == '\'') quote = 0; else PUSHC(c);
        } else if (quote == '"') {
            if (c == '"') quote = 0;
            else if (c == '\\' && i + 1 < len) {
                unsigned char n = (unsigned char)s[++i];
                if (n == 'n') PUSHC('\n');
                else if (n == 't') PUSHC('\t');
                else if (n == '"' || n == '\\') PUSHC(n);
                else { PUSHC('\\'); PUSHC(n); }
            } else PUSHC(c);
        } else if (c == '\'' || c == '"') {
            quote = (char)c;
            have = 1;
        } else if (c == '\\') {
            if (i + 1 >= len) { snprintf(err, cap, "trailing backslash"); rc = HFC_EINVAL; goto fail; }
            PUSHC((unsigned char)s[++i]);
        } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ENDWORD();
        } else {
            PUSHC(c);
        }
    }
    if (quote) { snprintf(err, cap, "unterminated %s quote", quote == '"' ? "double" : "single"); rc = HFC_EINVAL; goto fail; }
    ENDWORD();
    *out_argv = argv;
    *out_argc = argc;
    return HFC_OK;
oom:
    snprintf(err, cap, "out of memory");
    rc = HFC_ENOMEM;
fail:
    hfc_free(cur);
    hfc_argv_free(argv, argc);
    return rc;
#undef PUSHC
#undef ENDWORD
}

/* ---- output ---------------------------------------------------------------------- */

void hfc_opts_dump(FILE *f, const hfc_opts *o)
{
    size_t i;
    hfc_opts *w = (hfc_opts *)o;     /* read-only use of the slot helpers */
    for (i = 0; i < NDEFS; i++) {
        const opt_def *d = &defs[i];
        fprintf(f, "%s=", d->name);
        switch (d->kind) {
        case K_STR:  { const char *s = *str_slot(w, d); fprintf(f, "%s", s ? s : ""); break; }
        case K_INT:  case K_BOOL: fprintf(f, "%d", *(int *)slot(w, d)); break;
        case K_U64:  case K_SIZE: fprintf(f, "%llu", (unsigned long long)*(uint64_t *)slot(w, d)); break;
        case K_DBL:  fprintf(f, "%g", *(double *)slot(w, d)); break;
        case K_ENUM: fprintf(f, "%s", d->choices[*(int *)slot(w, d)]); break;
        case K_BYTE: fprintf(f, "%d", *(int *)slot(w, d)); break;
        }
        fputc('\n', f);
    }
}

void hfc_opts_usage(FILE *f)
{
    size_t i;
    fprintf(f, "usage: hfcpu [options]\n\nOptions (startup-only ones cannot appear in per-request lines):\n");
    for (i = 0; i < NDEFS; i++)
        fprintf(f, "  --%-16s %s%s\n", defs[i].name, defs[i].help,
                defs[i].scope == S_SESSION ? " [startup only]" : "");
    fprintf(f, "\nBoolean options also accept --no-NAME. Values may be given as --name value or --name=value.\n");
}
