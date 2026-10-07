/* ops.c - request dispatch: echo, inspect, doctor; generate is a stub. */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "ops.h"
#include "ggtype.h"
#include "gguf.h"
#include "pal.h"
#include "tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- helpers --------------------------------------------------------------- */

/* Split "key=value" lines of `text` into @name key=... value=... events. */
static void emit_lines(hfc_out *o, const char *name, const char *text)
{
    const char *p = text;
    while (*p) {
        const char *nl = strchr(p, '\n'), *eq;
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char *line = hfc_strndup(p, len);
        if (line) {
            eq = strchr(line, '=');
            if (eq) {
                *(char *)eq = '\0';
                hfc_out_event(o, name, "key", line, "value", eq + 1, (const char *)NULL);
            }
            hfc_free(line);
        }
        if (!nl) break;
        p = nl + 1;
    }
}

/* Capture what a FILE*-printing function writes. Returns heap string or NULL. */
static char *capture(void (*fn)(FILE *, const void *, const void *), const void *a, const void *b)
{
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    char *copy;
    if (!f) return NULL;
    fn(f, a, b);
    fclose(f);
    copy = buf ? hfc_strdup(buf) : NULL;       /* move into our allocator */
    free(buf);
    return copy;
}

static void dump_opts_fn(FILE *f, const void *a, const void *b) { (void)b; hfc_opts_dump(f, (const hfc_opts *)a); }
static void probe_fn(FILE *f, const void *a, const void *b)
{
    hfc_probe_print(f, (const hfc_cpu *)a, (const hfc_machine *)b);
}

/* Read a whole file, at most `max` bytes. */
static hfc_status read_file(const char *path, size_t max, char **out, size_t *len)
{
    FILE *f = fopen(path, "rb");
    char *buf, *nb;
    size_t cap = 4096, n = 0, r;
    *out = NULL;
    *len = 0;
    if (!f) return HFC_EIO;
    buf = (char *)hfc_malloc(cap);
    if (!buf) { fclose(f); return HFC_ENOMEM; }
    for (;;) {
        if (n + 1 >= cap) {
            if (cap > max) { hfc_free(buf); fclose(f); return HFC_ERANGE; }
            cap *= 2;
            nb = (char *)hfc_realloc(buf, cap);
            if (!nb) { hfc_free(buf); fclose(f); return HFC_ENOMEM; }
            buf = nb;
        }
        r = fread(buf + n, 1, cap - 1 - n, f);
        if (r == 0) break;
        n += r;
        if (n > max) { hfc_free(buf); fclose(f); return HFC_ERANGE; }
    }
    if (ferror(f)) { hfc_free(buf); fclose(f); return HFC_EIO; }
    fclose(f);
    buf[n] = '\0';
    *out = buf;
    *len = n;
    return HFC_OK;
}

/* ---- ops -------------------------------------------------------------------- */

typedef struct {
    char   *system, *prompt;
    size_t  system_len, prompt_len;
} texts_t;

static void texts_free(texts_t *t) { hfc_free(t->system); hfc_free(t->prompt); memset(t, 0, sizeof *t); }

static hfc_status resolve_texts(const hfc_opts *eff, const char *body, size_t body_len, int has_body,
                                texts_t *t, char *msg, size_t msgcap)
{
    hfc_status rc;
    memset(t, 0, sizeof *t);
    if (eff->system && eff->system_file) { snprintf(msg, msgcap, "give --system or --system-file, not both"); return HFC_EINVAL; }
    if ((has_body ? 1 : 0) + (eff->prompt ? 1 : 0) + (eff->prompt_file ? 1 : 0) > 1) {
        snprintf(msg, msgcap, "prompt given more than once (stdin record, --prompt, --prompt-file)");
        return HFC_EINVAL;
    }
    if (eff->system) {
        t->system = hfc_strdup(eff->system);
        if (!t->system) return HFC_ENOMEM;
        t->system_len = strlen(t->system);
    } else if (eff->system_file) {
        if ((rc = read_file(eff->system_file, (size_t)eff->max_record, &t->system, &t->system_len)) != HFC_OK) {
            snprintf(msg, msgcap, "cannot read --system-file '%s': %s", eff->system_file, hfc_strerror(rc));
            return rc;
        }
    }
    if (has_body) {
        t->prompt = hfc_strndup(body, body_len);
        if (!t->prompt) { texts_free(t); return HFC_ENOMEM; }
        t->prompt_len = body_len;
    } else if (eff->prompt) {
        t->prompt = hfc_strdup(eff->prompt);
        if (!t->prompt) { texts_free(t); return HFC_ENOMEM; }
        t->prompt_len = strlen(t->prompt);
    } else if (eff->prompt_file) {
        if ((rc = read_file(eff->prompt_file, (size_t)eff->max_record, &t->prompt, &t->prompt_len)) != HFC_OK) {
            snprintf(msg, msgcap, "cannot read --prompt-file '%s': %s", eff->prompt_file, hfc_strerror(rc));
            texts_free(t);
            return rc;
        }
    }
    return HFC_OK;
}

static hfc_status op_echo(hfc_session *s, const hfc_opts *eff, const char *body, size_t body_len,
                          int has_body, char *msg, size_t msgcap)
{
    texts_t t;
    hfc_status rc = resolve_texts(eff, body, body_len, has_body, &t, msg, msgcap);
    char *dump;
    if (rc != HFC_OK) return rc;
    dump = capture(dump_opts_fn, eff, NULL);
    if (dump) { emit_lines(&s->out, "opt", dump); hfc_free(dump); }
    hfc_out_payload(&s->out, "system", t.system ? t.system : "", t.system_len, (const char *)NULL);
    hfc_out_payload(&s->out, "prompt", t.prompt ? t.prompt : "", t.prompt_len, (const char *)NULL);
    texts_free(&t);
    return HFC_OK;
}

static int is_hparam_suffix(const char *key)
{
    static const char *suf[] = { ".block_count", ".embedding_length", ".feed_forward_length",
        ".context_length", ".attention.head_count", ".attention.head_count_kv", ".attention.key_length",
        ".attention.value_length", ".rope.freq_base", ".rope.dimension_count", ".vocab_size",
        ".expert_count", ".expert_used_count", ".attention.layer_norm_rms_epsilon", NULL };
    size_t i, kl = strlen(key);
    for (i = 0; suf[i]; i++) {
        size_t sl = strlen(suf[i]);
        if (kl > sl && strcmp(key + kl - sl, suf[i]) == 0) return 1;
    }
    return 0;
}

static void emit_kv(hfc_out *o, const gguf_kv *kv)
{
    char num[64];
    const char *tn = gguf_type_name(kv->type);
    if (kv->type == GGUF_T_STR) {
        size_t n = strlen(kv->str);
        if (n > 160) {
            char *cut = hfc_strndup(kv->str, 160);
            snprintf(num, sizeof num, "%lu", (unsigned long)n);
            hfc_out_event(o, "kv", "key", kv->key, "type", tn, "value", cut ? cut : "", "truncated_from", num, (const char *)NULL);
            hfc_free(cut);
        } else {
            hfc_out_event(o, "kv", "key", kv->key, "type", tn, "value", kv->str, (const char *)NULL);
        }
    } else if (kv->type == GGUF_T_ARR) {
        char cnt[32];
        snprintf(cnt, sizeof cnt, "%llu", (unsigned long long)kv->arr_count);
        hfc_out_event(o, "kv", "key", kv->key, "type", "array", "elem", gguf_type_name(kv->arr_type), "count", cnt, (const char *)NULL);
    } else {
        switch (kv->type) {
        case GGUF_T_F32: case GGUF_T_F64: snprintf(num, sizeof num, "%.9g", kv->v.f); break;
        case GGUF_T_I8: case GGUF_T_I16: case GGUF_T_I32: case GGUF_T_I64:
            snprintf(num, sizeof num, "%lld", (long long)kv->v.i); break;
        default: snprintf(num, sizeof num, "%llu", (unsigned long long)kv->v.u); break;
        }
        hfc_out_event(o, "kv", "key", kv->key, "type", tn, "value", num, (const char *)NULL);
    }
}

static hfc_status op_inspect(hfc_session *s, const hfc_opts *eff, char *msg, size_t msgcap)
{
    gguf_file g;
    hfc_status rc;
    hfc_out *o = &s->out;
    char a[64], b[64], c[64], d[64];
    uint64_t i, total = 0;
    uint64_t type_n[256], type_b[256];
    int unknown_or_unsupported = 0;
    const char *arch;

    if (!eff->model) { snprintf(msg, msgcap, "inspect needs --model PATH"); return HFC_EINVAL; }
    rc = gguf_open(&g, eff->model, msg, msgcap);
    if (rc != HFC_OK) return rc;

    snprintf(a, sizeof a, "%u", g.version);
    snprintf(b, sizeof b, "%llu", (unsigned long long)g.n_tensors);
    snprintf(c, sizeof c, "%llu", (unsigned long long)g.n_kv);
    snprintf(d, sizeof d, "%lu", (unsigned long)g.map.len);
    hfc_out_event(o, "file", "path", eff->model, "bytes", d, "gguf_version", a, "tensors", b, "kv", c, (const char *)NULL);
    snprintf(a, sizeof a, "%llu", (unsigned long long)g.align);
    snprintf(b, sizeof b, "%llu", (unsigned long long)g.data_off);
    hfc_out_event(o, "layout", "alignment", a, "data_offset", b, (const char *)NULL);

    arch = gguf_get_str(&g, "general.architecture");
    hfc_out_event(o, "arch", "name", arch ? arch : "(none)", "supported", "no", "reason", "inference not implemented yet", (const char *)NULL);

    for (i = 0; i < g.n_kv; i++) {
        const gguf_kv *kv = &g.kv[i];
        int show = strncmp(kv->key, "general.", 8) == 0 || is_hparam_suffix(kv->key) ||
                   strcmp(kv->key, "tokenizer.ggml.model") == 0 || strcmp(kv->key, "tokenizer.ggml.pre") == 0;
        if (show || eff->list_tensors) emit_kv(o, kv);
        else if (kv->type == GGUF_T_ARR || kv->type == GGUF_T_STR) emit_kv(o, kv);
    }

    memset(type_n, 0, sizeof type_n);
    memset(type_b, 0, sizeof type_b);
    for (i = 0; i < g.n_tensors; i++) {
        const gguf_tensor *t = &g.t[i];
        total += t->nbytes;
        if (t->type < 256) { type_n[t->type]++; type_b[t->type] += t->nbytes; }
        if (eff->list_tensors) {
            char dims[160] = "", off[32], nb[32], ty[32];
            uint32_t k;
            for (k = 0; k < t->n_dims; k++) {
                char one[32];
                snprintf(one, sizeof one, "%s%llu", k ? "x" : "", (unsigned long long)t->ne[k]);
                if (strlen(dims) + strlen(one) < sizeof dims) strcat(dims, one);
            }
            snprintf(off, sizeof off, "%llu", (unsigned long long)t->offset);
            snprintf(nb, sizeof nb, "%llu", (unsigned long long)t->nbytes);
            snprintf(ty, sizeof ty, "%s", hfc_type_name(t->type));
            hfc_out_event(o, "tensor", "name", t->name, "type", ty, "dims", dims, "offset", off, "bytes", nb, (const char *)NULL);
        }
        {
            const hfc_type_info *ti = hfc_type_lookup(t->type);
            if (!ti || !ti->dequant) unknown_or_unsupported = 1;
        }
    }
    for (i = 0; i < 256; i++) {
        if (type_n[i]) {
            const hfc_type_info *ti = hfc_type_lookup((uint32_t)i);
            snprintf(a, sizeof a, "%llu", (unsigned long long)type_n[i]);
            snprintf(b, sizeof b, "%llu", (unsigned long long)type_b[i]);
            hfc_out_event(o, "types", "name", hfc_type_name((uint32_t)i), "id", (snprintf(c, sizeof c, "%u", (unsigned)i), c),
                          "tensors", a, "bytes", b, "reference_dequant", ti && ti->dequant ? "yes" : "no", (const char *)NULL);
        }
    }
    snprintf(a, sizeof a, "%llu", (unsigned long long)total);
    hfc_out_event(o, "summary", "tensor_bytes", a, "all_types_supported", unknown_or_unsupported ? "no" : "yes", (const char *)NULL);
    gguf_close(&g);
    return HFC_OK;
}

static hfc_status op_tokenize(hfc_session *s, const hfc_opts *eff, const char *body, size_t body_len,
                            int has_body, char *msg, size_t msgcap)
{
    texts_t t;
    gguf_file g;
    hfc_tok *tok = NULL;
    uint32_t *ids = NULL;
    size_t n = 0, i, cap, pos = 0;
    char *list = NULL, cnt[32];
    hfc_status rc;

    if (!eff->model) { snprintf(msg, msgcap, "tokenize needs --model PATH (a GGUF with a tokenizer)"); return HFC_EINVAL; }
    if ((rc = resolve_texts(eff, body, body_len, has_body, &t, msg, msgcap)) != HFC_OK) return rc;
    if ((rc = gguf_open(&g, eff->model, msg, msgcap)) != HFC_OK) { texts_free(&t); return rc; }
    if ((rc = hfc_tok_load(&tok, &g, msg, msgcap)) != HFC_OK) goto out;
    rc = hfc_tok_encode(tok, t.prompt ? t.prompt : "", t.prompt_len, eff->parse_special ? HFC_TOK_PARSE_SPECIAL : 0, &ids, &n);
    if (rc != HFC_OK) { snprintf(msg, msgcap, "tokenization failed: %s", hfc_strerror(rc)); goto out; }
    if (!hfc_mul_size(n, 11, &cap) || !hfc_add_size(cap, 1, &cap)) { rc = HFC_ERANGE; goto out; }
    list = (char *)hfc_malloc(cap);
    if (!list) { rc = HFC_ENOMEM; snprintf(msg, msgcap, "out of memory"); goto out; }
    list[0] = '\0';
    for (i = 0; i < n; i++) pos += (size_t)sprintf(list + pos, i ? " %lu" : "%lu", (unsigned long)ids[i]);
    snprintf(cnt, sizeof cnt, "%lu", (unsigned long)n);
    hfc_out_event(&s->out, "tokens", "count", cnt, "ids", list, (const char *)NULL);
out:
    hfc_free(list); hfc_free(ids); hfc_tok_free(tok); gguf_close(&g); texts_free(&t);
    return rc;
}

static hfc_status op_doctor(hfc_session *s, const hfc_opts *eff, char *msg, size_t msgcap)
{
    pal_meminfo mi;
    char *dir = NULL, *dump;
    char a[64], b[64], c[64], d[64];
    hfc_status rc;
    int loaded = 0;
    uint64_t budget;

    if (!s->cpu_ready) { hfc_cpu_detect(&s->cpu); s->cpu_ready = 1; }
    memset(&mi, 0, sizeof mi);
    if ((rc = pal_meminfo_get(&mi)) != HFC_OK) hfc_log(HFC_LOG_WARN, "msg=\"memory info unavailable\" status=%s", hfc_status_name(rc));
    budget = pal_mem_budget(&mi);
    snprintf(a, sizeof a, "%llu", (unsigned long long)mi.total);
    snprintf(b, sizeof b, "%llu", (unsigned long long)mi.avail);
    snprintf(c, sizeof c, "%llu", (unsigned long long)budget);
    snprintf(d, sizeof d, "%d", mi.overcommit);
    hfc_out_event(&s->out, "mem", "total", a, "avail", b, "budget", c, "overcommit", d, (const char *)NULL);
    snprintf(a, sizeof a, "%llu", (unsigned long long)mi.commit_limit);
    snprintf(b, sizeof b, "%llu", (unsigned long long)mi.committed);
    snprintf(c, sizeof c, "%llu", (unsigned long long)mi.cgroup_limit);
    hfc_out_event(&s->out, "commit", "limit", a, "committed", b, "cgroup_limit", c, (const char *)NULL);

    dir = eff->cache_dir ? hfc_strdup(eff->cache_dir) : pal_default_cache_dir();
    if (dir && !eff->probe_force && s->session->probe_force == 0 &&
        hfc_probe_load(dir, &s->cpu, mi.total, &s->machine) == HFC_OK) {
        loaded = 1;
    }
    if (!loaded) {
        rc = hfc_probe_run(&s->cpu, 1, eff->probe_reps, &s->machine);
        if (rc != HFC_OK) { snprintf(msg, msgcap, "probe failed: %s", hfc_strerror(rc)); hfc_free(dir); return rc; }
        if (dir) {
            rc = hfc_probe_save(dir, &s->machine);
            if (rc != HFC_OK) hfc_log(HFC_LOG_WARN, "msg=\"could not save machine profile\" dir=%s status=%s", dir, hfc_status_name(rc));
        }
    }
    if (eff->probe_sustained > 0) {
        int k;
        rc = hfc_probe_sustained(&s->cpu, eff->probe_sustained, &s->machine);
        if (rc != HFC_OK) { snprintf(msg, msgcap, "sustained probe failed: %s", hfc_strerror(rc)); hfc_free(dir); return rc; }
        for (k = 0; k < s->machine.nsustained; k++) {
            char t[16], gf[32];
            snprintf(t, sizeof t, "%d", k + 1);
            snprintf(gf, sizeof gf, "%.1f", s->machine.sustained_trace[k]);
            hfc_out_event(&s->out, "sustained", "second", t, "gflops", gf, (const char *)NULL);
        }
        if (dir) (void)hfc_probe_save(dir, &s->machine);
    }
    s->machine_ready = 1;
    hfc_out_event(&s->out, "profile", "source", loaded ? "cache" : "measured", "dir", dir ? dir : "(none)", (const char *)NULL);
    dump = capture(probe_fn, &s->cpu, &s->machine);
    if (dump) { emit_lines(&s->out, "doctor", dump); hfc_free(dump); }
    hfc_free(dir);
    return HFC_OK;
}

/* ---- dispatch ----------------------------------------------------------------- */

void hfc_emit_failed_request(hfc_session *s, const char *id, hfc_status st, const char *msg)
{
    s->n_requests++;
    s->n_errors++;
    hfc_out_event(&s->out, "begin", "id", id ? id : "", "op", "?", (const char *)NULL);
    hfc_out_event(&s->out, "done", "status", "error", "code", hfc_status_name(st), "msg", msg ? msg : "", (const char *)NULL);
}

hfc_status hfc_run_request(hfc_session *s, const hfc_opts *eff, const char *body, size_t body_len, int has_body)
{
    char msg[512] = "";
    hfc_status rc;
    const char *op = eff->op ? eff->op : "generate";
    double t0 = pal_now();
    char ms[48];

    s->n_requests++;
    hfc_out_event(&s->out, "begin", "id", eff->id ? eff->id : "", "op", op, (const char *)NULL);
    if (strcmp(op, "echo") == 0)         rc = op_echo(s, eff, body, body_len, has_body, msg, sizeof msg);
    else if (strcmp(op, "inspect") == 0) rc = op_inspect(s, eff, msg, sizeof msg);
    else if (strcmp(op, "doctor") == 0)  rc = op_doctor(s, eff, msg, sizeof msg);
    else if (strcmp(op, "tokenize") == 0) rc = op_tokenize(s, eff, body, body_len, has_body, msg, sizeof msg);
    else if (strcmp(op, "generate") == 0) {
        rc = HFC_ENOTSUP;
        snprintf(msg, sizeof msg, "generation is not implemented yet (phase 1)");
    } else {
        rc = HFC_EINVAL;
        snprintf(msg, sizeof msg, "unknown --op '%s' (generate, inspect, doctor, tokenize, echo)", op);
    }
    snprintf(ms, sizeof ms, "%.1f", (pal_now() - t0) * 1000.0);
    if (rc == HFC_OK) {
        hfc_out_event(&s->out, "done", "status", "ok", "ms", ms, (const char *)NULL);
    } else {
        s->n_errors++;
        hfc_out_event(&s->out, "done", "status", "error", "code", hfc_status_name(rc), "msg", msg[0] ? msg : hfc_strerror(rc), "ms", ms, (const char *)NULL);
    }
    return rc;
}
