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
#include "model.h"
#include "pool.h"
#include "sample.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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

/* ---- chat templates ------------------------------------------------------------------------
 * The prompt becomes one user turn (after an optional system turn) followed by the opening of the
 * assistant turn. Template markers are tokenized as special tokens; the user's text never is
 * (whatever --parse-special says), so a prompt containing "<|im_end|>" cannot forge a turn boundary. */

typedef struct { uint32_t *ids; size_t n, cap; } idvec;

static hfc_status idvec_text(idvec *v, const hfc_tok *tok, const char *text, size_t len, unsigned flags)
{
    uint32_t *part = NULL;
    size_t np = 0;
    hfc_status rc;
    if (len == 0) return HFC_OK;
    if ((rc = hfc_tok_encode(tok, text, len, flags, &part, &np)) != HFC_OK) return rc;
    if (v->n + np > v->cap) {
        size_t nc = v->cap ? v->cap : 64;
        uint32_t *g;
        while (nc < v->n + np) nc *= 2;
        g = (uint32_t *)hfc_realloc(v->ids, nc * sizeof(uint32_t));
        if (!g) { hfc_free(part); return HFC_ENOMEM; }
        v->ids = g; v->cap = nc;
    }
    memcpy(v->ids + v->n, part, np * sizeof(uint32_t));
    v->n += np;
    hfc_free(part);
    return HFC_OK;
}

/* tokenize prefix + text as one string so the pre-tokenizer sees what the reference template engine would */
static hfc_status idvec_cat(idvec *v, const hfc_tok *tok, const char *pre, const char *text, size_t len, unsigned flags)
{
    size_t pl = strlen(pre);
    char *buf = (char *)hfc_malloc(pl + len + 1);
    hfc_status rc;
    if (!buf) return HFC_ENOMEM;
    memcpy(buf, pre, pl); memcpy(buf + pl, text, len);
    rc = idvec_text(v, tok, buf, pl + len, flags);
    hfc_free(buf);
    return rc;
}

static int tok_has_special(const hfc_tok *tok, const char *s)
{
    uint32_t *ids = NULL;
    size_t n = 0;
    int ok;
    if (hfc_tok_encode(tok, s, strlen(s), HFC_TOK_PARSE_SPECIAL, &ids, &n) != HFC_OK) return 0;
    ok = n == 1;
    if (ok) {                                     /* a lone token that decodes back to the marker is a real special token */
        const unsigned char *p; size_t pl;
        ok = hfc_tok_piece(tok, ids[0], 1, &p, &pl) == HFC_OK && pl == strlen(s) && memcmp(p, s, pl) == 0;
    }
    hfc_free(ids);
    return ok;
}

static hfc_status build_chat_ids(const hfc_tok *tok, const char *arch, const hfc_opts *eff, int mode, const texts_t *t, uint32_t **out, size_t *nout,
                                 char *msg, size_t cap)
{
    unsigned user_flags = 0;                      /* user text is never parsed for special tokens */
    const char *sys = t->system;
    size_t sys_len = t->system_len;
    idvec v = { NULL, 0, 0 };
    hfc_status rc = HFC_OK;
    const char *prompt = t->prompt ? t->prompt : "";
    (void)eff;

    if (mode == 1) {                              /* auto: pick by the markers the vocabulary knows */
        if (tok_has_special(tok, "<|im_start|>") && tok_has_special(tok, "<|im_end|>")) mode = 2;
        else if (tok_has_special(tok, "<|start_header_id|>") && tok_has_special(tok, "<|eot_id|>")) mode = 3;
        else { snprintf(msg, cap, "--chat auto: no known chat template matches this vocabulary; use --chat chatml|llama3 or --chat raw"); return HFC_ENOTSUP; }
    }
    if ((mode == 2 && !(tok_has_special(tok, "<|im_start|>") && tok_has_special(tok, "<|im_end|>"))) ||
        (mode == 3 && !(tok_has_special(tok, "<|start_header_id|>") && tok_has_special(tok, "<|eot_id|>") && tok_has_special(tok, "<|begin_of_text|>")))) {
        snprintf(msg, cap, "this vocabulary has no special tokens for the requested chat template");
        return HFC_ENOTSUP;
    }
    if (mode == 2) {
        static const char dflt[] = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        if (!sys && arch && strcmp(arch, "qwen2") == 0) { sys = dflt; sys_len = sizeof dflt - 1; }
#define MARK(s) if ((rc = idvec_text(&v, tok, s, strlen(s), HFC_TOK_PARSE_SPECIAL)) != HFC_OK) goto fail
#define TEXT(pre, p, l, f) if ((rc = idvec_cat(&v, tok, pre, p, l, f)) != HFC_OK) goto fail
        if (sys) { MARK("<|im_start|>"); TEXT("system\n", sys, sys_len, user_flags); MARK("<|im_end|>"); MARK("\n"); }
        MARK("<|im_start|>"); TEXT("user\n", prompt, t->prompt_len, user_flags); MARK("<|im_end|>"); MARK("\n");
        MARK("<|im_start|>"); TEXT("assistant\n", "", 0, 0);
    } else {                                      /* llama3 */
        MARK("<|begin_of_text|>");
        if (sys) { MARK("<|start_header_id|>"); MARK("system"); MARK("<|end_header_id|>"); TEXT("\n\n", sys, sys_len, user_flags); MARK("<|eot_id|>"); }
        MARK("<|start_header_id|>"); MARK("user"); MARK("<|end_header_id|>"); TEXT("\n\n", prompt, t->prompt_len, user_flags); MARK("<|eot_id|>");
        MARK("<|start_header_id|>"); MARK("assistant"); MARK("<|end_header_id|>"); TEXT("\n\n", "", 0, 0);
    }
#undef MARK
#undef TEXT
    *out = v.ids; *nout = v.n;
    return HFC_OK;
fail:
    hfc_free(v.ids);
    snprintf(msg, cap, "chat template tokenization failed: %s", hfc_strerror(rc));
    return rc;
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
    msg[0] = 0;
    hfc_status rc;

    if (!eff->model) { snprintf(msg, msgcap, "tokenize needs --model PATH (a GGUF with a tokenizer)"); return HFC_EINVAL; }
    if ((rc = resolve_texts(eff, body, body_len, has_body, &t, msg, msgcap)) != HFC_OK) return rc;
    if ((rc = gguf_open(&g, eff->model, msg, msgcap)) != HFC_OK) { texts_free(&t); return rc; }
    if ((rc = hfc_tok_load(&tok, &g, msg, msgcap)) != HFC_OK) goto out;
    if (eff->chat != 0 || t.system) rc = build_chat_ids(tok, gguf_get_str(&g, "general.architecture"), eff, eff->chat != 0 ? eff->chat : 1, &t, &ids, &n, msg, msgcap);
    else rc = hfc_tok_encode(tok, t.prompt ? t.prompt : "", t.prompt_len, eff->parse_special ? HFC_TOK_PARSE_SPECIAL : 0, &ids, &n);
    if (rc != HFC_OK) { if (!msg[0]) snprintf(msg, msgcap, "tokenization failed: %s", hfc_strerror(rc)); goto out; }
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

/* ---- resident model + generate ---------------------------------------------------------- */

typedef struct hfc_resident {
    char      *path;
    uint64_t   size;
    long       mtime;
    gguf_file  g;
    int        g_open;
    hfc_model *model;
    hfc_tok   *tok;
    char       tok_err[200];
    /* prefix cache: the KV state of the previous request, valid for kv_ids[0..kv_n) */
    hfc_ctx   *kvctx;
    hfc_pool  *kv_pool;
    uint32_t  *kv_ids;
    size_t     kv_n, kv_ctx_max, kv_batch;
} hfc_resident;

static void kv_drop(hfc_resident *r)
{
    hfc_ctx_free(r->kvctx);
    hfc_free(r->kv_ids);
    r->kvctx = NULL; r->kv_ids = NULL; r->kv_pool = NULL; r->kv_n = 0;
}

static void resident_free(hfc_resident *r)
{
    if (!r) return;
    kv_drop(r);
    hfc_tok_free(r->tok);
    hfc_model_free(r->model);
    if (r->g_open) gguf_close(&r->g);
    hfc_free(r->path);
    hfc_free(r);
}

void hfc_session_close(hfc_session *s)
{
    resident_free(s->res);
    s->res = NULL;
    hfc_pool_free(s->pool);
    s->pool = NULL;
}

/* The pool matches --threads (0 = one thread per physical core). */
static hfc_status session_pool(hfc_session *s, hfc_pool **out)
{
    int n = s->session->threads;
    if (n <= 0) {
        if (!s->cpu_ready) { hfc_cpu_detect(&s->cpu); s->cpu_ready = 1; }
        n = s->cpu.physical > 0 ? s->cpu.physical : 1;
    }
    if (n > 64) n = 64;
    if (!s->pool || s->pool_n != n) {
        hfc_pool *p;
        hfc_status rc = hfc_pool_new(&p, n);
        if (rc != HFC_OK) return rc;
        if (s->res) kv_drop(s->res);                  /* its context points into the old pool */
        hfc_pool_free(s->pool);
        s->pool = p;
        s->pool_n = n;
    }
    *out = s->pool;
    return HFC_OK;
}

static hfc_status resident_get(hfc_session *s, const char *path, hfc_resident **out, char *msg, size_t cap)
{
    struct stat st;
    hfc_resident *r;
    hfc_status rc;
    hfc_cpu *cpu;
    if (stat(path, &st) != 0) { snprintf(msg, cap, "cannot open '%s'", path); return HFC_EIO; }
    r = s->res;
    if (r && strcmp(r->path, path) == 0 && r->size == (uint64_t)st.st_size && r->mtime == (long)st.st_mtime) {
        *out = r;
        return HFC_OK;
    }
    resident_free(r);
    s->res = NULL;
    r = (hfc_resident *)hfc_calloc(1, sizeof *r);
    if (!r) { snprintf(msg, cap, "out of memory"); return HFC_ENOMEM; }
    r->path = hfc_strdup(path);
    if (!r->path) { resident_free(r); snprintf(msg, cap, "out of memory"); return HFC_ENOMEM; }
    r->size = (uint64_t)st.st_size;
    r->mtime = (long)st.st_mtime;
    if ((rc = gguf_open(&r->g, path, msg, cap)) != HFC_OK) { resident_free(r); return rc; }
    r->g_open = 1;
    if (!s->cpu_ready) { hfc_cpu_detect(&s->cpu); s->cpu_ready = 1; }
    cpu = &s->cpu;
    if ((rc = hfc_model_load(&r->model, &r->g, hfc_kernels_for(cpu), msg, cap)) != HFC_OK) { resident_free(r); return rc; }
    if (hfc_tok_load(&r->tok, &r->g, r->tok_err, sizeof r->tok_err) != HFC_OK) r->tok = NULL;   /* not fatal: --prompt-ids still works */
    s->res = r;
    *out = r;
    return HFC_OK;
}

static hfc_status parse_ids(const char *s, uint32_t **ids, size_t *n, uint32_t n_vocab, char *msg, size_t cap)
{
    size_t count = 0, i = 0;
    const char *p;
    uint32_t *v;
    for (p = s; *p; ) {
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;
        while (*p && *p != ' ' && *p != ',' && *p != '\t' && *p != '\n') p++;
        count++;
    }
    v = (uint32_t *)hfc_malloc((count ? count : 1) * sizeof(uint32_t));
    if (!v) { snprintf(msg, cap, "out of memory"); return HFC_ENOMEM; }
    for (p = s; *p && i < count; ) {
        char *end;
        unsigned long long x;
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;
        x = strtoull(p, &end, 10);
        if (end == p || x >= n_vocab) { hfc_free(v); snprintf(msg, cap, "bad token id in --prompt-ids (vocabulary has %u tokens)", n_vocab); return HFC_EINVAL; }
        v[i++] = (uint32_t)x;
        p = end;
    }
    *ids = v;
    *n = i;
    return HFC_OK;
}

#define HFC_MAX_SEQ 64

static void emit_profile(hfc_out *o, const char *stage, double extra_sample)
{
    static const char *names[HFC_PROF_N] = { "embed", "glue", "qkv", "attn", "wo", "gate_up", "silu", "down", "head", "quant" };
    double p[HFC_PROF_N];
    char v[HFC_PROF_N][24], smp[24];
    int i;
    hfc_prof_take(p);
    for (i = 0; i < HFC_PROF_N; i++) snprintf(v[i], sizeof v[i], "%.1f", p[i] * 1000.0);
    snprintf(smp, sizeof smp, "%.1f", extra_sample * 1000.0);
    hfc_out_event(o, "profile", "stage", stage, names[0], v[0], names[1], v[1], names[2], v[2], names[3], v[3], names[4], v[4],
                  names[5], v[5], names[6], v[6], names[7], v[7], names[8], v[8], names[9], v[9], "sample", smp, (const char *)NULL);
}

volatile int hfc_cancel;
volatile int hfc_busy;

typedef struct {
    hfc_rng        rng;
    hfc_ctx       *ctx;
    float         *logits;
    int            active;
    const char    *stop;
    size_t         generated, npend;
    unsigned char  pend[16];
    unsigned char *hb;          /* text held back until it cannot be the start of a stop string */
    size_t         hb_len, hb_cap;
} gseq;

typedef struct { unsigned char *buf; size_t n, maxlen; size_t off[8], len[8]; } stopset;

static int hexv(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

/* --stop: backslash escapes (\n \t \r \\ \xHH); several stop strings are separated by \x1f. */
static hfc_status stops_parse(const char *src, stopset *ss)
{
    size_t n = src ? strlen(src) : 0, i, o = 0, start = 0;
    memset(ss, 0, sizeof *ss);
    if (n == 0) return HFC_OK;
    ss->buf = (unsigned char *)hfc_malloc(n + 1);
    if (!ss->buf) return HFC_ENOMEM;
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c == '\\' && i + 1 < n) {
            char e = src[i + 1];
            if (e == 'n') { c = '\n'; i++; } else if (e == 't') { c = '\t'; i++; } else if (e == 'r') { c = '\r'; i++; }
            else if (e == '\\') { c = '\\'; i++; }
            else if (e == 'x' && i + 3 < n + 0 && hexv(src[i + 2]) >= 0 && hexv(src[i + 3]) >= 0) { c = (unsigned char)(hexv(src[i + 2]) * 16 + hexv(src[i + 3])); i += 3; }
        }
        if (c == 0x1f) {
            if (o > start && ss->n < 8) { ss->off[ss->n] = start; ss->len[ss->n] = o - start; ss->n++; }
            start = o;
            continue;
        }
        ss->buf[o++] = c;
    }
    if (o > start && ss->n < 8) { ss->off[ss->n] = start; ss->len[ss->n] = o - start; ss->n++; }
    for (i = 0; i < ss->n; i++) if (ss->len[i] > ss->maxlen) ss->maxlen = ss->len[i];
    return HFC_OK;
}

static const unsigned char *find_bytes(const unsigned char *h, size_t hl, const unsigned char *nd, size_t nl)
{
    size_t i;
    if (nl == 0 || hl < nl) return NULL;
    for (i = 0; i + nl <= hl; i++) if (h[i] == nd[0] && memcmp(h + i, nd, nl) == 0) return h + i;
    return NULL;
}

static void emit_text(hfc_out *o, const char *seq, const void *data, size_t len);

/* Append text for one sequence; releases what can no longer be part of a stop string. Returns 1 when a
 * stop string was found (the text before it is released, the sequence is finished). */
static int seq_feed(gseq *g, hfc_out *o, const char *sq, const unsigned char *data, size_t len, const stopset *ss, int flush)
{
    size_t safe, i;
    int hit = 0;
    if (g->hb_len + len > g->hb_cap) {
        size_t nc = g->hb_cap ? g->hb_cap : 64;
        unsigned char *nb;
        while (nc < g->hb_len + len) nc *= 2;
        nb = (unsigned char *)hfc_realloc(g->hb, nc);
        if (!nb) { emit_text(o, sq, data, len); return 0; }      /* out of memory: no stop checking, but no lost text */
        g->hb = nb; g->hb_cap = nc;
    }
    if (len) { memcpy(g->hb + g->hb_len, data, len); g->hb_len += len; }
    safe = g->hb_len;
    if (ss && ss->n && !flush) {
        const unsigned char *best = NULL;
        for (i = 0; i < ss->n; i++) {
            const unsigned char *p = find_bytes(g->hb, g->hb_len, ss->buf + ss->off[i], ss->len[i]);
            if (p && (!best || p < best)) best = p;
        }
        if (best) { safe = (size_t)(best - g->hb); hit = 1; }
        else if (g->hb_len >= ss->maxlen) safe = hfc_utf8_complete_prefix(g->hb, g->hb_len - (ss->maxlen - 1));
        else safe = 0;
    }
    if (safe) emit_text(o, sq, g->hb, safe);
    if (hit) g->hb_len = 0;
    else if (g->hb_len > safe) { memmove(g->hb, g->hb + safe, g->hb_len - safe); g->hb_len -= safe; }
    else g->hb_len = 0;
    return hit;
}

static void emit_text(hfc_out *o, const char *seq, const void *data, size_t len)
{
    if (seq) hfc_out_payload(o, "text", data, len, "seq", seq, (const char *)NULL);
    else hfc_out_payload(o, "text", data, len, (const char *)NULL);
}

static hfc_status op_generate(hfc_session *s, const hfc_opts *eff, const char *body, size_t body_len,
                              int has_body, char *msg, size_t cap)
{
    hfc_resident *res = NULL;
    texts_t t;
    uint32_t *ids = NULL;
    size_t n_prompt = 0, ngen_max, i, pos, cached = 0, nvocab;
    int touched = 0;
    hfc_ctx *ctx = NULL;
    hfc_pool *pool = NULL;
    float *logits = NULL;
    hfc_sampler sp;
    float *lbuf = NULL, *tmpl = NULL;
    gseq *gs = NULL;
    size_t nseq = 1, step;
    char d_n[24];
    hfc_status rc;
    double t0, t_prefill, t_dec0, t_sample = 0.0;
    const char *stop0 = "length";
    stopset ss;
    double t_req = pal_now();
    int abort_why = 0;
    char a[64], b[64], c[64], d[64], e[64];
    int top_n = eff->logprobs > 64 ? 64 : eff->logprobs;
    size_t batch = (size_t)(s->session->batch > 0 ? s->session->batch : 64);

    memset(&t, 0, sizeof t);
    memset(&ss, 0, sizeof ss);
    if (!eff->model) { snprintf(msg, cap, "generate needs --model PATH"); return HFC_EINVAL; }
    if ((rc = stops_parse(eff->stop, &ss)) != HFC_OK) { snprintf(msg, cap, "out of memory"); return rc; }
    if ((rc = resident_get(s, eff->model, &res, msg, cap)) != HFC_OK) { hfc_free(ss.buf); return rc; }
    nvocab = (size_t)res->model->hp.n_vocab;

    if (eff->prompt_ids) {
        if (has_body || eff->prompt || eff->prompt_file) { snprintf(msg, cap, "--prompt-ids cannot be combined with a text prompt"); rc = HFC_EINVAL; goto out; }
        if (eff->system || eff->system_file || eff->chat) { snprintf(msg, cap, "--prompt-ids is already tokenized: --system and --chat do not apply"); rc = HFC_EINVAL; goto out; }
        if ((rc = parse_ids(eff->prompt_ids, &ids, &n_prompt, (uint32_t)nvocab, msg, cap)) != HFC_OK) goto out;
    } else {
        if ((rc = resolve_texts(eff, body, body_len, has_body, &t, msg, cap)) != HFC_OK) return rc;
        if (!res->tok) { snprintf(msg, cap, "this model has no usable tokenizer (%s); use --prompt-ids", res->tok_err); rc = HFC_ENOTSUP; goto out; }
        if (eff->chat != 0 || t.system) {
            int mode = eff->chat != 0 ? eff->chat : 1;
            rc = build_chat_ids(res->tok, res->model->hp.arch, eff, mode, &t, &ids, &n_prompt, msg, cap);
            if (rc != HFC_OK) goto out;
        } else {
            rc = hfc_tok_encode(res->tok, t.prompt ? t.prompt : "", t.prompt_len,
                                HFC_TOK_ADD_BOS | (eff->parse_special ? HFC_TOK_PARSE_SPECIAL : 0), &ids, &n_prompt);
            if (rc != HFC_OK) { snprintf(msg, cap, "tokenization failed: %s", hfc_strerror(rc)); goto out; }
        }
    }
    if (n_prompt == 0) { snprintf(msg, cap, "empty prompt"); rc = HFC_EINVAL; goto out; }
    if (n_prompt >= (size_t)s->session->ctx_max) {
        snprintf(msg, cap, "prompt has %lu tokens but --ctx-max is %llu", (unsigned long)n_prompt, (unsigned long long)s->session->ctx_max);
        rc = HFC_ERANGE; goto out;
    }
    ngen_max = (size_t)eff->max_tokens;
    if (ngen_max > (size_t)s->session->ctx_max - n_prompt) { ngen_max = (size_t)s->session->ctx_max - n_prompt; stop0 = "ctx"; }
    if ((rc = session_pool(s, &pool)) != HFC_OK) { snprintf(msg, cap, "cannot start worker threads: %s", hfc_strerror(rc)); goto out; }
    /* one context lives with the resident model; it keeps the KV state of the previous request */
    if (!res->kvctx || res->kv_pool != pool || res->kv_ctx_max != (size_t)s->session->ctx_max || res->kv_batch != batch) {
        kv_drop(res);
        res->kv_ids = (uint32_t *)hfc_malloc((size_t)s->session->ctx_max * sizeof(uint32_t));
        if (!res->kv_ids) { rc = HFC_ENOMEM; snprintf(msg, cap, "out of memory"); goto out; }
        if ((rc = hfc_ctx_new(&res->kvctx, res->model, (size_t)s->session->ctx_max, batch, pool)) != HFC_OK) {
            snprintf(msg, cap, "cannot allocate the inference context: %s", hfc_strerror(rc)); goto out;
        }
        res->kv_pool = pool; res->kv_ctx_max = (size_t)s->session->ctx_max; res->kv_batch = batch;
    }
    ctx = res->kvctx;
    if (eff->prefix_cache) {
        while (cached < res->kv_n && cached < n_prompt && res->kv_ids[cached] == ids[cached]) cached++;
        if (cached >= n_prompt) cached = n_prompt - 1;        /* the last prompt token is always evaluated: it yields the logits */
    }
    hfc_ctx_truncate(ctx, cached);
    res->kv_n = cached;
    touched = 1;
    logits = (float *)hfc_malloc(nvocab * sizeof(float));
    if (!logits) { rc = HFC_ENOMEM; snprintf(msg, cap, "out of memory"); goto out; }

    snprintf(a, sizeof a, "%lu", (unsigned long)n_prompt);
    snprintf(b, sizeof b, "%d", hfc_pool_size(pool));
    hfc_out_event(&s->out, "prompt", "tokens", a, "threads", b, (const char *)NULL);

    hfc_prof_enable(eff->profile);
    if (eff->profile) { double junk[HFC_PROF_N]; hfc_prof_take(junk); }
    t0 = pal_now();
    for (pos = cached; pos < n_prompt; ) {
        size_t nb = n_prompt - pos < batch ? n_prompt - pos : batch;
        if (hfc_cancel) abort_why = 1; else if (eff->timeout > 0 && pal_now() - t_req > eff->timeout) abort_why = 2;
        if (abort_why) { snprintf(msg, cap, "%s during prefill", abort_why == 1 ? "cancelled" : "timed out"); rc = HFC_ECANCEL; goto out; }
        rc = hfc_ctx_forward(ctx, ids + pos, nb, pos + nb == n_prompt ? logits : NULL);
        if (rc != HFC_OK) { snprintf(msg, cap, "prefill failed at token %lu: %s", (unsigned long)pos, hfc_strerror(rc)); goto out; }
        pos += nb;
    }
    memcpy(res->kv_ids + cached, ids + cached, (n_prompt - cached) * sizeof(uint32_t));
    res->kv_n = n_prompt;
    t_prefill = pal_now() - t0;
    snprintf(a, sizeof a, "%lu", (unsigned long)n_prompt);
    snprintf(b, sizeof b, "%.1f", t_prefill * 1000.0);
    snprintf(c, sizeof c, "%.2f", t_prefill > 0 ? (double)(n_prompt - cached) / t_prefill : 0.0);
    snprintf(d, sizeof d, "%lu", (unsigned long)hfc_ctx_kv_bytes(ctx));
    snprintf(e, sizeof e, "%lu", (unsigned long)cached);
    hfc_out_event(&s->out, "prefill", "tokens", a, "cached", e, "ms", b, "tok_per_s", c, "kv_bytes", d, (const char *)NULL);

    sp.temp = (float)eff->temp; sp.top_k = eff->top_k; sp.top_p = (float)eff->top_p; sp.min_p = (float)eff->min_p;
    if (eff->profile) emit_profile(&s->out, "prefill", 0.0);
    nseq = (size_t)eff->n;
    snprintf(d_n, sizeof d_n, "%lu", (unsigned long)nseq);
    if (nseq > 1 && nseq > batch) { snprintf(msg, cap, "--n %lu exceeds --batch %lu", (unsigned long)nseq, (unsigned long)batch); rc = HFC_EINVAL; goto out; }
    gs = (gseq *)hfc_calloc(nseq, sizeof *gs);
    lbuf = (float *)hfc_malloc(nseq * nvocab * sizeof(float));
    tmpl = nseq > 1 ? (float *)hfc_malloc(nseq * nvocab * sizeof(float)) : NULL;
    if (!gs || !lbuf || (nseq > 1 && !tmpl)) { rc = HFC_ENOMEM; snprintf(msg, cap, "out of memory"); goto out; }
    for (i = 0; i < nseq; i++) {
        gs[i].logits = lbuf + i * nvocab;
        gs[i].active = 1;
        gs[i].stop = stop0;
        hfc_rng_seed(&gs[i].rng, eff->seed + (uint64_t)i);
        memcpy(gs[i].logits, logits, nvocab * sizeof(float));
        if (nseq == 1) gs[i].ctx = ctx;
        else {
            if ((rc = hfc_ctx_fork(ctx, &gs[i].ctx)) != HFC_OK) { snprintf(msg, cap, "cannot fork the context: %s", hfc_strerror(rc)); goto out; }
        }
    }
    t_dec0 = pal_now();
    for (step = 0; step < ngen_max; step++) {
        hfc_ctx *act_ctx[HFC_MAX_SEQ];
        if (hfc_cancel) abort_why = 1; else if (eff->timeout > 0 && pal_now() - t_req > eff->timeout) abort_why = 2;
        if (abort_why) {
            size_t k2;
            for (k2 = 0; k2 < nseq; k2++) if (gs[k2].active) { gs[k2].stop = abort_why == 1 ? "cancel" : "timeout"; gs[k2].active = 0; }
            break;
        }
        uint32_t act_tok[HFC_MAX_SEQ];
        size_t act_idx[HFC_MAX_SEQ], nact = 0, si;
        for (si = 0; si < nseq; si++) {
            gseq *g = &gs[si];
            uint32_t tok;
            const unsigned char *piece;
            size_t plen, take;
            char sq[24], pos_[24], ids_[24], lp_[32];
            if (!g->active) continue;
            snprintf(sq, sizeof sq, "%lu", (unsigned long)si);
            { double ts = eff->profile ? pal_now() : 0.0;
              rc = hfc_sample(g->logits, nvocab, &sp, &g->rng, &tok);
              if (eff->profile) t_sample += pal_now() - ts; }
            if (rc != HFC_OK) { snprintf(msg, cap, "sampling failed"); goto out; }
            if (res->tok && hfc_tok_is_eog(res->tok, tok)) { g->stop = "eos"; g->active = 0; continue; }
            snprintf(pos_, sizeof pos_, "%lu", (unsigned long)step);
            snprintf(ids_, sizeof ids_, "%u", tok);
            if (top_n > 0) {
                hfc_top top[64];
                char list[64 * 24], *w = list;
                int j;
                hfc_top_logprobs(g->logits, nvocab, top_n, top);
                *w = '\0';
                for (j = 0; j < top_n; j++) w += sprintf(w, j ? " %u:%.6f" : "%u:%.6f", top[j].id, top[j].logprob);
                snprintf(lp_, sizeof lp_, "%.6f", hfc_logprob(g->logits, nvocab, tok));
                if (nseq > 1) hfc_out_event(&s->out, "token", "seq", sq, "pos", pos_, "id", ids_, "logprob", lp_, "top", list, (const char *)NULL);
                else hfc_out_event(&s->out, "token", "pos", pos_, "id", ids_, "logprob", lp_, "top", list, (const char *)NULL);
            } else {
                if (nseq > 1) hfc_out_event(&s->out, "token", "seq", sq, "pos", pos_, "id", ids_, (const char *)NULL);
                else hfc_out_event(&s->out, "token", "pos", pos_, "id", ids_, (const char *)NULL);
            }
            g->generated++;
            if (res->tok && hfc_tok_piece(res->tok, tok, 0, &piece, &plen) == HFC_OK && plen > 0) {
                unsigned char *buf = (unsigned char *)hfc_malloc(g->npend + plen);
                if (buf) {
                    memcpy(buf, g->pend, g->npend);
                    memcpy(buf + g->npend, piece, plen);
                    take = hfc_utf8_complete_prefix(buf, g->npend + plen);
                    g->npend = g->npend + plen - take;
                    if (g->npend > sizeof g->pend) { take += g->npend; g->npend = 0; }      /* not valid UTF-8: pass the bytes on */
                    else memcpy(g->pend, buf + take, g->npend);
                    if (take && seq_feed(g, &s->out, nseq > 1 ? sq : NULL, buf, take, &ss, 0)) { g->stop = "stop"; g->active = 0; g->npend = 0; }
                    hfc_free(buf);
                }
            }
            if (!g->active) continue;
            if (step + 1 == ngen_max) { g->active = 0; continue; }
            act_ctx[nact] = g->ctx; act_tok[nact] = tok; act_idx[nact] = si; nact++;
        }
        if (nact == 0) break;
        if (nseq == 1) {
            rc = hfc_ctx_forward(ctx, &act_tok[0], 1, gs[0].logits);
            if (rc == HFC_OK) res->kv_ids[res->kv_n++] = act_tok[0];
        } else {
            rc = hfc_ctx_forward_multi(ctx, act_ctx, act_tok, nact, tmpl);
            if (rc == HFC_OK) { size_t j; for (j = 0; j < nact; j++) memcpy(gs[act_idx[j]].logits, tmpl + j * nvocab, nvocab * sizeof(float)); }
        }
        if (rc != HFC_OK) { snprintf(msg, cap, "decode failed at token %lu: %s", (unsigned long)step, hfc_strerror(rc)); goto out; }
    }
    if (eff->profile) emit_profile(&s->out, "decode", t_sample);
    {
        double dt = pal_now() - t_dec0;
        size_t si, total = 0;
        for (si = 0; si < nseq; si++) {
            gseq *g = &gs[si];
            char sq[24];
            snprintf(sq, sizeof sq, "%lu", (unsigned long)si);
            if (strcmp(g->stop, "stop") != 0) seq_feed(g, &s->out, nseq > 1 ? sq : NULL, g->pend, g->npend, &ss, 1);
            total += g->generated;
        }
        for (si = 0; si < nseq; si++) {
            gseq *g = &gs[si];
            char sq[24];
            snprintf(sq, sizeof sq, "%lu", (unsigned long)si);
            snprintf(a, sizeof a, "%lu", (unsigned long)g->generated);
            snprintf(b, sizeof b, "%.1f", dt * 1000.0);
            snprintf(c, sizeof c, "%.2f", dt > 0 ? (double)g->generated / dt : 0.0);
            if (nseq > 1) hfc_out_event(&s->out, "gen", "seq", sq, "tokens", a, "ms", b, "tok_per_s", c, "stop", g->stop, (const char *)NULL);
            else hfc_out_event(&s->out, "gen", "tokens", a, "ms", b, "tok_per_s", c, "stop", g->stop, (const char *)NULL);
        }
        if (nseq > 1) {
            snprintf(a, sizeof a, "%lu", (unsigned long)total);
            snprintf(b, sizeof b, "%.1f", dt * 1000.0);
            snprintf(c, sizeof c, "%.2f", dt > 0 ? (double)total / dt : 0.0);
            hfc_out_event(&s->out, "genall", "sequences", d_n, "tokens", a, "ms", b, "tok_per_s", c, (const char *)NULL);
        }
    }
    rc = HFC_OK;
out:
    hfc_free(ss.buf);
    hfc_prof_enable(0);
    if (gs) { size_t si; for (si = 0; si < nseq; si++) if (gs[si].ctx != ctx) hfc_ctx_free(gs[si].ctx); }
    if (gs) { size_t si; for (si = 0; si < nseq; si++) hfc_free(gs[si].hb); }
    hfc_free(gs); hfc_free(lbuf); hfc_free(tmpl);
    hfc_free(logits);
    if (rc != HFC_OK && res && touched) kv_drop(res);            /* the cached state may be half-written */
    hfc_free(ids);
    texts_free(&t);
    return rc;
}

static int cmp_dbl2(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static hfc_status op_bench(hfc_session *s, const hfc_opts *eff, char *msg, size_t cap)
{
    hfc_resident *res = NULL;
    hfc_status rc;
    int list[64], nl = 0, li, reps = eff->probe_reps > 0 ? eff->probe_reps : 1;
    size_t nprompt = (size_t)eff->bench_prompt, ngen = (size_t)eff->bench_gen, i;
    uint32_t *ids = NULL;
    float *logits = NULL;
    size_t nvocab, batch = (size_t)(s->session->batch > 0 ? s->session->batch : 64);

    if (!eff->model) { snprintf(msg, cap, "bench needs --model PATH"); return HFC_EINVAL; }
    if ((rc = resident_get(s, eff->model, &res, msg, cap)) != HFC_OK) return rc;
    nvocab = (size_t)res->model->hp.n_vocab;
    if (nprompt + ngen > (size_t)s->session->ctx_max) { snprintf(msg, cap, "bench-prompt + bench-gen exceeds --ctx-max"); return HFC_ERANGE; }
    if (eff->bench_threads) {
        const char *p = eff->bench_threads;
        while (*p && nl < 64) {
            char *end;
            long v = strtol(p, &end, 10);
            if (end == p || v < 1 || v > 64) { snprintf(msg, cap, "bad --bench-threads list"); return HFC_EINVAL; }
            list[nl++] = (int)v;
            p = end;
            while (*p == ',' || *p == ' ') p++;
        }
    } else {
        int phys, v;
        if (!s->cpu_ready) { hfc_cpu_detect(&s->cpu); s->cpu_ready = 1; }
        phys = s->cpu.physical > 0 ? s->cpu.physical : 1;
        for (v = 1; v <= phys && nl < 64; v++) list[nl++] = v;
        if (s->cpu.logical > phys && nl < 64) list[nl++] = s->cpu.logical;
    }
    if (nl == 0) { snprintf(msg, cap, "empty --bench-threads list"); return HFC_EINVAL; }
    ids = (uint32_t *)hfc_malloc((nprompt + 1) * sizeof(uint32_t));
    logits = (float *)hfc_malloc(nvocab * sizeof(float));
    if (!ids || !logits) { hfc_free(ids); hfc_free(logits); snprintf(msg, cap, "out of memory"); return HFC_ENOMEM; }
    { uint64_t x = 88172645463325252ull;
      uint32_t span = nvocab < 2000 ? (uint32_t)nvocab : 2000;
      for (i = 0; i < nprompt; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ids[i] = (uint32_t)(x % span); } }

    for (li = 0; li < nl; li++) {
        double pf[64], dc[64];
        int r;
        char a[32], b[32], c[32], d[32], e[32];
        hfc_pool *pool = NULL;
        if (reps > 64) reps = 64;
        if ((rc = hfc_pool_new(&pool, list[li])) != HFC_OK) { snprintf(msg, cap, "cannot start %d threads", list[li]); goto out; }
        for (r = 0; r < reps; r++) {
            hfc_ctx *ctx = NULL;
            size_t pos;
            double t0, t1, t2;
            if ((rc = hfc_ctx_new(&ctx, res->model, nprompt + ngen, batch > nprompt ? nprompt : batch, pool)) != HFC_OK) {
                snprintf(msg, cap, "cannot allocate the inference context: %s", hfc_strerror(rc)); hfc_pool_free(pool); goto out;
            }
            t0 = pal_now();
            for (pos = 0; pos < nprompt; ) {
                size_t nb = nprompt - pos < batch ? nprompt - pos : batch;
                rc = hfc_ctx_forward(ctx, ids + pos, nb, pos + nb == nprompt ? logits : NULL);
                if (rc != HFC_OK) { snprintf(msg, cap, "bench prefill failed: %s", hfc_strerror(rc)); hfc_ctx_free(ctx); hfc_pool_free(pool); goto out; }
                pos += nb;
            }
            t1 = pal_now();
            for (i = 0; i < ngen; i++) {
                uint32_t tok = 0;
                size_t j;
                for (j = 1; j < nvocab; j++) if (logits[j] > logits[tok]) tok = (uint32_t)j;
                rc = hfc_ctx_forward(ctx, &tok, 1, logits);
                if (rc != HFC_OK) { snprintf(msg, cap, "bench decode failed: %s", hfc_strerror(rc)); hfc_ctx_free(ctx); hfc_pool_free(pool); goto out; }
            }
            t2 = pal_now();
            pf[r] = (double)nprompt / (t1 - t0);
            dc[r] = (double)ngen / (t2 - t1);
            hfc_ctx_free(ctx);
        }
        hfc_pool_free(pool);
        qsort(pf, (size_t)reps, sizeof(double), cmp_dbl2);
        qsort(dc, (size_t)reps, sizeof(double), cmp_dbl2);
        snprintf(a, sizeof a, "%d", list[li]);
        snprintf(b, sizeof b, "%.2f", pf[reps / 2]);
        snprintf(c, sizeof c, "%.2f", dc[reps / 2]);
        snprintf(d, sizeof d, "%lu", (unsigned long)nprompt);
        snprintf(e, sizeof e, "%lu", (unsigned long)ngen);
        hfc_out_event(&s->out, "bench", "threads", a, "prefill_tok_s", b, "decode_tok_s", c, "prompt_tokens", d, "gen_tokens", e, (const char *)NULL);
    }
    rc = HFC_OK;
out:
    hfc_free(ids);
    hfc_free(logits);
    return rc;
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
    hfc_busy = 1;
    hfc_out_event(&s->out, "begin", "id", eff->id ? eff->id : "", "op", op, (const char *)NULL);
    if (strcmp(op, "echo") == 0)         rc = op_echo(s, eff, body, body_len, has_body, msg, sizeof msg);
    else if (strcmp(op, "inspect") == 0) rc = op_inspect(s, eff, msg, sizeof msg);
    else if (strcmp(op, "doctor") == 0)  rc = op_doctor(s, eff, msg, sizeof msg);
    else if (strcmp(op, "bench") == 0)    rc = op_bench(s, eff, msg, sizeof msg);
    else if (strcmp(op, "tokenize") == 0) rc = op_tokenize(s, eff, body, body_len, has_body, msg, sizeof msg);
    else if (strcmp(op, "generate") == 0) rc = op_generate(s, eff, body, body_len, has_body, msg, sizeof msg);
    else {
        rc = HFC_EINVAL;
        snprintf(msg, sizeof msg, "unknown --op '%s' (generate, inspect, doctor, tokenize, bench, echo)", op);
    }
    hfc_busy = 0;
    hfc_cancel = 0;
    snprintf(ms, sizeof ms, "%.1f", (pal_now() - t0) * 1000.0);
    if (rc == HFC_OK) {
        hfc_out_event(&s->out, "done", "status", "ok", "ms", ms, (const char *)NULL);
    } else {
        s->n_errors++;
        hfc_out_event(&s->out, "done", "status", "error", "code", hfc_status_name(rc), "msg", msg[0] ? msg : hfc_strerror(rc), "ms", ms, (const char *)NULL);
    }
    return rc;
}
