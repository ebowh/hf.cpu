/* tok.c - byte-level BPE tokenizer. See tok.h. */
#include "tok.h"
#include "unitab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NONE32 0xffffffffu

typedef struct { uint32_t len, id; } spec_t;

struct hfc_tok {
    uint32_t       n_vocab;
    unsigned char *text;       /* token strings (GPT-2 byte-to-unicode form) */
    size_t        *toff;       /* n_vocab + 1 offsets */
    unsigned char *dec;        /* decoded bytes of each token */
    size_t        *doff;       /* n_vocab + 1 offsets */
    int8_t        *type;       /* GGUF token types (1 normal, 3 control, 4 user defined, ...) */
    uint32_t      *vhash;      /* id + 1 per slot, open addressing */
    size_t         vcap;
    uint32_t       byte_tok[256];
    /* merges: (left,right) -> rank, result */
    uint64_t      *mkey;
    uint32_t      *mrank, *mres;
    size_t         mcap, n_merges;
    /* special tokens, longest first */
    spec_t        *spec;
    size_t         n_spec;
    unsigned char  spec_first[256];
    int64_t        bos, eos, pad;
    int            add_bos;
};

/* ---- byte <-> unicode (GPT-2) ----------------------------------------------------- */

static uint32_t b2u[256];
static int16_t  u2b[0x200];
static int      maps_ready;

static void init_maps(void)
{
    int b, n = 0, i;
    if (maps_ready) return;
    for (i = 0; i < 0x200; i++) u2b[i] = -1;
    for (b = 0; b < 256; b++) {
        int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        b2u[b] = keep ? (uint32_t)b : (uint32_t)(256 + n++);
        u2b[b2u[b]] = (int16_t)b;
    }
    maps_ready = 1;
}

static size_t utf8_put(uint32_t cp, unsigned char *o)
{
    if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) { o[0] = (unsigned char)(0xc0 | (cp >> 6)); o[1] = (unsigned char)(0x80 | (cp & 0x3f)); return 2; }
    if (cp < 0x10000) {
        o[0] = (unsigned char)(0xe0 | (cp >> 12)); o[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
        o[2] = (unsigned char)(0x80 | (cp & 0x3f)); return 3;
    }
    o[0] = (unsigned char)(0xf0 | (cp >> 18)); o[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3f));
    o[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f)); o[3] = (unsigned char)(0x80 | (cp & 0x3f));
    return 4;
}

/* Decode one UTF-8 character at s[i]. Invalid or truncated sequences give
 * cp = 0xffffffff with length 1 (they belong to no Unicode class). */
static uint32_t utf8_get(const unsigned char *s, size_t n, size_t i, size_t *len)
{
    unsigned c = s[i];
    uint32_t cp;
    size_t need, k;
    if (c < 0x80) { *len = 1; return c; }
    if (c >= 0xc2 && c <= 0xdf) { need = 1; cp = c & 0x1f; }
    else if (c >= 0xe0 && c <= 0xef) { need = 2; cp = c & 0x0f; }
    else if (c >= 0xf0 && c <= 0xf4) { need = 3; cp = c & 0x07; }
    else { *len = 1; return 0xffffffffu; }
    if (n - i < need + 1) { *len = 1; return 0xffffffffu; }
    for (k = 1; k <= need; k++) {
        unsigned cc = s[i + k];
        if ((cc & 0xc0) != 0x80) { *len = 1; return 0xffffffffu; }
        cp = (cp << 6) | (cc & 0x3f);
    }
    if ((need == 2 && cp < 0x800) || (need == 3 && (cp < 0x10000 || cp > 0x10ffff)) ||
        (cp >= 0xd800 && cp <= 0xdfff)) { *len = 1; return 0xffffffffu; }
    *len = need + 1;
    return cp;
}

size_t hfc_utf8_complete_prefix(const unsigned char *p, size_t n)
{
    size_t i, keep = 0;
    /* look at the last up-to-3 bytes for an unfinished lead sequence */
    for (i = 1; i <= 3 && i <= n; i++) {
        unsigned c = p[n - i];
        if ((c & 0xc0) == 0x80) continue;                    /* continuation byte */
        {
            size_t need = c >= 0xf0 && c <= 0xf4 ? 4 : c >= 0xe0 ? 3 : c >= 0xc2 && c < 0xe0 ? 2 : 1;
            if (need > i && c >= 0xc2 && c <= 0xf4) keep = i;   /* still incomplete */
        }
        break;
    }
    return n - keep;
}

/* ---- hashing ------------------------------------------------------------------------ */

static uint64_t fnv64(const unsigned char *s, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;
    size_t i;
    for (i = 0; i < n; i++) { h ^= s[i]; h *= 0x100000001b3ull; }
    return h;
}

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}

static uint32_t vocab_find(const hfc_tok *t, const unsigned char *s, size_t n)
{
    size_t h = (size_t)(fnv64(s, n) & (t->vcap - 1));
    while (t->vhash[h]) {
        uint32_t id = t->vhash[h] - 1;
        size_t l = t->toff[id + 1] - t->toff[id];
        if (l == n && memcmp(t->text + t->toff[id], s, n) == 0) return id;
        h = (h + 1) & (t->vcap - 1);
    }
    return NONE32;
}

static int merge_find(const hfc_tok *t, uint32_t l, uint32_t r, uint32_t *rank, uint32_t *res)
{
    uint64_t key = ((uint64_t)l << 32) | r | 0;
    size_t h = (size_t)(mix64(key + 1) & (t->mcap - 1));
    while (t->mkey[h]) {
        if (t->mkey[h] == key + 1) { *rank = t->mrank[h]; *res = t->mres[h]; return 1; }
        h = (h + 1) & (t->mcap - 1);
    }
    return 0;
}

/* ---- loading ------------------------------------------------------------------------ */

void hfc_tok_free(hfc_tok *t)
{
    if (!t) return;
    hfc_free(t->text); hfc_free(t->toff); hfc_free(t->dec); hfc_free(t->doff);
    hfc_free(t->type); hfc_free(t->vhash); hfc_free(t->mkey); hfc_free(t->mrank);
    hfc_free(t->mres); hfc_free(t->spec);
    hfc_free(t);
}

typedef struct { hfc_tok *t; size_t pos; } load_ctx;

static int load_cb(uint64_t idx, const char *s, uint64_t len, void *vctx)
{
    load_ctx *c = (load_ctx *)vctx;
    c->t->toff[idx] = c->pos;
    memcpy(c->t->text + c->pos, s, (size_t)len);
    c->pos += (size_t)len;
    return 0;
}

static int cmp_spec(const void *a, const void *b)
{
    const spec_t *x = (const spec_t *)a, *y = (const spec_t *)b;
    if (x->len != y->len) return x->len < y->len ? 1 : -1;     /* longest first */
    return x->id < y->id ? -1 : x->id > y->id;
}

static uint32_t rd_i32le(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

hfc_status hfc_tok_load(hfc_tok **out, const gguf_file *g, char *err, size_t cap)
{
    hfc_tok *t = NULL;
    const gguf_kv *kt, *kty, *km;
    const char *model, *pre;
    uint64_t n, i;
    size_t j;
    hfc_status rc = HFC_ENOMEM;
    load_ctx lc;
    const unsigned char *base = (const unsigned char *)g->map.base;

#define FAILMSG(code, ...) do { snprintf(err, cap, __VA_ARGS__); rc = (code); goto fail; } while (0)
    *out = NULL;
    init_maps();
    model = gguf_get_str(g, "tokenizer.ggml.model");
    pre = gguf_get_str(g, "tokenizer.ggml.pre");
    if (!model) { snprintf(err, cap, "model has no tokenizer metadata"); return HFC_EFORMAT; }
    if (strcmp(model, "gpt2") != 0) { snprintf(err, cap, "tokenizer model '%s' is not supported yet (only gpt2 BPE)", model); return HFC_ENOTSUP; }
    if (!pre || strcmp(pre, "qwen2") != 0) {
        snprintf(err, cap, "pre-tokenizer '%s' is not supported yet (only qwen2)", pre ? pre : "(none)");
        return HFC_ENOTSUP;
    }
    kt = gguf_find_kv(g, "tokenizer.ggml.tokens");
    km = gguf_find_kv(g, "tokenizer.ggml.merges");
    kty = gguf_find_kv(g, "tokenizer.ggml.token_type");
    if (!kt || kt->type != GGUF_T_ARR || kt->arr_type != GGUF_T_STR) { snprintf(err, cap, "tokenizer.ggml.tokens missing or not a string array"); return HFC_EFORMAT; }
    if (!km || km->type != GGUF_T_ARR || km->arr_type != GGUF_T_STR) { snprintf(err, cap, "tokenizer.ggml.merges missing or not a string array"); return HFC_EFORMAT; }
    n = kt->arr_count;
    if (n == 0 || n > (1u << 24)) { snprintf(err, cap, "implausible vocabulary size"); return HFC_EFORMAT; }

    t = (hfc_tok *)hfc_calloc(1, sizeof *t);
    if (!t) FAILMSG(HFC_ENOMEM, "out of memory");
    t->n_vocab = (uint32_t)n;
    t->bos = t->eos = t->pad = -1;
    t->text = (unsigned char *)hfc_malloc((size_t)kt->arr_bytes);
    t->dec = (unsigned char *)hfc_malloc((size_t)kt->arr_bytes);
    t->toff = (size_t *)hfc_calloc((size_t)n + 1, sizeof(size_t));
    t->doff = (size_t *)hfc_calloc((size_t)n + 1, sizeof(size_t));
    t->type = (int8_t *)hfc_calloc((size_t)n, 1);
    if (!t->text || !t->dec || !t->toff || !t->doff || !t->type) FAILMSG(HFC_ENOMEM, "out of memory");
    lc.t = t; lc.pos = 0;
    if (gguf_foreach_str(g, kt, load_cb, &lc) != HFC_OK) FAILMSG(HFC_EFORMAT, "corrupt token array");
    t->toff[n] = lc.pos;

    /* token types */
    for (i = 0; i < n; i++) t->type[i] = 1;
    if (kty && kty->type == GGUF_T_ARR && kty->arr_type == GGUF_T_I32 && kty->arr_count == n) {
        for (i = 0; i < n; i++) {
            int32_t v = (int32_t)rd_i32le(base + kty->arr_off + 4 * i);
            t->type[i] = (int8_t)(v >= 0 && v < 127 ? v : 1);
        }
    }

    /* vocabulary hash */
    t->vcap = 16;
    while (t->vcap < (size_t)n * 2) t->vcap <<= 1;
    t->vhash = (uint32_t *)hfc_calloc(t->vcap, sizeof(uint32_t));
    if (!t->vhash) FAILMSG(HFC_ENOMEM, "out of memory");
    for (i = 0; i < n; i++) {
        size_t h = (size_t)(fnv64(t->text + t->toff[i], t->toff[i + 1] - t->toff[i]) & (t->vcap - 1));
        int dup = 0;
        while (t->vhash[h]) {
            uint32_t o = t->vhash[h] - 1;
            if (t->toff[o + 1] - t->toff[o] == t->toff[i + 1] - t->toff[i] &&
                memcmp(t->text + t->toff[o], t->text + t->toff[i], t->toff[i + 1] - t->toff[i]) == 0) { dup = 1; break; }
            h = (h + 1) & (t->vcap - 1);
        }
        if (!dup) t->vhash[h] = (uint32_t)(i + 1);
    }

    /* decoded bytes + special list */
    {
        size_t pos = 0, nspec = 0;
        for (i = 0; i < n; i++) {
            const unsigned char *s = t->text + t->toff[i];
            size_t sl = t->toff[i + 1] - t->toff[i], k = 0;
            int raw = t->type[i] == 3 || t->type[i] == 4;
            t->doff[i] = pos;
            if (raw) {
                memcpy(t->dec + pos, s, sl);
                pos += sl;
                if (sl) nspec++;
            } else {
                while (k < sl) {
                    size_t l;
                    uint32_t cp = utf8_get(s, sl, k, &l);
                    if (cp < 0x200 && u2b[cp] >= 0) t->dec[pos++] = (unsigned char)u2b[cp];
                    else { memcpy(t->dec + pos, s + k, l); pos += l; }
                    k += l;
                }
            }
        }
        t->doff[n] = pos;
        if (nspec) {
            t->spec = (spec_t *)hfc_calloc(nspec, sizeof(spec_t));
            if (!t->spec) FAILMSG(HFC_ENOMEM, "out of memory");
            for (i = 0; i < n; i++)
                if ((t->type[i] == 3 || t->type[i] == 4) && t->toff[i + 1] > t->toff[i]) {
                    t->spec[t->n_spec].len = (uint32_t)(t->toff[i + 1] - t->toff[i]);
                    t->spec[t->n_spec].id = (uint32_t)i;
                    t->spec_first[t->text[t->toff[i]]] = 1;
                    t->n_spec++;
                }
            qsort(t->spec, t->n_spec, sizeof(spec_t), cmp_spec);
        }
    }

    /* single-byte tokens */
    for (j = 0; j < 256; j++) {
        unsigned char b[4];
        size_t bl = utf8_put(b2u[j], b);
        t->byte_tok[j] = vocab_find(t, b, bl);
        if (t->byte_tok[j] == NONE32) FAILMSG(HFC_EFORMAT, "vocabulary lacks the single-byte token for byte %lu", (unsigned long)j);
    }

    /* merges */
    t->n_merges = (size_t)km->arr_count;
    t->mcap = 16;
    while (t->mcap < t->n_merges * 2) t->mcap <<= 1;
    t->mkey = (uint64_t *)hfc_calloc(t->mcap, sizeof(uint64_t));
    t->mrank = (uint32_t *)hfc_calloc(t->mcap, sizeof(uint32_t));
    t->mres = (uint32_t *)hfc_calloc(t->mcap, sizeof(uint32_t));
    if (!t->mkey || !t->mrank || !t->mres) FAILMSG(HFC_ENOMEM, "out of memory");
    {
        size_t pos = (size_t)km->arr_off, end = pos + (size_t)km->arr_bytes;
        unsigned char *cat = NULL;
        size_t catcap = 0;
        for (i = 0; i < km->arr_count; i++) {
            uint64_t sl = 0, b;
            const unsigned char *s;
            const unsigned char *sp;
            uint32_t l, r, res;
            size_t cl, h;
            uint64_t key;
            for (b = 0; b < 8; b++) sl |= (uint64_t)base[pos + b] << (8 * b);
            pos += 8;
            s = base + pos;
            pos += (size_t)sl;
            if (pos > end) { hfc_free(cat); FAILMSG(HFC_EFORMAT, "corrupt merges array"); }
            sp = (const unsigned char *)memchr(s, ' ', (size_t)sl);
            if (!sp || sp == s) continue;
            l = vocab_find(t, s, (size_t)(sp - s));
            r = vocab_find(t, sp + 1, (size_t)(s + sl - sp - 1));
            if (l == NONE32 || r == NONE32) continue;
            cl = (size_t)sl - 1;
            if (cl > catcap) {
                unsigned char *nc = (unsigned char *)hfc_realloc(cat, cl);
                if (!nc) { hfc_free(cat); FAILMSG(HFC_ENOMEM, "out of memory"); }
                cat = nc; catcap = cl;
            }
            memcpy(cat, s, (size_t)(sp - s));
            memcpy(cat + (sp - s), sp + 1, (size_t)(s + sl - sp - 1));
            res = vocab_find(t, cat, cl);
            if (res == NONE32) continue;
            key = (((uint64_t)l << 32) | r) + 1;
            h = (size_t)(mix64(key) & (t->mcap - 1));
            while (t->mkey[h] && t->mkey[h] != key) h = (h + 1) & (t->mcap - 1);
            if (!t->mkey[h]) { t->mkey[h] = key; t->mrank[h] = (uint32_t)i; t->mres[h] = res; }
        }
        hfc_free(cat);
    }

    {
        uint64_t v;
        if (gguf_get_u64(g, "tokenizer.ggml.bos_token_id", &v) && v < n) t->bos = (int64_t)v;
        if (gguf_get_u64(g, "tokenizer.ggml.eos_token_id", &v) && v < n) t->eos = (int64_t)v;
        if (gguf_get_u64(g, "tokenizer.ggml.padding_token_id", &v) && v < n) t->pad = (int64_t)v;
        if (gguf_get_u64(g, "tokenizer.ggml.add_bos_token", &v)) t->add_bos = v != 0;
    }
    *out = t;
    return HFC_OK;
fail:
    hfc_tok_free(t);
    return rc;
#undef FAILMSG
}

uint32_t hfc_tok_n_vocab(const hfc_tok *t) { return t->n_vocab; }
int64_t  hfc_tok_bos(const hfc_tok *t)     { return t->bos; }
int64_t  hfc_tok_eos(const hfc_tok *t)     { return t->eos; }
int64_t  hfc_tok_pad(const hfc_tok *t)     { return t->pad; }
int      hfc_tok_add_bos(const hfc_tok *t) { return t->add_bos; }

/* ---- Qwen2 pre-tokenizer ------------------------------------------------------------------
 * Equivalent to the regular expression
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
 *   | ' '?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * with alternatives tried in order at each position. */

#define IS_L(cp) ((cp) != 0xffffffffu && HFC_IS_LETTER(cp))
#define IS_N(cp) ((cp) != 0xffffffffu && HFC_IS_NUMBER(cp))
#define IS_S(cp) ((cp) != 0xffffffffu && HFC_IS_SPACE(cp))
#define IS_NL(cp) ((cp) == '\r' || (cp) == '\n')
#define IS_X(cp)  (!IS_S(cp) && !IS_L(cp) && !IS_N(cp))      /* [^\s\p{L}\p{N}] */

static size_t pretok_next(const unsigned char *s, size_t n, size_t i)
{
    size_t l0, l1, j;
    uint32_t c0 = utf8_get(s, n, i, &l0), c1;

    /* 1. contractions */
    if (c0 == '\'' && i + 1 < n) {
        unsigned a = s[i + 1] | 0x20;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if (i + 2 < n) {
            unsigned b = s[i + 2] | 0x20;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
        }
    }
    /* 2. optional non-letter prefix, then letters */
    if (IS_L(c0)) {
        j = i + l0;
        while (j < n) { c1 = utf8_get(s, n, j, &l1); if (!IS_L(c1)) break; j += l1; }
        return j;
    }
    if (!IS_NL(c0) && !IS_N(c0) && i + l0 < n) {
        c1 = utf8_get(s, n, i + l0, &l1);
        if (IS_L(c1)) {
            j = i + l0 + l1;
            while (j < n) { c1 = utf8_get(s, n, j, &l1); if (!IS_L(c1)) break; j += l1; }
            return j;
        }
    }
    /* 3. a single number */
    if (IS_N(c0)) return i + l0;
    /* 4. optional space, symbols, trailing newlines */
    {
        size_t k = i;
        uint32_t ck = c0;
        size_t lk = l0;
        if (c0 == ' ' && i + 1 < n) {
            uint32_t cn = utf8_get(s, n, i + 1, &l1);
            if (IS_X(cn)) { k = i + 1; ck = cn; lk = l1; }
        }
        if (IS_X(ck)) {
            j = k + lk;
            while (j < n) { c1 = utf8_get(s, n, j, &l1); if (!IS_X(c1)) break; j += l1; }
            while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
            return j;
        }
    }
    /* 5-7. whitespace runs */
    {
        size_t end_run = i, last_nl_end = 0, last_char_start = i, count = 0;
        int have_nl = 0;
        j = i;
        while (j < n) {
            c1 = utf8_get(s, n, j, &l1);
            if (!IS_S(c1)) break;
            last_char_start = j;
            j += l1;
            count++;
            if (IS_NL(c1)) { have_nl = 1; last_nl_end = j; }
        }
        end_run = j;
        if (count == 0) return i + l0;                    /* unreachable for valid classes; stay safe */
        if (have_nl) return last_nl_end;                  /* \s*[\r\n]+ */
        if (end_run >= n) return end_run;                 /* \s+(?!\S) at end of text */
        if (count >= 2) return last_char_start;           /* leave the last space for the next word */
        return end_run;                                   /* \s+ */
    }
}

/* ---- BPE -------------------------------------------------------------------------------------- */

typedef struct { uint32_t rank; int32_t left, right; uint32_t lid, rid, res; } bigram_t;

typedef struct {
    uint32_t *id;
    int32_t  *prev, *next;
    bigram_t *heap;
    size_t    nheap, capheap, capsym;
} ws_t;

static void ws_free(ws_t *w) { hfc_free(w->id); hfc_free(w->prev); hfc_free(w->next); hfc_free(w->heap); }

static hfc_status ws_reserve(ws_t *w, size_t nsym)
{
    if (nsym > w->capsym) {
        size_t nc = nsym < 64 ? 64 : nsym * 2, sz;
        uint32_t *a; int32_t *b, *c;
        if (!hfc_mul_size(nc, sizeof(uint32_t), &sz)) return HFC_ERANGE;
        a = (uint32_t *)hfc_realloc(w->id, sz);   if (!a) return HFC_ENOMEM;   w->id = a;
        b = (int32_t *)hfc_realloc(w->prev, sz);  if (!b) return HFC_ENOMEM;   w->prev = b;
        c = (int32_t *)hfc_realloc(w->next, sz);  if (!c) return HFC_ENOMEM;   w->next = c;
        w->capsym = nc;
    }
    return HFC_OK;
}

static int bg_less(const bigram_t *a, const bigram_t *b)
{
    return a->rank < b->rank || (a->rank == b->rank && a->left < b->left);
}

static hfc_status heap_push(ws_t *w, bigram_t e)
{
    size_t i;
    if (w->nheap == w->capheap) {
        size_t nc = w->capheap ? w->capheap * 2 : 128, sz;
        bigram_t *h;
        if (!hfc_mul_size(nc, sizeof(bigram_t), &sz)) return HFC_ERANGE;
        h = (bigram_t *)hfc_realloc(w->heap, sz);
        if (!h) return HFC_ENOMEM;
        w->heap = h; w->capheap = nc;
    }
    i = w->nheap++;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!bg_less(&e, &w->heap[p])) break;
        w->heap[i] = w->heap[p];
        i = p;
    }
    w->heap[i] = e;
    return HFC_OK;
}

static bigram_t heap_pop(ws_t *w)
{
    bigram_t top = w->heap[0], last = w->heap[--w->nheap];
    size_t i = 0;
    for (;;) {
        size_t c = 2 * i + 1;
        if (c >= w->nheap) break;
        if (c + 1 < w->nheap && bg_less(&w->heap[c + 1], &w->heap[c])) c++;
        if (!bg_less(&w->heap[c], &last)) break;
        w->heap[i] = w->heap[c];
        i = c;
    }
    if (w->nheap) w->heap[i] = last;
    return top;
}

typedef struct { uint32_t *v; size_t n, cap; } idvec_t;

static hfc_status idvec_push(idvec_t *v, uint32_t id)
{
    if (v->n == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 256, sz;
        uint32_t *p;
        if (!hfc_mul_size(nc, sizeof(uint32_t), &sz)) return HFC_ERANGE;
        p = (uint32_t *)hfc_realloc(v->v, sz);
        if (!p) return HFC_ENOMEM;
        v->v = p; v->cap = nc;
    }
    v->v[v->n++] = id;
    return HFC_OK;
}

static hfc_status try_pair(const hfc_tok *t, ws_t *w, int32_t l, int32_t r)
{
    uint32_t rank, res;
    if (l < 0 || r < 0) return HFC_OK;
    if (!merge_find(t, w->id[l], w->id[r], &rank, &res)) return HFC_OK;
    {
        bigram_t e;
        e.rank = rank; e.left = l; e.right = r; e.lid = w->id[l]; e.rid = w->id[r]; e.res = res;
        return heap_push(w, e);
    }
}

static hfc_status bpe_word(const hfc_tok *t, ws_t *w, const unsigned char *s, size_t n, idvec_t *out)
{
    size_t i;
    hfc_status rc;
    int32_t k;
    if (n > 0x7ffffff0u) return HFC_ERANGE;
    if ((rc = ws_reserve(w, n)) != HFC_OK) return rc;
    w->nheap = 0;
    for (i = 0; i < n; i++) {
        w->id[i] = t->byte_tok[s[i]];
        w->prev[i] = (int32_t)i - 1;
        w->next[i] = i + 1 < n ? (int32_t)i + 1 : -1;
    }
    for (i = 0; i + 1 < n; i++)
        if ((rc = try_pair(t, w, (int32_t)i, (int32_t)i + 1)) != HFC_OK) return rc;
    while (w->nheap) {
        bigram_t e = heap_pop(w);
        int32_t nx;
        if (w->id[e.left] == NONE32 || w->id[e.right] == NONE32) continue;
        if (w->id[e.left] != e.lid || w->id[e.right] != e.rid || w->next[e.left] != e.right) continue;
        w->id[e.left] = e.res;
        w->id[e.right] = NONE32;
        nx = w->next[e.right];
        w->next[e.left] = nx;
        if (nx >= 0) w->prev[nx] = e.left;
        if ((rc = try_pair(t, w, w->prev[e.left], e.left)) != HFC_OK) return rc;
        if ((rc = try_pair(t, w, e.left, nx)) != HFC_OK) return rc;
    }
    for (k = 0; k >= 0 && (size_t)k < n; k = w->next[k])
        if ((rc = idvec_push(out, w->id[k])) != HFC_OK) return rc;
    return HFC_OK;
}

static hfc_status encode_plain(const hfc_tok *t, ws_t *w, const unsigned char *s, size_t n, idvec_t *out)
{
    size_t i = 0;
    while (i < n) {
        size_t e = pretok_next(s, n, i);
        hfc_status rc;
        if (e <= i) e = i + 1;                       /* defensive: always make progress */
        if ((rc = bpe_word(t, w, s + i, e - i, out)) != HFC_OK) return rc;
        i = e;
    }
    return HFC_OK;
}

hfc_status hfc_tok_encode(const hfc_tok *t, const char *text, size_t len, unsigned flags,
                          uint32_t **ids, size_t *nout)
{
    const unsigned char *s = (const unsigned char *)text;
    idvec_t out = { NULL, 0, 0 };
    ws_t w;
    size_t i = 0, seg = 0;
    hfc_status rc = HFC_OK;

    *ids = NULL;
    *nout = 0;
    memset(&w, 0, sizeof w);
    if ((flags & HFC_TOK_ADD_BOS) && t->add_bos && t->bos >= 0)
        if ((rc = idvec_push(&out, (uint32_t)t->bos)) != HFC_OK) goto done;
    while (i < len) {
        if ((flags & HFC_TOK_PARSE_SPECIAL) && t->n_spec && t->spec_first[s[i]]) {
            size_t k;
            int hit = 0;
            for (k = 0; k < t->n_spec; k++) {
                const spec_t *sp = &t->spec[k];
                if (sp->len <= len - i && memcmp(s + i, t->text + t->toff[sp->id], sp->len) == 0) {
                    if (i > seg && (rc = encode_plain(t, &w, s + seg, i - seg, &out)) != HFC_OK) goto done;
                    if ((rc = idvec_push(&out, sp->id)) != HFC_OK) goto done;
                    i += sp->len;
                    seg = i;
                    hit = 1;
                    break;
                }
            }
            if (hit) continue;
        }
        i++;
    }
    if (len > seg && (rc = encode_plain(t, &w, s + seg, len - seg, &out)) != HFC_OK) goto done;
done:
    ws_free(&w);
    if (rc != HFC_OK) { hfc_free(out.v); return rc; }
    if (!out.v) { out.v = (uint32_t *)hfc_malloc(sizeof(uint32_t)); if (!out.v) return HFC_ENOMEM; }
    *ids = out.v;
    *nout = out.n;
    return HFC_OK;
}

hfc_status hfc_tok_piece(const hfc_tok *t, uint32_t id, int render_special,
                         const unsigned char **bytes, size_t *len)
{
    if (id >= t->n_vocab) return HFC_EINVAL;
    if (!render_special && (t->type[id] == 3)) { *bytes = t->dec; *len = 0; return HFC_OK; }
    *bytes = t->dec + t->doff[id];
    *len = t->doff[id + 1] - t->doff[id];
    return HFC_OK;
}

hfc_status hfc_tok_decode(const hfc_tok *t, const uint32_t *ids, size_t n, int render_special,
                          char **out, size_t *len)
{
    size_t i, total = 0, pos = 0;
    char *buf;
    for (i = 0; i < n; i++) {
        const unsigned char *p; size_t l;
        hfc_status rc = hfc_tok_piece(t, ids[i], render_special, &p, &l);
        if (rc != HFC_OK) return rc;
        if (!hfc_add_size(total, l, &total)) return HFC_ERANGE;
    }
    if (!hfc_add_size(total, 1, &pos)) return HFC_ERANGE;
    buf = (char *)hfc_malloc(pos);
    if (!buf) return HFC_ENOMEM;
    pos = 0;
    for (i = 0; i < n; i++) {
        const unsigned char *p; size_t l;
        (void)hfc_tok_piece(t, ids[i], render_special, &p, &l);
        memcpy(buf + pos, p, l);
        pos += l;
    }
    buf[pos] = '\0';
    *out = buf;
    *len = pos;
    return HFC_OK;
}
