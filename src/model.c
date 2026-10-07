/* model.c - loader, KV cache and forward pass for dense transformers. */
#include "model.h"
#include "mathx.h"
#include "ggtype.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- loading ------------------------------------------------------------------------- */

static hfc_status owned_add(hfc_model *m, float *p)
{
    if (m->n_owned == m->cap_owned) {
        size_t nc = m->cap_owned ? m->cap_owned * 2 : 16;
        float **q = (float **)hfc_realloc(m->owned, nc * sizeof(float *));
        if (!q) return HFC_ENOMEM;
        m->owned = q;
        m->cap_owned = nc;
    }
    m->owned[m->n_owned++] = p;
    return HFC_OK;
}

void hfc_model_free(hfc_model *m)
{
    size_t i;
    if (!m) return;
    for (i = 0; i < m->n_owned; i++) hfc_free(m->owned[i]);
    hfc_free(m->owned);
    hfc_free(m->layers);
    hfc_free(m);
}

#define FAIL(code, ...) do { snprintf(err, cap, __VA_ARGS__); return (code); } while (0)

static hfc_status get_u64(const gguf_file *g, const char *arch, const char *suffix, int required,
                          uint64_t *out, char *err, size_t cap)
{
    char key[96];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    if (gguf_get_u64(g, key, out)) return HFC_OK;
    if (required) FAIL(HFC_EFORMAT, "metadata key '%s' is missing", key);
    return HFC_ENOTSUP;                                    /* optional and absent */
}

/* A 1-D float vector of exactly n elements. Returns a pointer into the file
 * mapping when it is f32 and aligned, else a converted heap copy owned by m. */
static hfc_status get_vec(hfc_model *m, const char *name, uint64_t n, int required,
                          const float **out, char *err, size_t cap)
{
    const gguf_tensor *t = gguf_find_tensor(m->g, name);
    const void *d;
    float *copy;
    hfc_status rc;
    *out = NULL;
    if (!t) { if (required) FAIL(HFC_EFORMAT, "tensor '%s' is missing", name); return HFC_OK; }
    if (t->nelem != n || t->ne[0] != n) FAIL(HFC_EFORMAT, "tensor '%s' has the wrong size (%llu, expected %llu)", name, (unsigned long long)t->nelem, (unsigned long long)n);
    d = gguf_tensor_data(m->g, t);
    if (!d || !t->known) FAIL(HFC_EFORMAT, "tensor '%s' has no readable data", name);
    if (t->type == 0 && ((uintptr_t)d & 3) == 0) { *out = (const float *)d; return HFC_OK; }
    copy = (float *)hfc_malloc((size_t)n * sizeof(float));
    if (!copy) FAIL(HFC_ENOMEM, "out of memory");
    rc = hfc_dequant_row(t->type, d, copy, (size_t)n);
    if (rc != HFC_OK) { hfc_free(copy); FAIL(rc, "tensor '%s' has unsupported type %s", name, hfc_type_name(t->type)); }
    if ((rc = owned_add(m, copy)) != HFC_OK) { hfc_free(copy); FAIL(rc, "out of memory"); }
    *out = copy;
    return HFC_OK;
}

static hfc_status get_mat(hfc_model *m, const char *name, uint64_t cols, uint64_t rows, hfc_mat *w,
                          char *err, size_t cap)
{
    const gguf_tensor *t = gguf_find_tensor(m->g, name);
    const hfc_type_info *ti;
    if (!t) FAIL(HFC_EFORMAT, "tensor '%s' is missing", name);
    if (t->n_dims != 2 || t->ne[0] != cols || t->ne[1] != rows)
        FAIL(HFC_EFORMAT, "tensor '%s' has shape %llux%llu, expected %llux%llu", name,
             (unsigned long long)t->ne[0], (unsigned long long)t->ne[1], (unsigned long long)cols, (unsigned long long)rows);
    ti = hfc_type_lookup(t->type);
    if (!ti || !ti->dequant) FAIL(HFC_ENOTSUP, "tensor '%s' uses type %s, which has no kernel yet", name, hfc_type_name(t->type));
    w->t = t;
    w->type = t->type;
    w->cols = cols;
    w->rows = rows;
    w->data = (const unsigned char *)gguf_tensor_data(m->g, t);
    if (!w->data) FAIL(HFC_EFORMAT, "tensor '%s' data is out of range", name);
    if (hfc_type_row_bytes(t->type, cols, &w->row_bytes) != HFC_OK) FAIL(HFC_EFORMAT, "bad row size for '%s'", name);
    return HFC_OK;
}

hfc_status hfc_model_load(hfc_model **out, const gguf_file *g, const hfc_kernels *k, char *err, size_t cap)
{
    hfc_model *m;
    hfc_hparams *hp;
    const char *arch;
    uint64_t v;
    hfc_status rc;
    int l;
    char name[96];
    double f;

    *out = NULL;
    arch = gguf_get_str(g, "general.architecture");
    if (!arch) FAIL(HFC_EFORMAT, "model has no general.architecture");
    if (strcmp(arch, "qwen2") && strcmp(arch, "qwen3") && strcmp(arch, "llama"))
        FAIL(HFC_ENOTSUP, "architecture '%s' is not supported yet (qwen2, qwen3 and llama are)", arch);

    m = (hfc_model *)hfc_calloc(1, sizeof *m);
    if (!m) FAIL(HFC_ENOMEM, "out of memory");
    m->g = g;
    m->k = k;
    hp = &m->hp;
    snprintf(hp->arch, sizeof hp->arch, "%s", arch);

#define REQ(suffix, field) do { if ((rc = get_u64(g, arch, suffix, 1, &v, err, cap)) != HFC_OK) goto fail; \
                                if (v == 0 || v > 1000000) { snprintf(err, cap, "implausible value for %s.%s", arch, suffix); rc = HFC_EFORMAT; goto fail; } \
                                hp->field = (int)v; } while (0)
    REQ("block_count", n_layer);
    REQ("embedding_length", n_embd);
    REQ("feed_forward_length", n_ff);
    REQ("attention.head_count", n_head);
    hp->n_head_kv = hp->n_head;
    if (get_u64(g, arch, "attention.head_count_kv", 0, &v, err, cap) == HFC_OK && v > 0 && v <= 100000) hp->n_head_kv = (int)v;
    hp->n_ctx_train = 0;
    if (get_u64(g, arch, "context_length", 0, &v, err, cap) == HFC_OK && v < 100000000) hp->n_ctx_train = (int)v;
    hp->head_dim = hp->n_embd / hp->n_head;
    if (get_u64(g, arch, "attention.key_length", 0, &v, err, cap) == HFC_OK && v > 0 && v <= 100000) hp->head_dim = (int)v;
    hp->n_rot = hp->head_dim;
    if (get_u64(g, arch, "rope.dimension_count", 0, &v, err, cap) == HFC_OK && v > 0 && v <= 100000) hp->n_rot = (int)v;
    hp->rms_eps = 1e-5f;
    snprintf(name, sizeof name, "%s.attention.layer_norm_rms_epsilon", arch);
    if (gguf_get_f64(g, name, &f) && f > 0 && f < 1) hp->rms_eps = (float)f;
    hp->rope_base = 10000.0f;
    snprintf(name, sizeof name, "%s.rope.freq_base", arch);
    if (gguf_get_f64(g, name, &f) && f > 0 && f < 1e12) hp->rope_base = (float)f;
    hp->rope_neox = strcmp(arch, "llama") != 0;
    if (hp->n_head % hp->n_head_kv != 0 || hp->n_rot > hp->head_dim || (hp->n_rot & 1) || hp->head_dim < 2 || (hp->head_dim & 1)) {
        snprintf(err, cap, "unsupported attention geometry (heads %d/%d, head_dim %d, rope dims %d)", hp->n_head, hp->n_head_kv, hp->head_dim, hp->n_rot);
        rc = HFC_ENOTSUP; goto fail;
    }
    if (hp->n_head * hp->head_dim > 1 << 20 || hp->n_head_kv * hp->head_dim > 1 << 20) { snprintf(err, cap, "implausible attention size"); rc = HFC_EFORMAT; goto fail; }

    /* token_embd rows (the vocabulary) are discovered from the tensor itself */
    {
        const gguf_tensor *te = gguf_find_tensor(g, "token_embd.weight");
        if (!te || te->n_dims != 2 || te->ne[0] != (uint64_t)hp->n_embd || te->ne[1] == 0 || te->ne[1] > (1u << 24)) {
            snprintf(err, cap, "token_embd.weight is missing or has an unexpected shape"); rc = HFC_EFORMAT; goto fail;
        }
        hp->n_vocab = (int)te->ne[1];
        if ((rc = get_mat(m, "token_embd.weight", (uint64_t)hp->n_embd, (uint64_t)hp->n_vocab, &m->tok_embd, err, cap)) != HFC_OK) goto fail;
    }
    if (gguf_find_tensor(g, "output.weight")) {
        if ((rc = get_mat(m, "output.weight", (uint64_t)hp->n_embd, (uint64_t)hp->n_vocab, &m->output, err, cap)) != HFC_OK) goto fail;
    } else { m->output = m->tok_embd; m->tied = 1; }
    if ((rc = get_vec(m, "output_norm.weight", (uint64_t)hp->n_embd, 1, &m->out_norm, err, cap)) != HFC_OK) goto fail;

    m->layers = (hfc_layer *)hfc_calloc((size_t)hp->n_layer, sizeof(hfc_layer));
    if (!m->layers) { snprintf(err, cap, "out of memory"); rc = HFC_ENOMEM; goto fail; }
    for (l = 0; l < hp->n_layer; l++) {
        hfc_layer *L = &m->layers[l];
        uint64_t E = (uint64_t)hp->n_embd, QD = (uint64_t)hp->n_head * hp->head_dim, KD = (uint64_t)hp->n_head_kv * hp->head_dim;
#define TN(s) (snprintf(name, sizeof name, "blk.%d.%s", l, s), name)
        if ((rc = get_vec(m, TN("attn_norm.weight"), E, 1, &L->attn_norm, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_vec(m, TN("ffn_norm.weight"), E, 1, &L->ffn_norm, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_mat(m, TN("attn_q.weight"), E, QD, &L->wq, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_mat(m, TN("attn_k.weight"), E, KD, &L->wk, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_mat(m, TN("attn_v.weight"), E, KD, &L->wv, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_mat(m, TN("attn_output.weight"), QD, E, &L->wo, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_mat(m, TN("ffn_gate.weight"), E, (uint64_t)hp->n_ff, &L->gate, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_mat(m, TN("ffn_up.weight"), E, (uint64_t)hp->n_ff, &L->up, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_mat(m, TN("ffn_down.weight"), (uint64_t)hp->n_ff, E, &L->down, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_vec(m, TN("attn_q.bias"), QD, 0, &L->bq, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_vec(m, TN("attn_k.bias"), KD, 0, &L->bk, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_vec(m, TN("attn_v.bias"), KD, 0, &L->bv, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_vec(m, TN("attn_q_norm.weight"), (uint64_t)hp->head_dim, 0, &L->q_norm, err, cap)) != HFC_OK) goto fail;
        if ((rc = get_vec(m, TN("attn_k_norm.weight"), (uint64_t)hp->head_dim, 0, &L->k_norm, err, cap)) != HFC_OK) goto fail;
        if (L->q_norm && L->k_norm) hp->has_qk_norm = 1;
        if ((L->q_norm != NULL) != (L->k_norm != NULL)) { snprintf(err, cap, "layer %d has only one of the QK norms", l); rc = HFC_EFORMAT; goto fail; }
#undef TN
    }
    *out = m;
    return HFC_OK;
fail:
    hfc_model_free(m);
    return rc;
#undef REQ
}

/* ---- context ----------------------------------------------------------------------------- */

struct hfc_ctx {
    const hfc_model *m;
    const hfc_kernels *k;
    size_t   ctx_max, max_batch, n_pos;
    size_t   max_blocks, n_blocks;         /* blocks allocated per layer */
    uint16_t **blk;                        /* [n_layer * max_blocks]; block = K then V */
    size_t   kvdim, block_u16;
    size_t   kv_bytes;
    /* scratch */
    float *x, *xn, *q, *kk, *vv, *att, *o, *gate, *up, *cs;      /* [max_batch * dim] */
    float *scores;                                              /* [ctx_max + 1] */
    unsigned char *qa, *qa2;                                    /* quantized activations */
    float *rowbuf;                                              /* dequantized weight row */
    size_t qa_bytes, qa2_bytes;
};

static void *xmalloc_f(size_t count) { size_t b; return hfc_mul_size(count, sizeof(float), &b) ? hfc_malloc(b) : NULL; }

void hfc_ctx_free(hfc_ctx *c)
{
    size_t i;
    if (!c) return;
    if (c->blk) {
        size_t total = (size_t)c->m->hp.n_layer * c->max_blocks;
        for (i = 0; i < total; i++) hfc_free(c->blk[i]);
        hfc_free(c->blk);
    }
    hfc_free(c->x); hfc_free(c->xn); hfc_free(c->q); hfc_free(c->kk); hfc_free(c->vv); hfc_free(c->att);
    hfc_free(c->o); hfc_free(c->gate); hfc_free(c->up); hfc_free(c->cs); hfc_free(c->scores);
    hfc_free(c->qa); hfc_free(c->qa2); hfc_free(c->rowbuf);
    hfc_free(c);
}

hfc_status hfc_ctx_new(hfc_ctx **out, const hfc_model *m, size_t ctx_max, size_t max_batch)
{
    const hfc_hparams *hp = &m->hp;
    hfc_ctx *c;
    size_t qd = (size_t)hp->n_head * hp->head_dim, E = (size_t)hp->n_embd, F = (size_t)hp->n_ff;
    size_t widest, total;
    *out = NULL;
    if (ctx_max == 0 || max_batch == 0 || max_batch > 4096) return HFC_EINVAL;
    c = (hfc_ctx *)hfc_calloc(1, sizeof *c);
    if (!c) return HFC_ENOMEM;
    c->m = m; c->k = m->k; c->ctx_max = ctx_max; c->max_batch = max_batch;
    c->kvdim = (size_t)hp->n_head_kv * hp->head_dim;
    c->block_u16 = 2 * HFC_KV_BLOCK_TOKENS * c->kvdim;
    c->max_blocks = (ctx_max + HFC_KV_BLOCK_TOKENS - 1) / HFC_KV_BLOCK_TOKENS;
    if (!hfc_mul_size((size_t)hp->n_layer, c->max_blocks, &total)) { hfc_free(c); return HFC_ERANGE; }
    c->blk = (uint16_t **)hfc_calloc(total, sizeof(uint16_t *));
    c->x = xmalloc_f(max_batch * E); c->xn = xmalloc_f(max_batch * E);
    c->q = xmalloc_f(max_batch * qd); c->kk = xmalloc_f(max_batch * c->kvdim); c->vv = xmalloc_f(max_batch * c->kvdim);
    c->att = xmalloc_f(max_batch * qd); c->o = xmalloc_f(max_batch * E);
    c->gate = xmalloc_f(max_batch * F); c->up = xmalloc_f(max_batch * F);
    c->cs = xmalloc_f((size_t)hp->n_rot);
    c->scores = xmalloc_f(ctx_max + 1);
    widest = qd > F ? qd : F;
    if (E > widest) widest = E;
    c->qa_bytes = max_batch * (widest / 32 + 1) * HFC_Q8_0_BLOCK;
    c->qa2_bytes = c->qa_bytes;
    c->qa = (unsigned char *)hfc_malloc(c->qa_bytes);
    c->qa2 = (unsigned char *)hfc_malloc(c->qa2_bytes);
    c->rowbuf = xmalloc_f(widest + 64);
    if (!c->blk || !c->x || !c->xn || !c->q || !c->kk || !c->vv || !c->att || !c->o || !c->gate || !c->up ||
        !c->cs || !c->scores || !c->qa || !c->qa2 || !c->rowbuf) { hfc_ctx_free(c); return HFC_ENOMEM; }
    *out = c;
    return HFC_OK;
}

size_t hfc_ctx_pos(const hfc_ctx *c)      { return c->n_pos; }
size_t hfc_ctx_kv_bytes(const hfc_ctx *c) { return c->kv_bytes; }
void   hfc_ctx_truncate(hfc_ctx *c, size_t n) { if (n < c->n_pos) c->n_pos = n; }

/* Make sure KV blocks exist for positions [0, upto). All layers grow together so a
 * failure leaves the cache consistent. */
static hfc_status kv_reserve(hfc_ctx *c, size_t upto)
{
    size_t need = (upto + HFC_KV_BLOCK_TOKENS - 1) / HFC_KV_BLOCK_TOKENS, b, l;
    size_t nl = (size_t)c->m->hp.n_layer, bytes;
    if (need > c->max_blocks) return HFC_ERANGE;
    for (b = c->n_blocks; b < need; b++) {
        for (l = 0; l < nl; l++) {
            uint16_t *p;
            if (!hfc_mul_size(c->block_u16, sizeof(uint16_t), &bytes)) return HFC_ERANGE;
            p = (uint16_t *)hfc_malloc(bytes);
            if (!p) {
                size_t j;
                for (j = 0; j < l; j++) { hfc_free(c->blk[j * c->max_blocks + b]); c->blk[j * c->max_blocks + b] = NULL; }
                return HFC_ENOMEM;
            }
            c->blk[l * c->max_blocks + b] = p;
        }
        c->n_blocks = b + 1;
        c->kv_bytes += nl * bytes;
    }
    return HFC_OK;
}

/* ---- matrix-vector products ------------------------------------------------------------------ */

/* Y[t][r] = dot(W row r, X[t]) (+ bias[r]) for t < n. Weight rows are read once and
 * reused across the n tokens. xf: f32 activations [n][cols]; xq: Q8_0 quantized copy. */
static hfc_status matmat(hfc_ctx *c, const hfc_mat *w, const float *xf, const unsigned char *xq,
                         size_t n, const float *bias, float *y)
{
    const hfc_kernels *k = c->k;
    size_t r, t, nblk = (size_t)(w->cols / 32), qstride = nblk * HFC_Q8_0_BLOCK, cols = (size_t)w->cols;
    size_t rows = (size_t)w->rows;
    for (r = 0; r < rows; r++) {
        const unsigned char *wr = w->data + r * (size_t)w->row_bytes;
        float b = bias ? bias[r] : 0.0f;
        switch (w->type) {
        case 8:                                              /* Q8_0 */
            for (t = 0; t < n; t++) y[t * rows + r] = k->dot_q8_0(wr, xq + t * qstride, nblk) + b;
            break;
        default:
            /* any other type: dequantize the row once, reuse it for every token */
            if (hfc_dequant_row(w->type, wr, c->rowbuf, cols) != HFC_OK) return HFC_ENOTSUP;
            for (t = 0; t < n; t++) y[t * rows + r] = k->dot_f32(c->rowbuf, xf + t * cols, cols) + b;
            break;
        }
    }
    return HFC_OK;
}

static void quantize_rows(hfc_ctx *c, const float *x, size_t n, size_t cols, unsigned char *dst)
{
    size_t t, stride = cols / 32 * HFC_Q8_0_BLOCK;
    for (t = 0; t < n; t++) c->k->quantize_q8_0(x + t * cols, dst + t * stride, cols);
}

/* ---- RoPE ------------------------------------------------------------------------------------ */

static void rope_cache(const hfc_hparams *hp, size_t pos, float *cs)      /* cs[2*i], cs[2*i+1] = cos, sin */
{
    size_t i, half = (size_t)hp->n_rot / 2;
    float theta_scale = (float)pow((double)hp->rope_base, -2.0 / (double)hp->n_rot);
    float theta = (float)pos;
    for (i = 0; i < half; i++) {
        cs[2 * i] = cosf(theta);
        cs[2 * i + 1] = sinf(theta);
        theta *= theta_scale;
    }
}

static void rope_apply(const hfc_hparams *hp, const float *cs, float *v, int nheads)
{
    int h;
    size_t i, half = (size_t)hp->n_rot / 2;
    for (h = 0; h < nheads; h++) {
        float *p = v + (size_t)h * hp->head_dim;
        for (i = 0; i < half; i++) {
            float c0 = cs[2 * i], s0 = cs[2 * i + 1], x0, x1;
            size_t a = hp->rope_neox ? i : 2 * i, b = hp->rope_neox ? i + half : 2 * i + 1;
            x0 = p[a]; x1 = p[b];
            p[a] = x0 * c0 - x1 * s0;
            p[b] = x0 * s0 + x1 * c0;
        }
    }
}

/* ---- forward pass ---------------------------------------------------------------------------- */

hfc_status hfc_ctx_forward(hfc_ctx *c, const uint32_t *tokens, size_t n, float *logits_last)
{
    const hfc_model *m = c->m;
    const hfc_hparams *hp = &m->hp;
    const hfc_kernels *k = c->k;
    size_t E = (size_t)hp->n_embd, QD = (size_t)hp->n_head * hp->head_dim, KD = c->kvdim, F = (size_t)hp->n_ff;
    size_t hd = (size_t)hp->head_dim, pos0 = c->n_pos, t, l, h;
    size_t group = (size_t)(hp->n_head / hp->n_head_kv);
    float scale = 1.0f / sqrtf((float)hd);
    hfc_status rc;

    if (n == 0 || n > c->max_batch) return HFC_EINVAL;
    if (pos0 + n > c->ctx_max) return HFC_ERANGE;
    if ((rc = kv_reserve(c, pos0 + n)) != HFC_OK) return rc;

    /* embeddings */
    for (t = 0; t < n; t++) {
        const unsigned char *row;
        if (tokens[t] >= (uint32_t)hp->n_vocab) return HFC_EINVAL;
        row = m->tok_embd.data + (size_t)tokens[t] * (size_t)m->tok_embd.row_bytes;
        if ((rc = hfc_dequant_row(m->tok_embd.type, row, c->x + t * E, E)) != HFC_OK) return rc;
    }

    for (l = 0; l < (size_t)hp->n_layer; l++) {
        const hfc_layer *L = &m->layers[l];
        /* --- attention --- */
        for (t = 0; t < n; t++) hfc_rmsnorm(c->x + t * E, L->attn_norm, c->xn + t * E, E, hp->rms_eps);
        quantize_rows(c, c->xn, n, E, c->qa);
        if ((rc = matmat(c, &L->wq, c->xn, c->qa, n, L->bq, c->q)) != HFC_OK) return rc;
        if ((rc = matmat(c, &L->wk, c->xn, c->qa, n, L->bk, c->kk)) != HFC_OK) return rc;
        if ((rc = matmat(c, &L->wv, c->xn, c->qa, n, L->bv, c->vv)) != HFC_OK) return rc;
        for (t = 0; t < n; t++) {
            float *q = c->q + t * QD, *kx = c->kk + t * KD;
            size_t pos = pos0 + t, bi = pos / HFC_KV_BLOCK_TOKENS, off = pos % HFC_KV_BLOCK_TOKENS;
            uint16_t *kb = c->blk[l * c->max_blocks + bi], *vb = kb + HFC_KV_BLOCK_TOKENS * KD;
            if (hp->has_qk_norm) {
                for (h = 0; h < (size_t)hp->n_head; h++) hfc_rmsnorm(q + h * hd, L->q_norm, q + h * hd, hd, hp->rms_eps);
                for (h = 0; h < (size_t)hp->n_head_kv; h++) hfc_rmsnorm(kx + h * hd, L->k_norm, kx + h * hd, hd, hp->rms_eps);
            }
            rope_cache(hp, pos, c->cs);
            rope_apply(hp, c->cs, q, hp->n_head);
            rope_apply(hp, c->cs, kx, hp->n_head_kv);
            k->f32_to_f16(kx, kb + off * KD, KD);
            k->f32_to_f16(c->vv + t * KD, vb + off * KD, KD);
        }
        for (t = 0; t < n; t++) {
            size_t pos = pos0 + t, len = pos + 1;
            for (h = 0; h < (size_t)hp->n_head; h++) {
                const float *qh = c->q + t * QD + h * hd;
                float *out = c->att + t * QD + h * hd;
                size_t kvh = h / group, p;
                for (p = 0; p < len; p++) {
                    const uint16_t *kb = c->blk[l * c->max_blocks + p / HFC_KV_BLOCK_TOKENS];
                    c->scores[p] = k->dot_f32_f16(qh, kb + (p % HFC_KV_BLOCK_TOKENS) * KD + kvh * hd, hd) * scale;
                }
                hfc_softmax(c->scores, len);
                memset(out, 0, hd * sizeof(float));
                for (p = 0; p < len; p++) {
                    const uint16_t *vb = c->blk[l * c->max_blocks + p / HFC_KV_BLOCK_TOKENS] + HFC_KV_BLOCK_TOKENS * KD;
                    k->axpy_f32_f16(out, c->scores[p], vb + (p % HFC_KV_BLOCK_TOKENS) * KD + kvh * hd, hd);
                }
            }
        }
        quantize_rows(c, c->att, n, QD, c->qa);
        if ((rc = matmat(c, &L->wo, c->att, c->qa, n, NULL, c->o)) != HFC_OK) return rc;
        { size_t i; for (i = 0; i < n * E; i++) c->x[i] += c->o[i]; }

        /* --- feed forward --- */
        for (t = 0; t < n; t++) hfc_rmsnorm(c->x + t * E, L->ffn_norm, c->xn + t * E, E, hp->rms_eps);
        quantize_rows(c, c->xn, n, E, c->qa);
        if ((rc = matmat(c, &L->gate, c->xn, c->qa, n, NULL, c->gate)) != HFC_OK) return rc;
        if ((rc = matmat(c, &L->up, c->xn, c->qa, n, NULL, c->up)) != HFC_OK) return rc;
        { size_t i; for (i = 0; i < n * F; i++) c->gate[i] = hfc_silu(c->gate[i]) * c->up[i]; }
        quantize_rows(c, c->gate, n, F, c->qa);
        if ((rc = matmat(c, &L->down, c->gate, c->qa, n, NULL, c->o)) != HFC_OK) return rc;
        { size_t i; for (i = 0; i < n * E; i++) c->x[i] += c->o[i]; }
    }
    c->n_pos = pos0 + n;

    if (logits_last) {
        hfc_rmsnorm(c->x + (n - 1) * E, m->out_norm, c->xn, E, hp->rms_eps);
        quantize_rows(c, c->xn, 1, E, c->qa);
        return matmat(c, &m->output, c->xn, c->qa, 1, NULL, logits_last);
    }
    return HFC_OK;
}
