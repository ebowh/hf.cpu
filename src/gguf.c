/* gguf.c - hardened GGUF reader. See gguf.h. */
#include "gguf.h"
#include "ggtype.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_KV       (1u << 20)
#define MAX_TENSORS  (1u << 20)
#define MAX_KEY      (1u << 16)
#define MAX_STRING   (1u << 28)

typedef struct {
    const unsigned char *p;
    uint64_t len, pos;
    char *err;
    size_t errcap;
} rd_t;

static hfc_status rfail(rd_t *r, hfc_status rc, const char *msg)
{
    if (r->err && r->errcap) snprintf(r->err, r->errcap, "%s (at byte %llu)", msg, (unsigned long long)r->pos);
    return rc;
}

static int rd_have(const rd_t *r, uint64_t n) { return n <= r->len - r->pos; }

static hfc_status rd_u32(rd_t *r, uint32_t *out)
{
    const unsigned char *p;
    if (!rd_have(r, 4)) return rfail(r, HFC_EFORMAT, "truncated file");
    p = r->p + r->pos;
    *out = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    r->pos += 4;
    return HFC_OK;
}

static hfc_status rd_u64(rd_t *r, uint64_t *out)
{
    uint32_t lo, hi;
    hfc_status rc;
    if ((rc = rd_u32(r, &lo)) != HFC_OK) return rc;
    if ((rc = rd_u32(r, &hi)) != HFC_OK) return rc;
    *out = (uint64_t)lo | ((uint64_t)hi << 32);
    return HFC_OK;
}

/* Reads a length-prefixed string, copying it to the heap (NUL terminated). */
static hfc_status rd_str(rd_t *r, uint64_t maxlen, char **out)
{
    uint64_t n;
    hfc_status rc;
    char *s;
    *out = NULL;
    if ((rc = rd_u64(r, &n)) != HFC_OK) return rc;
    if (n > maxlen) return rfail(r, HFC_EFORMAT, "string too long");
    if (!rd_have(r, n)) return rfail(r, HFC_EFORMAT, "string runs past end of file");
    s = hfc_strndup((const char *)r->p + r->pos, (size_t)n);
    if (!s) return rfail(r, HFC_ENOMEM, "out of memory");
    r->pos += n;
    *out = s;
    return HFC_OK;
}

/* Size of a fixed-width scalar element, 0 for string/array. */
static unsigned scalar_size(uint32_t t)
{
    switch (t) {
    case GGUF_T_U8: case GGUF_T_I8: case GGUF_T_BOOL: return 1;
    case GGUF_T_U16: case GGUF_T_I16: return 2;
    case GGUF_T_U32: case GGUF_T_I32: case GGUF_T_F32: return 4;
    case GGUF_T_U64: case GGUF_T_I64: case GGUF_T_F64: return 8;
    default: return 0;
    }
}

const char *gguf_type_name(uint32_t t)
{
    static const char *n[] = { "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool",
                               "string", "array", "u64", "i64", "f64" };
    return t <= GGUF_T_F64 ? n[t] : "?";
}

static hfc_status rd_scalar(rd_t *r, uint32_t t, gguf_kv *kv)
{
    unsigned sz = scalar_size(t);
    uint64_t raw = 0;
    unsigned i;
    if (!sz) return rfail(r, HFC_EFORMAT, "bad scalar type");
    if (!rd_have(r, sz)) return rfail(r, HFC_EFORMAT, "truncated value");
    for (i = 0; i < sz; i++) raw |= (uint64_t)r->p[r->pos + i] << (8 * i);
    r->pos += sz;
    switch (t) {
    case GGUF_T_I8:  kv->v.i = (int8_t)raw; break;
    case GGUF_T_I16: kv->v.i = (int16_t)raw; break;
    case GGUF_T_I32: kv->v.i = (int32_t)raw; break;
    case GGUF_T_I64: kv->v.i = (int64_t)raw; break;
    case GGUF_T_F32: { uint32_t b = (uint32_t)raw; float f; memcpy(&f, &b, 4); kv->v.f = f; break; }
    case GGUF_T_F64: { double d; memcpy(&d, &raw, 8); kv->v.f = d; break; }
    case GGUF_T_BOOL: kv->v.u = raw ? 1 : 0; break;
    default: kv->v.u = raw; break;
    }
    return HFC_OK;
}

static hfc_status read_kv(rd_t *r, gguf_kv *kv)
{
    hfc_status rc;
    uint32_t t;
    if ((rc = rd_str(r, MAX_KEY, &kv->key)) != HFC_OK) return rc;
    if ((rc = rd_u32(r, &t)) != HFC_OK) return rc;
    kv->type = t;
    if (t == GGUF_T_STR) return rd_str(r, MAX_STRING, &kv->str);
    if (t == GGUF_T_ARR) {
        uint32_t et;
        uint64_t cnt, bytes = 0, i;
        if ((rc = rd_u32(r, &et)) != HFC_OK) return rc;
        if ((rc = rd_u64(r, &cnt)) != HFC_OK) return rc;
        if (et == GGUF_T_ARR) return rfail(r, HFC_EFORMAT, "nested arrays are not supported");
        kv->arr_type = et;
        kv->arr_count = cnt;
        kv->arr_off = r->pos;
        if (et == GGUF_T_STR) {
            /* each string needs at least its 8-byte length */
            if (cnt > (r->len - r->pos) / 8) return rfail(r, HFC_EFORMAT, "string array count exceeds file size");
            for (i = 0; i < cnt; i++) {
                uint64_t n;
                if ((rc = rd_u64(r, &n)) != HFC_OK) return rc;
                if (n > MAX_STRING || !rd_have(r, n)) return rfail(r, HFC_EFORMAT, "array string runs past end of file");
                r->pos += n;
            }
        } else {
            unsigned sz = scalar_size(et);
            if (!sz) return rfail(r, HFC_EFORMAT, "bad array element type");
            if (!hfc_mul_u64(cnt, sz, &bytes)) return rfail(r, HFC_ERANGE, "array size overflow");
            if (!rd_have(r, bytes)) return rfail(r, HFC_EFORMAT, "array runs past end of file");
            r->pos += bytes;
        }
        kv->arr_bytes = r->pos - kv->arr_off;
        return HFC_OK;
    }
    return rd_scalar(r, t, kv);
}

static uint64_t fnv(const char *s)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 0x100000001b3ull; }
    return h;
}

static hfc_status build_hash(gguf_file *g, rd_t *r)
{
    uint64_t cap = 16, i;
    while (cap < g->n_tensors * 2) cap <<= 1;
    g->thash = (uint32_t *)hfc_calloc((size_t)cap, sizeof(uint32_t));
    if (!g->thash) return rfail(r, HFC_ENOMEM, "out of memory");
    g->thash_cap = cap;
    for (i = 0; i < g->n_tensors; i++) {
        uint64_t h = fnv(g->t[i].name) & (cap - 1);
        while (g->thash[h]) {
            if (strcmp(g->t[g->thash[h] - 1].name, g->t[i].name) == 0)
                return rfail(r, HFC_EFORMAT, "duplicate tensor name");
            h = (h + 1) & (cap - 1);
        }
        g->thash[h] = (uint32_t)(i + 1);
    }
    return HFC_OK;
}

static hfc_status parse(gguf_file *g, rd_t *r)
{
    uint32_t magic;
    hfc_status rc;
    uint64_t i, data_start, data_size, tmp;

    if ((rc = rd_u32(r, &magic)) != HFC_OK) return rc;
    if (magic != 0x46554747u) {
        return rfail(r, HFC_EFORMAT, magic == 0x47475546u ? "not a GGUF file (or big-endian, unsupported)"
                                                          : "not a GGUF file (bad magic)");
    }
    if ((rc = rd_u32(r, &g->version)) != HFC_OK) return rc;
    if (g->version < 2 || g->version > 3) {
        if (g->version == 0x03000000u || g->version == 0x02000000u)
            return rfail(r, HFC_EFORMAT, "big-endian GGUF is not supported");
        return rfail(r, HFC_EFORMAT, "unsupported GGUF version");
    }
    if ((rc = rd_u64(r, &g->n_tensors)) != HFC_OK) return rc;
    if ((rc = rd_u64(r, &g->n_kv)) != HFC_OK) return rc;
    if (g->n_tensors > MAX_TENSORS) return rfail(r, HFC_EFORMAT, "implausible tensor count");
    if (g->n_kv > MAX_KV) return rfail(r, HFC_EFORMAT, "implausible metadata count");
    /* each kv needs >= 12 bytes (len + type + ...), each tensor >= 24 */
    if (g->n_kv > (r->len - r->pos) / 12) return rfail(r, HFC_EFORMAT, "metadata count exceeds file size");
    if (g->n_tensors > (r->len - r->pos) / 24) return rfail(r, HFC_EFORMAT, "tensor count exceeds file size");

    g->kv = (gguf_kv *)hfc_calloc((size_t)g->n_kv ? (size_t)g->n_kv : 1, sizeof *g->kv);
    g->t = (gguf_tensor *)hfc_calloc((size_t)g->n_tensors ? (size_t)g->n_tensors : 1, sizeof *g->t);
    if (!g->kv || !g->t) return rfail(r, HFC_ENOMEM, "out of memory");

    g->align = 32;
    for (i = 0; i < g->n_kv; i++) {
        uint64_t j;
        if ((rc = read_kv(r, &g->kv[i])) != HFC_OK) return rc;
        for (j = 0; j < i; j++)
            if (strcmp(g->kv[j].key, g->kv[i].key) == 0) return rfail(r, HFC_EFORMAT, "duplicate metadata key");
        if (strcmp(g->kv[i].key, "general.alignment") == 0) {
            uint64_t a = g->kv[i].v.u;
            if (g->kv[i].type != GGUF_T_U32 && g->kv[i].type != GGUF_T_U64)
                return rfail(r, HFC_EFORMAT, "general.alignment has wrong type");
            if (a == 0 || (a & (a - 1)) != 0 || a > (1u << 20)) return rfail(r, HFC_EFORMAT, "bad general.alignment");
            g->align = a;
        }
    }

    for (i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->t[i];
        uint32_t d;
        uint64_t nel = 1;
        if ((rc = rd_str(r, MAX_KEY, &t->name)) != HFC_OK) return rc;
        if ((rc = rd_u32(r, &t->n_dims)) != HFC_OK) return rc;
        if (t->n_dims == 0 || t->n_dims > GGUF_MAX_DIMS) return rfail(r, HFC_EFORMAT, "bad tensor dimension count");
        for (d = 0; d < t->n_dims; d++) {
            if ((rc = rd_u64(r, &t->ne[d])) != HFC_OK) return rc;
            if (t->ne[d] == 0) return rfail(r, HFC_EFORMAT, "zero-sized tensor dimension");
            if (!hfc_mul_u64(nel, t->ne[d], &nel)) return rfail(r, HFC_ERANGE, "tensor element count overflow");
        }
        for (; d < GGUF_MAX_DIMS; d++) t->ne[d] = 1;
        if ((rc = rd_u32(r, &t->type)) != HFC_OK) return rc;
        if ((rc = rd_u64(r, &t->offset)) != HFC_OK) return rc;
        t->nelem = nel;
    }

    /* data section starts at the next aligned offset */
    if (!hfc_add_u64(r->pos, g->align - 1, &tmp)) return rfail(r, HFC_ERANGE, "offset overflow");
    data_start = tmp & ~(g->align - 1);
    if (data_start > r->len) {
        /* vocab-only files end right after the metadata: allowed when there are no tensors */
        if (g->n_tensors != 0) return rfail(r, HFC_EFORMAT, "file ends before the tensor data section");
        data_start = r->len;
    }
    g->data_off = data_start;
    data_size = r->len - data_start;

    for (i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->t[i];
        const hfc_type_info *ti = hfc_type_lookup(t->type);
        t->known = ti != NULL;
        if (ti) {
            uint64_t row_bytes, rows = t->nelem / t->ne[0], end;
            if (t->ne[0] % ti->blck != 0) return rfail(r, HFC_EFORMAT, "row length is not a multiple of the type's block size");
            if (hfc_type_row_bytes(t->type, t->ne[0], &row_bytes) != HFC_OK) return rfail(r, HFC_ERANGE, "row size overflow");
            if (!hfc_mul_u64(row_bytes, rows, &t->nbytes)) return rfail(r, HFC_ERANGE, "tensor size overflow");
            if (!hfc_add_u64(t->offset, t->nbytes, &end) || end > data_size)
                return rfail(r, HFC_EFORMAT, "tensor data extends past end of file");
        } else if (t->offset > data_size) {
            return rfail(r, HFC_EFORMAT, "tensor offset past end of file");
        }
    }
    return build_hash(g, r);
}

static hfc_status open_common(gguf_file *g, char *err, size_t errcap)
{
    rd_t r;
    hfc_status rc;
    r.p = (const unsigned char *)g->map.base;
    r.len = g->map.len;
    r.pos = 0;
    r.err = err;
    r.errcap = errcap;
    rc = parse(g, &r);
    if (rc != HFC_OK) gguf_close(g);
    return rc;
}

hfc_status gguf_open(gguf_file *g, const char *path, char *err, size_t errcap)
{
    hfc_status rc;
    memset(g, 0, sizeof *g);
    if (err && errcap) err[0] = '\0';
    rc = pal_map_file(path, &g->map);
    if (rc != HFC_OK) {
        if (err && errcap)
            snprintf(err, errcap, "cannot open '%s': %s", path, rc == HFC_EFORMAT ? "empty file" : hfc_strerror(rc));
        return rc;
    }
    g->owns_map = 1;
    return open_common(g, err, errcap);
}

hfc_status gguf_open_mem(gguf_file *g, const void *data, size_t len, char *err, size_t errcap)
{
    memset(g, 0, sizeof *g);
    if (err && errcap) err[0] = '\0';
    g->map.base = (void *)data;     /* never written; marked non-owning below */
    g->map.len = len;
    return open_common(g, err, errcap);
}

void gguf_close(gguf_file *g)
{
    uint64_t i;
    if (g->kv) {
        for (i = 0; i < g->n_kv; i++) { hfc_free(g->kv[i].key); hfc_free(g->kv[i].str); }
        hfc_free(g->kv);
    }
    if (g->t) {
        for (i = 0; i < g->n_tensors; i++) hfc_free(g->t[i].name);
        hfc_free(g->t);
    }
    hfc_free(g->thash);
    g->kv = NULL; g->t = NULL; g->thash = NULL;
    g->n_kv = g->n_tensors = 0;
    if (g->owns_map) pal_unmap(&g->map);
    g->owns_map = 0;
}

const gguf_kv *gguf_find_kv(const gguf_file *g, const char *key)
{
    uint64_t i;
    for (i = 0; i < g->n_kv; i++) if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    return NULL;
}

const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name)
{
    uint64_t h;
    if (!g->thash_cap) return NULL;
    h = fnv(name) & (g->thash_cap - 1);
    while (g->thash[h]) {
        const gguf_tensor *t = &g->t[g->thash[h] - 1];
        if (strcmp(t->name, name) == 0) return t;
        h = (h + 1) & (g->thash_cap - 1);
    }
    return NULL;
}

int gguf_get_u64(const gguf_file *g, const char *key, uint64_t *out)
{
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv) return 0;
    switch (kv->type) {
    case GGUF_T_U8: case GGUF_T_U16: case GGUF_T_U32: case GGUF_T_U64: case GGUF_T_BOOL:
        *out = kv->v.u; return 1;
    case GGUF_T_I8: case GGUF_T_I16: case GGUF_T_I32: case GGUF_T_I64:
        if (kv->v.i < 0) return 0;
        *out = (uint64_t)kv->v.i; return 1;
    default: return 0;
    }
}

int gguf_get_f64(const gguf_file *g, const char *key, double *out)
{
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv) return 0;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) { *out = kv->v.f; return 1; }
    return 0;
}

const char *gguf_get_str(const gguf_file *g, const char *key)
{
    const gguf_kv *kv = gguf_find_kv(g, key);
    return (kv && kv->type == GGUF_T_STR) ? kv->str : NULL;
}

const void *gguf_tensor_data(const gguf_file *g, const gguf_tensor *t)
{
    uint64_t start;
    if (!hfc_add_u64(g->data_off, t->offset, &start) || start > g->map.len) return NULL;
    return (const unsigned char *)g->map.base + start;
}

hfc_status gguf_foreach_str(const gguf_file *g, const gguf_kv *kv, gguf_str_cb f, void *ctx)
{
    const unsigned char *p = (const unsigned char *)g->map.base;
    uint64_t pos, i, end;
    if (kv->type != GGUF_T_ARR || kv->arr_type != GGUF_T_STR) return HFC_EINVAL;
    pos = kv->arr_off;
    end = kv->arr_off + kv->arr_bytes;
    for (i = 0; i < kv->arr_count; i++) {
        uint64_t n = 0, b;
        if (end - pos < 8) return HFC_EFORMAT;
        for (b = 0; b < 8; b++) n |= (uint64_t)p[pos + b] << (8 * b);
        pos += 8;
        if (n > end - pos) return HFC_EFORMAT;
        if (f(i, (const char *)p + pos, n, ctx)) return HFC_OK;
        pos += n;
    }
    return HFC_OK;
}
