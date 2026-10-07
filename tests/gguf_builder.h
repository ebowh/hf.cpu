/* gguf_builder.h - build GGUF images in memory for tests (little-endian). */
#ifndef GGUF_BUILDER_H
#define GGUF_BUILDER_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char *p; size_t len, cap; } gb_t;

static void gb_put(gb_t *b, const void *d, size_t n)
{
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 64;
        b->p = (unsigned char *)realloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, d, n);
    b->len += n;
}
static void gb_u8(gb_t *b, unsigned v)  { unsigned char c = (unsigned char)v; gb_put(b, &c, 1); }
static void gb_u32(gb_t *b, uint32_t v) { int i; for (i = 0; i < 4; i++) gb_u8(b, (v >> (8 * i)) & 0xff); }
static void gb_u64(gb_t *b, uint64_t v) { int i; for (i = 0; i < 8; i++) gb_u8(b, (unsigned)((v >> (8 * i)) & 0xff)); }
static void gb_str(gb_t *b, const char *s) { size_t n = strlen(s); gb_u64(b, n); gb_put(b, s, n); }

static void gb_header(gb_t *b, uint32_t version, uint64_t nt, uint64_t nkv)
{
    gb_put(b, "GGUF", 4); gb_u32(b, version); gb_u64(b, nt); gb_u64(b, nkv);
}
static void gb_kv_str(gb_t *b, const char *k, const char *v) { gb_str(b, k); gb_u32(b, 8); gb_str(b, v); }
static void gb_kv_u32(gb_t *b, const char *k, uint32_t v)    { gb_str(b, k); gb_u32(b, 4); gb_u32(b, v); }
static void gb_kv_f32(gb_t *b, const char *k, float v)
{
    uint32_t r; memcpy(&r, &v, 4); gb_str(b, k); gb_u32(b, 6); gb_u32(b, r);
}
static void gb_kv_arr_str(gb_t *b, const char *k, const char **v, uint64_t n)
{
    uint64_t i; gb_str(b, k); gb_u32(b, 9); gb_u32(b, 8); gb_u64(b, n);
    for (i = 0; i < n; i++) gb_str(b, v[i]);
}
static void gb_tensor(gb_t *b, const char *name, uint32_t nd, const uint64_t *dims, uint32_t type, uint64_t off)
{
    uint32_t i; gb_str(b, name); gb_u32(b, nd);
    for (i = 0; i < nd; i++) gb_u64(b, dims[i]);
    gb_u32(b, type); gb_u64(b, off);
}
static void gb_align(gb_t *b, size_t a) { while (b->len % a) gb_u8(b, 0); }

/* A small valid model: 3 kvs (+alignment), 2 tensors (q8_0 [64,4], f32 [8,2]). */
static void gb_sample(gb_t *b)
{
    static const char *toks[] = { "<s>", "hello", " world" };
    uint64_t d1[2] = { 64, 4 }, d2[2] = { 8, 2 };
    unsigned i, j;
    gb_header(b, 3, 2, 4);
    gb_kv_str(b, "general.architecture", "qwen2");
    gb_kv_u32(b, "qwen2.block_count", 2);
    gb_kv_f32(b, "qwen2.rope.freq_base", 1000000.0f);
    gb_kv_arr_str(b, "tokenizer.ggml.tokens", toks, 3);
    gb_tensor(b, "token_embd.weight", 2, d1, 8, 0);          /* 4 rows x 2 blocks x 34 B = 272 */
    gb_tensor(b, "blk.0.attn_q.weight", 2, d2, 0, 288);      /* 16 floats = 64 B */
    gb_align(b, 32);
    for (i = 0; i < 8; i++) {                                 /* q8_0 blocks: d = 1.0, qs[j] = j */
        gb_u8(b, 0x00); gb_u8(b, 0x3c);
        for (j = 0; j < 32; j++) gb_u8(b, j);
    }
    gb_align(b, 32);
    for (i = 0; i < 16; i++) { float f = (float)i; uint32_t r; memcpy(&r, &f, 4); gb_u32(b, r); }
}
#endif
