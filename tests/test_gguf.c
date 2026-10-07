#include "../src/gguf.h"
#include "../src/ggtype.h"
#include "gguf_builder.h"
#include "t.h"
#include <stdio.h>

static char err[256];

static int collect(uint64_t idx, const char *s, uint64_t len, void *ctx)
{
    char *out = (char *)ctx;
    size_t cur = strlen(out);
    (void)idx;
    memcpy(out + cur, s, (size_t)len);
    out[cur + len] = '|';
    out[cur + len + 1] = '\0';
    return 0;
}

static void test_valid(void)
{
    gb_t b = { 0 };
    gguf_file g;
    const gguf_tensor *t;
    char toks[64] = "";
    uint64_t u;
    double f;
    float row[64];
    unsigned i;

    gb_sample(&b);
    CHECK(gguf_open_mem(&g, b.p, b.len, err, sizeof err) == HFC_OK);
    CHECK(g.version == 3 && g.n_kv == 4 && g.n_tensors == 2 && g.align == 32);
    CHECK_STR(gguf_get_str(&g, "general.architecture"), "qwen2");
    CHECK(gguf_get_u64(&g, "qwen2.block_count", &u) && u == 2);
    CHECK(gguf_get_f64(&g, "qwen2.rope.freq_base", &f) && f == 1000000.0);
    CHECK(!gguf_get_u64(&g, "general.architecture", &u));
    CHECK(gguf_get_str(&g, "nope") == NULL);
    CHECK(gguf_foreach_str(&g, gguf_find_kv(&g, "tokenizer.ggml.tokens"), collect, toks) == HFC_OK);
    CHECK_STR(toks, "<s>|hello| world|");
    t = gguf_find_tensor(&g, "token_embd.weight");
    CHECK(t != NULL && t->known && t->nelem == 256 && t->nbytes == 272 && t->type == 8);
    CHECK(gguf_find_tensor(&g, "blk.0.attn_q.weight") != NULL);
    CHECK(gguf_find_tensor(&g, "missing") == NULL);
    CHECK(hfc_dequant_row(t->type, gguf_tensor_data(&g, t), row, 64) == HFC_OK);
    for (i = 0; i < 32; i++) CHECK_NEAR(row[i], (double)i, 0);
    {
        const float *fp = (const float *)gguf_tensor_data(&g, gguf_find_tensor(&g, "blk.0.attn_q.weight"));
        CHECK(fp[5] == 5.0f);
    }
    gguf_close(&g);
    CHECK(hfc_live_allocs() == 0);
    free(b.p);
}

static hfc_status open_img(const gb_t *b)
{
    gguf_file g;
    hfc_status rc = gguf_open_mem(&g, b->p, b->len, err, sizeof err);
    if (rc == HFC_OK) gguf_close(&g);
    return rc;
}

static void test_truncation(void)
{
    gb_t b = { 0 };
    size_t n;
    gb_sample(&b);
    for (n = 0; n < b.len; n++) {
        gb_t c = { b.p, n, n };
        CHECK(open_img(&c) != HFC_OK);
    }
    CHECK(hfc_live_allocs() == 0);
    free(b.p);
}

static void test_attacks(void)
{
    gb_t b;
    uint64_t dim[2] = { 4, 4 };
    uint64_t huge[2] = { 1ull << 40, 1ull << 40 };
    uint64_t dim9[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    memset(&b, 0, sizeof b); gb_put(&b, "GGUX", 4); gb_u32(&b, 3); gb_u64(&b, 0); gb_u64(&b, 0);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);
    memset(&b, 0, sizeof b); gb_put(&b, "FUGG", 4); gb_u32(&b, 3); gb_u64(&b, 0); gb_u64(&b, 0);
    CHECK(open_img(&b) == HFC_EFORMAT && strstr(err, "big-endian")); free(b.p);
    memset(&b, 0, sizeof b); gb_header(&b, 1, 0, 0);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);
    memset(&b, 0, sizeof b); gb_header(&b, 3, 0, 1ull << 40);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);
    memset(&b, 0, sizeof b); gb_header(&b, 3, 1ull << 40, 0);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);

    /* string length past end of file */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 0, 1);
    gb_u64(&b, 1ull << 62); gb_put(&b, "abc", 3);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);

    /* array count * element size overflows */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 0, 1);
    gb_str(&b, "a"); gb_u32(&b, 9); gb_u32(&b, 10); gb_u64(&b, 1ull << 62);
    CHECK(open_img(&b) != HFC_OK); free(b.p);

    /* string array count larger than the file */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 0, 1);
    gb_str(&b, "a"); gb_u32(&b, 9); gb_u32(&b, 8); gb_u64(&b, 1ull << 40);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);

    /* nested array */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 0, 1);
    gb_str(&b, "a"); gb_u32(&b, 9); gb_u32(&b, 9); gb_u64(&b, 0);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);

    /* duplicate keys */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 0, 2);
    gb_kv_u32(&b, "k", 1); gb_kv_u32(&b, "k", 2);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);

    /* bad alignment values */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 0, 1); gb_kv_u32(&b, "general.alignment", 3);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);
    memset(&b, 0, sizeof b); gb_header(&b, 3, 0, 1); gb_kv_u32(&b, "general.alignment", 0);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);

    /* tensor dims overflow, zero dims, too many dims */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 1, 0); gb_tensor(&b, "t", 2, huge, 0, 0); gb_align(&b, 32);
    CHECK(open_img(&b) == HFC_ERANGE); free(b.p);
    memset(&b, 0, sizeof b); gb_header(&b, 3, 1, 0); gb_tensor(&b, "t", 0, dim, 0, 0); gb_align(&b, 32);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);
    memset(&b, 0, sizeof b); gb_header(&b, 3, 1, 0); gb_tensor(&b, "t", 9, dim9, 0, 0); gb_align(&b, 32);
    CHECK(open_img(&b) != HFC_OK); free(b.p);

    /* tensor beyond end; offset overflow; row not multiple of block */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 1, 0); gb_tensor(&b, "t", 2, dim, 0, 1ull << 63); gb_align(&b, 32);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);
    memset(&b, 0, sizeof b); gb_header(&b, 3, 1, 0); gb_tensor(&b, "t", 2, dim, 0, ~0ull); gb_align(&b, 32);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);
    memset(&b, 0, sizeof b); gb_header(&b, 3, 1, 0); gb_tensor(&b, "t", 2, dim, 8, 0); gb_align(&b, 32);
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);   /* q8_0 needs rows of 32k */

    /* duplicate tensor names */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 2, 0);
    gb_tensor(&b, "t", 2, dim, 0, 0); gb_tensor(&b, "t", 2, dim, 0, 64); gb_align(&b, 32);
    { size_t i; for (i = 0; i < 128; i++) gb_u8(&b, 0); }
    CHECK(open_img(&b) == HFC_EFORMAT); free(b.p);

    /* unknown tensor type is tolerated at open (inspect can still list it) */
    memset(&b, 0, sizeof b); gb_header(&b, 3, 1, 0); gb_tensor(&b, "t", 2, dim, 9999, 0); gb_align(&b, 32);
    { size_t i; for (i = 0; i < 64; i++) gb_u8(&b, 0); }
    CHECK(open_img(&b) == HFC_OK); free(b.p);

    CHECK(hfc_live_allocs() == 0);
}

static uint64_t rs = 0x1234567887654321ull;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 16); }

static void test_fuzz(void)
{
    gb_t b = { 0 };
    int it, ok = 0, bad = 0;
    gb_sample(&b);
    for (it = 0; it < 60000; it++) {
        gb_t c;
        gguf_file g;
        int nm = 1 + (int)(rnd() % 4), k;
        c.cap = c.len = b.len;
        c.p = (unsigned char *)malloc(b.len);
        memcpy(c.p, b.p, b.len);
        for (k = 0; k < nm; k++) {
            size_t pos = rnd() % (b.len - 8);          /* mostly the header region */
            if (rnd() % 3 == 0) pos = rnd() % 200;
            switch (rnd() % 4) {
            case 0: c.p[pos] = (unsigned char)rnd(); break;
            case 1: c.p[pos] ^= (unsigned char)(1u << (rnd() % 8)); break;
            case 2: c.p[pos] = 0xff; break;
            default: c.p[pos] = 0; break;
            }
        }
        if (rnd() % 8 == 0) c.len -= rnd() % 16;
        if (gguf_open_mem(&g, c.p, c.len, err, sizeof err) == HFC_OK) {
            uint64_t i;
            ok++;
            for (i = 0; i < g.n_tensors; i++) {
                const gguf_tensor *t = &g.t[i];
                if (t->known) {
                    uint64_t end = g.data_off + t->offset + t->nbytes;
                    CHECK(end <= c.len);
                    CHECK(gguf_tensor_data(&g, t) != NULL);
                }
            }
            gguf_close(&g);
        } else bad++;
        free(c.p);
    }
    CHECK(ok > 0 && bad > 0);
    CHECK(hfc_live_allocs() == 0);
    free(b.p);
}

static void test_fault_injection(void)
{
    gb_t b = { 0 };
    long n;
    int fails = 0, oks = 0;
    gb_sample(&b);
    for (n = 1; n < 200; n++) {
        gguf_file g;
        hfc_status rc;
        hfc_fail_after(n);
        rc = gguf_open_mem(&g, b.p, b.len, err, sizeof err);
        hfc_fail_after(0);
        if (rc == HFC_OK) { oks++; gguf_close(&g); } else { fails++; CHECK(rc == HFC_ENOMEM); }
        CHECK(hfc_live_allocs() == 0);
    }
    CHECK(fails > 5 && oks > 0);
    free(b.p);
}

int main(void)
{
    test_valid();
    test_truncation();
    test_attacks();
    test_fuzz();
    test_fault_injection();
    return t_report("test_gguf");
}
