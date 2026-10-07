#include "../src/tok.h"
#include "../src/pal.h"
#include "t.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char err[256];
static gguf_file g;
static hfc_tok *tok;

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    char *b;
    long n;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    b = (char *)malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    b[n] = '\0';
    *len = (size_t)n;
    return b;
}

static int ids_equal(const uint32_t *a, size_t na, const char *spec)
{
    /* spec: space-separated decimal ids */
    size_t i = 0;
    const char *p = spec;
    while (*p) {
        char *end;
        unsigned long v;
        while (*p == ' ') p++;
        if (!*p) break;
        v = strtoul(p, &end, 10);
        if (end == p) return 0;
        if (i >= na || a[i] != (uint32_t)v) return 0;
        i++;
        p = end;
    }
    return i == na;
}

static void test_vocab_file(void)
{
    size_t li, lo, k;
    char *inp = slurp("tests/data/ggml-vocab-qwen2.gguf.inp", &li);
    char *out = slurp("tests/data/ggml-vocab-qwen2.gguf.out", &lo);
    const char *sep = "\n__ggml_vocab_test__\n";
    char *ip = inp, *op = out;
    int ncase = 0;
    (void)k;
    CHECK(inp && out);
    if (!inp || !out) return;
    for (;;) {
        char *hit = strstr(ip, sep);
        char *nl = strchr(op, '\n');
        char *text, *line;
        uint32_t *ids; size_t n;
        char *back; size_t blen;
        if (!hit || !nl) break;
        text = (char *)malloc((size_t)(hit - ip) + 1);
        memcpy(text, ip, (size_t)(hit - ip)); text[hit - ip] = '\0';
        line = (char *)malloc((size_t)(nl - op) + 1);
        memcpy(line, op, (size_t)(nl - op)); line[nl - op] = '\0';
        CHECK(hfc_tok_encode(tok, text, (size_t)(hit - ip), 0, &ids, &n) == HFC_OK);
        if (!ids_equal(ids, n, line)) {
            size_t q;
            t_fail_++;
            fprintf(stderr, "FAIL case %d: text=[%s]\n  expected: %s\n  got:     ", ncase, text, line);
            for (q = 0; q < n; q++) fprintf(stderr, " %u", ids[q]);
            fprintf(stderr, "\n");
        } else t_run_++;
        /* decoding gives the text back (inputs here contain no special tokens) */
        CHECK(hfc_tok_decode(tok, ids, n, 0, &back, &blen) == HFC_OK);
        CHECK(blen == (size_t)(hit - ip) && memcmp(back, text, blen) == 0);
        hfc_free(back); hfc_free(ids);
        free(text); free(line);
        ip = hit + strlen(sep);
        op = nl + 1;
        ncase++;
    }
    CHECK(ncase == 46);
    free(inp); free(out);
}

static void test_specials(void)
{
    const char *s = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    uint32_t *ids, *ids2; size_t n, n2;
    char *back; size_t bl;
    CHECK(hfc_tok_encode(tok, s, strlen(s), HFC_TOK_PARSE_SPECIAL, &ids, &n) == HFC_OK);
    CHECK(n >= 6 && ids[0] == 151644 && ids[n - 4] != 0);
    {
        size_t i, count_end = 0;
        for (i = 0; i < n; i++) if (ids[i] == 151645) count_end++;
        CHECK(count_end == 1);
    }
    CHECK(hfc_tok_decode(tok, ids, n, 1, &back, &bl) == HFC_OK);
    CHECK(bl == strlen(s) && memcmp(back, s, bl) == 0);
    hfc_free(back);
    CHECK(hfc_tok_decode(tok, ids, n, 0, &back, &bl) == HFC_OK);      /* control tokens hidden */
    CHECK(strstr(back, "<|im_start|>") == NULL && strstr(back, "user") != NULL);
    hfc_free(back);
    CHECK(hfc_tok_encode(tok, s, strlen(s), 0, &ids2, &n2) == HFC_OK);  /* not parsed: plain text */
    CHECK(n2 > n && ids2[0] != 151644);
    hfc_free(ids); hfc_free(ids2);
}

static void test_utf8_prefix(void)
{
    const unsigned char a[] = { 'a', 0xe4, 0xbd, 0xa0 };
    CHECK(hfc_utf8_complete_prefix(a, 4) == 4);
    CHECK(hfc_utf8_complete_prefix(a, 3) == 1);
    CHECK(hfc_utf8_complete_prefix(a, 2) == 1);
    CHECK(hfc_utf8_complete_prefix(a, 1) == 1);
    CHECK(hfc_utf8_complete_prefix(a, 0) == 0);
    {
        const unsigned char e[] = { 0xf0, 0x9f, 0x99, 0x82, 0xf0, 0x9f };
        CHECK(hfc_utf8_complete_prefix(e, 6) == 4);
        CHECK(hfc_utf8_complete_prefix(e, 5) == 4);
        CHECK(hfc_utf8_complete_prefix(e, 4) == 4);
    }
    {
        const unsigned char bad[] = { 0xff, 0xfe, 0x80 };           /* invalid bytes are "complete" */
        CHECK(hfc_utf8_complete_prefix(bad, 3) == 3);
    }
}

static void test_pathological(void)
{
    size_t n = 300000, i;
    char *buf = (char *)malloc(n);
    uint32_t *ids; size_t cnt;
    double t0 = pal_now();
    memset(buf, 'a', n);                                              /* one enormous word */
    CHECK(hfc_tok_encode(tok, buf, n, 0, &ids, &cnt) == HFC_OK);
    CHECK(cnt > 0 && cnt < n);
    hfc_free(ids);
    memset(buf, ' ', n);                                              /* one enormous whitespace run */
    CHECK(hfc_tok_encode(tok, buf, n, 0, &ids, &cnt) == HFC_OK);
    hfc_free(ids);
    for (i = 0; i < n; i++) buf[i] = (char)((i * 2654435761u) >> 13);   /* garbage bytes, invalid UTF-8 */
    CHECK(hfc_tok_encode(tok, buf, n, HFC_TOK_PARSE_SPECIAL, &ids, &cnt) == HFC_OK);
    {
        char *back; size_t bl;
        CHECK(hfc_tok_decode(tok, ids, cnt, 1, &back, &bl) == HFC_OK);
        CHECK(bl == n && memcmp(back, buf, n) == 0);                   /* byte-exact round trip */
        hfc_free(back);
    }
    hfc_free(ids);
    CHECK(pal_now() - t0 < 10.0);
    free(buf);
}

static void test_faults(void)
{
    long n;
    int fails = 0;
    for (n = 1; n < 40; n++) {
        hfc_tok *t2 = NULL;
        hfc_status rc;
        hfc_fail_after(n);
        rc = hfc_tok_load(&t2, &g, err, sizeof err);
        hfc_fail_after(0);
        if (rc == HFC_OK) hfc_tok_free(t2); else { fails++; CHECK(rc == HFC_ENOMEM && t2 == NULL); }
    }
    CHECK(fails > 20);
    for (n = 1; n < 120; n++) {
        uint32_t *ids = NULL; size_t cnt;
        hfc_status rc;
        hfc_fail_after(n);
        rc = hfc_tok_encode(tok, "Hello, world! 12345 <|im_end|>  \n\n tail", 40, HFC_TOK_PARSE_SPECIAL, &ids, &cnt);
        hfc_fail_after(0);
        if (rc == HFC_OK) hfc_free(ids); else CHECK(rc == HFC_ENOMEM && ids == NULL);
    }
}

static int hexval(int c) { return c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10; }

static void test_golden(const char *path)
{
    size_t len;
    char *d = slurp(path, &len), *p;
    char *text = NULL; size_t tl = 0;
    int checked = 0;
    if (!d) { fprintf(stderr, "(no golden file %s, skipped)\n", path); return; }
    for (p = d; p && *p; ) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = '\0';
        if (strncmp(p, "text_hex ", 9) == 0) {
            const char *h = p + 9;
            size_t i, hl = strlen(h);
            free(text);
            text = (char *)malloc(hl / 2 + 1);
            for (i = 0; i + 1 < hl; i += 2) text[i / 2] = (char)(hexval(h[i]) * 16 + hexval(h[i + 1]));
            tl = hl / 2;
        } else if (strncmp(p, "prompt_tokens ", 14) == 0 && text) {
            uint32_t *ids; size_t n;
            CHECK(hfc_tok_encode(tok, text, tl, HFC_TOK_PARSE_SPECIAL, &ids, &n) == HFC_OK);
            if (!ids_equal(ids, n, p + 14)) { t_fail_++; fprintf(stderr, "FAIL golden prompt tokens differ (text length %lu)\n", (unsigned long)tl); }
            else t_run_++;
            hfc_free(ids);
            checked++;
        }
        p = nl ? nl + 1 : NULL;
    }
    CHECK(checked > 0);
    free(text); free(d);
}

int main(int argc, char **argv)
{
    int i;
    if (gguf_open(&g, "tests/data/ggml-vocab-qwen2.gguf", err, sizeof err) != HFC_OK) {
        fprintf(stderr, "cannot open vocab file (run from the repository root): %s\n", err);
        return 1;
    }
    if (hfc_tok_load(&tok, &g, err, sizeof err) != HFC_OK) { fprintf(stderr, "tokenizer load: %s\n", err); return 1; }
    CHECK(hfc_tok_n_vocab(tok) > 150000);
    test_vocab_file();
    test_specials();
    test_utf8_prefix();
    test_pathological();
    test_faults();
    test_golden("tests/golden/qwen2.5-0.5b-q8_0.golden");
    for (i = 1; i < argc; i++) test_golden(argv[i]);
    hfc_tok_free(tok);
    gguf_close(&g);
    CHECK(hfc_live_allocs() == 0);
    return t_report("test_tok");
}
