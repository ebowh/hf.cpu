#include "../src/kern.h"
#include "../src/mathx.h"
#include "../src/probe.h"
#include "../src/ggtype.h"
#include "t.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x2545f4914f6cdd1dull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float frand(void) { return (float)((int)(rnd() % 20001) - 10000) / 5000.0f; }

static float h2f(uint16_t h)       /* independent reference conversion */
{
    int e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexpf((float)(1024 + m), e - 25);
    return (h & 0x8000) ? -v : v;
}

static void test_expf(void)
{
    float x, worst = 0;
    for (x = -87.0f; x < 88.0f; x += 0.0137f) {
        double ref = exp((double)x), got = (double)hfc_expf(x);
        double rel = fabs(got - ref) / ref;
        if (rel > worst) worst = (float)rel;
    }
    CHECK(worst < 4e-7f);
    CHECK(hfc_expf(0.0f) == 1.0f);
    CHECK(hfc_expf(-100.0f) == 0.0f);
    CHECK(isfinite(hfc_expf(1000.0f)));
    CHECK(hfc_silu(0.0f) == 0.0f);
    CHECK_NEAR(hfc_silu(1.0f), 0.7310585786, 1e-6);
}

static void test_f16(const hfc_kernels *k)
{
    uint32_t b;
    int bad = 0;
    uint16_t h;
    for (b = 0; b < 0xffffffffu - 16; b += 17) {
        float f; uint16_t a, c;
        memcpy(&f, &b, 4);
        if (f != f) continue;
        a = hfc_f32_to_f16(f);
        k->f32_to_f16(&f, &c, 1);
        if (a != c) { if (bad++ < 5) fprintf(stderr, "f16 mismatch for bits %08x: %04x vs %04x\n", b, a, c); }
    }
    /* exact halfway cases between adjacent halves */
    for (h = 0; h < 0x7bff; h++) {
        float lo = h2f(h), hi = h2f((uint16_t)(h + 1)), mid = (float)(((double)lo + (double)hi) / 2.0);
        uint16_t a = hfc_f32_to_f16(mid), c;
        k->f32_to_f16(&mid, &c, 1);
        if (a != c) { if (bad++ < 5) fprintf(stderr, "f16 tie mismatch at %04x: %04x vs %04x\n", h, a, c); }
        if (a != h && a != (uint16_t)(h + 1)) bad++;
        if ((a & 1) != 0) bad++;                         /* ties go to even */
    }
    CHECK(bad == 0);
    /* every finite half converts back exactly through dot_f32_f16 */
    for (h = 0; h < 0x7c00; h++) {
        float one = 1.0f, got;
        got = k->dot_f32_f16(&one, &h, 1);
        if (got != h2f(h)) bad++;
    }
    CHECK(bad == 0);
}

static void test_q8(const hfc_kernels *k, const hfc_kernels *g)
{
    size_t n, nb, i, trial;
    for (trial = 0; trial < 200; trial++) {
        float *x, *y;
        unsigned char *qa, *qb, *qa2;
        double ref = 0;
        float got, got_g;
        nb = 1 + trial % 40;
        n = nb * 32;
        x = (float *)malloc(n * sizeof(float)); y = (float *)malloc(n * sizeof(float));
        qa = (unsigned char *)malloc(nb * HFC_Q8_0_BLOCK); qb = (unsigned char *)malloc(nb * HFC_Q8_0_BLOCK);
        qa2 = (unsigned char *)malloc(nb * HFC_Q8_0_BLOCK);
        for (i = 0; i < n; i++) { x[i] = frand() * (1 + (float)(i / 32 % 5)); y[i] = frand(); }
        g->quantize_q8_0(x, qa, n);
        k->quantize_q8_0(x, qa2, n);
        CHECK(memcmp(qa, qa2, nb * HFC_Q8_0_BLOCK) == 0);
        g->quantize_q8_0(y, qb, n);
        /* reference: dequantize both and multiply in double */
        for (i = 0; i < nb; i++) {
            uint16_t da = (uint16_t)(qa[i * 34] | (qa[i * 34 + 1] << 8)), db = (uint16_t)(qb[i * 34] | (qb[i * 34 + 1] << 8));
            int j;
            for (j = 0; j < 32; j++) ref += (double)h2f(da) * (int8_t)qa[i * 34 + 2 + j] * (double)h2f(db) * (int8_t)qb[i * 34 + 2 + j];
        }
        got = k->dot_q8_0(qa, qb, nb);
        got_g = g->dot_q8_0(qa, qb, nb);
        CHECK(fabs((double)got - ref) <= 1e-4 * (fabs(ref) + 1.0));
        CHECK(fabs((double)got_g - ref) <= 1e-4 * (fabs(ref) + 1.0));
        /* quantization error is at most half a step */
        for (i = 0; i < nb; i++) {
            float d = h2f((uint16_t)(qa[i * 34] | (qa[i * 34 + 1] << 8)));
            int j;
            for (j = 0; j < 32; j++) {
                float dq = d * (int8_t)qa[i * 34 + 2 + j];
                if (fabsf(dq - x[i * 32 + j]) > d * 0.57f + 1e-6f * fabsf(x[i * 32 + j])) { t_fail_++; fprintf(stderr, "quantization error too large\n"); goto next; }
            }
        }
next:
        free(x); free(y); free(qa); free(qb); free(qa2);
    }
    {   /* all-zero block must not divide by zero */
        float z[32]; unsigned char q[34]; int j;
        memset(z, 0, sizeof z);
        k->quantize_q8_0(z, q, 32);
        for (j = 0; j < 34; j++) CHECK(q[j] == 0);
    }
}

static void test_f32(const hfc_kernels *k)
{
    size_t n;
    for (n = 0; n < 150; n++) {
        float a[160], b[160], y[160], y2[160];
        uint16_t h[160];
        double ref = 0, ref2 = 0;
        size_t i;
        float s = frand();
        for (i = 0; i < n; i++) { a[i] = frand(); b[i] = frand(); y[i] = y2[i] = frand(); }
        k->f32_to_f16(b, h, n);
        for (i = 0; i < n; i++) { ref += (double)a[i] * b[i]; ref2 += (double)a[i] * h2f(h[i]); y2[i] += s * h2f(h[i]); }
        CHECK(fabs((double)k->dot_f32(a, b, n) - ref) < 1e-4 * (fabs(ref) + 1));
        CHECK(fabs((double)k->dot_f32_f16(a, h, n) - ref2) < 1e-4 * (fabs(ref2) + 1));
        k->axpy_f32_f16(y, s, h, n);
        for (i = 0; i < n; i++) CHECK(fabsf(y[i] - y2[i]) < 1e-5f * (fabsf(y2[i]) + 1));
    }
}

static void test_sincos(void)
{
    int i, bad = 0;
    float worst = 0;
    for (i = 0; i < 400000; i++) {
        float th = (i < 1000) ? (float)i * 0.01f : (float)(rnd() % 40000000) / 1000.0f;   /* up to 40000 rad */
        float c, s;
        double rc = cos((double)th), rs_ = sin((double)th);
        float e1, e2;
        hfc_cos_sin(th, &c, &s);
        e1 = fabsf(c - (float)rc); e2 = fabsf(s - (float)rs_);
        if (e1 > worst) worst = e1;
        if (e2 > worst) worst = e2;
        if (e1 > 1.2e-7f || e2 > 1.2e-7f) bad++;          /* within 1 ulp of the correctly rounded value */
    }
    CHECK(bad == 0);
    {   float c, s;
        hfc_cos_sin(0.0f, &c, &s);   CHECK(c == 1.0f && s == 0.0f);
        hfc_cos_sin(-3.0f, &c, &s);  CHECK_NEAR(c, cos(-3.0), 1e-7); CHECK_NEAR(s, sin(-3.0), 1e-7);
    }
    CHECK_NEAR(hfc_rope_theta_scale(1000000.0f, 64), pow(1000000.0, -2.0 / 64.0), 7e-8);
    CHECK_NEAR(hfc_rope_theta_scale(10000.0f, 128), pow(10000.0, -2.0 / 128.0), 7e-8);
    CHECK_NEAR(hfc_rope_theta_scale(10000.0f, 96), pow(10000.0, -2.0 / 96.0), 7e-8);
}

static void rand_half_at(unsigned char *p)
{
    uint16_t h = (uint16_t)(((8 + rnd() % 8) << 10) | (rnd() & 0x3ff));         /* finite, 2^-7 .. 2^0 */
    p[0] = (unsigned char)(h & 0xff); p[1] = (unsigned char)(h >> 8);
}

typedef float (*dotfn)(const void *, const void *, size_t);

static void test_quant_dot(const hfc_kernels *k, const char *label)
{
    struct { uint32_t type; int act_k; size_t blk, bytes; int halves[3]; dotfn fn; } cases[] = {
        { 2,  0, 32,  18,  { 0, -1, -1 },   k->dot_q4_0 },
        { 6,  0, 32,  22,  { 0, -1, -1 },   k->dot_q5_0 },
        { 12, 1, 256, 144, { 0, 2, -1 },     k->dot_q4_K },
        { 13, 1, 256, 176, { 0, 2, -1 },     k->dot_q5_K },
        { 14, 1, 256, 210, { 208, -1, -1 },  k->dot_q6_K },
    };
    size_t ci;
    for (ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
        int trial;
        for (trial = 0; trial < 150; trial++) {
            size_t nb = 1 + (size_t)(rnd() % 6), n, i;
            unsigned char *w, *qa;
            float *x, *wf, *af;
            double ref = 0;
            float got;
            int h;
            n = nb * cases[ci].blk;
            w = (unsigned char *)malloc(nb * cases[ci].bytes);
            qa = (unsigned char *)malloc(nb * 292 + 64);
            x = (float *)malloc(n * sizeof(float)); wf = (float *)malloc(n * sizeof(float)); af = (float *)malloc(n * sizeof(float));
            for (i = 0; i < nb * cases[ci].bytes; i++) w[i] = (unsigned char)rnd();
            for (i = 0; i < nb; i++)
                for (h = 0; h < 3; h++) if (cases[ci].halves[h] >= 0) rand_half_at(w + i * cases[ci].bytes + cases[ci].halves[h]);
            for (i = 0; i < n; i++) x[i] = frand() * (1 + (float)(i / 32 % 7));
            if (cases[ci].act_k) {
                k->quantize_q8_K(x, qa, n);
                for (i = 0; i < nb; i++) {
                    float d; size_t j;
                    memcpy(&d, qa + i * 292, 4);
                    for (j = 0; j < 256; j++) af[i * 256 + j] = d * (float)(int8_t)qa[i * 292 + 4 + j];
                }
            } else {
                k->quantize_q8_0(x, qa, n);
                for (i = 0; i < nb; i++) {
                    float d = h2f((uint16_t)(qa[i * 34] | (qa[i * 34 + 1] << 8))); size_t j;
                    for (j = 0; j < 32; j++) af[i * 32 + j] = d * (float)(int8_t)qa[i * 34 + 2 + j];
                }
            }
            if (hfc_dequant_row(cases[ci].type, w, wf, n) != HFC_OK) { t_fail_++; fprintf(stderr, "no dequant for type %u\n", cases[ci].type); return; }
            for (i = 0; i < n; i++) ref += (double)wf[i] * (double)af[i];
            got = cases[ci].fn(w, qa, nb);
            if (!(fabs((double)got - ref) <= 2e-4 * (fabs(ref) + 1.0))) {
                t_fail_++;
                fprintf(stderr, "FAIL %s type %u trial %d: kernel %g reference %g\n", label, cases[ci].type, trial, (double)got, ref);
                trial = 1000;
            } else t_run_++;
            free(w); free(qa); free(x); free(wf); free(af);
        }
    }
    {   /* Q8_K quantizer: error within half a step, block sums exact, zero block ok */
        float x[512]; unsigned char q[2 * 292]; size_t i, b; int j;
        for (i = 0; i < 512; i++) x[i] = frand() * (i < 256 ? 1.0f : 30.0f);
        k->quantize_q8_K(x, q, 512);
        for (b = 0; b < 2; b++) {
            float d; memcpy(&d, q + b * 292, 4);
            for (j = 0; j < 256; j++) CHECK(fabsf(d * (float)(int8_t)q[b * 292 + 4 + j] - x[b * 256 + j]) <= fabsf(d) * 0.5001f + 1e-6f);
            for (j = 0; j < 16; j++) {
                int s = 0, e;
                for (e = 0; e < 16; e++) s += (int8_t)q[b * 292 + 4 + 16 * j + e];
                CHECK((int16_t)(q[b * 292 + 260 + 2 * j] | (q[b * 292 + 261 + 2 * j] << 8)) == s);
            }
        }
        memset(x, 0, sizeof x);
        k->quantize_q8_K(x, q, 256);
        for (j = 0; j < 292; j++) CHECK(q[j] == 0);
    }
}

static void test_norm_softmax(void)
{
    float x[5] = { 1, 2, 3, 4, 5 }, w[5] = { 1, 1, 2, 2, 0.5f }, y[5], s[4] = { 1, 2, 3, 4 }, tot = 0;
    int i;
    hfc_rmsnorm(x, w, y, 5, 1e-6f);
    {
        double ms = (1 + 4 + 9 + 16 + 25) / 5.0, sc = 1.0 / sqrt(ms + 1e-6);
        for (i = 0; i < 5; i++) CHECK_NEAR(y[i], x[i] * sc * w[i], 1e-5);
    }
    hfc_softmax(s, 4);
    for (i = 0; i < 4; i++) tot += s[i];
    CHECK_NEAR(tot, 1.0, 1e-6);
    CHECK(s[3] > s[2] && s[2] > s[1]);
    CHECK_NEAR(s[3] / s[2], exp(1.0), 1e-5);
}

int main(void)
{
    hfc_cpu cpu;
    const hfc_kernels *g = hfc_kernels_generic(), *k;
    hfc_cpu_detect(&cpu);
    k = hfc_kernels_for(&cpu);
    fprintf(stderr, "kernel sets under test: generic, %s\n", k->isa);
    test_expf();
    test_norm_softmax();
    test_sincos();
    test_f16(g);
    test_q8(g, g);
    test_f32(g);
    test_quant_dot(g, "generic");
    if (k != g) { test_f16(k); test_q8(k, g); test_f32(k); test_quant_dot(k, k->isa); }
    return t_report("test_kern");
}
