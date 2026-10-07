/* kern_generic.c - portable reference kernels. */
#include "kern.h"
#include "mathx.h"

#include <string.h>

static uint64_t read_sum_generic(const void *p, size_t bytes)
{
    const uint64_t *w = (const uint64_t *)p;
    size_t n = bytes / 8, i;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (i = 0; i + 4 <= n; i += 4) {
        a0 += w[i]; a1 += w[i + 1]; a2 += w[i + 2]; a3 += w[i + 3];
    }
    return a0 + a1 + a2 + a3;
}

static float fma_burn_generic(long iters)
{
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float m = 0.999999f, c = 1e-7f;
    long i;
    int k;
    for (i = 0; i < iters; i++)
        for (k = 0; k < 8; k++)
            a[k] = a[k] * m + c;
    return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
}

/* ---- model kernels (portable C) ---- */

static float f16f(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu, r;
    float f;
    if (e == 0) {
        if (m == 0) r = s;
        else { e = 127 - 15 + 1; while (!(m & 0x400u)) { m <<= 1; e--; } m &= 0x3ffu; r = s | (e << 23) | (m << 13); }
    } else if (e == 31) r = s | 0x7f800000u | (m << 13);
    else r = s | ((e + 127 - 15) << 23) | (m << 13);
    memcpy(&f, &r, sizeof f);
    return f;
}

static float rne(float v)               /* round to nearest even, |v| < 2^22 */
{
    volatile float t = v + 12582912.0f;
    return t - 12582912.0f;
}

static void quantize_q8_0_generic(const float *x, void *y, size_t n)
{
    unsigned char *out = (unsigned char *)y;
    size_t nb = n / 32, b;
    int j;
    for (b = 0; b < nb; b++, x += 32, out += HFC_Q8_0_BLOCK) {
        float amax = 0.0f, d, id;
        uint16_t h;
        for (j = 0; j < 32; j++) { float a = x[j] < 0 ? -x[j] : x[j]; if (a > amax) amax = a; }
        d = amax / 127.0f;
        id = d != 0.0f ? 1.0f / d : 0.0f;
        h = hfc_f32_to_f16(d);
        out[0] = (unsigned char)(h & 0xff);
        out[1] = (unsigned char)(h >> 8);
        for (j = 0; j < 32; j++) out[2 + j] = (unsigned char)(int8_t)(int)rne(x[j] * id);
    }
}

static float dot_q8_0_generic(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    float sum = 0.0f;
    size_t b;
    int j;
    for (b = 0; b < nb; b++, pw += HFC_Q8_0_BLOCK, pa += HFC_Q8_0_BLOCK) {
        int isum = 0;
        float dw = f16f((uint16_t)(pw[0] | (pw[1] << 8))), da = f16f((uint16_t)(pa[0] | (pa[1] << 8)));
        for (j = 0; j < 32; j++) isum += (int)(int8_t)pw[2 + j] * (int)(int8_t)pa[2 + j];
        sum += (dw * da) * (float)isum;
    }
    return sum;
}

static float dot_f32_generic(const float *a, const float *b, size_t n)
{
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i;
    for (i = 0; i + 4 <= n; i += 4) { s0 += a[i] * b[i]; s1 += a[i+1] * b[i+1]; s2 += a[i+2] * b[i+2]; s3 += a[i+3] * b[i+3]; }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

static float dot_f32_f16_generic(const float *a, const uint16_t *b, size_t n)
{
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i;
    for (i = 0; i + 4 <= n; i += 4) { s0 += a[i] * f16f(b[i]); s1 += a[i+1] * f16f(b[i+1]); s2 += a[i+2] * f16f(b[i+2]); s3 += a[i+3] * f16f(b[i+3]); }
    for (; i < n; i++) s0 += a[i] * f16f(b[i]);
    return (s0 + s1) + (s2 + s3);
}

static void axpy_f32_f16_generic(float *y, float a, const uint16_t *x, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) y[i] += a * f16f(x[i]);
}

static void f32_to_f16_generic(const float *x, uint16_t *y, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) y[i] = hfc_f32_to_f16(x[i]);
}


/* ---- K-quants and 4/5-bit legacy dots (portable C) ---- */

static float rd_f16(const unsigned char *p) { return f16f((uint16_t)(p[0] | (p[1] << 8))); }

static void quantize_q8_K_generic(const float *x, void *y, size_t n)
{
    unsigned char *out = (unsigned char *)y;
    size_t nb = n / 256, b;
    int j;
    for (b = 0; b < nb; b++, x += 256, out += HFC_Q8_K_BLOCK) {
        float amax = 0.0f, max = 0.0f, iscale, d;
        int8_t *qs = (int8_t *)(out + 4);
        for (j = 0; j < 256; j++) {
            float a = x[j] < 0 ? -x[j] : x[j];
            if (a > amax) { amax = a; max = x[j]; }
        }
        if (amax == 0.0f) { memset(out, 0, HFC_Q8_K_BLOCK); continue; }
        iscale = -127.0f / max;
        for (j = 0; j < 256; j++) {
            int v = (int)rne(iscale * x[j]);
            qs[j] = (int8_t)(v > 127 ? 127 : v);
        }
        for (j = 0; j < 16; j++) {
            int s = 0, i;
            int16_t bs;
            for (i = 0; i < 16; i++) s += qs[16 * j + i];
            bs = (int16_t)s;
            out[260 + 2 * j] = (unsigned char)(bs & 0xff);
            out[261 + 2 * j] = (unsigned char)((bs >> 8) & 0xff);
        }
        d = 1.0f / iscale;
        memcpy(out, &d, 4);
    }
}

static int16_t rd_i16(const unsigned char *p) { return (int16_t)(p[0] | (p[1] << 8)); }

static float dot_q4_0_generic(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    float sum = 0.0f;
    size_t b;
    int j;
    for (b = 0; b < nb; b++, pw += 18, pa += HFC_Q8_0_BLOCK) {
        int isum = 0;
        for (j = 0; j < 16; j++)
            isum += ((pw[2 + j] & 0xF) - 8) * (int)(int8_t)pa[2 + j] + ((pw[2 + j] >> 4) - 8) * (int)(int8_t)pa[18 + j];
        sum += (rd_f16(pw) * rd_f16(pa)) * (float)isum;
    }
    return sum;
}

static float dot_q5_0_generic(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    float sum = 0.0f;
    size_t b;
    int j;
    for (b = 0; b < nb; b++, pw += 22, pa += HFC_Q8_0_BLOCK) {
        uint32_t qh = (uint32_t)pw[2] | ((uint32_t)pw[3] << 8) | ((uint32_t)pw[4] << 16) | ((uint32_t)pw[5] << 24);
        int isum = 0;
        for (j = 0; j < 16; j++) {
            int x0 = (int)(((pw[6 + j] & 0xF) | (((qh >> j) & 1u) << 4))) - 16;
            int x1 = (int)(((pw[6 + j] >> 4) | (((qh >> (j + 16)) & 1u) << 4))) - 16;
            isum += x0 * (int)(int8_t)pa[2 + j] + x1 * (int)(int8_t)pa[18 + j];
        }
        sum += (rd_f16(pw) * rd_f16(pa)) * (float)isum;
    }
    return sum;
}

/* 8 six-bit scales and 8 six-bit mins packed in 12 bytes (ggml's get_scale_min_k4) */
static void unpack_scales_k4(const unsigned char *q, unsigned char *sc, unsigned char *mn)
{
    int j;
    for (j = 0; j < 8; j++) {
        if (j < 4) { sc[j] = q[j] & 63; mn[j] = q[j + 4] & 63; }
        else {
            sc[j] = (unsigned char)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            mn[j] = (unsigned char)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }
}

static float dot_q4_K_generic(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    float sumf = 0.0f;
    size_t b;
    for (b = 0; b < nb; b++, pw += 144, pa += HFC_Q8_K_BLOCK) {
        unsigned char sc[8], mn[8];
        const unsigned char *q4 = pw + 16;
        const int8_t *q8 = (const int8_t *)(pa + 4);
        float yd, d, dmin;
        int isum = 0, imin = 0, jj, l, j;
        memcpy(&yd, pa, 4);
        d = yd * rd_f16(pw);
        dmin = yd * rd_f16(pw + 2);
        unpack_scales_k4(pw + 4, sc, mn);
        for (jj = 0; jj < 4; jj++, q4 += 32, q8 += 64) {
            int lo = 0, hi = 0;
            for (l = 0; l < 32; l++) { lo += (q4[l] & 0xF) * q8[l]; hi += (q4[l] >> 4) * q8[32 + l]; }
            isum += sc[2 * jj] * lo + sc[2 * jj + 1] * hi;
        }
        for (j = 0; j < 8; j++) imin += mn[j] * (rd_i16(pa + 260 + 4 * j) + rd_i16(pa + 262 + 4 * j));
        sumf += d * (float)isum - dmin * (float)imin;
    }
    return sumf;
}

static float dot_q5_K_generic(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    float sumf = 0.0f;
    size_t b;
    for (b = 0; b < nb; b++, pw += 176, pa += HFC_Q8_K_BLOCK) {
        unsigned char sc[8], mn[8];
        const unsigned char *qh = pw + 16, *ql = pw + 48;
        const int8_t *q8 = (const int8_t *)(pa + 4);
        float yd, d, dmin;
        int isum = 0, imin = 0, jj, l, j;
        unsigned u1 = 1, u2 = 2;
        memcpy(&yd, pa, 4);
        d = yd * rd_f16(pw);
        dmin = yd * rd_f16(pw + 2);
        unpack_scales_k4(pw + 4, sc, mn);
        for (jj = 0; jj < 4; jj++, ql += 32, q8 += 64, u1 <<= 2, u2 <<= 2) {
            int lo = 0, hi = 0;
            for (l = 0; l < 32; l++) {
                lo += ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) * q8[l];
                hi += ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) * q8[32 + l];
            }
            isum += sc[2 * jj] * lo + sc[2 * jj + 1] * hi;
        }
        for (j = 0; j < 8; j++) imin += mn[j] * (rd_i16(pa + 260 + 4 * j) + rd_i16(pa + 262 + 4 * j));
        sumf += d * (float)isum - dmin * (float)imin;
    }
    return sumf;
}

static float dot_q6_K_generic(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    float sumf = 0.0f;
    size_t b;
    for (b = 0; b < nb; b++, pw += 210, pa += HFC_Q8_K_BLOCK) {
        const unsigned char *ql = pw, *qh = pw + 128;
        const int8_t *sc = (const int8_t *)(pw + 192), *q8 = (const int8_t *)(pa + 4);
        float yd, d;
        int isum = 0, n, l;
        memcpy(&yd, pa, 4);
        d = yd * rd_f16(pw + 208);
        for (n = 0; n < 2; n++, ql += 64, qh += 32, sc += 8, q8 += 128) {
            for (l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                isum += sc[is] * q1 * q8[l] + sc[is + 2] * q2 * q8[l + 32] + sc[is + 4] * q3 * q8[l + 64] + sc[is + 6] * q4 * q8[l + 96];
            }
        }
        sumf += d * (float)isum;
    }
    return sumf;
}

static const hfc_kernels k_generic = {
    "generic", read_sum_generic, fma_burn_generic, 16.0,
    quantize_q8_0_generic, dot_q8_0_generic, dot_f32_generic, dot_f32_f16_generic,
    axpy_f32_f16_generic, f32_to_f16_generic,
    quantize_q8_K_generic, dot_q4_0_generic, dot_q5_0_generic, dot_q4_K_generic, dot_q5_K_generic, dot_q6_K_generic
};

const hfc_kernels *hfc_kernels_generic(void) { return &k_generic; }
