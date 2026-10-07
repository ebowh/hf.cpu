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

static const hfc_kernels k_generic = {
    "generic", read_sum_generic, fma_burn_generic, 16.0,
    quantize_q8_0_generic, dot_q8_0_generic, dot_f32_generic, dot_f32_f16_generic,
    axpy_f32_f16_generic, f32_to_f16_generic
};

const hfc_kernels *hfc_kernels_generic(void) { return &k_generic; }
