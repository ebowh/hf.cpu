/* mathx.c - deterministic math helpers. See mathx.h. */
#include "mathx.h"

#include <math.h>
#include <string.h>

float hfc_expf(float x)
{
    /* Cephes-style: exp(x) = 2^n * exp(r), r = x - n*ln2 in [-ln2/2, ln2/2]. */
    float fx, z, y;
    int n;
    uint32_t bits;
    float p2;
    if (x != x) return x;
    if (x > 88.3762626647949f) x = 88.3762626647949f;
    if (x < -87.3365478515625f) return 0.0f;
    fx = floorf(x * 1.44269504088896341f + 0.5f);
    n = (int)fx;
    x = x - fx * 0.693359375f;
    x = x - fx * -2.12194440e-4f;
    z = x * x;
    y = 1.9875691500e-4f;
    y = y * x + 1.3981999507e-3f;
    y = y * x + 8.3334519073e-3f;
    y = y * x + 4.1665795894e-2f;
    y = y * x + 1.6666665459e-1f;
    y = y * x + 5.0000001201e-1f;
    y = y * z + x + 1.0f;
    bits = (uint32_t)(n + 127) << 23;
    memcpy(&p2, &bits, sizeof p2);
    return y * p2;
}

float hfc_silu(float x) { return x / (1.0f + hfc_expf(-x)); }

uint16_t hfc_f32_to_f16(float f)
{
    uint32_t x, sign, man, half, rem;
    int e;
    uint32_t exp;
    memcpy(&x, &f, sizeof x);
    sign = (x >> 16) & 0x8000u;
    exp = (x >> 23) & 0xffu;
    man = x & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (man ? (0x200u | (man >> 13)) : 0u));
    e = (int)exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        uint32_t shift, halfway;
        if (e < -10) return (uint16_t)sign;
        man |= 0x800000u;
        shift = (uint32_t)(14 - e);
        half = man >> shift;
        rem = man & ((1u << shift) - 1u);
        halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) half++;
        return (uint16_t)(sign | half);
    }
    half = ((uint32_t)e << 10) | (man >> 13);
    rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
    return (uint16_t)(sign | half);
}

void hfc_rmsnorm(const float *x, const float *w, float *y, size_t n, float eps)
{
    double sum = 0.0;
    float mean, scale;
    size_t i;
    for (i = 0; i < n; i++) sum += (double)(x[i] * x[i]);
    mean = (float)(sum / (double)n);
    scale = 1.0f / sqrtf(mean + eps);
    if (w) for (i = 0; i < n; i++) y[i] = (x[i] * scale) * w[i];
    else   for (i = 0; i < n; i++) y[i] = x[i] * scale;
}

void hfc_softmax(float *x, size_t n)
{
    float mx = x[0], inv;
    double sum = 0.0;
    size_t i;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) { x[i] = hfc_expf(x[i] - mx); sum += (double)x[i]; }
    inv = (float)(1.0 / sum);
    for (i = 0; i < n; i++) x[i] *= inv;
}
