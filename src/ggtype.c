/* ggtype.c - type table and scalar reference dequantizers.
 * Block layouts follow ggml-common.h; all reads are byte-wise so blocks may
 * sit at any alignment inside a memory-mapped file.
 */
#include "ggtype.h"

#include <stdio.h>
#include <string.h>

enum {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8, T_Q8_1 = 9,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_Q8_K = 15,
    T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19, T_IQ4_NL = 20, T_IQ3_S = 21,
    T_IQ2_S = 22, T_IQ4_XS = 23, T_I8 = 24, T_I16 = 25, T_I32 = 26, T_I64 = 27, T_F64 = 28,
    T_IQ1_M = 29, T_BF16 = 30, T_TQ1_0 = 34, T_TQ2_0 = 35, T_MXFP4 = 39, T_NVFP4 = 40,
    T_Q1_0 = 41, T_Q2_0 = 42, T_PQ2_0 = 142, T_PTQ1_0 = 143
};

static const hfc_type_info types[] = {
    { T_F32,     "f32",     1,   4,   1 },
    { T_F16,     "f16",     1,   2,   1 },
    { T_Q4_0,    "q4_0",    32,  18,  1 },
    { T_Q4_1,    "q4_1",    32,  20,  1 },
    { T_Q5_0,    "q5_0",    32,  22,  1 },
    { T_Q5_1,    "q5_1",    32,  24,  1 },
    { T_Q8_0,    "q8_0",    32,  34,  1 },
    { T_Q8_1,    "q8_1",    32,  36,  0 },
    { T_Q2_K,    "q2_K",    256, 84,  1 },
    { T_Q3_K,    "q3_K",    256, 110, 1 },
    { T_Q4_K,    "q4_K",    256, 144, 1 },
    { T_Q5_K,    "q5_K",    256, 176, 1 },
    { T_Q6_K,    "q6_K",    256, 210, 1 },
    { T_Q8_K,    "q8_K",    256, 292, 0 },
    { T_IQ2_XXS, "iq2_xxs", 256, 66,  0 },
    { T_IQ2_XS,  "iq2_xs",  256, 74,  0 },
    { T_IQ3_XXS, "iq3_xxs", 256, 98,  0 },
    { T_IQ1_S,   "iq1_s",   256, 50,  0 },
    { T_IQ4_NL,  "iq4_nl",  32,  18,  1 },
    { T_IQ3_S,   "iq3_s",   256, 110, 0 },
    { T_IQ2_S,   "iq2_s",   256, 82,  0 },
    { T_IQ4_XS,  "iq4_xs",  256, 136, 1 },
    { T_I8,      "i8",      1,   1,   0 },
    { T_I16,     "i16",     1,   2,   0 },
    { T_I32,     "i32",     1,   4,   0 },
    { T_I64,     "i64",     1,   8,   0 },
    { T_F64,     "f64",     1,   8,   0 },
    { T_IQ1_M,   "iq1_m",   256, 56,  0 },
    { T_BF16,    "bf16",    1,   2,   1 },
    { T_TQ1_0,   "tq1_0",   256, 54,  0 },
    { T_TQ2_0,   "tq2_0",   256, 66,  0 },
    { T_MXFP4,   "mxfp4",   32,  17,  0 },
    { T_NVFP4,   "nvfp4",   64,  36,  0 },
    { T_Q1_0,    "q1_0",    128, 18,  1 },
    { T_Q2_0,    "q2_0",    64,  18,  1 },
    /* PrismML fork-private types (from the fork's ggml-common.h) */
    { T_PQ2_0,   "pq2_0",   128, 34,  1 },
    { T_PTQ1_0,  "ptq1_0",  128, 28,  0 },
};
#define NTYPES (sizeof types / sizeof types[0])

const hfc_type_info *hfc_type_lookup(uint32_t id)
{
    size_t i;
    for (i = 0; i < NTYPES; i++) if (types[i].id == id) return &types[i];
    return NULL;
}

const char *hfc_type_name(uint32_t id)
{
    static char buf[24];
    const hfc_type_info *t = hfc_type_lookup(id);
    if (t) return t->name;
    snprintf(buf, sizeof buf, "type_%lu", (unsigned long)id);
    return buf;
}

hfc_status hfc_type_row_bytes(uint32_t id, uint64_t n, uint64_t *out)
{
    const hfc_type_info *t = hfc_type_lookup(id);
    if (!t) return HFC_ENOTSUP;
    if (n % t->blck != 0) return HFC_EFORMAT;
    if (!hfc_mul_u64(n / t->blck, t->bytes, out)) return HFC_ERANGE;
    return HFC_OK;
}

/* ---- scalar helpers ---------------------------------------------------------- */

float hfc_f16_to_f32(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu, r;
    float f;
    if (e == 0) {
        if (m == 0) r = s;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3ffu;
            r = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        r = s | 0x7f800000u | (m << 13);
    } else {
        r = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    memcpy(&f, &r, sizeof f);
    return f;
}

float hfc_bf16_to_f32(uint16_t h)
{
    uint32_t r = (uint32_t)h << 16;
    float f;
    memcpy(&f, &r, sizeof f);
    return f;
}

static uint16_t rd16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static float    rdh(const unsigned char *p)  { return hfc_f16_to_f32(rd16(p)); }

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static void get_scale_min_k4(int j, const unsigned char *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

/* ---- dequantizers ------------------------------------------------------------- */

static void dq_q4_0(const unsigned char *x, float *y, size_t nb)
{
    size_t i; int j;
    for (i = 0; i < nb; i++, x += 18, y += 32) {
        float d = rdh(x);
        for (j = 0; j < 16; j++) {
            y[j]      = (float)((x[2 + j] & 0x0F) - 8) * d;
            y[j + 16] = (float)((x[2 + j] >> 4) - 8) * d;
        }
    }
}

static void dq_q4_1(const unsigned char *x, float *y, size_t nb)
{
    size_t i; int j;
    for (i = 0; i < nb; i++, x += 20, y += 32) {
        float d = rdh(x), m = rdh(x + 2);
        for (j = 0; j < 16; j++) {
            y[j]      = (float)(x[4 + j] & 0x0F) * d + m;
            y[j + 16] = (float)(x[4 + j] >> 4) * d + m;
        }
    }
}

static void dq_q5_0(const unsigned char *x, float *y, size_t nb)
{
    size_t i; int j;
    for (i = 0; i < nb; i++, x += 22, y += 32) {
        float d = rdh(x);
        uint32_t qh = (uint32_t)x[2] | ((uint32_t)x[3] << 8) | ((uint32_t)x[4] << 16) | ((uint32_t)x[5] << 24);
        for (j = 0; j < 16; j++) {
            uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh1 = (uint8_t)(((qh >> (j + 12))) & 0x10);
            y[j]      = (float)(((x[6 + j] & 0x0F) | xh0) - 16) * d;
            y[j + 16] = (float)(((x[6 + j] >> 4) | xh1) - 16) * d;
        }
    }
}

static void dq_q5_1(const unsigned char *x, float *y, size_t nb)
{
    size_t i; int j;
    for (i = 0; i < nb; i++, x += 24, y += 32) {
        float d = rdh(x), m = rdh(x + 2);
        uint32_t qh = (uint32_t)x[4] | ((uint32_t)x[5] << 8) | ((uint32_t)x[6] << 16) | ((uint32_t)x[7] << 24);
        for (j = 0; j < 16; j++) {
            uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh1 = (uint8_t)(((qh >> (j + 12))) & 0x10);
            y[j]      = (float)((x[8 + j] & 0x0F) | xh0) * d + m;
            y[j + 16] = (float)((x[8 + j] >> 4) | xh1) * d + m;
        }
    }
}

static void dq_q8_0(const unsigned char *x, float *y, size_t nb)
{
    size_t i; int j;
    for (i = 0; i < nb; i++, x += 34, y += 32) {
        float d = rdh(x);
        for (j = 0; j < 32; j++) y[j] = (float)(int8_t)x[2 + j] * d;
    }
}

static void dq_q2_k(const unsigned char *x, float *y, size_t nb)
{
    size_t i;
    for (i = 0; i < nb; i++, x += 84) {
        const unsigned char *scales = x, *q = x + 16;
        float d = rdh(x + 80), min = rdh(x + 82);
        int is = 0, n, j, l;
        for (n = 0; n < 256; n += 128) {
            int shift = 0;
            for (j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (float)(sc & 0xF), ml = min * (float)(sc >> 4);
                for (l = 0; l < 16; l++) *y++ = dl * (float)((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (float)(sc & 0xF); ml = min * (float)(sc >> 4);
                for (l = 0; l < 16; l++) *y++ = dl * (float)((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

static void dq_q3_k(const unsigned char *x, float *y, size_t nb)
{
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    size_t i;
    for (i = 0; i < nb; i++, x += 110) {
        const unsigned char *hm = x, *q = x + 32;
        float d_all = rdh(x + 108);
        uint32_t aux[4], tmp;
        int8_t scales[16];
        int is = 0, n, j, l, k;
        uint8_t m = 1;
        for (k = 0; k < 3; k++)
            aux[k] = (uint32_t)x[96 + 4*k] | ((uint32_t)x[97 + 4*k] << 8) |
                     ((uint32_t)x[98 + 4*k] << 16) | ((uint32_t)x[99 + 4*k] << 24);
        tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (k = 0; k < 4; k++) {
            scales[4*k + 0] = (int8_t)(aux[k] & 0xff);
            scales[4*k + 1] = (int8_t)((aux[k] >> 8) & 0xff);
            scales[4*k + 2] = (int8_t)((aux[k] >> 16) & 0xff);
            scales[4*k + 3] = (int8_t)((aux[k] >> 24) & 0xff);
        }
        for (n = 0; n < 256; n += 128) {
            int shift = 0;
            for (j = 0; j < 4; j++) {
                float dl = d_all * (float)(scales[is++] - 32);
                for (l = 0; l < 16; l++)
                    *y++ = dl * (float)((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (float)(scales[is++] - 32);
                for (l = 0; l < 16; l++)
                    *y++ = dl * (float)((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

static void dq_q4_k(const unsigned char *x, float *y, size_t nb)
{
    size_t i;
    for (i = 0; i < nb; i++, x += 144) {
        const unsigned char *scales = x + 4, *q = x + 16;
        float d = rdh(x), min = rdh(x + 2);
        int is = 0, j, l;
        uint8_t sc, m;
        for (j = 0; j < 256; j += 64) {
            float d1, m1, d2, m2;
            get_scale_min_k4(is + 0, scales, &sc, &m); d1 = d * sc; m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m); d2 = d * sc; m2 = min * m;
            for (l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
            for (l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static void dq_q5_k(const unsigned char *x, float *y, size_t nb)
{
    size_t i;
    for (i = 0; i < nb; i++, x += 176) {
        const unsigned char *scales = x + 4, *qh = x + 16, *ql = x + 48;
        float d = rdh(x), min = rdh(x + 2);
        int is = 0, j, l;
        uint8_t sc, m, u1 = 1, u2 = 2;
        for (j = 0; j < 256; j += 64) {
            float d1, m1, d2, m2;
            get_scale_min_k4(is + 0, scales, &sc, &m); d1 = d * sc; m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m); d2 = d * sc; m2 = min * m;
            for (l = 0; l < 32; l++) *y++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (l = 0; l < 32; l++) *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
    }
}

static void dq_q6_k(const unsigned char *x, float *y, size_t nb)
{
    size_t i;
    for (i = 0; i < nb; i++, x += 210) {
        const unsigned char *ql = x, *qh = x + 128;
        const int8_t *sc = (const int8_t *)(x + 192);
        float d = rdh(x + 208);
        int n, l;
        for (n = 0; n < 256; n += 128) {
            for (l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

static void dq_iq4_nl(const unsigned char *x, float *y, size_t nb)
{
    size_t i; int j;
    for (i = 0; i < nb; i++, x += 18, y += 32) {
        float d = rdh(x);
        for (j = 0; j < 16; j++) {
            y[j]      = d * (float)kvalues_iq4nl[x[2 + j] & 0xf];
            y[j + 16] = d * (float)kvalues_iq4nl[x[2 + j] >> 4];
        }
    }
}

static void dq_iq4_xs(const unsigned char *x, float *y, size_t nb)
{
    size_t i; int ib, j;
    for (i = 0; i < nb; i++, x += 136) {
        float d = rdh(x);
        unsigned scales_h = rd16(x + 2);
        const unsigned char *scales_l = x + 4, *qs = x + 8;
        for (ib = 0; ib < 8; ib++) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (int)(((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            for (j = 0; j < 16; j++) {
                y[j]      = dl * (float)kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32; qs += 16;
        }
    }
}

/* 2-bit codec shared by Q2_0 (group 64) and the fork's PQ2_0 (group 128):
 * element j lives in byte j/4 at bit offset 2*(j%4); value = (q - 1) * d. */
static void dq_2bit(const unsigned char *x, float *y, size_t nb, int group)
{
    size_t i; int j;
    size_t bsz = 2 + (size_t)group / 4;
    for (i = 0; i < nb; i++, x += bsz, y += group) {
        float d = rdh(x);
        for (j = 0; j < group; j++) {
            int q = (x[2 + j / 4] >> ((j % 4) * 2)) & 3;
            y[j] = (float)(q - 1) * d;
        }
    }
}

static void dq_q1_0(const unsigned char *x, float *y, size_t nb)
{
    size_t i; int j;
    for (i = 0; i < nb; i++, x += 18, y += 128) {
        float d = rdh(x);
        for (j = 0; j < 128; j++) y[j] = ((x[2 + j / 8] >> (j % 8)) & 1) ? d : -d;
    }
}

hfc_status hfc_dequant_row(uint32_t id, const void *src, float *dst, size_t n)
{
    const hfc_type_info *t = hfc_type_lookup(id);
    const unsigned char *x = (const unsigned char *)src;
    size_t i, nb;
    if (!t) return HFC_ENOTSUP;
    if (n % t->blck != 0) return HFC_EFORMAT;
    nb = n / t->blck;
    switch (id) {
    case T_F32:
        for (i = 0; i < n; i++) {
            uint32_t r = (uint32_t)x[4*i] | ((uint32_t)x[4*i+1] << 8) | ((uint32_t)x[4*i+2] << 16) | ((uint32_t)x[4*i+3] << 24);
            memcpy(&dst[i], &r, 4);
        }
        return HFC_OK;
    case T_F16:  for (i = 0; i < n; i++) dst[i] = hfc_f16_to_f32(rd16(x + 2 * i)); return HFC_OK;
    case T_BF16: for (i = 0; i < n; i++) dst[i] = hfc_bf16_to_f32(rd16(x + 2 * i)); return HFC_OK;
    case T_Q4_0:   dq_q4_0(x, dst, nb); return HFC_OK;
    case T_Q4_1:   dq_q4_1(x, dst, nb); return HFC_OK;
    case T_Q5_0:   dq_q5_0(x, dst, nb); return HFC_OK;
    case T_Q5_1:   dq_q5_1(x, dst, nb); return HFC_OK;
    case T_Q8_0:   dq_q8_0(x, dst, nb); return HFC_OK;
    case T_Q2_K:   dq_q2_k(x, dst, nb); return HFC_OK;
    case T_Q3_K:   dq_q3_k(x, dst, nb); return HFC_OK;
    case T_Q4_K:   dq_q4_k(x, dst, nb); return HFC_OK;
    case T_Q5_K:   dq_q5_k(x, dst, nb); return HFC_OK;
    case T_Q6_K:   dq_q6_k(x, dst, nb); return HFC_OK;
    case T_IQ4_NL: dq_iq4_nl(x, dst, nb); return HFC_OK;
    case T_IQ4_XS: dq_iq4_xs(x, dst, nb); return HFC_OK;
    case T_Q1_0:   dq_q1_0(x, dst, nb); return HFC_OK;
    case T_Q2_0:   dq_2bit(x, dst, nb, 64); return HFC_OK;
    case T_PQ2_0:  dq_2bit(x, dst, nb, 128); return HFC_OK;
    default:       return HFC_ENOTSUP;
    }
}
