/* kern_avx2.c - AVX2/FMA kernels. Compiled with -mavx2 -mfma (x86_64 only);
 * only ever called after run-time feature and OS-support detection. */
#include "kern.h"

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#include <string.h>

static uint64_t read_sum_avx2(const void *p, size_t bytes)
{
    const __m256i *v = (const __m256i *)p;
    size_t n = bytes / 128, i;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    uint64_t t[4];
    for (i = 0; i < n; i++) {
        a0 = _mm256_add_epi64(a0, _mm256_load_si256(v + 4 * i));
        a1 = _mm256_add_epi64(a1, _mm256_load_si256(v + 4 * i + 1));
        a2 = _mm256_add_epi64(a2, _mm256_load_si256(v + 4 * i + 2));
        a3 = _mm256_add_epi64(a3, _mm256_load_si256(v + 4 * i + 3));
    }
    a0 = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
    _mm256_storeu_si256((__m256i *)t, a0);
    return t[0] + t[1] + t[2] + t[3];
}

static float fma_burn_avx2(long iters)
{
    __m256 a0 = _mm256_set1_ps(1), a1 = _mm256_set1_ps(2), a2 = _mm256_set1_ps(3),
           a3 = _mm256_set1_ps(4), a4 = _mm256_set1_ps(5), a5 = _mm256_set1_ps(6),
           a6 = _mm256_set1_ps(7), a7 = _mm256_set1_ps(8), a8 = _mm256_set1_ps(9),
           a9 = _mm256_set1_ps(10);
    const __m256 m = _mm256_set1_ps(0.999999f), c = _mm256_set1_ps(1e-7f);
    float t[8];
    long i;
    for (i = 0; i < iters; i++) {
        a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c);
        a2 = _mm256_fmadd_ps(a2, m, c); a3 = _mm256_fmadd_ps(a3, m, c);
        a4 = _mm256_fmadd_ps(a4, m, c); a5 = _mm256_fmadd_ps(a5, m, c);
        a6 = _mm256_fmadd_ps(a6, m, c); a7 = _mm256_fmadd_ps(a7, m, c);
        a8 = _mm256_fmadd_ps(a8, m, c); a9 = _mm256_fmadd_ps(a9, m, c);
    }
    a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    a4 = _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7));
    a8 = _mm256_add_ps(a8, a9);
    a0 = _mm256_add_ps(_mm256_add_ps(a0, a4), a8);
    _mm256_storeu_ps(t, a0);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

static float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

static float dot_q8_0_avx2(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    size_t b;
    for (b = 0; b < nb; b++, pw += HFC_Q8_0_BLOCK, pa += HFC_Q8_0_BLOCK) {
        uint16_t hw = (uint16_t)(pw[0] | (pw[1] << 8)), ha = (uint16_t)(pa[0] | (pa[1] << 8));
        float d = _cvtsh_ss(hw) * _cvtsh_ss(ha);
        __m256i qx = _mm256_loadu_si256((const __m256i *)(pw + 2));
        __m256i qy = _mm256_loadu_si256((const __m256i *)(pa + 2));
        __m256i ax = _mm256_sign_epi8(qx, qx);
        __m256i sy = _mm256_sign_epi8(qy, qx);
        __m256i p16 = _mm256_maddubs_epi16(ax, sy);
        __m256i p32 = _mm256_madd_epi16(p16, ones);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc);
    }
    return hsum256(acc);
}

static float dot_f32_avx2(const float *a, const float *b, size_t n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    size_t i;
    float r;
    for (i = 0; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    r = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}

static float dot_f32_f16_avx2(const float *a, const uint16_t *b, size_t n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    size_t i;
    float r;
    for (i = 0; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i))), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i + 8))), s1);
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i))), s0);
    r = hsum256(_mm256_add_ps(s0, s1));
    for (; i < n; i++) r += a[i] * _cvtsh_ss(b[i]);
    return r;
}

static void axpy_f32_f16_avx2(float *y, float a, const uint16_t *x, size_t n)
{
    const __m256 va = _mm256_set1_ps(a);
    size_t i;
    for (i = 0; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * _cvtsh_ss(x[i]);
}

static void f32_to_f16_avx2(const float *x, uint16_t *y, size_t n)
{
    size_t i;
    for (i = 0; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(y + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    for (; i < n; i++) y[i] = (uint16_t)_cvtss_sh(x[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}


/* ---- K-quants and 4/5-bit legacy dots (AVX2) ---- */

static inline float rd_h(const unsigned char *p) { return _cvtsh_ss((uint16_t)(p[0] | (p[1] << 8))); }

static inline __m256 sum_i8_pairs_ps(__m256i x, __m256i y)     /* signed x signed -> 8 float partial sums */
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i ax = _mm256_sign_epi8(x, x), sy = _mm256_sign_epi8(y, x);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), ones));
}

static float dot_q4_0_avx2(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i off = _mm256_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    size_t b;
    for (b = 0; b < nb; b++, pw += 18, pa += HFC_Q8_0_BLOCK) {
        __m128i raw = _mm_loadu_si128((const __m128i *)(pw + 2));
        __m128i lo = _mm_and_si128(raw, m4), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        __m256i qx = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), off);
        __m256i qy = _mm256_loadu_si256((const __m256i *)(pa + 2));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(rd_h(pw) * rd_h(pa)), sum_i8_pairs_ps(qx, qy), acc);
    }
    return hsum256(acc);
}

static float dot_q5_0_avx2(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const __m256i bitmask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    __m256 acc = _mm256_setzero_ps();
    size_t b;
    for (b = 0; b < nb; b++, pw += 22, pa += HFC_Q8_0_BLOCK) {
        uint32_t qh = (uint32_t)pw[2] | ((uint32_t)pw[3] << 8) | ((uint32_t)pw[4] << 16) | ((uint32_t)pw[5] << 24);
        __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), shuf);
        __m256i bitset = _mm256_cmpeq_epi8(_mm256_or_si256(bytes, bitmask), _mm256_set1_epi64x(-1));   /* 0xFF where the bit is set */
        __m256i hi5 = _mm256_andnot_si256(bitset, _mm256_set1_epi8((char)0xF0));                          /* 0xF0 where it is clear */
        __m128i raw = _mm_loadu_si128((const __m128i *)(pw + 6));
        __m128i lo = _mm_and_si128(raw, m4), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        __m256i qx = _mm256_or_si256(_mm256_set_m128i(hi, lo), hi5);        /* nibble | 0xF0 reads as nibble - 16 */
        __m256i qy = _mm256_loadu_si256((const __m256i *)(pa + 2));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(rd_h(pw) * rd_h(pa)), sum_i8_pairs_ps(qx, qy), acc);
    }
    return hsum256(acc);
}

static inline int16_t rd_s16(const unsigned char *p) { return (int16_t)(p[0] | (p[1] << 8)); }

static inline int hsum_epi32(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

/* sum over j < 8 of mn[j] * (bsums[2j] + bsums[2j+1]) -- the Q4_K / Q5_K minimum term */
static inline int min_term_k(const unsigned char *mn, const unsigned char *bsums)
{
    __m256i m32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)mn));
    __m256i pair = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)bsums), _mm256_set1_epi16(1));
    return hsum_epi32(_mm256_mullo_epi32(m32, pair));
}

/* sum over j < 16 of sc[j] * bsums[j] -- the Q6_K offset term */
static inline int off_term_q6(const int8_t *sc, const unsigned char *bsums)
{
    __m256i s16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
    return hsum_epi32(_mm256_madd_epi16(s16, _mm256_loadu_si256((const __m256i *)bsums)));
}

static inline void unpack_k4(const unsigned char *q, unsigned char *sc, unsigned char *mn)
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

static float dot_q4_K_avx2(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    float minacc = 0.0f;
    size_t b;
    for (b = 0; b < nb; b++, pw += 144, pa += HFC_Q8_K_BLOCK) {
        unsigned char sc[8], mn[8];
        const unsigned char *q4 = pw + 16, *q8 = pa + 4;
        __m256i sumi = _mm256_setzero_si256();
        float yd, d, dmin;
        int imin = 0, jj;
        memcpy(&yd, pa, 4);
        d = yd * rd_h(pw);
        dmin = yd * rd_h(pw + 2);
        unpack_k4(pw + 4, sc, mn);
        for (jj = 0; jj < 4; jj++, q4 += 32, q8 += 64) {
            __m256i bits = _mm256_loadu_si256((const __m256i *)q4);
            __m256i ql = _mm256_and_si256(bits, m4), qhh = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            __m256i pl = _mm256_maddubs_epi16(ql, _mm256_loadu_si256((const __m256i *)q8));
            __m256i ph = _mm256_maddubs_epi16(qhh, _mm256_loadu_si256((const __m256i *)(q8 + 32)));
            pl = _mm256_madd_epi16(_mm256_set1_epi16(sc[2 * jj]), pl);
            ph = _mm256_madd_epi16(_mm256_set1_epi16(sc[2 * jj + 1]), ph);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
        }
        imin = min_term_k(mn, pa + 260);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        minacc += dmin * (float)imin;
    }
    return hsum256(acc) - minacc;
}

static float dot_q5_K_avx2(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    const __m256i m4 = _mm256_set1_epi8(0xF), mone = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float minacc = 0.0f;
    size_t b;
    for (b = 0; b < nb; b++, pw += 176, pa += HFC_Q8_K_BLOCK) {
        unsigned char sc[8], mn[8];
        const unsigned char *ql = pw + 48, *q8 = pa + 4;
        __m256i hbits = _mm256_loadu_si256((const __m256i *)(pw + 16));
        __m256i sumi = _mm256_setzero_si256();
        float yd, d, dmin;
        int imin = 0, jj;
        memcpy(&yd, pa, 4);
        d = yd * rd_h(pw);
        dmin = yd * rd_h(pw + 2);
        unpack_k4(pw + 4, sc, mn);
        for (jj = 0; jj < 4; jj++, ql += 32, q8 += 64) {
            __m256i bits = _mm256_loadu_si256((const __m256i *)ql);
            __m256i q0 = _mm256_or_si256(_mm256_and_si256(bits, m4), _mm256_slli_epi16(_mm256_and_si256(hbits, mone), 4));
            __m256i p0, p1, q1;
            hbits = _mm256_srli_epi16(hbits, 1);
            q1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), _mm256_slli_epi16(_mm256_and_si256(hbits, mone), 4));
            hbits = _mm256_srli_epi16(hbits, 1);
            p0 = _mm256_maddubs_epi16(q0, _mm256_loadu_si256((const __m256i *)q8));
            p1 = _mm256_maddubs_epi16(q1, _mm256_loadu_si256((const __m256i *)(q8 + 32)));
            p0 = _mm256_madd_epi16(_mm256_set1_epi16(sc[2 * jj]), p0);
            p1 = _mm256_madd_epi16(_mm256_set1_epi16(sc[2 * jj + 1]), p1);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
        }
        imin = min_term_k(mn, pa + 260);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        minacc += dmin * (float)imin;
    }
    return hsum256(acc) - minacc;
}

static float dot_q6_K_avx2(const void *w, const void *a, size_t nb)
{
    const unsigned char *pw = (const unsigned char *)w, *pa = (const unsigned char *)a;
    const __m256i m4 = _mm256_set1_epi8(0xF), m30 = _mm256_set1_epi8(0x30);
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.0f;
    size_t b;
    for (b = 0; b < nb; b++, pw += 210, pa += HFC_Q8_K_BLOCK) {
        const unsigned char *ql = pw, *qh = pw + 128;
        const int8_t *sc = (const int8_t *)(pw + 192);
        const unsigned char *q8 = pa + 4;
        __m256i sumi = _mm256_setzero_si256();
        float yd, d;
        int off = 0, n;
        memcpy(&yd, pa, 4);
        d = yd * rd_h(pw + 208);
        off = off_term_q6(sc, pa + 260);     /* the -32 offset: 32 * sum(scale * bsum) */
        for (n = 0; n < 2; n++, ql += 64, qh += 32, sc += 8, q8 += 128) {
            __m256i l0 = _mm256_loadu_si256((const __m256i *)ql), l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i hv = _mm256_loadu_si256((const __m256i *)qh);
            __m256i g0 = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_and_si256(_mm256_slli_epi16(hv, 4), m30));
            __m256i g1 = _mm256_or_si256(_mm256_and_si256(l1, m4), _mm256_and_si256(_mm256_slli_epi16(hv, 2), m30));
            __m256i g2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), _mm256_and_si256(hv, m30));
            __m256i g3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4), _mm256_and_si256(_mm256_srli_epi16(hv, 2), m30));
            __m256i s0 = _mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0]));
            __m256i s1 = _mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2]));
            __m256i s2 = _mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4]));
            __m256i s3 = _mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6]));
            __m256i p0 = _mm256_maddubs_epi16(g0, _mm256_loadu_si256((const __m256i *)q8));
            __m256i p1 = _mm256_maddubs_epi16(g1, _mm256_loadu_si256((const __m256i *)(q8 + 32)));
            __m256i p2 = _mm256_maddubs_epi16(g2, _mm256_loadu_si256((const __m256i *)(q8 + 64)));
            __m256i p3 = _mm256_maddubs_epi16(g3, _mm256_loadu_si256((const __m256i *)(q8 + 96)));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_madd_epi16(s0, p0), _mm256_madd_epi16(s1, p1)));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_madd_epi16(s2, p2), _mm256_madd_epi16(s3, p3)));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        corr += d * (float)(32 * off);
    }
    return hsum256(acc) - corr;
}

/* ---- four-token variants: decode the weight block once, dot it with four activation rows.
 * Each token's arithmetic is exactly that of the single-token kernel above, so the results are
 * bit-identical to it; only the weight unpacking (and loop overhead) is shared. ---- */

#define NT 4

static void dot4_q8_0_avx2(const void *w, const void *a, size_t as, size_t nb, float *out)
{
    const unsigned char *pw = (const unsigned char *)w, *pa[NT];
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NT];
    size_t b;
    int t;
    for (t = 0; t < NT; t++) { pa[t] = (const unsigned char *)a + (size_t)t * as; acc[t] = _mm256_setzero_ps(); }
    for (b = 0; b < nb; b++, pw += HFC_Q8_0_BLOCK) {
        float dw = rd_h(pw);
        __m256i qx = _mm256_loadu_si256((const __m256i *)(pw + 2));
        __m256i ax = _mm256_sign_epi8(qx, qx);
        for (t = 0; t < NT; t++, pa[t - 1] += HFC_Q8_0_BLOCK) {
            __m256i qy = _mm256_loadu_si256((const __m256i *)(pa[t] + 2));
            __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(qy, qx)), ones);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * rd_h(pa[t])), _mm256_cvtepi32_ps(p32), acc[t]);
        }
    }
    for (t = 0; t < NT; t++) out[t] = hsum256(acc[t]);
}

static void dot4_q4_0_avx2(const void *w, const void *a, size_t as, size_t nb, float *out)
{
    const unsigned char *pw = (const unsigned char *)w, *pa[NT];
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i off = _mm256_set1_epi8(8);
    __m256 acc[NT];
    size_t b;
    int t;
    for (t = 0; t < NT; t++) { pa[t] = (const unsigned char *)a + (size_t)t * as; acc[t] = _mm256_setzero_ps(); }
    for (b = 0; b < nb; b++, pw += 18) {
        __m128i raw = _mm_loadu_si128((const __m128i *)(pw + 2));
        __m128i lo = _mm_and_si128(raw, m4), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        __m256i qx = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), off);
        float dw = rd_h(pw);
        for (t = 0; t < NT; t++, pa[t - 1] += HFC_Q8_0_BLOCK) {
            __m256i qy = _mm256_loadu_si256((const __m256i *)(pa[t] + 2));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * rd_h(pa[t])), sum_i8_pairs_ps(qx, qy), acc[t]);
        }
    }
    for (t = 0; t < NT; t++) out[t] = hsum256(acc[t]);
}

static void dot4_q5_0_avx2(const void *w, const void *a, size_t as, size_t nb, float *out)
{
    const unsigned char *pw = (const unsigned char *)w, *pa[NT];
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const __m256i bitmask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    __m256 acc[NT];
    size_t b;
    int t;
    for (t = 0; t < NT; t++) { pa[t] = (const unsigned char *)a + (size_t)t * as; acc[t] = _mm256_setzero_ps(); }
    for (b = 0; b < nb; b++, pw += 22) {
        uint32_t qh = (uint32_t)pw[2] | ((uint32_t)pw[3] << 8) | ((uint32_t)pw[4] << 16) | ((uint32_t)pw[5] << 24);
        __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), shuf);
        __m256i bitset = _mm256_cmpeq_epi8(_mm256_or_si256(bytes, bitmask), _mm256_set1_epi64x(-1));
        __m256i hi5 = _mm256_andnot_si256(bitset, _mm256_set1_epi8((char)0xF0));
        __m128i raw = _mm_loadu_si128((const __m128i *)(pw + 6));
        __m128i lo = _mm_and_si128(raw, m4), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        __m256i qx = _mm256_or_si256(_mm256_set_m128i(hi, lo), hi5);
        float dw = rd_h(pw);
        for (t = 0; t < NT; t++, pa[t - 1] += HFC_Q8_0_BLOCK) {
            __m256i qy = _mm256_loadu_si256((const __m256i *)(pa[t] + 2));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * rd_h(pa[t])), sum_i8_pairs_ps(qx, qy), acc[t]);
        }
    }
    for (t = 0; t < NT; t++) out[t] = hsum256(acc[t]);
}

static void dot4_q4_K_avx2(const void *w, const void *a, size_t as, size_t nb, float *out)
{
    const unsigned char *pw = (const unsigned char *)w, *pa[NT];
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[NT];
    float minacc[NT];
    size_t b;
    int t;
    for (t = 0; t < NT; t++) { pa[t] = (const unsigned char *)a + (size_t)t * as; acc[t] = _mm256_setzero_ps(); minacc[t] = 0.0f; }
    for (b = 0; b < nb; b++, pw += 144) {
        unsigned char sc[8], mn[8];
        const unsigned char *q4 = pw + 16;
        __m256i sumi[NT];
        float dwd = rd_h(pw), dwm = rd_h(pw + 2);
        int jj;
        unpack_k4(pw + 4, sc, mn);
        for (t = 0; t < NT; t++) sumi[t] = _mm256_setzero_si256();
        for (jj = 0; jj < 4; jj++, q4 += 32) {
            __m256i bits = _mm256_loadu_si256((const __m256i *)q4);
            __m256i ql = _mm256_and_si256(bits, m4), qhh = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            __m256i s0 = _mm256_set1_epi16(sc[2 * jj]), s1 = _mm256_set1_epi16(sc[2 * jj + 1]);
            for (t = 0; t < NT; t++) {
                const unsigned char *q8 = pa[t] + 4 + 64 * jj;
                __m256i pl = _mm256_maddubs_epi16(ql, _mm256_loadu_si256((const __m256i *)q8));
                __m256i ph = _mm256_maddubs_epi16(qhh, _mm256_loadu_si256((const __m256i *)(q8 + 32)));
                pl = _mm256_madd_epi16(s0, pl);
                ph = _mm256_madd_epi16(s1, ph);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(pl, ph));
            }
        }
        for (t = 0; t < NT; t++) {
            float yd;
            int imin = 0;
            memcpy(&yd, pa[t], 4);
            imin = min_term_k(mn, pa[t] + 260);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * dwd), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            minacc[t] += (yd * dwm) * (float)imin;
            pa[t] += HFC_Q8_K_BLOCK;
        }
    }
    for (t = 0; t < NT; t++) out[t] = hsum256(acc[t]) - minacc[t];
}

static void dot4_q5_K_avx2(const void *w, const void *a, size_t as, size_t nb, float *out)
{
    const unsigned char *pw = (const unsigned char *)w, *pa[NT];
    const __m256i m4 = _mm256_set1_epi8(0xF), mone = _mm256_set1_epi8(1);
    __m256 acc[NT];
    float minacc[NT];
    size_t b;
    int t;
    for (t = 0; t < NT; t++) { pa[t] = (const unsigned char *)a + (size_t)t * as; acc[t] = _mm256_setzero_ps(); minacc[t] = 0.0f; }
    for (b = 0; b < nb; b++, pw += 176) {
        unsigned char sc[8], mn[8];
        const unsigned char *ql = pw + 48;
        __m256i hbits = _mm256_loadu_si256((const __m256i *)(pw + 16));
        __m256i sumi[NT];
        float dwd = rd_h(pw), dwm = rd_h(pw + 2);
        int jj;
        unpack_k4(pw + 4, sc, mn);
        for (t = 0; t < NT; t++) sumi[t] = _mm256_setzero_si256();
        for (jj = 0; jj < 4; jj++, ql += 32) {
            __m256i bits = _mm256_loadu_si256((const __m256i *)ql);
            __m256i q0 = _mm256_or_si256(_mm256_and_si256(bits, m4), _mm256_slli_epi16(_mm256_and_si256(hbits, mone), 4));
            __m256i q1, s0 = _mm256_set1_epi16(sc[2 * jj]), s1 = _mm256_set1_epi16(sc[2 * jj + 1]);
            hbits = _mm256_srli_epi16(hbits, 1);
            q1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), _mm256_slli_epi16(_mm256_and_si256(hbits, mone), 4));
            hbits = _mm256_srli_epi16(hbits, 1);
            for (t = 0; t < NT; t++) {
                const unsigned char *q8 = pa[t] + 4 + 64 * jj;
                __m256i p0 = _mm256_maddubs_epi16(q0, _mm256_loadu_si256((const __m256i *)q8));
                __m256i p1 = _mm256_maddubs_epi16(q1, _mm256_loadu_si256((const __m256i *)(q8 + 32)));
                p0 = _mm256_madd_epi16(s0, p0);
                p1 = _mm256_madd_epi16(s1, p1);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p0, p1));
            }
        }
        for (t = 0; t < NT; t++) {
            float yd;
            int imin = 0;
            memcpy(&yd, pa[t], 4);
            imin = min_term_k(mn, pa[t] + 260);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * dwd), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            minacc[t] += (yd * dwm) * (float)imin;
            pa[t] += HFC_Q8_K_BLOCK;
        }
    }
    for (t = 0; t < NT; t++) out[t] = hsum256(acc[t]) - minacc[t];
}

static void dot4_q6_K_avx2(const void *w, const void *a, size_t as, size_t nb, float *out)
{
    const unsigned char *pw = (const unsigned char *)w, *pa[NT];
    const __m256i m4 = _mm256_set1_epi8(0xF), m30 = _mm256_set1_epi8(0x30);
    __m256 acc[NT];
    float corr[NT];
    size_t b;
    int t;
    for (t = 0; t < NT; t++) { pa[t] = (const unsigned char *)a + (size_t)t * as; acc[t] = _mm256_setzero_ps(); corr[t] = 0.0f; }
    for (b = 0; b < nb; b++, pw += 210) {
        const unsigned char *ql = pw, *qh = pw + 128;
        const int8_t *sc = (const int8_t *)(pw + 192);
        __m256i sumi[NT];
        float dwd = rd_h(pw + 208);
        int off[NT], n;
        for (t = 0; t < NT; t++) {
            sumi[t] = _mm256_setzero_si256();
            off[t] = 0;
            off[t] = off_term_q6(sc, pa[t] + 260);
        }
        for (n = 0; n < 2; n++, ql += 64, qh += 32, sc += 8) {
            __m256i l0 = _mm256_loadu_si256((const __m256i *)ql), l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i hv = _mm256_loadu_si256((const __m256i *)qh);
            __m256i g0 = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_and_si256(_mm256_slli_epi16(hv, 4), m30));
            __m256i g1 = _mm256_or_si256(_mm256_and_si256(l1, m4), _mm256_and_si256(_mm256_slli_epi16(hv, 2), m30));
            __m256i g2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), _mm256_and_si256(hv, m30));
            __m256i g3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4), _mm256_and_si256(_mm256_srli_epi16(hv, 2), m30));
            __m256i s0 = _mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0]));
            __m256i s1 = _mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2]));
            __m256i s2 = _mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4]));
            __m256i s3 = _mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6]));
            for (t = 0; t < NT; t++) {
                const unsigned char *q8 = pa[t] + 4 + 128 * n;
                __m256i p0 = _mm256_maddubs_epi16(g0, _mm256_loadu_si256((const __m256i *)q8));
                __m256i p1 = _mm256_maddubs_epi16(g1, _mm256_loadu_si256((const __m256i *)(q8 + 32)));
                __m256i p2 = _mm256_maddubs_epi16(g2, _mm256_loadu_si256((const __m256i *)(q8 + 64)));
                __m256i p3 = _mm256_maddubs_epi16(g3, _mm256_loadu_si256((const __m256i *)(q8 + 96)));
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(s0, p0), _mm256_madd_epi16(s1, p1)));
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(s2, p2), _mm256_madd_epi16(s3, p3)));
            }
        }
        for (t = 0; t < NT; t++) {
            float yd, d;
            memcpy(&yd, pa[t], 4);
            d = yd * dwd;
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            corr[t] += d * (float)(32 * off[t]);
            pa[t] += HFC_Q8_K_BLOCK;
        }
    }
    for (t = 0; t < NT; t++) out[t] = hsum256(acc[t]) - corr[t];
}

/* ---- activation quantizers: same arithmetic as the portable versions, eight lanes at a time ---- */

static inline __m256i pack_i32x4_to_i8(__m256i a, __m256i b, __m256i c, __m256i d)     /* values already within int8 */
{
    __m256i ab = _mm256_packs_epi32(a, b), cd = _mm256_packs_epi32(c, d);
    __m256i r = _mm256_packs_epi16(ab, cd);
    return _mm256_permutevar8x32_epi32(r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

static void quantize_q8_0_avx2(const float *x, void *y, size_t n)
{
    unsigned char *out = (unsigned char *)y;
    const __m256 sign = _mm256_set1_ps(-0.0f);
    size_t nb = n / 32, b;
    for (b = 0; b < nb; b++, x += 32, out += HFC_Q8_0_BLOCK) {
        __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                 _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        float amax, d, id;
        uint16_t h;
        __m256 vid;
        __m256i q;
        __m128 t = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        t = _mm_max_ps(t, _mm_movehl_ps(t, t));
        t = _mm_max_ss(t, _mm_shuffle_ps(t, t, 1));
        amax = _mm_cvtss_f32(t);
        d = amax / 127.0f;
        id = d != 0.0f ? 1.0f / d : 0.0f;
        h = (uint16_t)_cvtss_sh(d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        out[0] = (unsigned char)(h & 0xff);
        out[1] = (unsigned char)(h >> 8);
        vid = _mm256_set1_ps(id);
        q = pack_i32x4_to_i8(_mm256_cvtps_epi32(_mm256_mul_ps(v0, vid)), _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid)),
                             _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid)), _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid)));
        _mm256_storeu_si256((__m256i *)(out + 2), q);
    }
}

static void quantize_q8_K_avx2(const float *x, void *y, size_t n)
{
    unsigned char *out = (unsigned char *)y;
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i hi = _mm256_set1_epi32(127);
    size_t nb = n / 256, b;
    int j;
    for (b = 0; b < nb; b++, x += 256, out += HFC_Q8_K_BLOCK) {
        __m256 m = _mm256_setzero_ps();
        float amax, max = 0.0f, iscale, d;
        int8_t *qs = (int8_t *)(out + 4);
        __m128 t;
        __m256 vs;
        for (j = 0; j < 256; j += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + j)));
        t = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        t = _mm_max_ps(t, _mm_movehl_ps(t, t));
        t = _mm_max_ss(t, _mm_shuffle_ps(t, t, 1));
        amax = _mm_cvtss_f32(t);
        if (amax == 0.0f) { memset(out, 0, HFC_Q8_K_BLOCK); continue; }
        for (j = 0; j < 256; j++) { float a = x[j] < 0 ? -x[j] : x[j]; if (a == amax) { max = x[j]; break; } }   /* first element of largest magnitude */
        iscale = -127.0f / max;
        vs = _mm256_set1_ps(iscale);
        for (j = 0; j < 256; j += 32) {
            __m256i a0 = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j))), hi);
            __m256i a1 = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 8))), hi);
            __m256i a2 = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 16))), hi);
            __m256i a3 = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 24))), hi);
            _mm256_storeu_si256((__m256i *)(qs + j), pack_i32x4_to_i8(a0, a1, a2, a3));
        }
        for (j = 0; j < 16; j++) {
            __m128i v = _mm_loadu_si128((const __m128i *)(qs + 16 * j));
            __m256i w16 = _mm256_cvtepi8_epi16(v);
            __m128i s = _mm_add_epi16(_mm256_castsi256_si128(w16), _mm256_extracti128_si256(w16, 1));    /* 8 x int16 */
            int s32;
            int16_t bs;
            s = _mm_madd_epi16(s, _mm_set1_epi16(1));                         /* 4 x int32 */
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            s32 = _mm_cvtsi128_si32(s);
            bs = (int16_t)s32;
            out[260 + 2 * j] = (unsigned char)(bs & 0xff);
            out[261 + 2 * j] = (unsigned char)((bs >> 8) & 0xff);
        }
        d = 1.0f / iscale;
        memcpy(out, &d, 4);
    }
}

/* 10 vectors x 8 lanes x 2 flops; quantization still uses the portable code */
static hfc_kernels k_avx2;

const hfc_kernels *hfc_kernels_avx2(void)
{
    if (!k_avx2.isa) {
        const hfc_kernels *g = hfc_kernels_generic();
        k_avx2 = *g;
        k_avx2.isa = "avx2";
        k_avx2.read_sum = read_sum_avx2;
        k_avx2.fma_burn = fma_burn_avx2;
        k_avx2.flops_per_iter = 160.0;
        k_avx2.quantize_q8_0 = quantize_q8_0_avx2; k_avx2.quantize_q8_K = quantize_q8_K_avx2;
        k_avx2.dot_q8_0 = dot_q8_0_avx2;
        k_avx2.dot_f32 = dot_f32_avx2;
        k_avx2.dot_f32_f16 = dot_f32_f16_avx2;
        k_avx2.axpy_f32_f16 = axpy_f32_f16_avx2;
        k_avx2.f32_to_f16 = f32_to_f16_avx2;
        k_avx2.dot_q4_0 = dot_q4_0_avx2;
        k_avx2.dot_q5_0 = dot_q5_0_avx2;
        k_avx2.dot_q4_K = dot_q4_K_avx2;
        k_avx2.dot_q5_K = dot_q5_K_avx2;
        k_avx2.dot_q6_K = dot_q6_K_avx2;
        k_avx2.dot4_q8_0 = dot4_q8_0_avx2; k_avx2.dot4_q4_0 = dot4_q4_0_avx2; k_avx2.dot4_q5_0 = dot4_q5_0_avx2;
        k_avx2.dot4_q4_K = dot4_q4_K_avx2; k_avx2.dot4_q5_K = dot4_q5_K_avx2; k_avx2.dot4_q6_K = dot4_q6_K_avx2;
    }
    return &k_avx2;
}
#endif
