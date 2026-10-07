/* kern_avx2.c - AVX2/FMA kernels. Compiled with -mavx2 -mfma (x86_64 only);
 * only ever called after run-time feature and OS-support detection. */
#include "kern.h"

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

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
        k_avx2.dot_q8_0 = dot_q8_0_avx2;
        k_avx2.dot_f32 = dot_f32_avx2;
        k_avx2.dot_f32_f16 = dot_f32_f16_avx2;
        k_avx2.axpy_f32_f16 = axpy_f32_f16_avx2;
        k_avx2.f32_to_f16 = f32_to_f16_avx2;
    }
    return &k_avx2;
}
#endif
