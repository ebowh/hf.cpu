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

/* 10 vectors x 8 lanes x 2 flops */
static const hfc_kernels k_avx2 = { "avx2", read_sum_avx2, fma_burn_avx2, 160.0 };

const hfc_kernels *hfc_kernels_avx2(void) { return &k_avx2; }
#endif
