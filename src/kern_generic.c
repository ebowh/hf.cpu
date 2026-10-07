/* kern_generic.c - portable reference kernels. */
#include "kern.h"

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

static const hfc_kernels k_generic = { "generic", read_sum_generic, fma_burn_generic, 16.0 };

const hfc_kernels *hfc_kernels_generic(void) { return &k_generic; }
