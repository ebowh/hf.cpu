/* sample.c - sampling. */
#include "sample.h"
#include "mathx.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void hfc_rng_seed(hfc_rng *r, uint64_t seed)
{
    r->s = seed * 0x9e3779b97f4a7c15ull + 0x1234567ull;
    if (r->s == 0) r->s = 0x2545f4914f6cdd1dull;
}

static uint64_t rng_next(hfc_rng *r)
{
    uint64_t x = r->s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    r->s = x;
    return x * 0x2545f4914f6cdd1dull;
}

float hfc_rng_uniform(hfc_rng *r) { return (float)(rng_next(r) >> 40) / 16777216.0f; }

double hfc_logprob(const float *logits, size_t n, uint32_t tok)
{
    float mx = logits[0];
    double sum = 0.0;
    size_t i;
    for (i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (i = 0; i < n; i++) sum += (double)hfc_expf(logits[i] - mx);
    return (double)(logits[tok] - mx) - log(sum);
}

void hfc_top_logprobs(const float *logits, size_t n, int k, hfc_top *out)
{
    int cnt = 0, j;
    size_t i;
    double lse;
    float mx = logits[0];
    double sum = 0.0;
    uint32_t ids[64];
    if (k > 64) k = 64;
    for (i = 0; i < n; i++) {
        float v = logits[i];
        if (cnt == k && v <= logits[ids[cnt - 1]]) continue;
        j = cnt < k ? cnt++ : cnt - 1;
        while (j > 0 && logits[ids[j - 1]] < v) { ids[j] = ids[j - 1]; j--; }
        ids[j] = (uint32_t)i;
    }
    for (i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (i = 0; i < n; i++) sum += (double)hfc_expf(logits[i] - mx);
    lse = log(sum);
    for (j = 0; j < cnt; j++) { out[j].id = ids[j]; out[j].logprob = (float)((double)(logits[ids[j]] - mx) - lse); }
}

typedef struct { uint32_t id; float p; } cand_t;

static int cmp_desc(const void *a, const void *b)
{
    const cand_t *x = (const cand_t *)a, *y = (const cand_t *)b;
    if (x->p != y->p) return x->p < y->p ? 1 : -1;
    return x->id < y->id ? -1 : x->id > y->id;
}

hfc_status hfc_sample(const float *logits, size_t n, const hfc_sampler *p, hfc_rng *rng, uint32_t *tok)
{
    cand_t *c;
    size_t i, m = n, kept;
    float mx, sum = 0.0f, pmax, cum, r;
    if (n == 0) return HFC_EINVAL;
    if (p->temp <= 0.0f) {
        size_t best = 0;
        for (i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        *tok = (uint32_t)best;
        return HFC_OK;
    }
    c = (cand_t *)hfc_malloc(n * sizeof(cand_t));
    if (!c) return HFC_ENOMEM;
    for (i = 0; i < n; i++) { c[i].id = (uint32_t)i; c[i].p = logits[i]; }
    if (p->top_k > 0 && (size_t)p->top_k < n) {
        /* partial selection: keep the top_k largest by insertion into a sorted prefix */
        size_t k = (size_t)p->top_k, cnt = 0, j;
        cand_t *t = (cand_t *)hfc_malloc(k * sizeof(cand_t));
        if (!t) { hfc_free(c); return HFC_ENOMEM; }
        for (i = 0; i < n; i++) {
            cand_t v = c[i];
            if (cnt == k && cmp_desc(&v, &t[cnt - 1]) >= 0) continue;
            j = cnt < k ? cnt++ : cnt - 1;
            while (j > 0 && cmp_desc(&v, &t[j - 1]) < 0) { t[j] = t[j - 1]; j--; }
            t[j] = v;
        }
        memcpy(c, t, cnt * sizeof(cand_t));
        m = cnt;
        hfc_free(t);
    } else {
        qsort(c, n, sizeof(cand_t), cmp_desc);
    }
    mx = c[0].p;
    for (i = 0; i < m; i++) { c[i].p = hfc_expf((c[i].p - mx) / p->temp); sum += c[i].p; }
    for (i = 0; i < m; i++) c[i].p /= sum;
    pmax = c[0].p;
    kept = m;
    if (p->min_p > 0.0f) { kept = 0; while (kept < m && c[kept].p >= p->min_p * pmax) kept++; if (kept == 0) kept = 1; }
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        cum = 0.0f;
        for (i = 0; i < kept; i++) { cum += c[i].p; if (cum >= p->top_p) { kept = i + 1; break; } }
    }
    sum = 0.0f;
    for (i = 0; i < kept; i++) sum += c[i].p;
    r = hfc_rng_uniform(rng) * sum;
    cum = 0.0f;
    *tok = c[kept - 1].id;
    for (i = 0; i < kept; i++) { cum += c[i].p; if (r < cum) { *tok = c[i].id; break; } }
    hfc_free(c);
    return HFC_OK;
}
