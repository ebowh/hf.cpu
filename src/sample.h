/* sample.h - token sampling and log-probabilities. */
#ifndef HFC_SAMPLE_H
#define HFC_SAMPLE_H

#include "hfc.h"

typedef struct { uint64_t s; } hfc_rng;
void hfc_rng_seed(hfc_rng *r, uint64_t seed);
float hfc_rng_uniform(hfc_rng *r);            /* [0, 1) */

typedef struct {
    float    temp;          /* <= 0: greedy */
    int      top_k;         /* 0 = off */
    float    top_p, min_p;  /* 1 / 0 = off */
} hfc_sampler;

/* Pick a token from logits[0..n). Greedy ties go to the lowest id. */
hfc_status hfc_sample(const float *logits, size_t n, const hfc_sampler *p, hfc_rng *rng, uint32_t *tok);

/* log-softmax of one token: logits[tok] - logsumexp(logits). */
double hfc_logprob(const float *logits, size_t n, uint32_t tok);

typedef struct { uint32_t id; float logprob; } hfc_top;
/* The best `k` tokens (k <= 64) by logit, with their log-probabilities. */
void hfc_top_logprobs(const float *logits, size_t n, int k, hfc_top *out);

#endif
