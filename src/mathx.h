/* mathx.h - small deterministic math helpers.
 *
 * Everything here uses only IEEE +, -, *, / (no libm transcendental except
 * floorf, which is exact), so results are identical on every platform and
 * compiler when built with -ffp-contract=off.
 */
#ifndef HFC_MATHX_H
#define HFC_MATHX_H

#include <stddef.h>
#include <stdint.h>

float    hfc_expf(float x);                 /* ~1 ulp; 0 for x < -87.3, finite max for large x */
float    hfc_silu(float x);                 /* x / (1 + exp(-x)) */
uint16_t hfc_f32_to_f16(float f);           /* round to nearest even, like F16C */

/* y[i] = x[i] * (1/sqrt(mean(x^2) + eps)) * w[i]  (w may be NULL). n any size. */
void hfc_rmsnorm(const float *x, const float *w, float *y, size_t n, float eps);

/* In-place softmax over x[0..n). */
void hfc_softmax(float *x, size_t n);

/* Deterministic cos and sin of a float angle (|theta| up to ~1e6): computed in
 * double with only + - * / so every platform gets the same bits. */
void hfc_cos_sin(float theta, float *c, float *s);

/* base^(-2/n_rot) as a float, deterministic (square-root chain when n_rot is a
 * power of two). */
float hfc_rope_theta_scale(float base, int n_rot);

#endif
