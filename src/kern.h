/* kern.h - runtime-dispatched SIMD kernels.
 *
 * Every kernel has a portable C version (kern_generic.c). ISA-specific
 * versions live in their own translation units compiled with the matching
 * -m flags, and are selected at run time from the detected CPU features.
 */
#ifndef HFC_KERN_H
#define HFC_KERN_H

#include <stddef.h>
#include <stdint.h>

struct hfc_cpu;

typedef struct {
    const char *isa;                                  /* "generic", "avx2" ... */
    /* Sum of 64-bit words over a buffer; bytes must be a multiple of 128 and
     * p 32-byte aligned. Used as a memory-bandwidth probe. */
    uint64_t (*read_sum)(const void *p, size_t bytes);
    /* Register-resident FMA burn: returns a checksum; each call performs
     * iters * flops_per_iter floating point operations. */
    float    (*fma_burn)(long iters);
    double     flops_per_iter;

    /* ---- model kernels ---- */
    /* Q8_0 block = { uint16 fp16 scale; int8 q[32] } = 34 bytes, no padding. */
    void  (*quantize_q8_0)(const float *x, void *y, size_t n);       /* n multiple of 32 */
    float (*dot_q8_0)(const void *w, const void *a, size_t nblocks);   /* sum over blocks of dw*da*sum(wq*aq) */
    float (*dot_f32)(const float *a, const float *b, size_t n);
    float (*dot_f32_f16)(const float *a, const uint16_t *b, size_t n);
    void  (*axpy_f32_f16)(float *y, float a, const uint16_t *x, size_t n);   /* y += a*x */
    void  (*f32_to_f16)(const float *x, uint16_t *y, size_t n);
} hfc_kernels;

#define HFC_Q8_0_BLOCK 34
#define HFC_Q8_0_QK    32

const hfc_kernels *hfc_kernels_for(const struct hfc_cpu *cpu);
const hfc_kernels *hfc_kernels_generic(void);

#if defined(__x86_64__)
const hfc_kernels *hfc_kernels_avx2(void);            /* kern_avx2.c */
#endif

#endif
