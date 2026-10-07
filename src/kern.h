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
} hfc_kernels;

const hfc_kernels *hfc_kernels_for(const struct hfc_cpu *cpu);
const hfc_kernels *hfc_kernels_generic(void);

#if defined(__x86_64__)
const hfc_kernels *hfc_kernels_avx2(void);            /* kern_avx2.c */
#endif

#endif
