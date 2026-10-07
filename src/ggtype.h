/* ggtype.h - ggml tensor type table and scalar reference dequantization.
 *
 * Type ids follow ggml's `enum ggml_type`. PrismML's fork adds 142 (PQ2_0)
 * and 143 (PTQ1_0). The scalar dequantizers are the correctness oracle for
 * the SIMD kernels and are differentially tested against upstream
 * ggml-quants.c (tests/diff_dequant.sh).
 */
#ifndef HFC_GGTYPE_H
#define HFC_GGTYPE_H

#include "hfc.h"

typedef struct {
    uint32_t    id;
    const char *name;
    uint32_t    blck;       /* elements per block */
    uint32_t    bytes;      /* bytes per block */
    int         dequant;    /* 1 if a reference dequantizer exists */
} hfc_type_info;

const hfc_type_info *hfc_type_lookup(uint32_t id);   /* NULL for unknown ids */
const char          *hfc_type_name(uint32_t id);     /* "type_N" if unknown, static buffer */

/* Bytes for a row of n elements (n must be a multiple of the block size). */
hfc_status hfc_type_row_bytes(uint32_t id, uint64_t n, uint64_t *out);

/* Convert one row of n elements of `id` at src into floats. */
hfc_status hfc_dequant_row(uint32_t id, const void *src, float *dst, size_t n);

float hfc_f16_to_f32(uint16_t h);
float hfc_bf16_to_f32(uint16_t h);

#endif
