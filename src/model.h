/* model.h - dense decoder-only transformer (Qwen2, Qwen3, Llama layouts):
 * hyper-parameters and tensors read from GGUF, a paged f16 KV cache, and the
 * forward pass. Weights stay in the read-only file mapping.
 */
#ifndef HFC_MODEL_H
#define HFC_MODEL_H

#include "gguf.h"
#include "hfc.h"
#include "kern.h"

struct hfc_pool;

typedef struct {
    const gguf_tensor   *t;
    const unsigned char *data;
    uint32_t             type;
    uint64_t             cols;        /* input dimension  (ne[0]) */
    uint64_t             rows;        /* output dimension (ne[1]) */
    uint64_t             row_bytes;
} hfc_mat;

typedef struct {
    char   arch[24];
    int    n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab, n_ctx_train, n_rot;
    float  rms_eps, rope_base;
    int    rope_neox;                 /* 1: pairs (i, i+n_rot/2); 0: pairs (2i, 2i+1) */
    int    has_qk_norm;
} hfc_hparams;

typedef struct {
    hfc_mat wq, wk, wv, wo, gate, up, down;
    const float *bq, *bk, *bv;        /* NULL when the model has no bias */
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm;
} hfc_layer;

typedef struct hfc_model {
    const gguf_file *g;
    hfc_hparams      hp;
    hfc_mat          tok_embd, output;     /* output aliases tok_embd when embeddings are tied */
    int              tied;
    const float     *out_norm;
    hfc_layer       *layers;
    float          **owned;                /* converted/copied vectors, freed with the model */
    size_t           n_owned, cap_owned;
    const hfc_kernels *k;
} hfc_model;

hfc_status hfc_model_load(hfc_model **out, const gguf_file *g, const hfc_kernels *k, char *err, size_t errcap);
void       hfc_model_free(hfc_model *m);

/* ---- inference context: KV cache + scratch ---- */

typedef struct hfc_ctx hfc_ctx;

#define HFC_KV_BLOCK_TOKENS 64

hfc_status hfc_ctx_new(hfc_ctx **out, const hfc_model *m, size_t ctx_max, size_t max_batch, struct hfc_pool *pool);
void       hfc_ctx_free(hfc_ctx *c);
size_t     hfc_ctx_pos(const hfc_ctx *c);          /* tokens currently in the KV cache */
size_t     hfc_ctx_kv_bytes(const hfc_ctx *c);     /* bytes of KV memory allocated */
void       hfc_ctx_truncate(hfc_ctx *c, size_t n); /* roll back to the first n tokens */

/* Process n tokens (n <= max_batch) appended at the current position.
 * If logits_last is non-NULL, it receives n_vocab logits for the LAST token. */
hfc_status hfc_ctx_forward(hfc_ctx *c, const uint32_t *tokens, size_t n, float *logits_last);

#endif
