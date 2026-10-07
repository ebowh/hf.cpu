/* tok.h - byte-level BPE tokenizer (GPT-2 style) read from GGUF metadata.
 *
 * Supported: tokenizer.ggml.model = "gpt2" with tokenizer.ggml.pre = "qwen2"
 * (Qwen2 / Qwen2.5 / Qwen3 families). The pre-tokenizer is a hand-written
 * scanner equivalent to the Qwen2 regular expression, so no regex engine or
 * PCRE dependency is needed.
 */
#ifndef HFC_TOK_H
#define HFC_TOK_H

#include "gguf.h"
#include "hfc.h"

typedef struct hfc_tok hfc_tok;

hfc_status hfc_tok_load(hfc_tok **out, const gguf_file *g, char *err, size_t errcap);
void       hfc_tok_free(hfc_tok *t);

uint32_t   hfc_tok_n_vocab(const hfc_tok *t);
/* Special token ids from the GGUF; -1 if absent. */
int64_t    hfc_tok_bos(const hfc_tok *t);
int64_t    hfc_tok_eos(const hfc_tok *t);
int64_t    hfc_tok_pad(const hfc_tok *t);
int        hfc_tok_add_bos(const hfc_tok *t);

#define HFC_TOK_PARSE_SPECIAL 1u   /* recognise "<|im_start|>"-style tokens in the text */
#define HFC_TOK_ADD_BOS       2u   /* prepend BOS if the model asks for it */

/* Encode text. *ids is allocated with hfc_malloc (free with hfc_free). */
hfc_status hfc_tok_encode(const hfc_tok *t, const char *text, size_t len, unsigned flags,
                          uint32_t **ids, size_t *n);

/* Bytes of one token. Control tokens give an empty piece unless
 * render_special is set. Pointer is into tokenizer storage. */
hfc_status hfc_tok_piece(const hfc_tok *t, uint32_t id, int render_special,
                         const unsigned char **bytes, size_t *len);

/* Decode a whole sequence into a heap buffer (NUL terminated, *len excludes it). */
hfc_status hfc_tok_decode(const hfc_tok *t, const uint32_t *ids, size_t n, int render_special,
                          char **out, size_t *len);

/* Number of leading bytes of p[0..n) that form complete UTF-8 characters; a
 * truncated multi-byte sequence at the end is held back so streamed text
 * never splits a character. Invalid bytes count as complete. */
size_t hfc_utf8_complete_prefix(const unsigned char *p, size_t n);

#endif
