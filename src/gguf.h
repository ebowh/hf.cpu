/* gguf.h - hardened, bounds-checked GGUF (v2/v3) reader over a read-only mmap.
 *
 * The file is treated as hostile: every count, length, offset and dimension is
 * validated against the file size before it is used, all arithmetic is
 * overflow checked, and no allocation size is taken from the file without a
 * cap derived from the remaining bytes.
 */
#ifndef HFC_GGUF_H
#define HFC_GGUF_H

#include "hfc.h"
#include "pal.h"

enum {
    GGUF_T_U8 = 0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16, GGUF_T_U32, GGUF_T_I32,
    GGUF_T_F32, GGUF_T_BOOL, GGUF_T_STR, GGUF_T_ARR, GGUF_T_U64, GGUF_T_I64, GGUF_T_F64
};

#define GGUF_MAX_DIMS 8

typedef struct {
    char    *key;
    uint32_t type;             /* GGUF_T_* */
    union { uint64_t u; int64_t i; double f; } v;   /* scalars (bool in u) */
    char    *str;              /* GGUF_T_STR value */
    uint32_t arr_type;         /* element type for GGUF_T_ARR */
    uint64_t arr_count;
    uint64_t arr_off;          /* file offset of the first element */
    uint64_t arr_bytes;        /* payload size in bytes */
} gguf_kv;

typedef struct {
    char    *name;
    uint32_t n_dims;
    uint64_t ne[GGUF_MAX_DIMS];
    uint32_t type;             /* ggml type id */
    uint64_t offset;           /* relative to data section */
    uint64_t nelem;
    uint64_t nbytes;           /* 0 if the type is unknown */
    int      known;            /* type found in the type table */
} gguf_tensor;

typedef struct {
    pal_map      map;
    uint32_t     version;
    uint64_t     n_kv, n_tensors;
    gguf_kv     *kv;
    gguf_tensor *t;
    uint64_t     align;
    uint64_t     data_off;     /* absolute file offset of tensor data */
    uint32_t    *thash;        /* open-addressing table of tensor indices + 1 */
    uint64_t     thash_cap;
    int          owns_map;     /* gguf_close unmaps the file (gguf_open only) */
} gguf_file;

/* Open and fully validate. On failure *err has a human-readable reason and
 * the returned status is HFC_EFORMAT / HFC_EIO / HFC_ENOMEM / HFC_ERANGE. */
hfc_status gguf_open(gguf_file *g, const char *path, char *err, size_t errcap);
/* Same, but over an in-memory image (tests, fuzzing). The image is not copied
 * and must outlive the gguf_file; gguf_close will not unmap it. */
hfc_status gguf_open_mem(gguf_file *g, const void *data, size_t len, char *err, size_t errcap);
void       gguf_close(gguf_file *g);

const gguf_kv     *gguf_find_kv(const gguf_file *g, const char *key);
const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name);

/* Scalar accessors; return 1 on success (type compatible), 0 otherwise. */
int gguf_get_u64(const gguf_file *g, const char *key, uint64_t *out);
int gguf_get_f64(const gguf_file *g, const char *key, double *out);
const char *gguf_get_str(const gguf_file *g, const char *key);   /* NULL if absent */

/* Pointer to tensor data inside the mapping (NULL if out of range). */
const void *gguf_tensor_data(const gguf_file *g, const gguf_tensor *t);

/* Walk a string array: calls f(index, ptr, len, ctx); stops if f returns
 * non-zero. Strings are bounds checked. */
typedef int (*gguf_str_cb)(uint64_t idx, const char *s, uint64_t len, void *ctx);
hfc_status gguf_foreach_str(const gguf_file *g, const gguf_kv *kv, gguf_str_cb f, void *ctx);

const char *gguf_type_name(uint32_t t);

#endif
