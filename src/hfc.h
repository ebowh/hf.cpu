/* hfc.h - common types, status codes and checked allocation.
 *
 * C99. Every allocation can fail and every failure is reported to the
 * caller; nothing in this code base may assume malloc succeeds.
 */
#ifndef HFC_H
#define HFC_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    HFC_OK = 0,
    HFC_ENOMEM,    /* allocation or address-space failure */
    HFC_EINVAL,    /* bad argument or option */
    HFC_EIO,       /* system call failed (errno has details) */
    HFC_EFORMAT,   /* malformed or hostile input file */
    HFC_ERANGE,    /* size or offset out of range / overflow */
    HFC_ENOTSUP,   /* valid but not implemented */
    HFC_EEOF       /* clean end of input */
} hfc_status;

const char *hfc_strerror(hfc_status s);

/* ---- checked allocation ------------------------------------------------ */

void  *hfc_malloc(size_t n);
void  *hfc_calloc(size_t count, size_t size);   /* overflow checked */
void  *hfc_realloc(void *p, size_t n);
void   hfc_free(void *p);
char  *hfc_strdup(const char *s);               /* NULL on failure */
char  *hfc_strndup(const char *s, size_t n);

/* Fault injection: make the n-th allocation from now on fail (1 = next).
 * n == 0 disables. The env var HFC_FAIL_AFTER sets this at startup. */
void   hfc_fail_after(long n);
/* Number of live allocations (leak check in tests). */
long   hfc_live_allocs(void);

/* ---- overflow-checked arithmetic (return 1 on success) ------------------ */

int hfc_mul_size(size_t a, size_t b, size_t *out);
int hfc_add_size(size_t a, size_t b, size_t *out);
int hfc_mul_u64(uint64_t a, uint64_t b, uint64_t *out);
int hfc_add_u64(uint64_t a, uint64_t b, uint64_t *out);

/* ---- bump arena (per-request scratch; reset instead of free) ----------- */

typedef struct {
    unsigned char *base;
    size_t         cap;
    size_t         used;
} hfc_arena;

hfc_status hfc_arena_init(hfc_arena *a, size_t cap);
void      *hfc_arena_alloc(hfc_arena *a, size_t n, size_t align); /* NULL if full */
void       hfc_arena_reset(hfc_arena *a);
void       hfc_arena_free(hfc_arena *a);

/* ---- logging (stderr) --------------------------------------------------- */

typedef enum { HFC_LOG_ERROR = 0, HFC_LOG_WARN, HFC_LOG_INFO, HFC_LOG_DEBUG } hfc_loglevel;
void hfc_log_set_level(hfc_loglevel lv);
hfc_loglevel hfc_log_get_level(void);
/* Emits "level key=value ... msg=\"...\"" style lines. */
void hfc_log(hfc_loglevel lv, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#endif /* HFC_H */
