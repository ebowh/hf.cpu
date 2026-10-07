/* mem.c - checked allocation, fault injection, arena, logging. */
#include "hfc.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long g_fail_after = -1;   /* -1: uninitialised (read env on first use) */
static long g_live = 0;

static void fail_init(void)
{
    if (g_fail_after == -1) {
        const char *e = getenv("HFC_FAIL_AFTER");
        g_fail_after = e ? strtol(e, NULL, 10) : 0;
        if (g_fail_after < 0) g_fail_after = 0;
    }
}

void hfc_fail_after(long n) { g_fail_after = n < 0 ? 0 : n; }
long hfc_live_allocs(void)  { return g_live; }

/* Returns 1 if this allocation should be failed. */
static int inject(void)
{
    fail_init();
    if (g_fail_after > 0 && --g_fail_after == 0) return 1;
    return 0;
}

const char *hfc_strerror(hfc_status s)
{
    switch (s) {
    case HFC_OK:      return "ok";
    case HFC_ENOMEM:  return "out of memory";
    case HFC_EINVAL:  return "invalid argument";
    case HFC_EIO:     return "i/o error";
    case HFC_EFORMAT: return "malformed input";
    case HFC_ERANGE:  return "size out of range";
    case HFC_ENOTSUP: return "not supported";
    case HFC_ECANCEL: return "cancelled";
    case HFC_EEOF:    return "end of input";
    }
    return "unknown error";
}

const char *hfc_status_name(hfc_status s)
{
    switch (s) {
    case HFC_OK:      return "ok";
    case HFC_ENOMEM:  return "ENOMEM";
    case HFC_EINVAL:  return "EINVAL";
    case HFC_EIO:     return "EIO";
    case HFC_EFORMAT: return "EFORMAT";
    case HFC_ERANGE:  return "ERANGE";
    case HFC_ENOTSUP: return "ENOTSUP";
    case HFC_ECANCEL: return "ECANCEL";
    case HFC_EEOF:    return "EEOF";
    }
    return "EUNKNOWN";
}

void *hfc_malloc(size_t n)
{
    void *p;
    if (inject()) return NULL;
    p = malloc(n ? n : 1);
    if (p) g_live++;
    return p;
}

int hfc_mul_size(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > (size_t)-1 / a) return 0;
    *out = a * b;
    return 1;
}

int hfc_add_size(size_t a, size_t b, size_t *out)
{
    if (a > (size_t)-1 - b) return 0;
    *out = a + b;
    return 1;
}

int hfc_mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a != 0 && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

int hfc_add_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

void *hfc_calloc(size_t count, size_t size)
{
    size_t n;
    void *p;
    if (!hfc_mul_size(count, size, &n)) return NULL;
    p = hfc_malloc(n);
    if (p) memset(p, 0, n);
    return p;
}

void *hfc_realloc(void *p, size_t n)
{
    void *q;
    if (!p) return hfc_malloc(n);
    if (inject()) return NULL;           /* original block stays valid */
    q = realloc(p, n ? n : 1);
    return q;                            /* NULL: caller still owns p */
}

void hfc_free(void *p)
{
    if (!p) return;
    g_live--;
    free(p);
}

char *hfc_strndup(const char *s, size_t n)
{
    char *p;
    size_t need;
    if (!hfc_add_size(n, 1, &need)) return NULL;
    p = (char *)hfc_malloc(need);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

char *hfc_strdup(const char *s) { return hfc_strndup(s, strlen(s)); }

/* ---- arena -------------------------------------------------------------- */

hfc_status hfc_arena_init(hfc_arena *a, size_t cap)
{
    a->base = (unsigned char *)hfc_malloc(cap);
    a->cap = a->base ? cap : 0;
    a->used = 0;
    return a->base ? HFC_OK : HFC_ENOMEM;
}

void *hfc_arena_alloc(hfc_arena *a, size_t n, size_t align)
{
    size_t start, end;
    if (align == 0 || (align & (align - 1)) != 0) return NULL;
    /* align the absolute address, not just the offset */
    start = (size_t)(((uintptr_t)a->base + a->used + (align - 1)) & ~(uintptr_t)(align - 1))
            - (size_t)(uintptr_t)a->base;
    if (!hfc_add_size(start, n, &end) || end > a->cap) return NULL;
    a->used = end;
    return a->base + start;
}

void hfc_arena_reset(hfc_arena *a) { a->used = 0; }

void hfc_arena_free(hfc_arena *a)
{
    hfc_free(a->base);
    a->base = NULL;
    a->cap = a->used = 0;
}

/* ---- logging ------------------------------------------------------------ */

static hfc_loglevel g_level = HFC_LOG_INFO;

void hfc_log_set_level(hfc_loglevel lv) { g_level = lv; }
hfc_loglevel hfc_log_get_level(void)    { return g_level; }

void hfc_log(hfc_loglevel lv, const char *fmt, ...)
{
    static const char *names[] = { "error", "warn", "info", "debug" };
    va_list ap;
    if (lv > g_level) return;
    fprintf(stderr, "%s ", names[lv]);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
