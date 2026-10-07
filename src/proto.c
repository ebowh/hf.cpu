/* proto.c - request reader and event writer. */
#include "proto.h"

#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

hfc_status hfc_reader_init(hfc_reader *r, int fd)
{
    memset(r, 0, sizeof *r);
    r->fd = fd;
    r->cap = 1 << 16;
    r->buf = (unsigned char *)hfc_malloc(r->cap);
    if (!r->buf) return HFC_ENOMEM;
    r->rec_cap = 1 << 12;
    r->rec = (unsigned char *)hfc_malloc(r->rec_cap);
    if (!r->rec) { hfc_free(r->buf); r->buf = NULL; return HFC_ENOMEM; }
    return HFC_OK;
}

void hfc_reader_free(hfc_reader *r)
{
    hfc_free(r->buf);
    hfc_free(r->rec);
    r->buf = r->rec = NULL;
}

static hfc_status rec_append(hfc_reader *r, const unsigned char *p, size_t n)
{
    size_t need;
    if (!hfc_add_size(r->rec_len, n, &need) || !hfc_add_size(need, 1, &need)) return HFC_ERANGE;
    if (need > r->rec_cap) {
        size_t nc = r->rec_cap;
        unsigned char *q;
        while (nc < need) {
            if (nc > (size_t)-1 / 2) return HFC_ERANGE;
            nc *= 2;
        }
        q = (unsigned char *)hfc_realloc(r->rec, nc);
        if (!q) return HFC_ENOMEM;
        r->rec = q;
        r->rec_cap = nc;
    }
    memcpy(r->rec + r->rec_len, p, n);
    r->rec_len += n;
    return HFC_OK;
}

hfc_status hfc_reader_next(hfc_reader *r, int sep, size_t max, int *interrupted)
{
    int too_big = 0;
    hfc_status rc, end_status = HFC_ERANGE;
    *interrupted = 0;
    r->rec_len = 0;
    for (;;) {
        if (r->pos == r->end) {
            ssize_t n;
            if (r->eof) {
                if (too_big) return end_status;
                if (r->rec_len == 0) return HFC_EEOF;
                r->rec[r->rec_len] = '\0';
                return HFC_OK;
            }
            n = read(r->fd, r->buf, r->cap);
            if (n < 0) {
                if (errno == EINTR) { *interrupted = 1; return HFC_OK; }
                return HFC_EIO;
            }
            if (n == 0) { r->eof = 1; continue; }
            r->pos = 0;
            r->end = (size_t)n;
        }
        {
            const unsigned char *s = r->buf + r->pos;
            size_t avail = r->end - r->pos;
            const unsigned char *hit = (const unsigned char *)memchr(s, sep, avail);
            size_t take = hit ? (size_t)(hit - s) : avail;
            if (!too_big) {
                if (r->rec_len + take > max) {
                    too_big = 1;
                    r->rec_len = 0;
                } else if ((rc = rec_append(r, s, take)) != HFC_OK) {
                    /* out of memory: drop the record but keep stream sync */
                    too_big = 1;
                    r->rec_len = 0;
                    if (rc == HFC_ENOMEM) end_status = HFC_ENOMEM;
                }
            }
            r->pos += take + (hit ? 1 : 0);
            if (hit) {
                if (too_big) return end_status;
                r->rec[r->rec_len] = '\0';
                return HFC_OK;
            }
        }
    }
}

/* ---- output ---------------------------------------------------------------- */

void hfc_out_init(hfc_out *o, FILE *f) { o->f = f; o->err = 0; }

static void put(hfc_out *o, const void *p, size_t n)
{
    if (o->err) return;
    if (n && fwrite(p, 1, n, o->f) != n) o->err = 1;
}

char *hfc_event_quote(const char *v)
{
    size_t n = 0, i, len = strlen(v);
    int need = len == 0;
    char *out, *w;
    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)v[i];
        if (c <= ' ' || c == '"' || c == '\\' || c == '=' || c == 0x7f) need = 1;
    }
    if (!need) return hfc_strdup(v);
    out = (char *)hfc_malloc(len * 4 + 3);
    if (!out) return NULL;
    w = out;
    *w++ = '"';
    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)v[i];
        if (c == '"' || c == '\\') { *w++ = '\\'; *w++ = (char)c; }
        else if (c == '\n') { *w++ = '\\'; *w++ = 'n'; }
        else if (c == '\r') { *w++ = '\\'; *w++ = 'r'; }
        else if (c == '\t') { *w++ = '\\'; *w++ = 't'; }
        else if (c < 0x20 || c == 0x7f) { w += sprintf(w, "\\x%02x", c); }
        else *w++ = (char)c;
    }
    *w++ = '"';
    *w = '\0';
    (void)n;
    return out;
}

static void put_pairs(hfc_out *o, va_list ap)
{
    for (;;) {
        const char *k = va_arg(ap, const char *), *v;
        char *q;
        if (!k) break;
        v = va_arg(ap, const char *);
        if (!v) v = "";
        q = hfc_event_quote(v);
        put(o, " ", 1);
        put(o, k, strlen(k));
        put(o, "=", 1);
        if (q) { put(o, q, strlen(q)); hfc_free(q); }
        else { put(o, "\"\"", 2); }       /* out of memory: degrade, never crash */
    }
}

void hfc_out_event(hfc_out *o, const char *name, ...)
{
    va_list ap;
    put(o, "@", 1);
    put(o, name, strlen(name));
    va_start(ap, name);
    put_pairs(o, ap);
    va_end(ap);
    put(o, "\n", 1);
    hfc_out_flush(o);
}

void hfc_out_payload(hfc_out *o, const char *name, const void *data, size_t len, ...)
{
    va_list ap;
    char num[32];
    put(o, "@", 1);
    put(o, name, strlen(name));
    snprintf(num, sizeof num, " len=%lu", (unsigned long)len);
    put(o, num, strlen(num));
    va_start(ap, len);
    put_pairs(o, ap);
    va_end(ap);
    put(o, "\n", 1);
    put(o, data, len);
    put(o, "\n", 1);
    hfc_out_flush(o);
}

void hfc_out_raw_byte(hfc_out *o, int byte)
{
    unsigned char c = (unsigned char)byte;
    put(o, &c, 1);
}

void hfc_out_flush(hfc_out *o)
{
    if (!o->err && fflush(o->f) != 0) o->err = 1;
}
