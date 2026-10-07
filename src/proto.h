/* proto.h - stdin request reader and stdout event writer.
 *
 * stdout carries a stream of event lines:
 *     @name key=value key="value with spaces" ...\n
 * A payload event adds len=N and is followed by exactly N raw bytes and a
 * newline:   @text len=5\nhello\n
 * Every request ends with an @done event; in record mode the response is
 * then terminated by the record separator byte. stderr carries logs.
 */
#ifndef HFC_PROTO_H
#define HFC_PROTO_H

#include <stdio.h>
#include "hfc.h"

/* ---- input ---------------------------------------------------------------- */

typedef struct {
    int            fd;
    unsigned char *buf;     /* raw read buffer */
    size_t         cap, pos, end;
    int            eof;
    unsigned char *rec;     /* current record (owned) */
    size_t         rec_len, rec_cap;
} hfc_reader;

hfc_status hfc_reader_init(hfc_reader *r, int fd);
void       hfc_reader_free(hfc_reader *r);

/* Next record terminated by byte `sep` (or EOF). Returns HFC_OK with
 * r->rec / r->rec_len set (NUL terminated, valid until the next call),
 * HFC_EEOF when no more data, HFC_ERANGE when the record exceeded max bytes
 * (the rest of it has been skipped), HFC_ENOMEM, or HFC_EIO. If *interrupted
 * is set the read was interrupted by a signal and the caller should check its
 * flags and call again. */
hfc_status hfc_reader_next(hfc_reader *r, int sep, size_t max, int *interrupted);

/* ---- output --------------------------------------------------------------- */

typedef struct {
    FILE *f;
    int   err;              /* set on any write failure (e.g. EPIPE) */
} hfc_out;

void hfc_out_init(hfc_out *o, FILE *f);
/* Variadic key,value string pairs terminated by NULL. */
void hfc_out_event(hfc_out *o, const char *name, ...);
void hfc_out_payload(hfc_out *o, const char *name, const void *data, size_t len, ...);
void hfc_out_raw_byte(hfc_out *o, int byte);
void hfc_out_flush(hfc_out *o);

/* Quote a value for an event line if needed; returns heap string or NULL. */
char *hfc_event_quote(const char *v);

#endif
