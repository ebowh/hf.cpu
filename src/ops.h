/* ops.h - request dispatch and the operations available in phase 0. */
#ifndef HFC_OPS_H
#define HFC_OPS_H

#include "hfc.h"
#include "opts.h"
#include "probe.h"
#include "proto.h"

typedef struct {
    const hfc_opts *session;      /* command line + settings (never modified) */
    hfc_out         out;
    hfc_cpu         cpu;
    int             cpu_ready;
    hfc_machine     machine;
    int             machine_ready;
    unsigned long   n_requests, n_errors;
    struct hfc_resident *res;     /* model kept loaded between requests */
} hfc_session;

void hfc_session_close(hfc_session *s);

/* Run one request with effective options `eff`. `body` is the prompt carried
 * on stdin in prompts mode (has_body = 1), else NULL. Emits @begin ... @done
 * and returns the request's status. */
hfc_status hfc_run_request(hfc_session *s, const hfc_opts *eff, const char *body, size_t body_len, int has_body);

/* Emit a failed request without running anything (bad option line etc.). */
void hfc_emit_failed_request(hfc_session *s, const char *id, hfc_status st, const char *msg);

#endif
