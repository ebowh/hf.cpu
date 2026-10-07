/* main.c - hfcpu: command line, stdin request loop, signals. */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "ops.h"
#include "pal.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HFC_VERSION "0.0.1-phase0"

static volatile sig_atomic_t g_sigint = 0;
/* SIGINT ends the running request (generation stops and reports stop=cancel); when idle it ends the process.
 * SIGTERM always ends both. */
static void on_sigint(int sig)
{
    if (sig == SIGINT && hfc_busy) hfc_cancel = 1;
    else { g_sigint = 1; hfc_cancel = 1; }
}

static void install_signals(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint;           /* no SA_RESTART: reads return EINTR */
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
}

/* Option line / record header -> effective options for one request. */
static hfc_status make_effective(const hfc_opts *session, const char *line, size_t len,
                                 hfc_opts *eff, char *err, size_t errcap)
{
    char **argv = NULL;
    int argc = 0;
    hfc_status rc = hfc_opts_copy(eff, session);
    if (rc != HFC_OK) { snprintf(err, errcap, "out of memory"); return rc; }
    rc = hfc_shell_split(line, len, &argv, &argc, err, errcap);
    if (rc == HFC_OK) rc = hfc_opts_apply(eff, argc, argv, HFC_CTX_REQUEST, err, errcap);
    hfc_argv_free(argv, argc);
    if (rc != HFC_OK) hfc_opts_free(eff);
    return rc;
}

static void run_prompts_mode(hfc_session *s, const hfc_opts *o)
{
    hfc_reader rd;
    char err[512];
    if (hfc_reader_init(&rd, 0) != HFC_OK) { hfc_log(HFC_LOG_ERROR, "msg=\"out of memory\""); return; }
    while (!g_sigint && !s->out.err) {
        int intr = 0;
        hfc_status rc = hfc_reader_next(&rd, o->record_sep, (size_t)o->max_record, &intr);
        const char *body;
        size_t blen;
        hfc_opts eff;
        const unsigned char *us;

        if (intr) continue;
        if (rc == HFC_EEOF) break;
        if (rc == HFC_EIO) { hfc_log(HFC_LOG_ERROR, "msg=\"read error on stdin\""); break; }
        if (rc != HFC_OK) {
            snprintf(err, sizeof err, "request rejected: %s (limit --max-record %llu bytes)",
                     rc == HFC_ERANGE ? "record too large" : "out of memory", (unsigned long long)o->max_record);
            hfc_emit_failed_request(s, NULL, rc, err);
            hfc_out_raw_byte(&s->out, o->record_sep);
            hfc_out_flush(&s->out);
            continue;
        }
        body = (const char *)rd.rec;
        blen = rd.rec_len;
        us = (const unsigned char *)memchr(rd.rec, 0x1f, rd.rec_len);
        if (us) {
            size_t hlen = (size_t)(us - rd.rec);
            rc = make_effective(o, (const char *)rd.rec, hlen, &eff, err, sizeof err);
            body = (const char *)us + 1;
            blen = rd.rec_len - hlen - 1;
        } else {
            rc = hfc_opts_copy(&eff, o);
            if (rc != HFC_OK) snprintf(err, sizeof err, "out of memory");
        }
        if (rc != HFC_OK) {
            hfc_emit_failed_request(s, NULL, rc, err);
        } else {
            (void)hfc_run_request(s, &eff, body, blen, 1);
            hfc_opts_free(&eff);
        }
        hfc_out_raw_byte(&s->out, o->record_sep);
        hfc_out_flush(&s->out);
    }
    hfc_reader_free(&rd);
}

static void run_args_mode(hfc_session *s, const hfc_opts *o)
{
    hfc_reader rd;
    char err[512];
    int quit = 0;
    if (hfc_reader_init(&rd, 0) != HFC_OK) { hfc_log(HFC_LOG_ERROR, "msg=\"out of memory\""); return; }
    while (!g_sigint && !s->out.err && !quit) {
        int intr = 0;
        hfc_status rc = hfc_reader_next(&rd, '\n', (size_t)o->max_record, &intr);
        hfc_opts eff;
        size_t len;

        if (intr) continue;
        if (rc == HFC_EEOF) break;
        if (rc == HFC_EIO) { hfc_log(HFC_LOG_ERROR, "msg=\"read error on stdin\""); break; }
        if (rc != HFC_OK) {
            snprintf(err, sizeof err, "request line rejected: %s (limit --max-record %llu bytes)",
                     rc == HFC_ERANGE ? "line too long" : "out of memory", (unsigned long long)o->max_record);
            hfc_emit_failed_request(s, NULL, rc, err);
            continue;
        }
        len = rd.rec_len;
        if (len && rd.rec[len - 1] == '\r') len--;
        { size_t i = 0; while (i < len && (rd.rec[i] == ' ' || rd.rec[i] == '\t')) i++; if (i == len) continue; }
        if (rd.rec[0] == '#') continue;
        if (rd.rec[0] == '!') {
            char buf[64];
            if (strncmp((const char *)rd.rec, "!quit", 5) == 0) quit = 1;
            else if (strncmp((const char *)rd.rec, "!stats", 6) == 0) {
                char a[32], b[32], c[32];
                snprintf(a, sizeof a, "%lu", s->n_requests);
                snprintf(b, sizeof b, "%lu", s->n_errors);
                snprintf(c, sizeof c, "%ld", hfc_live_allocs());
                hfc_out_event(&s->out, "stats", "requests", a, "errors", b, "live_allocs", c, (const char *)NULL);
            } else if (strncmp((const char *)rd.rec, "!cancel", 7) == 0) {
                hfc_out_event(&s->out, "info", "msg", "no request is running", (const char *)NULL);
            } else {
                snprintf(buf, sizeof buf, "unknown control line");
                hfc_out_event(&s->out, "error", "msg", buf, (const char *)NULL);
            }
            continue;
        }
        rc = make_effective(o, (const char *)rd.rec, len, &eff, err, sizeof err);
        if (rc != HFC_OK) {
            hfc_emit_failed_request(s, NULL, rc, err);
            continue;
        }
        (void)hfc_run_request(s, &eff, NULL, 0, 0);
        hfc_opts_free(&eff);
    }
    hfc_reader_free(&rd);
}

int main(int argc, char **argv)
{
    hfc_opts o;
    hfc_session s;
    char err[512] = "";
    int i, rc_exit = 0;
    const char *settings = NULL;
    hfc_status rc;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { hfc_opts_usage(stdout); return 0; }
        if (!strcmp(argv[i], "--version")) { printf("hfcpu %s\n", HFC_VERSION); return 0; }
        if (!strcmp(argv[i], "--settings") && i + 1 < argc) settings = argv[i + 1];
        else if (!strncmp(argv[i], "--settings=", 11)) settings = argv[i] + 11;
    }

    hfc_opts_defaults(&o);
    if (settings && (rc = hfc_opts_load_settings(&o, settings, err, sizeof err)) != HFC_OK) {
        fprintf(stderr, "hfcpu: %s\n", err);
        hfc_opts_free(&o);
        return 2;
    }
    if ((rc = hfc_opts_apply(&o, argc - 1, argv + 1, HFC_CTX_CLI, err, sizeof err)) != HFC_OK) {
        fprintf(stderr, "hfcpu: %s\n(try --help)\n", err);
        hfc_opts_free(&o);
        return 2;
    }
    hfc_log_set_level((hfc_loglevel)o.log_level);
    install_signals();

    memset(&s, 0, sizeof s);
    s.session = &o;
    hfc_out_init(&s.out, stdout);

    if (o.stdin_mode == HFC_STDIN_NONE) {
        rc = hfc_run_request(&s, &o, NULL, 0, 0);
        rc_exit = rc == HFC_OK ? 0 : 1;
    } else if (o.stdin_mode == HFC_STDIN_ARGS) {
        run_args_mode(&s, &o);
    } else {
        run_prompts_mode(&s, &o);
    }
    hfc_out_flush(&s.out);
    if (s.out.err) { hfc_log(HFC_LOG_ERROR, "msg=\"stdout closed\""); rc_exit = 1; }
    if (g_sigint) rc_exit = 130;
    hfc_session_close(&s);
    hfc_opts_free(&o);
    if (hfc_live_allocs() != 0)
        hfc_log(HFC_LOG_DEBUG, "msg=\"allocations still live at exit\" count=%ld", hfc_live_allocs());
    return rc_exit;
}
