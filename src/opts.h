/* opts.h - one option vocabulary for the command line, the settings file and
 * per-request option lines.
 *
 * Effective options for a request are layered:
 *   defaults < settings file < real command line < per-request options
 * The same parser serves all of them, so there is one set of flags and one
 * set of validation errors.
 */
#ifndef HFC_OPTS_H
#define HFC_OPTS_H

#include <stdio.h>
#include "hfc.h"

typedef enum { HFC_STDIN_PROMPTS = 0, HFC_STDIN_ARGS, HFC_STDIN_NONE } hfc_stdin_mode;

typedef struct {
    /* --- session-only options (command line / settings file) --- */
    char    *settings;          /* settings file path */
    int      stdin_mode;        /* hfc_stdin_mode */
    int      record_sep;        /* byte value 0..255 */
    int      control_fd;        /* -1 = none */
    char    *cache_dir;
    uint64_t cache_quota;       /* bytes; 0 = persistent cache disabled */
    uint64_t cache_min_free;    /* bytes of free space to always leave */
    int      threads;           /* 0 = auto */
    uint64_t ctx_init, ctx_step, ctx_max;
    uint64_t max_record;        /* largest accepted prompt/line in bytes */
    int      probe_force;
    int      log_level;         /* hfc_loglevel */

    /* --- options valid on the command line and in every request --- */
    char    *op;
    char    *model;
    char    *id;
    char    *system;            /* system prompt text */
    char    *system_file;
    char    *prompt;
    char    *prompt_file;
    int      n;                 /* parallel continuations */
    double   temp;
    int      top_k;
    double   top_p, min_p;
    uint64_t seed;
    int      max_tokens;
    int      logprobs;          /* top-N logprobs per token, 0 = off */
    int      stop_on_repeat;
    double   timeout;           /* seconds, 0 = none */
    int      list_tensors;      /* inspect: also list every tensor */
    int      probe_sustained;   /* doctor: seconds of all-core load to expose throttling, 0 = skip */
} hfc_opts;

void       hfc_opts_defaults(hfc_opts *o);
hfc_status hfc_opts_copy(hfc_opts *dst, const hfc_opts *src);  /* deep; dst is overwritten */
void       hfc_opts_free(hfc_opts *o);

typedef enum { HFC_CTX_CLI = 0, HFC_CTX_REQUEST = 1 } hfc_opts_ctx;

/* Apply argv[0..argc-1] (no program name) on top of *o. On error returns
 * HFC_EINVAL (or HFC_ENOMEM) and writes a message to err; *o may be partially
 * updated, so callers apply request options to a copy. */
hfc_status hfc_opts_apply(hfc_opts *o, int argc, char **argv, hfc_opts_ctx ctx,
                          char *err, size_t errcap);

/* Parse a settings file of `key = value` lines ('#' comments). */
hfc_status hfc_opts_load_settings(hfc_opts *o, const char *path, char *err, size_t errcap);

/* POSIX-shell-style word splitting with NO expansion. Result must be freed
 * with hfc_argv_free. */
hfc_status hfc_shell_split(const char *s, size_t len, char ***argv, int *argc,
                           char *err, size_t errcap);
void       hfc_argv_free(char **argv, int argc);

hfc_status hfc_parse_size(const char *s, uint64_t *out);   /* "80G", "512M", "4096" */

void hfc_opts_dump(FILE *f, const hfc_opts *o);   /* name=value lines */
void hfc_opts_usage(FILE *f);

#endif
