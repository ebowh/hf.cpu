#include "../src/opts.h"
#include "t.h"
#include <stdlib.h>

static char err[256];

static void test_split(void)
{
    char **v; int c;
    CHECK(hfc_shell_split("a b  c", 6, &v, &c, err, sizeof err) == HFC_OK);
    CHECK(c == 3); CHECK_STR(v[0], "a"); CHECK_STR(v[2], "c");
    hfc_argv_free(v, c);

    {
        const char *s = "--system 'be \"terse\"' --prompt \"line1\\nline2 \\\"q\\\"\" x\\ y \"\" ''";
        CHECK(hfc_shell_split(s, strlen(s), &v, &c, err, sizeof err) == HFC_OK);
        CHECK(c == 7);
        CHECK_STR(v[1], "be \"terse\"");
        CHECK_STR(v[3], "line1\nline2 \"q\"");
        CHECK_STR(v[4], "x y");
        CHECK_STR(v[5], "");
        CHECK_STR(v[6], "");
        hfc_argv_free(v, c);
    }
    /* no expansion */
    CHECK(hfc_shell_split("$HOME `x` *", 11, &v, &c, err, sizeof err) == HFC_OK);
    CHECK(c == 3); CHECK_STR(v[0], "$HOME"); CHECK_STR(v[1], "`x`"); CHECK_STR(v[2], "*");
    hfc_argv_free(v, c);

    CHECK(hfc_shell_split("'abc", 4, &v, &c, err, sizeof err) == HFC_EINVAL);
    CHECK(v == NULL && c == 0);
    CHECK(hfc_shell_split("\"abc", 4, &v, &c, err, sizeof err) == HFC_EINVAL);
    CHECK(hfc_shell_split("abc\\", 4, &v, &c, err, sizeof err) == HFC_EINVAL);
    CHECK(hfc_shell_split("a\0b", 3, &v, &c, err, sizeof err) == HFC_EINVAL);
    CHECK(hfc_shell_split("", 0, &v, &c, err, sizeof err) == HFC_OK);
    CHECK(c == 0);
    hfc_argv_free(v, c);
}

static void test_apply(void)
{
    hfc_opts o;
    char *a1[] = { "--n", "4", "--temp=0.5", "--cache-quota", "80G", "--no-stop-on-repeat",
                   "--record-sep", "RS", "--stdin-mode", "args", "--system", "hi" };
    char *bad1[] = { "--n", "0" };
    char *bad2[] = { "--n", "abc" };
    char *bad3[] = { "--temp", "nan" };
    char *bad4[] = { "--nosuch" };
    char *bad5[] = { "--n" };
    char *bad6[] = { "positional" };
    char *bad7[] = { "--cache-quota", "12X" };
    char *bad8[] = { "--stdin-mode", "bogus" };
    char *req_bad[] = { "--cache-dir", "/x" };
    char *req_ok[] = { "--n", "2", "--prompt-file", "/tmp/p" };
    hfc_opts r;
    uint64_t sz;

    hfc_opts_defaults(&o);
    CHECK(o.n == 1 && o.stdin_mode == HFC_STDIN_PROMPTS && o.record_sep == 0x1e);
    CHECK(hfc_opts_apply(&o, 12, a1, HFC_CTX_CLI, err, sizeof err) == HFC_OK);
    CHECK(o.n == 4); CHECK_NEAR(o.temp, 0.5, 1e-9);
    CHECK(o.cache_quota == (uint64_t)80 << 30); CHECK(o.stop_on_repeat == 0);
    CHECK(o.record_sep == 30); CHECK(o.stdin_mode == HFC_STDIN_ARGS); CHECK_STR(o.system, "hi");

    CHECK(hfc_opts_apply(&o, 2, bad1, HFC_CTX_CLI, err, sizeof err) == HFC_EINVAL);
    CHECK(strstr(err, "out of range") != NULL);
    CHECK(hfc_opts_apply(&o, 2, bad2, HFC_CTX_CLI, err, sizeof err) == HFC_EINVAL);
    CHECK(hfc_opts_apply(&o, 2, bad3, HFC_CTX_CLI, err, sizeof err) == HFC_EINVAL);
    CHECK(hfc_opts_apply(&o, 1, bad4, HFC_CTX_CLI, err, sizeof err) == HFC_EINVAL);
    CHECK(strstr(err, "unknown option") != NULL);
    CHECK(hfc_opts_apply(&o, 1, bad5, HFC_CTX_CLI, err, sizeof err) == HFC_EINVAL);
    CHECK(hfc_opts_apply(&o, 1, bad6, HFC_CTX_CLI, err, sizeof err) == HFC_EINVAL);
    CHECK(hfc_opts_apply(&o, 2, bad7, HFC_CTX_CLI, err, sizeof err) == HFC_EINVAL);
    CHECK(hfc_opts_apply(&o, 2, bad8, HFC_CTX_CLI, err, sizeof err) == HFC_EINVAL);

    /* layering: request options apply to a copy and do not touch the session */
    CHECK(hfc_opts_copy(&r, &o) == HFC_OK);
    CHECK(hfc_opts_apply(&r, 2, req_bad, HFC_CTX_REQUEST, err, sizeof err) == HFC_EINVAL);
    CHECK(strstr(err, "startup") != NULL);
    CHECK(hfc_opts_apply(&r, 4, req_ok, HFC_CTX_REQUEST, err, sizeof err) == HFC_OK);
    CHECK(r.n == 2); CHECK(o.n == 4); CHECK_STR(r.system, "hi"); CHECK_STR(r.prompt_file, "/tmp/p");
    CHECK(o.prompt_file == NULL);
    hfc_opts_free(&r);
    hfc_opts_free(&o);

    CHECK(hfc_parse_size("4096", &sz) == HFC_OK && sz == 4096);
    CHECK(hfc_parse_size("512M", &sz) == HFC_OK && sz == (uint64_t)512 << 20);
    CHECK(hfc_parse_size("2GiB", &sz) == HFC_OK && sz == (uint64_t)2 << 30);
    CHECK(hfc_parse_size("99999999999T", &sz) == HFC_ERANGE);
    CHECK(hfc_parse_size("-1", &sz) != HFC_OK);
    CHECK(hfc_parse_size("", &sz) != HFC_OK);
}

static void test_settings(void)
{
    const char *path = "/tmp/hfc_test_settings.conf";
    FILE *f = fopen(path, "w");
    hfc_opts o;
    fputs("# comment\n\nthreads = 3\ncache-quota=1G\nstop-on-repeat = no\nsystem = You are terse.\n", f);
    fclose(f);
    hfc_opts_defaults(&o);
    CHECK(hfc_opts_load_settings(&o, path, err, sizeof err) == HFC_OK);
    CHECK(o.threads == 3); CHECK(o.cache_quota == (uint64_t)1 << 30);
    CHECK(o.stop_on_repeat == 0); CHECK_STR(o.system, "You are terse.");
    hfc_opts_free(&o);

    f = fopen(path, "w");
    fputs("bogus = 1\n", f);
    fclose(f);
    hfc_opts_defaults(&o);
    CHECK(hfc_opts_load_settings(&o, path, err, sizeof err) == HFC_EINVAL);
    hfc_opts_free(&o);
    CHECK(hfc_opts_load_settings(&o, "/nonexistent/file", err, sizeof err) == HFC_EIO);
    remove(path);
}

/* Fail the n-th allocation for every n and check nothing leaks or crashes. */
static void test_fault_injection(void)
{
    long n, base = hfc_live_allocs();
    int saw_fail = 0, saw_ok = 0;
    for (n = 1; n < 400; n++) {
        char **v = NULL; int c = 0;
        hfc_opts o, r;
        char *argv1[] = { "--system", "a long enough system prompt to allocate", "--n", "2" };
        hfc_status rc;
        hfc_opts_defaults(&o);
        hfc_fail_after(n);
        rc = hfc_shell_split("--prompt 'x y' --id abc \"\" q", 28, &v, &c, err, sizeof err);
        if (rc == HFC_OK) {
            hfc_status r2 = hfc_opts_apply(&o, 4, argv1, HFC_CTX_CLI, err, sizeof err);
            hfc_status r3 = (r2 == HFC_OK) ? hfc_opts_copy(&r, &o) : r2;
            if (r2 != HFC_OK || r3 != HFC_OK) saw_fail = 1; else { saw_ok = 1; hfc_opts_free(&r); }
        } else {
            saw_fail = 1;
            CHECK(rc == HFC_ENOMEM);
        }
        hfc_fail_after(0);
        hfc_argv_free(v, c);
        hfc_opts_free(&o);
        CHECK(hfc_live_allocs() == base);
    }
    CHECK(saw_fail && saw_ok);
}

int main(void)
{
    test_split();
    test_apply();
    test_settings();
    test_fault_injection();
    return t_report("test_opts");
}
