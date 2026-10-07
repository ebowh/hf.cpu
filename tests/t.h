/* t.h - tiny test harness. */
#ifndef T_H
#define T_H
#include <stdio.h>
#include <string.h>

static int t_run_, t_fail_;
#define CHECK(c) do { t_run_++; if (!(c)) { t_fail_++; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CHECK_STR(a, b) do { const char *a_ = (a), *b_ = (b); t_run_++; \
    if (!a_ || !b_ || strcmp(a_, b_) != 0) { t_fail_++; \
    fprintf(stderr, "FAIL %s:%d: \"%s\" != \"%s\"\n", __FILE__, __LINE__, a_ ? a_ : "(null)", b_ ? b_ : "(null)"); } } while (0)
#define CHECK_NEAR(a, b, tol) do { double a_ = (a), b_ = (b); t_run_++; \
    if (!(a_ - b_ <= (tol) && b_ - a_ <= (tol))) { t_fail_++; \
    fprintf(stderr, "FAIL %s:%d: %g != %g\n", __FILE__, __LINE__, a_, b_); } } while (0)
static int t_report(const char *name)
{
    fprintf(stderr, "%s: %d checks, %d failed\n", name, t_run_, t_fail_);
    return t_fail_ ? 1 : 0;
}
#endif
