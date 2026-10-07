/* probe.c - CPU detection and machine micro-benchmarks. */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "probe.h"
#include "kern.h"
#include "pal.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define HFC_X86 1
#endif
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/sysctl.h>
#endif

/* ---- CPU detection ------------------------------------------------------ */

#ifdef HFC_X86
static unsigned long long xgetbv0(void)
{
    unsigned int lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}

static void detect_x86(hfc_cpu *c)
{
    unsigned a = 0, b = 0, cc = 0, d = 0, maxl = 0, maxe;
    int os_avx = 0, os_avx512 = 0;

    if (!__get_cpuid(0, &maxl, &b, &cc, &d)) return;
    memcpy(c->vendor, &b, 4);
    memcpy(c->vendor + 4, &d, 4);
    memcpy(c->vendor + 8, &cc, 4);
    c->vendor[12] = '\0';

    if (maxl >= 1) {
        __get_cpuid(1, &a, &b, &cc, &d);
        c->stepping = (int)(a & 0xf);
        c->model = (int)((a >> 4) & 0xf);
        c->family = (int)((a >> 8) & 0xf);
        if (c->family == 0xf) c->family += (int)((a >> 20) & 0xff);
        if (c->family == 0x6 || c->family >= 0xf) c->model += (int)(((a >> 16) & 0xf) << 4);
        c->sse42  = (cc >> 20) & 1;
        c->popcnt = (cc >> 23) & 1;
        if ((cc >> 27) & 1) {                      /* OSXSAVE */
            unsigned long long x = xgetbv0();
            os_avx = (x & 6) == 6;                 /* XMM+YMM state enabled */
            os_avx512 = os_avx && (x & 0xe0) == 0xe0;
        }
        c->avx = os_avx && ((cc >> 28) & 1);
        c->fma = os_avx && ((cc >> 12) & 1);
        c->f16c = os_avx && ((cc >> 29) & 1);
    }
    if (maxl >= 7) {
        unsigned a7, b7, c7, d7;
        __cpuid_count(7, 0, a7, b7, c7, d7);
        c->avx2 = os_avx && ((b7 >> 5) & 1);
        c->bmi2 = (b7 >> 8) & 1;
        c->avx512f = os_avx512 && ((b7 >> 16) & 1);
        c->avx512bw = os_avx512 && ((b7 >> 30) & 1);
        c->avx512vnni = os_avx512 && ((c7 >> 11) & 1);
        __cpuid_count(7, 1, a7, b7, c7, d7);
        c->avxvnni = os_avx && ((a7 >> 4) & 1);
    }
    maxe = __get_cpuid_max(0x80000000u, NULL);
    if (maxe >= 0x80000004u) {
        unsigned r[12], i;
        for (i = 0; i < 3; i++) __get_cpuid(0x80000002u + i, &r[4*i], &r[4*i+1], &r[4*i+2], &r[4*i+3]);
        memcpy(c->brand, r, 48);
        c->brand[48] = '\0';
    }
    /* caches: Intel leaf 4, AMD leaf 0x8000001D */
    {
        unsigned leaf = 0;
        if (strcmp(c->vendor, "GenuineIntel") == 0 && maxl >= 4) leaf = 4;
        else if (maxe >= 0x8000001Du) leaf = 0x8000001Du;
        if (leaf) {
            unsigned sub;
            for (sub = 0; sub < 16; sub++) {
                unsigned type, level, ways, parts, line, sets;
                __cpuid_count(leaf, sub, a, b, cc, d);
                type = a & 0x1f;
                if (type == 0) break;
                level = (a >> 5) & 7;
                ways = ((b >> 22) & 0x3ff) + 1;
                parts = ((b >> 12) & 0x3ff) + 1;
                line = (b & 0xfff) + 1;
                sets = cc + 1;
                {
                    unsigned long long sz = (unsigned long long)ways * parts * line * sets;
                    if (sz > 0xffffffffull) continue;
                    c->line = line;
                    if (level == 1 && type == 1) c->l1d = (unsigned)sz;
                    else if (level == 2 && (type == 3 || type == 1)) c->l2 = (unsigned)sz;
                    else if (level == 3) c->l3 = (unsigned)sz;
                }
            }
        }
    }
    /* SMT width: leaf 0xB level type 1 gives logical processors per core */
    if (maxl >= 0xb) {
        unsigned sub;
        for (sub = 0; sub < 4; sub++) {
            unsigned a2, b2, c2, d2;
            __cpuid_count(0xb, sub, a2, b2, c2, d2);
            if (((c2 >> 8) & 0xff) == 1 && (b2 & 0xffff) > 0) {
                int tpc = (int)(b2 & 0xffff);
                if (tpc > 0 && c->logical % tpc == 0) c->physical = c->logical / tpc;
                break;
            }
        }
    }
}
#endif /* HFC_X86 */

static void detect_os_caches(hfc_cpu *c)
{
#if defined(__APPLE__) || defined(__FreeBSD__)
    {
        unsigned long long v;
        size_t sz = sizeof v;
        v = 0; sz = sizeof v;
        if (!c->l1d && sysctlbyname("hw.l1dcachesize", &v, &sz, NULL, 0) == 0) c->l1d = (unsigned)v;
        v = 0; sz = sizeof v;
        if (!c->l2 && sysctlbyname("hw.l2cachesize", &v, &sz, NULL, 0) == 0) c->l2 = (unsigned)v;
        v = 0; sz = sizeof v;
        if (!c->l3 && sysctlbyname("hw.l3cachesize", &v, &sz, NULL, 0) == 0) c->l3 = (unsigned)v;
        v = 0; sz = sizeof v;
        if (!c->line && sysctlbyname("hw.cachelinesize", &v, &sz, NULL, 0) == 0) c->line = (unsigned)v;
    }
#elif defined(_SC_LEVEL1_DCACHE_SIZE)
    {
        long v;
        if (!c->l1d && (v = sysconf(_SC_LEVEL1_DCACHE_SIZE)) > 0) c->l1d = (unsigned)v;
        if (!c->l2 && (v = sysconf(_SC_LEVEL2_CACHE_SIZE)) > 0) c->l2 = (unsigned)v;
        if (!c->l3 && (v = sysconf(_SC_LEVEL3_CACHE_SIZE)) > 0) c->l3 = (unsigned)v;
        if (!c->line && (v = sysconf(_SC_LEVEL1_DCACHE_LINESIZE)) > 0) c->line = (unsigned)v;
    }
#else
    (void)c;
#endif
}

void hfc_cpu_detect(hfc_cpu *c)
{
    memset(c, 0, sizeof *c);
    c->logical = pal_ncpu_online();
    c->physical = c->logical;
    strcpy(c->vendor, "unknown");
    strcpy(c->brand, "unknown");
#ifdef HFC_X86
    detect_x86(c);
#endif
    detect_os_caches(c);
    if (!c->line) c->line = 64;
    if (c->physical < 1) c->physical = 1;
}

static uint64_t fnv64(uint64_t h, const void *p, size_t n)
{
    const unsigned char *s = (const unsigned char *)p;
    size_t i;
    for (i = 0; i < n; i++) { h ^= s[i]; h *= 0x100000001b3ull; }
    return h;
}

void hfc_cpu_signature(const hfc_cpu *c, uint64_t mem_total, char out[17])
{
    uint64_t h = 0xcbf29ce484222325ull;
    uint64_t gib = (mem_total + (1ull << 29)) >> 30;     /* rounded to GiB */
    int v[6];
    v[0] = c->family; v[1] = c->model; v[2] = c->stepping;
    v[3] = c->logical; v[4] = c->physical; v[5] = (int)c->line;
    h = fnv64(h, c->vendor, strlen(c->vendor));
    h = fnv64(h, c->brand, strlen(c->brand));
    h = fnv64(h, v, sizeof v);
    h = fnv64(h, &c->l1d, sizeof c->l1d);
    h = fnv64(h, &c->l2, sizeof c->l2);
    h = fnv64(h, &c->l3, sizeof c->l3);
    h = fnv64(h, &gib, sizeof gib);
    {
        static const char hx[] = "0123456789abcdef";
        int i;
        for (i = 0; i < 16; i++) out[i] = hx[(h >> (60 - 4 * i)) & 0xf];
        out[16] = '\0';
    }
}

/* ---- kernel selection --------------------------------------------------- */

const hfc_kernels *hfc_kernels_for(const hfc_cpu *c)
{
#if defined(__x86_64__)
    if (c->avx2 && c->fma) {
        const hfc_kernels *k = hfc_kernels_avx2();
        if (k) return k;
    }
#endif
    return hfc_kernels_generic();
}

/* ---- benchmarks ----------------------------------------------------------- */

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int             go;
    int             ready;
} gate_t;

typedef struct {
    gate_t *gate;
    const hfc_kernels *k;
    const unsigned char *buf;
    size_t bytes;
    int passes;
    long fma_iters;
    int mode;                         /* 0 = bandwidth, 1 = fma */
    double t0, t1;
    uint64_t sink;
} job_t;

static void *job_main(void *arg)
{
    job_t *j = (job_t *)arg;
    int p;
    pthread_mutex_lock(&j->gate->mu);
    j->gate->ready++;
    pthread_cond_broadcast(&j->gate->cv);
    while (!j->gate->go) pthread_cond_wait(&j->gate->cv, &j->gate->mu);
    pthread_mutex_unlock(&j->gate->mu);
    j->t0 = pal_now();
    if (j->mode == 0) {
        for (p = 0; p < j->passes; p++) j->sink += j->k->read_sum(j->buf, j->bytes);
    } else {
        j->sink += (uint64_t)j->k->fma_burn(j->fma_iters);
    }
    j->t1 = pal_now();
    return NULL;
}

/* Run nthreads jobs concurrently; returns wall time from the first start to
 * the last finish, or <0 on failure. */
static double run_jobs(job_t *jobs, int n)
{
    pthread_t th[HFC_MAX_BW + 64];
    gate_t g;
    int i, started = 0;
    double t0 = 1e300, t1 = 0;

    if (n > (int)(sizeof th / sizeof th[0])) return -1;
    pthread_mutex_init(&g.mu, NULL);
    pthread_cond_init(&g.cv, NULL);
    g.go = 0;
    g.ready = 0;
    for (i = 0; i < n; i++) {
        jobs[i].gate = &g;
        if (pthread_create(&th[i], NULL, job_main, &jobs[i]) != 0) break;
        started++;
    }
    pthread_mutex_lock(&g.mu);
    while (g.ready < started) pthread_cond_wait(&g.cv, &g.mu);
    g.go = 1;
    pthread_cond_broadcast(&g.cv);
    pthread_mutex_unlock(&g.mu);
    for (i = 0; i < started; i++) pthread_join(th[i], NULL);
    pthread_mutex_destroy(&g.mu);
    pthread_cond_destroy(&g.cv);
    if (started != n) return -1;
    for (i = 0; i < n; i++) {
        if (jobs[i].t0 < t0) t0 = jobs[i].t0;
        if (jobs[i].t1 > t1) t1 = jobs[i].t1;
    }
    return t1 - t0;
}

static void *aligned_alloc_hfc(size_t bytes, void **raw)
{
    unsigned char *p = (unsigned char *)hfc_malloc(bytes + 64);
    *raw = p;
    if (!p) return NULL;
    return (void *)(((uintptr_t)p + 63) & ~(uintptr_t)63);
}

static double bench_bw(const hfc_kernels *k, const unsigned char *buf, size_t bytes, int nthreads)
{
    job_t jobs[HFC_MAX_BW + 64];
    double best = 0;
    size_t slice = (bytes / (size_t)nthreads) & ~(size_t)127;
    int rep, i;
    const int passes = 2;
    if (nthreads > HFC_MAX_BW + 64 || slice < 128) return 0;
    for (rep = 0; rep < 3; rep++) {
        double dt, gbps;
        memset(jobs, 0, sizeof jobs);
        for (i = 0; i < nthreads; i++) {
            jobs[i].k = k;
            jobs[i].buf = buf + (size_t)i * slice;
            jobs[i].bytes = slice;
            jobs[i].passes = passes;
            jobs[i].mode = 0;
        }
        dt = run_jobs(jobs, nthreads);
        if (dt <= 0) return 0;
        gbps = (double)slice * nthreads * passes / dt / 1e9;
        if (gbps > best) best = gbps;
    }
    return best;
}

static double fma_run(const hfc_kernels *k, int nthreads, long iters)
{
    job_t jobs[HFC_MAX_BW + 64];
    double dt;
    int i;
    if (nthreads > HFC_MAX_BW + 64) return 0;
    memset(jobs, 0, sizeof jobs);
    for (i = 0; i < nthreads; i++) { jobs[i].k = k; jobs[i].fma_iters = iters; jobs[i].mode = 1; }
    dt = run_jobs(jobs, nthreads);
    if (dt <= 0) return 0;
    return k->flops_per_iter * (double)iters * nthreads / dt / 1e9;
}

static double bench_fma(const hfc_kernels *k, int nthreads, double seconds)
{
    job_t jobs[HFC_MAX_BW + 64];
    long iters = 1000000;
    double dt = 0;
    int i, tries;
    if (nthreads > HFC_MAX_BW + 64) return 0;
    /* calibrate iteration count to roughly `seconds` */
    for (tries = 0; tries < 8; tries++) {
        memset(jobs, 0, sizeof jobs);
        for (i = 0; i < nthreads; i++) { jobs[i].k = k; jobs[i].fma_iters = iters; jobs[i].mode = 1; }
        dt = run_jobs(jobs, nthreads);
        if (dt <= 0) return 0;
        if (dt >= seconds * 0.5) break;
        iters = (long)((double)iters * (seconds / (dt > 1e-6 ? dt : 1e-6)));
        if (iters > 2000000000L) iters = 2000000000L;
    }
    return k->flops_per_iter * (double)iters * nthreads / dt / 1e9;
}

typedef struct { size_t next; unsigned char pad[56]; } node_t;

static double bench_latency(size_t bytes, long loads)
{
    size_t n = bytes / sizeof(node_t), i;
    void *raw = NULL;
    node_t *nodes = (node_t *)aligned_alloc_hfc(n * sizeof(node_t), &raw);
    size_t *perm;
    uint64_t s = 88172645463325252ull;
    size_t cur = 0;
    long k;
    double t0, t1;
    if (!nodes) return -1;
    perm = (size_t *)hfc_malloc(n * sizeof *perm);
    if (!perm) { hfc_free(raw); return -1; }
    for (i = 0; i < n; i++) perm[i] = i;
    for (i = n - 1; i > 0; i--) {                 /* Fisher-Yates, xorshift64 */
        size_t j;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        j = (size_t)(s % (i + 1));
        { size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    }
    for (i = 0; i + 1 < n; i++) nodes[perm[i]].next = perm[i + 1];
    nodes[perm[n - 1]].next = perm[0];
    cur = perm[0];
    for (k = 0; k < (long)n && k < 1000000; k++) cur = nodes[cur].next;   /* warm */
    t0 = pal_now();
    for (k = 0; k < loads; k++) cur = nodes[cur].next;
    t1 = pal_now();
    hfc_free(perm);
    hfc_free(raw);
    if (cur == (size_t)-1) return -1;                                    /* keep cur live */
    return (t1 - t0) / (double)loads * 1e9;
}

hfc_status hfc_probe_run(const hfc_cpu *c, int quick, hfc_machine *m)
{
    const hfc_kernels *k = hfc_kernels_for(c);
    pal_meminfo mi;
    size_t bytes = (size_t)128 << 20;
    void *raw = NULL;
    unsigned char *buf = NULL;
    int tlist[HFC_MAX_BW], nt = 0, i, t;

    memset(m, 0, sizeof *m);
    memset(&mi, 0, sizeof mi);
    (void)pal_meminfo_get(&mi);
    hfc_cpu_signature(c, mi.total, m->sig);
    strncpy(m->isa, k->isa, sizeof m->isa - 1);
    m->when = (long)time(NULL);

    /* shrink the buffer if memory is tight; never touch more than 1/4 of avail */
    if (mi.avail && (uint64_t)bytes > mi.avail / 4) bytes = (size_t)(mi.avail / 4) & ~(size_t)4095;
    while (bytes >= ((size_t)8 << 20) && !(buf = (unsigned char *)aligned_alloc_hfc(bytes, &raw)))
        bytes /= 2;
    if (!buf) return HFC_ENOMEM;
    memset(buf, 1, bytes);                        /* first touch */

    if (quick) {
        tlist[nt++] = 1;
        if (c->physical > 2) tlist[nt++] = 2;     /* shows where bandwidth saturates */
        if (c->physical > 1) tlist[nt++] = c->physical;
        if (c->logical > c->physical) tlist[nt++] = c->logical;
    } else {
        for (t = 1; t <= c->logical && nt < HFC_MAX_BW; t++) tlist[nt++] = t;
    }
    for (i = 0; i < nt; i++) {
        double g = bench_bw(k, buf, bytes, tlist[i]);
        m->bw_threads[m->nbw] = tlist[i];
        m->bw_gbps[m->nbw] = g;
        if (g > m->bw_best_gbps * 1.03) { m->bw_best_gbps = g; m->bw_best_threads = tlist[i]; }
        m->nbw++;
    }
    hfc_free(raw);

    m->fma_gflops_1t = bench_fma(k, 1, quick ? 0.25 : 0.5);
    m->fma_gflops_all = c->logical > 1 ? bench_fma(k, c->physical, quick ? 0.25 : 0.5) : m->fma_gflops_1t;
    m->fma_gflops_logical = c->logical > c->physical ? bench_fma(k, c->logical, quick ? 0.25 : 0.5) : m->fma_gflops_all;
    {
        int ft[4], nf = 0, q;
        ft[nf++] = 1;
        if (c->physical > 2) ft[nf++] = 2;
        if (c->physical > 3) ft[nf++] = 3;
        if (c->physical > 1) ft[nf++] = c->physical;
        for (q = 0; q < nf && m->nfma < HFC_MAX_BW; q++) {
            m->fma_threads[m->nfma] = ft[q];
            m->fma_gflops_n[m->nfma++] = bench_fma(k, ft[q], quick ? 0.25 : 0.5);
        }
    }

    {
        static const size_t full_sz[] = { 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
            1u << 20, 2u << 20, 4u << 20, 8u << 20, 16u << 20, 32u << 20, 64u << 20 };
        static const size_t quick_sz[] = { 4096, 16384, 32768, 65536, 262144, 524288,
            1u << 20, 2u << 20, 4u << 20, 8u << 20, 16u << 20, 64u << 20 };
        const size_t *lst = quick ? quick_sz : full_sz;
        size_t cnt = quick ? sizeof quick_sz / sizeof quick_sz[0] : sizeof full_sz / sizeof full_sz[0];
        size_t li;
        for (li = 0; li < cnt && m->nlat < HFC_MAX_LAT; li++) {
            double ns = bench_latency(lst[li], quick ? 1000000 : 4000000);
            if (ns > 0) { m->lat_bytes[m->nlat] = lst[li]; m->lat_ns[m->nlat] = ns; m->nlat++; }
        }
    }
    return HFC_OK;
}

/* ---- profile persistence -------------------------------------------------- */

static void prof_path(const char *dir, const char *sig, char *out, size_t cap)
{
    snprintf(out, cap, "%s/machine-%s.prof", dir, sig);
}

hfc_status hfc_probe_save(const char *dir, const hfc_machine *m)
{
    char path[1024], *buf;
    size_t cap = 8192, off = 0;
    int i;
    hfc_status rc;

    if ((rc = pal_mkdir_p(dir)) != HFC_OK) return rc;
    prof_path(dir, m->sig, path, sizeof path);
    buf = (char *)hfc_malloc(cap);
    if (!buf) return HFC_ENOMEM;
#define APP(...) do { int w_ = snprintf(buf + off, cap - off, __VA_ARGS__); \
                      if (w_ < 0 || (size_t)w_ >= cap - off) { hfc_free(buf); return HFC_ERANGE; } \
                      off += (size_t)w_; } while (0)
    APP("format=3\nsig=%s\nisa=%s\nwhen=%ld\n", m->sig, m->isa, m->when);
    APP("bw_best_threads=%d\nbw_best_gbps=%.3f\n", m->bw_best_threads, m->bw_best_gbps);
    APP("fma_gflops_1t=%.3f\nfma_gflops_all=%.3f\nfma_gflops_logical=%.3f\n", m->fma_gflops_1t, m->fma_gflops_all, m->fma_gflops_logical);
    for (i = 0; i < m->nfma; i++) APP("fma.%d=%.3f\n", m->fma_threads[i], m->fma_gflops_n[i]);
    for (i = 0; i < m->nbw; i++) APP("bw.%d=%.3f\n", m->bw_threads[i], m->bw_gbps[i]);
    if (m->sustained_seconds) APP("sustained_seconds=%d\nfma_sustained_gflops=%.3f\nfma_sustained_ratio=%.4f\n", m->sustained_seconds, m->fma_sustained_gflops, m->fma_sustained_ratio);
    for (i = 0; i < m->nlat; i++) APP("lat.%lu=%.3f\n", (unsigned long)m->lat_bytes[i], m->lat_ns[i]);
#undef APP
    rc = pal_write_file_atomic(path, buf, off);
    hfc_free(buf);
    return rc;
}

hfc_status hfc_probe_load(const char *dir, const hfc_cpu *c, uint64_t mem_total, hfc_machine *m)
{
    char sig[17], path[1024], line[256];
    FILE *f;
    int saw_format = 0;

    memset(m, 0, sizeof *m);
    hfc_cpu_signature(c, mem_total, sig);
    prof_path(dir, sig, path, sizeof path);
    f = fopen(path, "r");
    if (!f) return HFC_EIO;
    while (fgets(line, sizeof line, f)) {
        char *eq = strchr(line, '=');
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (!eq) continue;
        *eq++ = '\0';
        if (strcmp(line, "format") == 0) saw_format = atoi(eq) == 3;     /* bump when the probe changes so stale profiles are re-measured */
        else if (strcmp(line, "sig") == 0) { strncpy(m->sig, eq, 16); m->sig[16] = '\0'; }
        else if (strcmp(line, "isa") == 0) { strncpy(m->isa, eq, sizeof m->isa - 1); }
        else if (strcmp(line, "when") == 0) m->when = atol(eq);
        else if (strcmp(line, "bw_best_threads") == 0) m->bw_best_threads = atoi(eq);
        else if (strcmp(line, "bw_best_gbps") == 0) m->bw_best_gbps = atof(eq);
        else if (strcmp(line, "fma_gflops_1t") == 0) m->fma_gflops_1t = atof(eq);
        else if (strcmp(line, "fma_gflops_all") == 0) m->fma_gflops_all = atof(eq);
        else if (strcmp(line, "fma_gflops_logical") == 0) m->fma_gflops_logical = atof(eq);
        else if (strcmp(line, "sustained_seconds") == 0) m->sustained_seconds = atoi(eq);
        else if (strcmp(line, "fma_sustained_gflops") == 0) m->fma_sustained_gflops = atof(eq);
        else if (strcmp(line, "fma_sustained_ratio") == 0) m->fma_sustained_ratio = atof(eq);
        else if (strncmp(line, "fma.", 4) == 0 && m->nfma < HFC_MAX_BW) {
            m->fma_threads[m->nfma] = atoi(line + 4);
            m->fma_gflops_n[m->nfma++] = atof(eq);
        }
        else if (strncmp(line, "bw.", 3) == 0 && m->nbw < HFC_MAX_BW) {
            m->bw_threads[m->nbw] = atoi(line + 3);
            m->bw_gbps[m->nbw++] = atof(eq);
        } else if (strncmp(line, "lat.", 4) == 0 && m->nlat < HFC_MAX_LAT) {
            m->lat_bytes[m->nlat] = (size_t)strtoul(line + 4, NULL, 10);
            m->lat_ns[m->nlat++] = atof(eq);
        }
    }
    fclose(f);
    if (!saw_format || strcmp(m->sig, sig) != 0) return HFC_EFORMAT;
    return HFC_OK;
}

void hfc_probe_print(FILE *f, const hfc_cpu *c, const hfc_machine *m)
{
    int i;
    fprintf(f, "cpu.vendor=%s\ncpu.brand=%s\n", c->vendor, c->brand);
    fprintf(f, "cpu.family=%d\ncpu.model=%d\ncpu.stepping=%d\n", c->family, c->model, c->stepping);
    fprintf(f, "cpu.logical=%d\ncpu.physical=%d\n", c->logical, c->physical);
    {
        static const char *nm[] = { "sse4.2", "popcnt", "avx", "avx2", "fma", "f16c", "bmi2", "avx512f", "avx512bw", "avx512vnni", "avx-vnni" };
        unsigned fl[11];
        int k, first = 1;
        fl[0] = c->sse42; fl[1] = c->popcnt; fl[2] = c->avx; fl[3] = c->avx2; fl[4] = c->fma; fl[5] = c->f16c;
        fl[6] = c->bmi2; fl[7] = c->avx512f; fl[8] = c->avx512bw; fl[9] = c->avx512vnni; fl[10] = c->avxvnni;
        fprintf(f, "cpu.features=");
        for (k = 0; k < 11; k++) if (fl[k]) { fprintf(f, "%s%s", first ? "" : " ", nm[k]); first = 0; }
        fputc('\n', f);
    }
    fprintf(f, "cache.l1d=%u\ncache.l2=%u\ncache.l3=%u\ncache.line=%u\n", c->l1d, c->l2, c->l3, c->line);
    fprintf(f, "machine.sig=%s\nmachine.isa=%s\nmachine.measured=%ld\n", m->sig, m->isa, m->when);
    for (i = 0; i < m->nbw; i++) fprintf(f, "bw.threads.%d=%.2f GB/s\n", m->bw_threads[i], m->bw_gbps[i]);
    fprintf(f, "bw.best=%.2f GB/s at %d threads\n", m->bw_best_gbps, m->bw_best_threads);
    for (i = 0; i < m->nfma; i++) fprintf(f, "fma.threads.%d=%.1f GFLOP/s\n", m->fma_threads[i], m->fma_gflops_n[i]);
    fprintf(f, "fma.gflops.1thread=%.1f\nfma.gflops.allcores=%.1f\nfma.gflops.allthreads=%.1f\n", m->fma_gflops_1t, m->fma_gflops_all, m->fma_gflops_logical);
    if (m->sustained_seconds)
        fprintf(f, "fma.gflops.sustained=%.1f over %ds (%.0f%% of first second)\n", m->fma_sustained_gflops, m->sustained_seconds, m->fma_sustained_ratio * 100.0);
    for (i = 0; i < m->nlat; i++) fprintf(f, "lat.%lu=%.2f ns\n", (unsigned long)m->lat_bytes[i], m->lat_ns[i]);
}

hfc_status hfc_probe_sustained(const hfc_cpu *c, int seconds, hfc_machine *m)
{
    const hfc_kernels *k = hfc_kernels_for(c);
    int nthreads = c->physical, s;
    long iters = 1000000;
    double g = 0, first, tail = 0;
    int tail_n = 0, from;

    if (seconds < 1) return HFC_OK;
    if (seconds > HFC_MAX_TRACE) seconds = HFC_MAX_TRACE;
    /* calibrate to about one second per sample */
    for (s = 0; s < 8; s++) {
        double t0 = pal_now(), dt;
        g = fma_run(k, nthreads, iters);
        dt = pal_now() - t0;
        if (g <= 0) return HFC_EIO;
        if (dt >= 0.5) break;
        iters = (long)((double)iters * (1.0 / (dt > 1e-6 ? dt : 1e-6)));
        if (iters > 2000000000L) iters = 2000000000L;
    }
    m->nsustained = 0;
    for (s = 0; s < seconds; s++) {
        double t0 = pal_now(), dt;
        g = fma_run(k, nthreads, iters);
        dt = pal_now() - t0;
        if (g <= 0) return HFC_EIO;
        /* rescale if a sample ran noticeably off one second (throttling lowers
         * throughput, so the sample gets longer; keep the iteration count) */
        m->sustained_trace[m->nsustained++] = g;
        (void)dt;
    }
    first = m->sustained_trace[0];
    from = m->nsustained - (m->nsustained + 3) / 4;
    for (s = from; s < m->nsustained; s++) { tail += m->sustained_trace[s]; tail_n++; }
    m->sustained_seconds = seconds;
    m->fma_sustained_gflops = tail_n ? tail / tail_n : 0;
    m->fma_sustained_ratio = first > 0 ? m->fma_sustained_gflops / first : 0;
    return HFC_OK;
}
