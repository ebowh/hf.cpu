/* probe.h - CPU feature / cache detection and machine micro-benchmarks. */
#ifndef HFC_PROBE_H
#define HFC_PROBE_H

#include <stdio.h>
#include "hfc.h"

typedef struct hfc_cpu {
    char vendor[16];
    char brand[64];
    int  family, model, stepping;
    int  logical, physical;          /* online hardware threads, cores */
    unsigned sse42, popcnt, avx, avx2, fma, f16c, bmi2;
    unsigned avx512f, avx512bw, avx512vnni, avxvnni;
    unsigned l1d, l2, l3;            /* bytes per instance (0 unknown) */
    unsigned line;                   /* cache line bytes */
} hfc_cpu;

void hfc_cpu_detect(hfc_cpu *c);
/* Stable 16-hex-digit signature of CPU model + caches + RAM size. */
void hfc_cpu_signature(const hfc_cpu *c, uint64_t mem_total, char out[17]);

#define HFC_MAX_BW   32
#define HFC_MAX_LAT  32
#define HFC_MAX_TRACE 600

typedef struct {
    char   sig[17];
    char   isa[16];                  /* kernel set used for the probe */
    long   when;                     /* unix time of the measurement */
    int    nbw;
    int    bw_threads[HFC_MAX_BW];
    double bw_gbps[HFC_MAX_BW];      /* aggregate read bandwidth, median over reps */
    double bw_lo[HFC_MAX_BW], bw_hi[HFC_MAX_BW];   /* spread over reps */
    int    bw_reps;
    int    bw_best_threads;
    double bw_best_gbps;
    double fma_gflops_1t, fma_gflops_all, fma_gflops_logical;
    int    nfma;
    int    fma_threads[HFC_MAX_BW];
    double fma_gflops_n[HFC_MAX_BW];   /* aggregate fp32 FMA throughput at each thread count */
    int    sustained_seconds;         /* 0 = not measured */
    double fma_sustained_gflops;      /* mean over the last quarter of the run */
    double fma_sustained_ratio;       /* sustained / first-second throughput */
    double sustained_trace[HFC_MAX_TRACE];
    int    nsustained;
    int    nlat;
    size_t lat_bytes[HFC_MAX_LAT];
    double lat_ns[HFC_MAX_LAT];
} hfc_machine;

/* quick != 0: fewer thread counts and sizes (about 2 s). reps > 1 repeats the
 * bandwidth sweep round-robin and reports median and spread, because run-to-run
 * variation on a busy machine is 10-40%. */
hfc_status hfc_probe_run(const hfc_cpu *c, int quick, int reps, hfc_machine *m);
hfc_status hfc_probe_load(const char *dir, const hfc_cpu *c, uint64_t mem_total, hfc_machine *m);
hfc_status hfc_probe_save(const char *dir, const hfc_machine *m);
/* All physical cores running the FMA kernel for `seconds` seconds, one
 * throughput sample per second: exposes power and thermal throttling. */
hfc_status hfc_probe_sustained(const hfc_cpu *c, int seconds, hfc_machine *m);
void hfc_probe_print(FILE *f, const hfc_cpu *c, const hfc_machine *m);

#endif
