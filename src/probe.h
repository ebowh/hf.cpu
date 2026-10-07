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

typedef struct {
    char   sig[17];
    char   isa[16];                  /* kernel set used for the probe */
    long   when;                     /* unix time of the measurement */
    int    nbw;
    int    bw_threads[HFC_MAX_BW];
    double bw_gbps[HFC_MAX_BW];      /* aggregate read bandwidth */
    int    bw_best_threads;
    double bw_best_gbps;
    double fma_gflops_1t, fma_gflops_all;
    int    nlat;
    size_t lat_bytes[HFC_MAX_LAT];
    double lat_ns[HFC_MAX_LAT];
} hfc_machine;

/* quick != 0: fewer thread counts and sizes (about 2 s). */
hfc_status hfc_probe_run(const hfc_cpu *c, int quick, hfc_machine *m);
hfc_status hfc_probe_load(const char *dir, const hfc_cpu *c, uint64_t mem_total, hfc_machine *m);
hfc_status hfc_probe_save(const char *dir, const hfc_machine *m);
void hfc_probe_print(FILE *f, const hfc_cpu *c, const hfc_machine *m);

#endif
