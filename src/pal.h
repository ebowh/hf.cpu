/* pal.h - platform abstraction layer (Linux, macOS, FreeBSD).
 *
 * All #ifdef lives in pal.c / probe.c. Everything here has a portable
 * fallback that is correct if slower. Linux is the primary, tested port;
 * the macOS and FreeBSD branches are written from the documented APIs and
 * need testing on those systems.
 */
#ifndef HFC_PAL_H
#define HFC_PAL_H

#include "hfc.h"

typedef struct {
    void  *base;
    size_t len;
} pal_map;

/* Map a regular file read-only. Read-only shared file mappings are not
 * charged against the commit limit, so this works under strict overcommit. */
hfc_status pal_map_file(const char *path, pal_map *m);
void       pal_unmap(pal_map *m);

typedef enum { PAL_ADV_NORMAL, PAL_ADV_SEQUENTIAL, PAL_ADV_RANDOM,
               PAL_ADV_WILLNEED, PAL_ADV_DONTNEED } pal_advice;
void   pal_advise(void *p, size_t len, pal_advice adv);   /* best effort */

size_t pal_page_size(void);
double pal_now(void);              /* monotonic seconds */
int    pal_ncpu_online(void);

typedef struct {
    uint64_t total;          /* physical RAM, bytes (0 if unknown) */
    uint64_t avail;          /* reclaimable without swapping, bytes */
    uint64_t commit_limit;   /* strict-overcommit limit (0 if n/a) */
    uint64_t committed;      /* currently committed (0 if n/a) */
    int      overcommit;     /* Linux vm.overcommit_memory (0,1,2), -1 unknown */
    uint64_t cgroup_limit;   /* container limit, 0 = none */
    uint64_t cgroup_used;
} pal_meminfo;

hfc_status pal_meminfo_get(pal_meminfo *mi);

/* Bytes the process may expect to allocate: min of avail and cgroup room,
 * and under strict overcommit also the free commit headroom. */
uint64_t pal_mem_budget(const pal_meminfo *mi);

/* Default per-user cache directory (heap string, caller frees), or NULL. */
char *pal_default_cache_dir(void);

/* mkdir -p; returns HFC_OK if the directory exists afterwards. */
hfc_status pal_mkdir_p(const char *path);

/* Durable replace: write len bytes to path via temp file + fsync + rename. */
hfc_status pal_write_file_atomic(const char *path, const void *data, size_t len);

#endif
