/* pool.h - fixed thread pool for data-parallel loops.
 *
 * One job at a time: hfc_pool_run() calls fn(ctx, tid, nthreads) on every
 * thread (tid 0 is the caller) and returns when all have finished. Workers
 * spin briefly between jobs, then sleep, so back-to-back matrix products pay
 * microseconds, not a wake-up each.
 *
 * Work is always split by output element, never by reduction dimension, so
 * results are bit-identical for any thread count.
 */
#ifndef HFC_POOL_H
#define HFC_POOL_H

#include "hfc.h"

typedef struct hfc_pool hfc_pool;
typedef void (*hfc_task_fn)(void *ctx, int tid, int nthreads);

/* nthreads includes the calling thread; 1 means no worker threads. */
hfc_status hfc_pool_new(hfc_pool **p, int nthreads);
void       hfc_pool_free(hfc_pool *p);
int        hfc_pool_size(const hfc_pool *p);
void       hfc_pool_run(hfc_pool *p, hfc_task_fn fn, void *ctx);

/* Contiguous share [*lo, *hi) of n items for thread tid of nth. */
void hfc_split(size_t n, int tid, int nth, size_t *lo, size_t *hi);

#endif
