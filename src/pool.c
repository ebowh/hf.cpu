/* pool.c - fixed thread pool. See pool.h. */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "pool.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#define LOAD(p)      __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define STORE(p, v)  __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#define FETCH_ADD(p) __atomic_fetch_add((p), 1, __ATOMIC_ACQ_REL)
#else
#error "hfcpu needs GCC or Clang atomics for the thread pool"
#endif

#if defined(__x86_64__) || defined(__i386__)
#define RELAX() __asm__ volatile("pause" ::: "memory")
#elif defined(__aarch64__)
#define RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define RELAX() ((void)0)
#endif

#define SPIN_WORKER 40000      /* iterations of pause before a worker sleeps */
#define SPIN_CALLER 200000     /* iterations before the caller starts yielding */

struct hfc_pool {
    int              n;
    pthread_t       *th;
    int              started;           /* worker threads actually created */
    pthread_mutex_t  mu;
    pthread_cond_t   cv;
    volatile unsigned gen;              /* bumped for each job */
    volatile int      done;             /* workers finished with the current job */
    volatile int      quit;
    int               sleeping;
    hfc_task_fn       fn;
    void             *ctx;
};

typedef struct { hfc_pool *p; int tid; } worker_arg;

void hfc_split(size_t n, int tid, int nth, size_t *lo, size_t *hi)
{
    *lo = n * (size_t)tid / (size_t)nth;
    *hi = n * (size_t)(tid + 1) / (size_t)nth;
}

static void *worker_main(void *a)
{
    worker_arg *wa = (worker_arg *)a;
    hfc_pool *p = wa->p;
    int tid = wa->tid;
    unsigned my_gen = 0;
    hfc_task_fn fn;
    void *ctx;
    free(wa);
    for (;;) {
        int i, got = 0;
        for (i = 0; i < SPIN_WORKER; i++) {
            if (LOAD(&p->gen) != my_gen || LOAD(&p->quit)) { got = 1; break; }
            RELAX();
        }
        if (!got) {
            pthread_mutex_lock(&p->mu);
            p->sleeping++;
            while (LOAD(&p->gen) == my_gen && !LOAD(&p->quit)) pthread_cond_wait(&p->cv, &p->mu);
            p->sleeping--;
            pthread_mutex_unlock(&p->mu);
        }
        if (LOAD(&p->quit)) break;
        my_gen = LOAD(&p->gen);
        fn = p->fn;
        ctx = p->ctx;
        fn(ctx, tid, p->n);
        FETCH_ADD(&p->done);
    }
    return NULL;
}

hfc_status hfc_pool_new(hfc_pool **out, int nthreads)
{
    hfc_pool *p;
    int i;
    *out = NULL;
    if (nthreads < 1 || nthreads > 256) return HFC_EINVAL;
    p = (hfc_pool *)hfc_calloc(1, sizeof *p);
    if (!p) return HFC_ENOMEM;
    p->n = nthreads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    if (nthreads > 1) {
        p->th = (pthread_t *)hfc_calloc((size_t)nthreads, sizeof(pthread_t));
        if (!p->th) { hfc_pool_free(p); return HFC_ENOMEM; }
        for (i = 1; i < nthreads; i++) {
            worker_arg *wa = (worker_arg *)malloc(sizeof *wa);
            if (!wa) break;
            wa->p = p; wa->tid = i;
            if (pthread_create(&p->th[i], NULL, worker_main, wa) != 0) { free(wa); break; }
            p->started++;
        }
        if (p->started != nthreads - 1) { hfc_pool_free(p); return HFC_ENOMEM; }
    }
    *out = p;
    return HFC_OK;
}

void hfc_pool_free(hfc_pool *p)
{
    int i;
    if (!p) return;
    if (p->started) {
        pthread_mutex_lock(&p->mu);
        STORE(&p->quit, 1);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
        for (i = 1; i <= p->started; i++) pthread_join(p->th[i], NULL);
    }
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    hfc_free(p->th);
    hfc_free(p);
}

int hfc_pool_size(const hfc_pool *p) { return p ? p->n : 1; }

void hfc_pool_run(hfc_pool *p, hfc_task_fn fn, void *ctx)
{
    int i;
    if (!p || p->n == 1) { fn(ctx, 0, 1); return; }
    p->fn = fn;
    p->ctx = ctx;
    STORE(&p->done, 0);
    pthread_mutex_lock(&p->mu);
    STORE(&p->gen, p->gen + 1);
    if (p->sleeping) pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    fn(ctx, 0, p->n);
    for (i = 0; LOAD(&p->done) != p->n - 1; i++) {
        if (i < SPIN_CALLER) RELAX(); else sched_yield();
    }
}
