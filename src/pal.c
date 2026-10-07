/* pal.c - platform layer. See pal.h. */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "pal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__FreeBSD__)
#include <sys/sysctl.h>
#endif

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

hfc_status pal_map_file(const char *path, pal_map *m)
{
    struct stat st;
    void *p;
    int fd;

    m->base = NULL;
    m->len = 0;
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return HFC_EIO;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) { close(fd); return HFC_EIO; }
    if (st.st_size <= 0) { close(fd); return HFC_EFORMAT; }
    if ((uint64_t)st.st_size > (uint64_t)(size_t)-1) { close(fd); return HFC_ERANGE; }
    p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return errno == ENOMEM ? HFC_ENOMEM : HFC_EIO;
    m->base = p;
    m->len = (size_t)st.st_size;
    return HFC_OK;
}

void pal_unmap(pal_map *m)
{
    if (m->base) munmap(m->base, m->len);
    m->base = NULL;
    m->len = 0;
}

void pal_advise(void *p, size_t len, pal_advice adv)
{
    int a;
    switch (adv) {
    case PAL_ADV_SEQUENTIAL: a = MADV_SEQUENTIAL; break;
    case PAL_ADV_RANDOM:     a = MADV_RANDOM;     break;
    case PAL_ADV_WILLNEED:   a = MADV_WILLNEED;   break;
    case PAL_ADV_DONTNEED:   a = MADV_DONTNEED;   break;
    default:                 a = MADV_NORMAL;     break;
    }
    (void)madvise(p, len, a);
}

size_t pal_page_size(void)
{
    long v = sysconf(_SC_PAGESIZE);
    return v > 0 ? (size_t)v : 4096;
}

double pal_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int pal_ncpu_online(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

/* ---- memory info -------------------------------------------------------- */

#if defined(__linux__)
static int read_small(const char *path, char *buf, size_t cap)
{
    FILE *f = fopen(path, "r");
    size_t n;
    if (!f) return 0;
    n = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[n] = '\0';
    return 1;
}

static uint64_t read_u64_file(const char *path, int *ok)
{
    char buf[64];
    char *end;
    unsigned long long v;
    *ok = 0;
    if (!read_small(path, buf, sizeof buf)) return 0;
    if (strncmp(buf, "max", 3) == 0) return 0;
    v = strtoull(buf, &end, 10);
    if (end == buf) return 0;
    *ok = 1;
    return (uint64_t)v;
}
#endif

hfc_status pal_meminfo_get(pal_meminfo *mi)
{
    memset(mi, 0, sizeof *mi);
    mi->overcommit = -1;
#if defined(__linux__)
    {
        FILE *f = fopen("/proc/meminfo", "r");
        char line[160];
        int ok;
        uint64_t v;
        if (!f) return HFC_EIO;
        while (fgets(line, sizeof line, f)) {
            unsigned long long kb;
            if (sscanf(line, "MemTotal: %llu kB", &kb) == 1)           mi->total = (uint64_t)kb * 1024;
            else if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1)  mi->avail = (uint64_t)kb * 1024;
            else if (sscanf(line, "CommitLimit: %llu kB", &kb) == 1)   mi->commit_limit = (uint64_t)kb * 1024;
            else if (sscanf(line, "Committed_AS: %llu kB", &kb) == 1)  mi->committed = (uint64_t)kb * 1024;
        }
        fclose(f);
        v = read_u64_file("/proc/sys/vm/overcommit_memory", &ok);
        if (ok) mi->overcommit = (int)v;
        /* cgroup v2, then v1 */
        v = read_u64_file("/sys/fs/cgroup/memory.max", &ok);
        if (ok) {
            mi->cgroup_limit = v;
            mi->cgroup_used = read_u64_file("/sys/fs/cgroup/memory.current", &ok);
        } else {
            v = read_u64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes", &ok);
            /* v1 reports a huge number when unlimited */
            if (ok && mi->total && v < mi->total * 4) {
                mi->cgroup_limit = v;
                mi->cgroup_used = read_u64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes", &ok);
            }
        }
        if (mi->total == 0) return HFC_EIO;
        return HFC_OK;
    }
#elif defined(__APPLE__)
    {
        uint64_t memsize = 0;
        size_t sz = sizeof memsize;
        vm_statistics64_data_t vs;
        mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
        if (sysctlbyname("hw.memsize", &memsize, &sz, NULL, 0) != 0) return HFC_EIO;
        mi->total = memsize;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &cnt) == KERN_SUCCESS) {
            uint64_t pages = (uint64_t)vs.free_count + (uint64_t)vs.inactive_count
                           + (uint64_t)vs.purgeable_count;
            mi->avail = pages * (uint64_t)pal_page_size();
        } else {
            mi->avail = memsize / 4;
        }
        return HFC_OK;
    }
#elif defined(__FreeBSD__)
    {
        unsigned long physmem = 0;
        unsigned int freec = 0, inactc = 0, cachec = 0;
        size_t sz = sizeof physmem, s4 = sizeof freec;
        if (sysctlbyname("hw.physmem", &physmem, &sz, NULL, 0) != 0) return HFC_EIO;
        mi->total = physmem;
        (void)sysctlbyname("vm.stats.vm.v_free_count", &freec, &s4, NULL, 0);
        s4 = sizeof inactc;
        (void)sysctlbyname("vm.stats.vm.v_inactive_count", &inactc, &s4, NULL, 0);
        s4 = sizeof cachec;
        (void)sysctlbyname("vm.stats.vm.v_cache_count", &cachec, &s4, NULL, 0);
        mi->avail = ((uint64_t)freec + inactc + cachec) * (uint64_t)pal_page_size();
        return HFC_OK;
    }
#else
    {
        long pages = sysconf(_SC_PHYS_PAGES), ap = sysconf(_SC_AVPHYS_PAGES);
        if (pages <= 0) return HFC_ENOTSUP;
        mi->total = (uint64_t)pages * pal_page_size();
        mi->avail = ap > 0 ? (uint64_t)ap * pal_page_size() : mi->total / 4;
        return HFC_OK;
    }
#endif
}

uint64_t pal_mem_budget(const pal_meminfo *mi)
{
    uint64_t b = mi->avail ? mi->avail : mi->total;
    if (mi->cgroup_limit) {
        uint64_t room = mi->cgroup_limit > mi->cgroup_used ? mi->cgroup_limit - mi->cgroup_used : 0;
        if (room < b) b = room;
    }
    if (mi->overcommit == 2 && mi->commit_limit) {
        uint64_t head = mi->commit_limit > mi->committed ? mi->commit_limit - mi->committed : 0;
        if (head < b) b = head;
    }
    return b;
}

/* ---- files -------------------------------------------------------------- */

char *pal_default_cache_dir(void)
{
    const char *x = getenv("XDG_CACHE_HOME");
    const char *h = getenv("HOME");
    char buf[1024];
    if (x && x[0] == '/') snprintf(buf, sizeof buf, "%s/hfcpu", x);
    else if (h && h[0])   snprintf(buf, sizeof buf, "%s/.cache/hfcpu", h);
    else return NULL;
    return hfc_strdup(buf);
}

hfc_status pal_mkdir_p(const char *path)
{
    char *tmp = hfc_strdup(path);
    char *p;
    struct stat st;
    if (!tmp) return HFC_ENOMEM;
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) { hfc_free(tmp); return HFC_EIO; }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) { hfc_free(tmp); return HFC_EIO; }
    hfc_free(tmp);
    return (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) ? HFC_OK : HFC_EIO;
}

hfc_status pal_write_file_atomic(const char *path, const void *data, size_t len)
{
    char *tmp;
    size_t n;
    int fd;
    const unsigned char *p = (const unsigned char *)data;
    hfc_status rc = HFC_EIO;

    if (!hfc_add_size(strlen(path), 32, &n)) return HFC_ERANGE;
    tmp = (char *)hfc_malloc(n);
    if (!tmp) return HFC_ENOMEM;
    snprintf(tmp, n, "%s.tmp.%ld", path, (long)getpid());
    fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) { hfc_free(tmp); return HFC_EIO; }
    while (len > 0) {
        ssize_t w = write(fd, p, len);
        if (w < 0) { if (errno == EINTR) continue; goto out; }
        p += w;
        len -= (size_t)w;
    }
#if defined(__APPLE__) && defined(F_FULLFSYNC)
    if (fcntl(fd, F_FULLFSYNC) != 0 && fsync(fd) != 0) goto out;
#else
    if (fsync(fd) != 0) goto out;
#endif
    if (close(fd) != 0) { fd = -1; goto out; }
    fd = -1;
    if (rename(tmp, path) != 0) goto out;
    rc = HFC_OK;
out:
    if (fd >= 0) close(fd);
    if (rc != HFC_OK) unlink(tmp);
    hfc_free(tmp);
    return rc;
}
