#define _GNU_SOURCE
#include <cuda.h>
#include <fcntl.h>
#include <link.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* glibc's audit namespace observes dlopen/dlsym without replacing libcuda. */
typedef CUresult (*ResolverV1)(const char *, void **, int, cuuint64_t);
typedef CUresult (*ResolverV2)(const char *, void **, int, cuuint64_t,
                               CUdriverProcAddressQueryResult *);
static ResolverV1 resolve_v1;
static ResolverV2 resolve_v2;
static int trace_fd = -1;
static _Atomic unsigned long sequence;
static _Atomic unsigned long write_errors;
static CUresult audited_resolve_v1(const char *, void **, int, cuuint64_t);
static CUresult audited_resolve_v2(const char *, void **, int, cuuint64_t,
                                  CUdriverProcAddressQueryResult *);

static void wrap_resolver(const char *name, void **fn, int version, CUresult result) {
    if (result != CUDA_SUCCESS || !*fn || strcmp(name, "cuGetProcAddress") != 0)
        return;
    if (version >= 12000) {
        resolve_v2 = (ResolverV2)*fn;
        *fn = (void *)audited_resolve_v2;
    } else {
        resolve_v1 = (ResolverV1)*fn;
        *fn = (void *)audited_resolve_v1;
    }
}

static void record(const char *kind, const char *name, uintptr_t address,
                   int version, unsigned long long flags, int result) {
    if (trace_fd < 0) return;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    char line[2048];
    int size = snprintf(line, sizeof(line),
        "{\"seq\":%lu,\"pid\":%d,\"ns\":%llu,\"kind\":\"%s\","
        "\"name\":\"%s\",\"address\":%lu,\"version\":%d,"
        "\"flags\":%llu,\"result\":%d}\n",
        atomic_fetch_add(&sequence, 1), getpid(),
        (unsigned long long)now.tv_sec * 1000000000ULL + now.tv_nsec,
        kind, name, (unsigned long)address, version, flags, result);
    if (size < 0 || (size_t)size >= sizeof(line) ||
        write(trace_fd, line, (size_t)size) != size)
        atomic_fetch_add(&write_errors, 1);
}

static CUresult audited_resolve_v1(const char *name, void **fn, int version,
                                  cuuint64_t flags) {
    CUresult result = resolve_v1(name, fn, version, flags);
    record("resolve", name, result == CUDA_SUCCESS ? (uintptr_t)*fn : 0,
           version, flags, result);
    wrap_resolver(name, fn, version, result);
    return result;
}

static CUresult audited_resolve_v2(const char *name, void **fn, int version,
                                  cuuint64_t flags,
                                  CUdriverProcAddressQueryResult *status) {
    CUresult result = resolve_v2(name, fn, version, flags, status);
    record("resolve_v2", name, result == CUDA_SUCCESS ? (uintptr_t)*fn : 0,
           version, flags, result);
    wrap_resolver(name, fn, version, result);
    return result;
}

unsigned int la_version(unsigned int version) {
    const char *path = getenv("SIM_LOADER_TRACE");
    if (path) trace_fd = open(path, O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0600);
    return version < LAV_CURRENT ? version : LAV_CURRENT;
}

unsigned int la_objopen(struct link_map *map, Lmid_t namespace_id,
                       uintptr_t *cookie) {
    (void)namespace_id;
    (void)cookie;
    if (strstr(map->l_name, "libcuda") || strstr(map->l_name, "libnccl") ||
        strstr(map->l_name, "libnvidia"))
        record("library", map->l_name, map->l_addr, 0, 0, 0);
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

uintptr_t la_symbind64(Elf64_Sym *symbol, unsigned int index,
                      uintptr_t *ref_cookie, uintptr_t *def_cookie,
                      unsigned int *flags, const char *name) {
    (void)index;
    (void)ref_cookie;
    (void)def_cookie;
    (void)flags;
    uintptr_t address = symbol->st_value;
    if (strncmp(name, "cu", 2) == 0 || strncmp(name, "nccl", 4) == 0 ||
        strncmp(name, "nvml", 4) == 0)
        record("binding", name, address, 0, 0, 0);
    if (strcmp(name, "cuGetProcAddress") == 0) {
        resolve_v1 = (ResolverV1)address;
        return (uintptr_t)audited_resolve_v1;
    }
    if (strcmp(name, "cuGetProcAddress_v2") == 0) {
        resolve_v2 = (ResolverV2)address;
        return (uintptr_t)audited_resolve_v2;
    }
    return address;
}

__attribute__((destructor)) static void finish(void) {
    record("summary", "write_errors", 0, 0, atomic_load(&write_errors), 0);
    if (trace_fd >= 0) close(trace_fd);
}
