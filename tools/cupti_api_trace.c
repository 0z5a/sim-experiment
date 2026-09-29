#define _GNU_SOURCE
#include <cupti.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static CUpti_SubscriberHandle subscriber;
static int trace_fd = -1;
static _Atomic unsigned long sequence;
static _Atomic unsigned long write_errors;
static int subscribed;

static void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                             CUpti_CallbackId id, const void *data) {
    (void)userdata;
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const CUpti_CallbackData *call = data;
    if (call->callbackSite != CUPTI_API_EXIT) return;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    char line[512];
    int size = snprintf(line, sizeof(line),
        "{\"seq\":%lu,\"ns\":%llu,\"cbid\":%u,\"name\":\"%s\","
        "\"correlation\":%u,\"context\":%u,\"result\":%d}\n",
        atomic_fetch_add(&sequence, 1),
        (unsigned long long)now.tv_sec * 1000000000ULL + now.tv_nsec,
        id, call->functionName, call->correlationId, call->contextUid,
        *(const CUresult *)call->functionReturnValue);
    if (size < 0 || (size_t)size >= sizeof(line) ||
        write(trace_fd, line, (size_t)size) != size)
        atomic_fetch_add(&write_errors, 1);
}

__attribute__((constructor)) static void start(void) {
    const char *path = getenv("SIM_CUPTI_TRACE");
    if (!path) return;
    trace_fd = open(path, O_CREAT | O_WRONLY | O_EXCL | O_CLOEXEC, 0600);
    if (trace_fd < 0) {
        perror("SIM_CUPTI_TRACE");
        return;
    }
    CUptiResult status = cuptiSubscribe(&subscriber, callback, NULL);
    if (status == CUPTI_SUCCESS) {
        subscribed = 1;
        status = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    }
    dprintf(trace_fd, "{\"kind\":\"startup\",\"status\":%d}\n", status);
}

__attribute__((destructor)) static void finish(void) {
    if (subscribed) cuptiUnsubscribe(subscriber);
    if (trace_fd >= 0) {
        dprintf(trace_fd, "{\"kind\":\"summary\",\"calls\":%lu,\"write_errors\":%lu}\n",
                atomic_load(&sequence), atomic_load(&write_errors));
        close(trace_fd);
    }
}
