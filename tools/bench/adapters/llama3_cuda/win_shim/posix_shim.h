/* Minimal POSIX surface used by llama3.cu, implemented in posix_shim.c via Win32.
 * Kept free of <windows.h> so it does not leak min/max/near/far macros into CUDA/CUB. */
#ifndef KOBA_BENCH_POSIX_SHIM_H
#define KOBA_BENCH_POSIX_SHIM_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif

typedef int clockid_t;
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1

/* CLOCK_REALTIME: GetSystemTimePreciseAsFileTime; CLOCK_MONOTONIC: QueryPerformanceCounter. */
int clock_gettime(clockid_t clk_id, struct timespec *tp);

#define PROT_NONE 0x0
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FAILED ((void *) -1)

/* fd must come from _open/open; mapped with CreateFileMapping + MapViewOfFile. */
void *mmap(void *addr, size_t length, int prot, int flags, int fd, int64_t offset);
int munmap(void *addr, size_t length);

#ifdef __cplusplus
}
#endif

#endif
