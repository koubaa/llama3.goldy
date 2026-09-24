#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <errno.h>
#include <io.h>

#include "posix_shim.h"

#define FILETIME_UNIX_EPOCH_100NS 116444736000000000ULL

int clock_gettime(clockid_t clk_id, struct timespec *tp) {
    if (!tp) {
        errno = EINVAL;
        return -1;
    }
    if (clk_id == CLOCK_MONOTONIC) {
        static LARGE_INTEGER freq;
        LARGE_INTEGER now;
        if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&now);
        tp->tv_sec = (time_t) (now.QuadPart / freq.QuadPart);
        tp->tv_nsec = (long) ((now.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
        return 0;
    }
    if (clk_id == CLOCK_REALTIME) {
        FILETIME ft;
        ULARGE_INTEGER t;
        GetSystemTimePreciseAsFileTime(&ft);
        t.LowPart = ft.dwLowDateTime;
        t.HighPart = ft.dwHighDateTime;
        unsigned long long since_epoch = t.QuadPart - FILETIME_UNIX_EPOCH_100NS;
        tp->tv_sec = (time_t) (since_epoch / 10000000ULL);
        tp->tv_nsec = (long) ((since_epoch % 10000000ULL) * 100ULL);
        return 0;
    }
    errno = EINVAL;
    return -1;
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, int64_t offset) {
    (void) addr;
    if (length == 0 || (prot & ~(PROT_READ | PROT_WRITE)) != 0) {
        errno = EINVAL;
        return MAP_FAILED;
    }
    HANDLE file = (HANDLE) _get_osfhandle(fd);
    if (file == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return MAP_FAILED;
    }
    DWORD page_prot = PAGE_READONLY;
    DWORD view_access = FILE_MAP_READ;
    if (prot & PROT_WRITE) {
        if (flags & MAP_PRIVATE) {
            page_prot = PAGE_WRITECOPY;
            view_access = FILE_MAP_COPY;
        } else {
            page_prot = PAGE_READWRITE;
            view_access = FILE_MAP_WRITE;
        }
    }
    unsigned long long end = (unsigned long long) offset + length;
    HANDLE mapping = CreateFileMappingW(file, NULL, page_prot, (DWORD) (end >> 32), (DWORD) end, NULL);
    if (!mapping) {
        errno = EACCES;
        return MAP_FAILED;
    }
    void *view = MapViewOfFile(mapping, view_access, (DWORD) ((unsigned long long) offset >> 32),
                               (DWORD) offset, length);
    CloseHandle(mapping);
    if (!view) {
        errno = ENOMEM;
        return MAP_FAILED;
    }
    return view;
}

int munmap(void *addr, size_t length) {
    (void) length;
    if (!UnmapViewOfFile(addr)) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}
