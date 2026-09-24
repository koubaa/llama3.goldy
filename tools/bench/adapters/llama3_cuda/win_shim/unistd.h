/* Windows stand-in for <unistd.h>: open/close/read from the UCRT plus the shimmed
 * clock_gettime (POSIX puts it in <time.h>, which MSVC does not let us shadow). */
#ifndef KOBA_BENCH_UNISTD_H
#define KOBA_BENCH_UNISTD_H

#include <io.h>
#include <fcntl.h>

#include "posix_shim.h"

#endif
