/* Pre-included (nvcc -include) so the pinned llama3.cu builds unchanged against CCCL 3.x
 * (CUDA 13), which removed cub::Max. No-op on older CCCL where cub::Max still exists. */
#ifndef KOBA_BENCH_CCCL_COMPAT_H
#define KOBA_BENCH_CCCL_COMPAT_H

#include <cub/version.cuh>

#if CUB_VERSION >= 300000
namespace cub {
struct Max {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a < b ? b : a;
    }
};
}  // namespace cub
#endif

#endif
