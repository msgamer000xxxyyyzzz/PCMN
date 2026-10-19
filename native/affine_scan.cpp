// PCMN v0.4: optional portable CPU inference kernel.
// C ABI avoids torch / pybind dependency. Inputs are contiguous [B,T,D] float32.
// Not for training: output has no autograd backward. Build with CMake.
#include <cstddef>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_WIN32)
  #define PCMN_API extern "C" __declspec(dllexport)
#else
  #define PCMN_API extern "C" __attribute__((visibility("default")))
#endif
PCMN_API int pcmn_scan_f32(const float* a, const float* b, const float* initial,
                            float* output, int64_t B, int64_t T, int64_t D) {
    if (!a || !b || !output || B <= 0 || T <= 0 || D <= 0) return -1;
    // Batch parallelism; inner feature loop contiguous / autovectorizable.
    #pragma omp parallel for schedule(static) if (B > 1)
    for (int64_t batch = 0; batch < B; ++batch) {
        for (int64_t j = 0; j < D; ++j)
            output[(batch * T) * D + j] = a[(batch * T) * D + j] *
                (initial ? initial[batch * D + j] : 0.f) + b[(batch * T) * D + j];
        for (int64_t t = 1; t < T; ++t) {
            size_t offset = (static_cast<size_t>(batch) * T + t) * D;
            for (int64_t j = 0; j < D; ++j)
                output[offset + j] = a[offset + j] * output[offset - D + j] + b[offset + j];
        }
    }
    return 0;
}
