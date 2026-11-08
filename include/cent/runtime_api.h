#ifndef CENT_RUNTIME_API_H_
#define CENT_RUNTIME_API_H_

// Interface C ABI publica da libcent_runtime: despacho de GEMV, inicializacao e hex dump.

#include <stddef.h>
#include "cent/telemetry.h"

#ifdef __cplusplus
extern "C" {
#define CENT_NOEXCEPT noexcept
#else
#define CENT_NOEXCEPT
#endif

enum CentStatus {
    CENT_OK                =  0,
    CENT_ERR_NULL_PTR      = -1,
    CENT_ERR_INVALID_SHAPE = -2,
    CENT_ERR_SRAM_OVERFLOW = -3,
    CENT_ERR_TRACE_IO      = -4
};

int cent_init_runtime(void) CENT_NOEXCEPT;

int cent_dispatch_gemv(
    const float* x,
    int M,
    int K,
    const float* W,
    float* y,
    int layer_id,
    const char* trace_path,
    CentDispatchTelemetry* out_telem
) CENT_NOEXCEPT;

int cent_get_instruction_buffer_hex(char* out_hex_str, size_t max_len) CENT_NOEXCEPT;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CENT_RUNTIME_API_H_
