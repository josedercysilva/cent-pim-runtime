#include "cent/runtime_api.h"

#include <stddef.h>
#include <stdint.h>

#include "cent/isa_emitter.hpp"
#include "cent/sram_buffer.hpp"
#include "cent/topology.hpp"

extern "C" {

int cent_init_runtime(void) noexcept {
    cent::get_global_sram_buffer().reset();
    return CENT_OK;
}

int cent_dispatch_gemv(
    const float* x,
    int M,
    int K,
    const float* W,
    float* y,
    int layer_id,
    const char* trace_path,
    CentDispatchTelemetry* out_telem
) noexcept {
    if (x == nullptr || W == nullptr || y == nullptr) {
        return CENT_ERR_NULL_PTR;
    }
    if (M <= 0 || K <= 0) {
        return CENT_ERR_INVALID_SHAPE;
    }

    const uint8_t dvid = cent::compute_dvid(layer_id);
    const uint32_t channel_mask = cent::compute_channel_mask(K);

    const CentStatus emit_status = cent::emit_gemv_isa(
        M,
        K,
        dvid,
        channel_mask,
        trace_path,
        cent::get_global_sram_buffer(),
        out_telem
    );
    if (emit_status != CENT_OK) {
        return emit_status;
    }

    // Cálculo numérico de referência FP64 -> FP32 com localidade espacial de cache (row-major)
    const size_t m_dim = static_cast<size_t>(M);
    const size_t k_dim = static_cast<size_t>(K);

    for (size_t i = 0U; i < m_dim; ++i) {
        const float* w_row = W + (i * k_dim);
        double acc = 0.0;
        for (size_t j = 0U; j < k_dim; ++j) {
            acc += static_cast<double>(x[j]) * static_cast<double>(w_row[j]);
        }
        y[i] = static_cast<float>(acc);
    }

    return CENT_OK;
}

int cent_get_instruction_buffer_hex(char* out_hex_str, size_t max_len) noexcept {
    return cent::get_global_sram_buffer().format_hex_dump(out_hex_str, max_len);
}

} // extern "C"
