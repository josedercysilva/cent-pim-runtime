#ifndef CENT_ISA_EMITTER_HPP_
#define CENT_ISA_EMITTER_HPP_

#include <stdint.h>

#include "cent/runtime_api.h"
#include "cent/sram_buffer.hpp"
#include "cent/telemetry.h"

namespace cent {

/**
 * Decompõe uma operação GEMV (dimensões M x K) em blocos Near-Bank PIM
 * (16 bancos DRAM operando em paralelo via All-Bank MAC), grava os pacotes
 * binários CentInstruction na SRAM de 2 MB, gera opcionalmente o arquivo
 * de traço (.trace) na gramática AiM do Ramulator 2.0 e preenche os contadores
 * de instruções na estrutura de telemetria.
 */
CentStatus emit_gemv_isa(
    int M,
    int K,
    uint8_t dvid,
    uint32_t channel_mask,
    const char* trace_path,
    SramBuffer& sram,
    CentDispatchTelemetry* out_telem
) noexcept;

} // namespace cent

#endif // CENT_ISA_EMITTER_HPP_
