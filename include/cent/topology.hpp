#ifndef CENT_TOPOLOGY_HPP_
#define CENT_TOPOLOGY_HPP_

#include <stdint.h>

namespace cent {

/**
 * Mapeia o índice da camada Transformer (layer_id) para um dos 32 dispositivos
 * CXL 3.0 (DVid no intervalo [0, 31]) usando aritmética modular segura.
 */
uint8_t compute_dvid(int layer_id) noexcept;

/**
 * Determina a máscara binária de 32 bits dos canais GDDR6-AiM ativos (CHmask)
 * com base na dimensão oculta de entrada K:
 *  - K >= 1024 : 0xFFFFFFFFu (32 canais ativos)
 *  - K >= 512  : 0x0000FFFFu (16 canais ativos)
 *  - K <  512  : 0x000000FFu (8 canais ativos)
 */
uint32_t compute_channel_mask(int K) noexcept;

} // namespace cent

#endif // CENT_TOPOLOGY_HPP_
