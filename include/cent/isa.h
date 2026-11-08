#ifndef CENT_ISA_H_
#define CENT_ISA_H_

// Definicao da CENT ISA: constantes fisicas, opcodes e pacote binario de 14 bytes para GDDR6-AiM.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
constexpr size_t   CENT_SRAM_BUFFER_BYTES     = 2U * 1024U * 1024U;
constexpr uint32_t CENT_TOTAL_CXL_DEVICES     = 32U;
constexpr uint32_t CENT_CHANNELS_PER_DEVICE   = 32U;
constexpr uint16_t CENT_BANKS_PER_CHANNEL     = 16U;
constexpr uint16_t CENT_DEFAULT_OPSIZE_BURSTS = 64U;
#else
#define CENT_SRAM_BUFFER_BYTES     (2U * 1024U * 1024U)
#define CENT_TOTAL_CXL_DEVICES     32U
#define CENT_CHANNELS_PER_DEVICE   32U
#define CENT_BANKS_PER_CHANNEL     16U
#define CENT_DEFAULT_OPSIZE_BURSTS 64U
#endif

#ifdef __cplusplus
enum CentOpcode : uint8_t {
#else
enum CentOpcode {
#endif
    CENT_OP_WR_GB   = 0x02,
    CENT_OP_WR_BIAS = 0x03,
    CENT_OP_RD_MAC  = 0x05,
    CENT_OP_MAC_ABK = 0x0F,
    CENT_OP_EOC     = 0xFF
};

#pragma pack(push, 1)
typedef struct CentInstruction {
    uint8_t  opcode;
    uint8_t  dvid;
    uint32_t channel_mask;
    uint16_t bank_index;
    uint32_t row_addr;
    uint16_t opsize;
} CentInstruction;
#pragma pack(pop)

#ifdef __cplusplus
static_assert(sizeof(CentInstruction) == 14,
              "ABI Error: CentInstruction must be exactly 14 bytes packed");
} // extern "C"
#endif

#endif // CENT_ISA_H_
