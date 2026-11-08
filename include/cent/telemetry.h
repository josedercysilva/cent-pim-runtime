#ifndef CENT_TELEMETRY_H_
#define CENT_TELEMETRY_H_

// Telemetria de despacho: contadores de instrucao e comparativo de trafego CXL vs. GPU.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CentDispatchTelemetry {
    uint8_t  dvid;
    uint32_t channel_mask;
    uint32_t total_instructions;
    uint32_t instruction_buffer_bytes;
    uint32_t wr_gb_count;
    uint32_t wr_bias_count;
    uint32_t mac_abk_count;
    uint32_t rd_mac_count;
    double   memory_traffic_saved_pct;
    double   pcie_traffic_bytes;
    double   gpu_traffic_bytes;
} CentDispatchTelemetry;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CENT_TELEMETRY_H_
