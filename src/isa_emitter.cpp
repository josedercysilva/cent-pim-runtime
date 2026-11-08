#include "cent/isa_emitter.hpp"

#include <cstdio>
#include <fstream>

#include "cent/isa.h"

namespace cent {

CentStatus emit_gemv_isa(
    int M,
    int K,
    uint8_t dvid,
    uint32_t channel_mask,
    const char* trace_path,
    SramBuffer& sram,
    CentDispatchTelemetry* out_telem
) noexcept {
    if (M <= 0 || K <= 0) {
        return CENT_ERR_INVALID_SHAPE;
    }

    sram.reset();

    constexpr int kBanksPerChannel = static_cast<int>(CENT_BANKS_PER_CHANNEL);
    const int num_blocks = (M + kBanksPerChannel - 1) / kBanksPerChannel;

    const bool write_trace = (trace_path != nullptr && trace_path[0] != '\0');
    std::ofstream trace_file;
    char ch_mask_str[16] = {0};

    if (write_trace) {
        trace_file.open(trace_path, std::ios::out | std::ios::trunc);
        if (!trace_file.is_open()) {
            return CENT_ERR_TRACE_IO;
        }
        std::snprintf(ch_mask_str, sizeof(ch_mask_str), "0x%08x", channel_mask);
    }

    for (int blk = 0; blk < num_blocks; ++blk) {
        const uint32_t row_addr = static_cast<uint32_t>(blk);

        const CentInstruction wr_gb_inst{
            CENT_OP_WR_GB, dvid, channel_mask, 0U, 0U, CENT_DEFAULT_OPSIZE_BURSTS
        };
        const CentInstruction wr_bias_inst{
            CENT_OP_WR_BIAS, dvid, channel_mask, 0U, 0U, 0U
        };
        const CentInstruction mac_abk_inst{
            CENT_OP_MAC_ABK, dvid, channel_mask, 0U, row_addr, CENT_DEFAULT_OPSIZE_BURSTS
        };
        const CentInstruction rd_mac_inst{
            CENT_OP_RD_MAC, dvid, channel_mask, 0U, 0U, 0U
        };

        CentStatus st = sram.push_instruction(wr_gb_inst);
        if (st != CENT_OK) {
            return st;
        }
        st = sram.push_instruction(wr_bias_inst);
        if (st != CENT_OK) {
            return st;
        }
        st = sram.push_instruction(mac_abk_inst);
        if (st != CENT_OK) {
            return st;
        }
        st = sram.push_instruction(rd_mac_inst);
        if (st != CENT_OK) {
            return st;
        }

        if (write_trace) {
            trace_file << "AiM WR_GB 64 0 " << ch_mask_str << '\n'
                       << "AiM WR_BIAS 0 " << ch_mask_str << '\n'
                       << "AiM MAC_ABK 64 " << ch_mask_str << ' ' << row_addr << '\n'
                       << "AiM RD_MAC 0 " << ch_mask_str << '\n';
            if (!trace_file.good()) {
                return CENT_ERR_TRACE_IO;
            }
        }
    }

    const CentInstruction eoc_inst{
        CENT_OP_EOC, dvid, channel_mask, 0U, 0U, 0U
    };
    const CentStatus eoc_status = sram.push_instruction(eoc_inst);
    if (eoc_status != CENT_OK) {
        return eoc_status;
    }

    if (write_trace) {
        trace_file << "AiM EOC\n";
        trace_file.flush();
        if (!trace_file.good()) {
            return CENT_ERR_TRACE_IO;
        }
    }

    if (out_telem != nullptr) {
        const uint32_t blocks_u32 = static_cast<uint32_t>(num_blocks);
        out_telem->dvid = dvid;
        out_telem->channel_mask = channel_mask;
        out_telem->total_instructions = sram.instruction_count();
        out_telem->instruction_buffer_bytes = static_cast<uint32_t>(sram.size());
        out_telem->wr_gb_count = blocks_u32;
        out_telem->wr_bias_count = blocks_u32;
        out_telem->mac_abk_count = blocks_u32;
        out_telem->rd_mac_count = blocks_u32;

        const double m_f64 = static_cast<double>(M);
        const double k_f64 = static_cast<double>(K);
        out_telem->gpu_traffic_bytes = (m_f64 * k_f64 + k_f64) * 2.0;
        out_telem->pcie_traffic_bytes = (k_f64 + m_f64) * 2.0;
        out_telem->memory_traffic_saved_pct =
            100.0 * (1.0 - (out_telem->pcie_traffic_bytes / out_telem->gpu_traffic_bytes));
    }

    return CENT_OK;
}

} // namespace cent
