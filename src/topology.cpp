#include "cent/topology.hpp"
#include "cent/isa.h"

namespace cent {

uint8_t compute_dvid(int layer_id) noexcept {
    constexpr int kTotalDevices = static_cast<int>(CENT_TOTAL_CXL_DEVICES);
    return static_cast<uint8_t>(((layer_id % kTotalDevices) + kTotalDevices) % kTotalDevices);
}

uint32_t compute_channel_mask(int K) noexcept {
    if (K >= 1024) {
        return 0xFFFFFFFFu;
    }
    if (K >= 512) {
        return 0x0000FFFFu;
    }
    return 0x000000FFu;
}

} // namespace cent
