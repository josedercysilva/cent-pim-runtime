#include "cent/sram_buffer.hpp"

#include <cstring>

namespace cent {

void SramBuffer::reset() noexcept {
    write_offset_ = 0U;
}

CentStatus SramBuffer::push_instruction(const CentInstruction& inst) noexcept {
    if (write_offset_ + sizeof(CentInstruction) > CENT_SRAM_BUFFER_BYTES) {
        return CENT_ERR_SRAM_OVERFLOW;
    }
    std::memcpy(buffer_.data() + write_offset_, &inst, sizeof(CentInstruction));
    write_offset_ += sizeof(CentInstruction);
    return CENT_OK;
}

int SramBuffer::format_hex_dump(char* out_hex_str, size_t max_len) const noexcept {
    if (out_hex_str == nullptr) {
        return CENT_ERR_NULL_PTR;
    }
    if (max_len == 0U) {
        return CENT_ERR_INVALID_SHAPE;
    }

    constexpr size_t kMaxDumpBytes = 64U;
    constexpr char kHexDigits[] = "0123456789ABCDEF";

    const size_t bytes_to_dump = (write_offset_ < kMaxDumpBytes) ? write_offset_ : kMaxDumpBytes;
    if (bytes_to_dump == 0U) {
        out_hex_str[0] = '\0';
        return 0;
    }

    size_t pos = 0U;
    size_t dumped_bytes = 0U;

    for (size_t i = 0U; i < bytes_to_dump; ++i) {
        if (pos + 2U >= max_len) {
            break;
        }
        const uint8_t byte_val = buffer_[i];
        out_hex_str[pos++] = kHexDigits[(byte_val >> 4U) & 0x0FU];
        out_hex_str[pos++] = kHexDigits[byte_val & 0x0FU];
        ++dumped_bytes;

        if (i + 1U < bytes_to_dump) {
            if (pos + 1U >= max_len) {
                break;
            }
            out_hex_str[pos++] = (((i + 1U) % 16U) == 0U) ? '\n' : ' ';
        }
    }

    out_hex_str[pos] = '\0';
    return static_cast<int>(dumped_bytes);
}

SramBuffer& get_global_sram_buffer() noexcept {
    static SramBuffer instance;
    return instance;
}

} // namespace cent
