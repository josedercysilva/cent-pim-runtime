#ifndef CENT_SRAM_BUFFER_HPP_
#define CENT_SRAM_BUFFER_HPP_

// Buffer estatico de SRAM de 2 MB: armazenamento contiguo O(1) de instrucoes e formatacao de dump.

#include <array>
#include <stddef.h>
#include <stdint.h>

#include "cent/isa.h"
#include "cent/runtime_api.h"

namespace cent {

class SramBuffer {
public:
    SramBuffer() noexcept = default;
    ~SramBuffer() noexcept = default;

    SramBuffer(const SramBuffer&) = delete;
    SramBuffer& operator=(const SramBuffer&) = delete;

    void reset() noexcept;
    CentStatus push_instruction(const CentInstruction& inst) noexcept;
    int format_hex_dump(char* out_hex_str, size_t max_len) const noexcept;

    size_t size() const noexcept { return write_offset_; }
    constexpr size_t capacity() const noexcept { return CENT_SRAM_BUFFER_BYTES; }
    const uint8_t* data() const noexcept { return buffer_.data(); }

    uint32_t instruction_count() const noexcept {
        return static_cast<uint32_t>(write_offset_ / sizeof(CentInstruction));
    }

private:
    std::array<uint8_t, CENT_SRAM_BUFFER_BYTES> buffer_{};
    size_t write_offset_{0};
};

SramBuffer& get_global_sram_buffer() noexcept;

} // namespace cent

#endif // CENT_SRAM_BUFFER_HPP_
