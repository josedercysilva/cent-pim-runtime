#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cent/isa.h"
#include "cent/runtime_api.h"
#include "cent/telemetry.h"

#define TEST_ASSERT(cond, msg)                                                \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg                    \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define TEST_PASS(msg)                                                        \
    std::cout << "\033[1;32m[PASS]\033[0m " << msg << "\n"

void test_abi_and_packing() {
    TEST_ASSERT(sizeof(CentInstruction) == 14, "CentInstruction deve ter exatamente 14 bytes");
    TEST_ASSERT(offsetof(CentInstruction, opcode) == 0, "offset opcode");
    TEST_ASSERT(offsetof(CentInstruction, dvid) == 1, "offset dvid");
    TEST_ASSERT(offsetof(CentInstruction, channel_mask) == 2, "offset channel_mask");
    TEST_ASSERT(offsetof(CentInstruction, bank_index) == 6, "offset bank_index");
    TEST_ASSERT(offsetof(CentInstruction, row_addr) == 8, "offset row_addr");
    TEST_ASSERT(offsetof(CentInstruction, opsize) == 12, "offset opsize");

    TEST_ASSERT(CENT_OP_WR_GB == 0x02, "CENT_OP_WR_GB deve ser 0x02");
    TEST_ASSERT(CENT_OP_WR_BIAS == 0x03, "CENT_OP_WR_BIAS deve ser 0x03");
    TEST_ASSERT(CENT_OP_RD_MAC == 0x05, "CENT_OP_RD_MAC deve ser 0x05");
    TEST_ASSERT(CENT_OP_MAC_ABK == 0x0F, "CENT_OP_MAC_ABK deve ser 0x0F");
    TEST_ASSERT(CENT_OP_EOC == 0xFF, "CENT_OP_EOC deve ser 0xFF");

    TEST_PASS("ABI e empacotamento da CENT ISA (14 bytes, alinhamentos e opcodes)");
}

void test_guard_clauses() {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float W[8] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    float y[2] = {0.0f, 0.0f};

    TEST_ASSERT(cent_dispatch_gemv(nullptr, 2, 4, W, y, 0, nullptr, nullptr) == CENT_ERR_NULL_PTR,
                "x nulo deve retornar CENT_ERR_NULL_PTR");
    TEST_ASSERT(cent_dispatch_gemv(x, 2, 4, nullptr, y, 0, nullptr, nullptr) == CENT_ERR_NULL_PTR,
                "W nulo deve retornar CENT_ERR_NULL_PTR");
    TEST_ASSERT(cent_dispatch_gemv(x, 2, 4, W, nullptr, 0, nullptr, nullptr) == CENT_ERR_NULL_PTR,
                "y nulo deve retornar CENT_ERR_NULL_PTR");
    TEST_ASSERT(cent_dispatch_gemv(x, 0, 4, W, y, 0, nullptr, nullptr) == CENT_ERR_INVALID_SHAPE,
                "M <= 0 deve retornar CENT_ERR_INVALID_SHAPE");
    TEST_ASSERT(cent_dispatch_gemv(x, 2, -1, W, y, 0, nullptr, nullptr) == CENT_ERR_INVALID_SHAPE,
                "K <= 0 deve retornar CENT_ERR_INVALID_SHAPE");

    TEST_PASS("Guard Clauses e tratamento de erros (zero exceptions, return codes)");
}

void test_numerical_accuracy() {
    const int M = 2;
    const int K = 3;
    const float x[3] = {1.0f, 2.0f, 3.0f};
    const float W[6] = {
        1.0f, 0.0f, 2.0f,  // linha 0: 1*1 + 2*0 + 3*2 = 7.0
        0.0f, 3.0f, 1.0f   // linha 1: 1*0 + 2*3 + 3*1 = 9.0
    };
    float y[2] = {0.0f, 0.0f};

    int status = cent_dispatch_gemv(x, M, K, W, y, 0, nullptr, nullptr);
    TEST_ASSERT(status == CENT_OK, "cent_dispatch_gemv falhou em GEMV simples");

    TEST_ASSERT(std::abs(y[0] - 7.0f) < 1e-5f, "y[0] incorreto");
    TEST_ASSERT(std::abs(y[1] - 9.0f) < 1e-5f, "y[1] incorreto");

    TEST_PASS("Precisao numerica FP64->FP32 com localidade sequencial de cache");
}

void test_llama2_attention_layer() {
    const int M = 4096;
    const int K = 4096;
    const int layer_id = 0;

    std::vector<float> x(K, 0.01f);
    std::vector<float> W(M * K, 0.001f);
    std::vector<float> y(M, 0.0f);

    CentDispatchTelemetry telem{};
    int status = cent_dispatch_gemv(x.data(), M, K, W.data(), y.data(), layer_id, nullptr, &telem);
    TEST_ASSERT(status == CENT_OK, "cent_dispatch_gemv falhou na camada de Atencao Llama-2 7B");

    TEST_ASSERT(telem.dvid == 0, "DVid da camada 0 deve ser 0");
    TEST_ASSERT(telem.channel_mask == 0xFFFFFFFFu, "CHmask para K=4096 deve ser 0xFFFFFFFF");
    TEST_ASSERT(telem.wr_gb_count == 256, "wr_gb_count deve ser 256");
    TEST_ASSERT(telem.wr_bias_count == 256, "wr_bias_count deve ser 256");
    TEST_ASSERT(telem.mac_abk_count == 256, "mac_abk_count deve ser 256");
    TEST_ASSERT(telem.rd_mac_count == 256, "rd_mac_count deve ser 256");
    TEST_ASSERT(telem.total_instructions == 1025, "total_instructions deve ser 1025 (256*4 + 1 EOC)");
    TEST_ASSERT(telem.instruction_buffer_bytes == 14350, "instruction_buffer_bytes deve ser 14350 (1025 * 14)");

    TEST_ASSERT(telem.memory_traffic_saved_pct >= 99.90 && telem.memory_traffic_saved_pct <= 99.99,
                "Economia de memoria na atencao deve ser aproximadamente 99.95%");

    char hex_dump[512] = {0};
    int dumped = cent_get_instruction_buffer_hex(hex_dump, sizeof(hex_dump));
    TEST_ASSERT(dumped == 64, "Dump da SRAM deve formatar os primeiros 64 bytes");
    TEST_ASSERT(hex_dump[0] == '0' && hex_dump[1] == '2', "Primeira instrucao deve ser WR_GB (0x02)");

    TEST_PASS("Validacao Llama-2 7B - Atencao (4Kx4K): 1025 inst, 14350 B na SRAM, DVid=0, ~99.95% economia");
}

void test_llama2_mlp_layer() {
    const int M = 11008;
    const int K = 4096;
    const int layer_id = 1;

    std::vector<float> x(K, 0.01f);
    std::vector<float> W(M * K, 0.001f);
    std::vector<float> y(M, 0.0f);

    CentDispatchTelemetry telem{};
    int status = cent_dispatch_gemv(x.data(), M, K, W.data(), y.data(), layer_id, nullptr, &telem);
    TEST_ASSERT(status == CENT_OK, "cent_dispatch_gemv falhou na camada MLP Llama-2 7B");

    TEST_ASSERT(telem.dvid == 1, "DVid da camada 1 deve ser 1");
    TEST_ASSERT(telem.channel_mask == 0xFFFFFFFFu, "CHmask para K=4096 deve ser 0xFFFFFFFF");
    TEST_ASSERT(telem.wr_gb_count == 688, "wr_gb_count deve ser 688 (11008 / 16)");
    TEST_ASSERT(telem.wr_bias_count == 688, "wr_bias_count deve ser 688");
    TEST_ASSERT(telem.mac_abk_count == 688, "mac_abk_count deve ser 688");
    TEST_ASSERT(telem.rd_mac_count == 688, "rd_mac_count deve ser 688");
    TEST_ASSERT(telem.total_instructions == 2753, "total_instructions deve ser 2753 (688*4 + 1 EOC)");
    TEST_ASSERT(telem.instruction_buffer_bytes == 38542, "instruction_buffer_bytes deve ser 38542 (2753 * 14)");

    TEST_PASS("Validacao Llama-2 7B - MLP (11Kx4K): 2753 inst, 38542 B na SRAM, DVid=1");
}

void test_trace_generation() {
    const int M = 16;   // 16 bancos = 1 bloco PIM (4 inst) + 1 EOC = 5 linhas
    const int K = 1024;
    const float x[1024] = {0.0f};
    std::vector<float> W(16 * 1024, 0.0f);
    float y[16] = {0.0f};

    const char* trace_file = "build/test_sample.trace";
    int status = cent_dispatch_gemv(x, M, K, W.data(), y, 0, trace_file, nullptr);
    TEST_ASSERT(status == CENT_OK, "Falha ao gerar trace");

    std::ifstream file(trace_file);
    TEST_ASSERT(file.is_open(), "Arquivo .trace nao foi criado");

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    file.close();

    TEST_ASSERT(lines.size() == 5, "Arquivo .trace para M=16 deve conter exatamente 5 linhas");
    TEST_ASSERT(lines[0] == "AiM WR_GB 64 0 0xffffffff", "Linha 0 incorreta no .trace");
    TEST_ASSERT(lines[1] == "AiM WR_BIAS 0 0xffffffff", "Linha 1 incorreta no .trace");
    TEST_ASSERT(lines[2] == "AiM MAC_ABK 64 0xffffffff 0", "Linha 2 incorreta no .trace");
    TEST_ASSERT(lines[3] == "AiM RD_MAC 0 0xffffffff", "Linha 3 incorreta no .trace");
    TEST_ASSERT(lines[4] == "AiM EOC", "Linha 4 incorreta no .trace (deve ser AiM EOC)");

    std::remove(trace_file);
    TEST_PASS("Sintaxe e emissao de .trace compatível com Ramulator 2.0 (4 comandos + EOC)");
}

int main() {
    std::cout << "\n======================================================\n";
    std::cout << "  CENT Runtime - Suíte de Testbench C++ / C ABI Nativa\n";
    std::cout << "======================================================\n\n";

    test_abi_and_packing();
    test_guard_clauses();
    test_numerical_accuracy();
    test_llama2_attention_layer();
    test_llama2_mlp_layer();
    test_trace_generation();

    std::cout << "\n\033[1;32mTODOS OS TESTES DO MOTOR C/C++ PASSARAM COM SUCESSO!\033[0m\n\n";
    return 0;
}
