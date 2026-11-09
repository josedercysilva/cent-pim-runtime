"""
Suíte oficial de testes automatizados (unittest) para a biblioteca cent_runtime.
Cobre:
  - Integridade da C ABI e empacotamento da estrutura de telemetria;
  - Precisão numérica da multiplicação Vetor-Matriz (GEMV) vs. NumPy e PyTorch;
  - Validação defensiva (rejeição de batch_size > 1 / GEMM, ranks inválidos e dimensões incompatíveis);
  - Métricas microarquiteturais exatas do Llama-2 7B (Atenção 4Kx4K e MLP 11Kx4K);
  - Buffer de instruções SRAM de 2 MB e formatação do Hex Dump;
  - Gramática e emissão do traço (.trace) da CENT ISA;
  - Parser estruturado de estatísticas do simulador Ramulator 2.0.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import cent_runtime
from cent_runtime._bindings import CentDispatchTelemetry, get_instruction_buffer_hex
from cent_runtime.simulator import (
    parse_simulator_output,
    resolve_config_path,
    resolve_ramulator_bin,
    simulate_on_ramulator,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


class TestCentRuntime(unittest.TestCase):
    """Testes unitários cobrindo todos os módulos do cent_runtime."""

    def test_abi_and_telemetry_structure(self):
        """Valida tamanho e alinhamento C ABI x86_64 da struct de telemetria."""
        self.assertEqual(ctypes.sizeof(CentDispatchTelemetry), 56)
        telem = CentDispatchTelemetry()
        telem.dvid = 3
        telem.channel_mask = 0xFFFFFFFF
        telem.total_instructions = 100
        d = telem.to_dict()
        self.assertEqual(d["dvid"], 3)
        self.assertEqual(d["channel_mask_hex"], "0xffffffff")
        self.assertEqual(d["total_instructions"], 100)

    def test_numerical_precision_numpy_1d(self):
        """Valida y = x * W^T para vetor de entrada 1D (K,) contra NumPy."""
        rng = np.random.default_rng(42)
        M, K = 32, 64
        x = rng.standard_normal(K).astype(np.float32)
        W = rng.standard_normal((M, K)).astype(np.float32)

        y, telem = cent_runtime.matmul(x, W, layer_id=2)

        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.shape, (M,))
        expected = W @ x
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(telem["dvid"], 2)

    def test_numerical_precision_numpy_2d_batch1(self):
        """Valida y = x * W^T para vetor de entrada 2D (1, K) preservando rank 2."""
        rng = np.random.default_rng(123)
        M, K = 16, 128
        x = rng.standard_normal((1, K)).astype(np.float32)
        W = rng.standard_normal((M, K)).astype(np.float32)

        y, telem = cent_runtime.matmul(x, W, layer_id=5)

        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape, (1, M))
        expected = x @ W.T
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(telem["dvid"], 5)

    @unittest.skipUnless(_HAS_TORCH, "PyTorch não está instalado no ambiente")
    def test_torch_tensor_support(self):
        """Valida transparência com torch.Tensor preservando tipo e dispositivo."""
        M, K = 16, 32
        x = torch.randn(1, K, dtype=torch.float32)
        W = torch.randn(M, K, dtype=torch.float32)

        y, telem = cent_runtime.matmul(x, W)

        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, (1, M))
        expected = torch.matmul(x, W.T)
        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)

    def test_defensive_batch_size_rejection(self):
        """Rejeita execuções com batch_size > 1 explicando a restrição de hardware GEMV."""
        W = np.ones((16, 32), dtype=np.float32)
        x_batch2 = np.ones((2, 32), dtype=np.float32)
        x_batch32 = np.ones((32, 32), dtype=np.float32)

        with self.assertRaises(ValueError) as ctx2:
            cent_runtime.matmul(x_batch2, W)
        self.assertIn("GEMV", str(ctx2.exception))
        self.assertIn("batch_size=1", str(ctx2.exception))

        with self.assertRaises(ValueError) as ctx32:
            cent_runtime.matmul(x_batch32, W)
        self.assertIn("batch_size=32", str(ctx32.exception))

    def test_defensive_rank_rejections(self):
        """Rejeita tensores de rank 3+ ou matrizes com rank inválido."""
        W = np.ones((16, 32), dtype=np.float32)
        x_3d = np.ones((1, 1, 32), dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            cent_runtime.matmul(x_3d, W)
        self.assertIn("rank 3", str(ctx.exception))

        x_1d = np.ones((32,), dtype=np.float32)
        W_1d = np.ones((32,), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx_w:
            cent_runtime.matmul(x_1d, W_1d)
        self.assertIn("Matriz de pesos 'W' deve ter rank 2", str(ctx_w.exception))

    def test_defensive_dimension_mismatch(self):
        """Rejeita quando dimensão oculta K difere entre ativação e matriz de pesos."""
        x = np.ones((64,), dtype=np.float32)
        W = np.ones((32, 128), dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            cent_runtime.matmul(x, W)
        self.assertIn("incompatível", str(ctx.exception))

    def test_defensive_invalid_types(self):
        """Rejeita tipos não suportados com TypeError."""
        with self.assertRaises(TypeError):
            cent_runtime.matmul([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])

    def test_llama2_7b_attention_layer_metrics(self):
        """
        Valida os números auditados da Camada de Atenção do Llama-2 7B:
        K=4096, M=4096, layer_id=0 -> DVid=0, CHmask=0xFFFFFFFF, 1025 inst, 14350 B, ~99.95% economia.
        """
        M, K = 4096, 4096
        x = np.full((K,), 0.01, dtype=np.float32)
        W = np.full((M, K), 0.001, dtype=np.float32)

        y, telem = cent_runtime.matmul(x, W, layer_id=0)

        self.assertEqual(telem["dvid"], 0)
        self.assertEqual(telem["channel_mask"], 0xFFFFFFFF)
        self.assertEqual(telem["channel_mask_hex"], "0xffffffff")
        self.assertEqual(telem["wr_gb_count"], 256)
        self.assertEqual(telem["wr_bias_count"], 256)
        self.assertEqual(telem["mac_abk_count"], 256)
        self.assertEqual(telem["rd_mac_count"], 256)
        self.assertEqual(telem["total_instructions"], 1025)
        self.assertEqual(telem["instruction_buffer_bytes"], 14350)
        self.assertAlmostEqual(telem["memory_traffic_saved_pct"], 99.95, delta=0.05)

    def test_llama2_7b_mlp_layer_metrics(self):
        """
        Valida os números auditados da Camada MLP / Feed-Forward do Llama-2 7B:
        K=4096, M=11008, layer_id=1 -> DVid=1, CHmask=0xFFFFFFFF, 2753 inst, 38542 B.
        """
        M, K = 11008, 4096
        x = np.full((K,), 0.01, dtype=np.float32)
        W = np.full((M, K), 0.001, dtype=np.float32)

        y, telem = cent_runtime.matmul(x, W, layer_id=1)

        self.assertEqual(telem["dvid"], 1)
        self.assertEqual(telem["channel_mask_hex"], "0xffffffff")
        self.assertEqual(telem["wr_gb_count"], 688)
        self.assertEqual(telem["wr_bias_count"], 688)
        self.assertEqual(telem["mac_abk_count"], 688)
        self.assertEqual(telem["rd_mac_count"], 688)
        self.assertEqual(telem["total_instructions"], 2753)
        self.assertEqual(telem["instruction_buffer_bytes"], 38542)

    def test_sram_hex_dump(self):
        """Verifica a integridade da formatação do Hex Dump dos primeiros 64 bytes da SRAM."""
        x = np.array([1.0, 2.0], dtype=np.float32)
        W = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        cent_runtime.matmul(x, W, layer_id=0)

        dump_str = cent_runtime.get_instruction_buffer_hex(512)
        self.assertIsInstance(dump_str, str)
        self.assertTrue(len(dump_str) > 0)
        # Primeiro byte da primeira instrução deve ser o opcode CENT_OP_WR_GB (0x02)
        self.assertTrue(dump_str.startswith("02"))

    def test_trace_file_syntax_and_eoc(self):
        """Valida que o arquivo .trace gerado obedece à sintaxe exata da CENT ISA."""
        with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as tmp:
            trace_path = tmp.name

        try:
            M, K = 32, 1024  # 2 blocos de 16 bancos DRAM
            x = np.ones((K,), dtype=np.float32)
            W = np.ones((M, K), dtype=np.float32)

            cent_runtime.matmul(x, W, layer_id=0, trace_path=trace_path)

            with open(trace_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            # 2 blocos * 4 comandos + 1 EOC = 9 linhas
            self.assertEqual(len(lines), 9)
            self.assertEqual(lines[0], "AiM WR_GB 64 0 0xffffffff")
            self.assertEqual(lines[1], "AiM WR_BIAS 0 0xffffffff")
            self.assertEqual(lines[2], "AiM MAC_ABK 64 0xffffffff 0")
            self.assertEqual(lines[3], "AiM RD_MAC 0 0xffffffff")
            self.assertEqual(lines[4], "AiM WR_GB 64 0 0xffffffff")
            self.assertEqual(lines[5], "AiM WR_BIAS 0 0xffffffff")
            self.assertEqual(lines[6], "AiM MAC_ABK 64 0xffffffff 1")
            self.assertEqual(lines[7], "AiM RD_MAC 0 0xffffffff")
            self.assertEqual(lines[8], "AiM EOC")
        finally:
            if os.path.exists(trace_path):
                os.remove(trace_path)

    def test_save_trace_flag(self):
        """Valida que save_trace=True gera cent_layer_{layer_id}.trace automaticamente."""
        expected_file = "cent_layer_3.trace"
        if os.path.exists(expected_file):
            os.remove(expected_file)

        try:
            x = np.ones((1024,), dtype=np.float32)
            W = np.ones((16, 1024), dtype=np.float32)
            _, telem = cent_runtime.matmul(x, W, layer_id=3, save_trace=True)

            self.assertTrue(os.path.exists(expected_file))
            self.assertEqual(telem.get("trace_path"), expected_file)
            with open(expected_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertTrue("AiM WR_GB" in content)
            self.assertTrue("AiM EOC" in content)
        finally:
            if os.path.exists(expected_file):
                os.remove(expected_file)

    def test_simulator_output_parser_mock(self):
        """Valida o parser estruturado de estatísticas com saída simulada do Ramulator 2.0."""
        mock_output = """
        # Ramulator 2.0 Simulation Stats
        Frontend_total_cycles: 32800
        CH31_AiM_ISR_WR_GB_cycles: 16416
        CH31_AiM_ISR_MAC_ABK_cycles: 16384
        memory_system_cycles: 32800
        dram_energy_pJ: 1450.75
        """
        parsed = parse_simulator_output(mock_output)
        self.assertEqual(parsed["status"], "success")
        self.assertEqual(parsed["wr_gb_cycles"], 16416)
        self.assertEqual(parsed["mac_abk_cycles"], 16384)
        self.assertEqual(parsed["memory_system_cycles"], 32800)
        self.assertAlmostEqual(parsed["parsed_stats"]["dram_energy_pJ"], 1450.75)

    def test_simulator_config_resolver(self):
        """Valida a resolução do arquivo YAML oficial GDDR6-AiM."""
        cfg_path = resolve_config_path()
        self.assertTrue(cfg_path.is_file())
        self.assertTrue(cfg_path.name.endswith(".yaml"))


if __name__ == "__main__":
    unittest.main()
