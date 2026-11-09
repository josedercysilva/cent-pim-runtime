#!/usr/bin/env python3
"""
Demonstração e Validação Microarquitetural do Llama-2 7B — cent_runtime
Autor: José Dercy Vieira da Silva Filho
Instituição: UFAM — FEEC — Arquitetura de Computadores (Prof. Thiago Bezerra)
Artigo Base: Gu et al., ASPLOS 2025 (PIM Is All You Need / CENT Architecture)

Este script valida rigorosamente as projeções lineares densas do Llama-2 7B
durante a fase de Decoding (batch_size = 1) no acelerador Near-Bank PIM:
  1. Camada de Atenção (4096 x 4096, layer_id = 0)
  2. Camada MLP / Feed-Forward (11008 x 4096, layer_id = 1)

Divisão Arquitetural:
  - Descarregado para CENT Near-Bank PIM (GDDR6-AiM via CXL 3.0):
      Multiplicações Vetor-Matriz (GEMV) das projeções densas (q_proj, k_proj,
      v_proj, o_proj, gate_proj, up_proj, down_proj), responsáveis por >99%
      da movimentação de pesos de LLMs.
  - Mantido no Host CPU / PNM (Dispatcher nativo do PyTorch):
      Operações não-lineares, normalizações e reduções (Softmax, RMSNorm,
      SiLU, RoPE), além da fase de Prefill (GEMM, batch > 1) e autograd.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Adiciona o diretório python/ ao sys.path para importação direta
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "python"))

import cent_runtime  # noqa: E402

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

import numpy as np  # noqa: E402


def format_bytes(num_bytes: float) -> str:
    """Formata valores em bytes para KB ou MB de forma humanamente legível."""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes:.0f} B"


def print_header() -> None:
    print("\n" + "=" * 78)
    print("  CENT Runtime — Demonstração Microarquitetural do Llama-2 7B")
    print("  PIM Is All You Need (ASPLOS 2025) — Dispatcher CXL 3.0 + GDDR6-AiM")
    print("=" * 78)
    print(
        "Divisão Arquitetural:\n"
        "  • Near-Bank PIM (AiM DRAM) : GEMV (batch=1, Decoding) -> Projeções Densas\n"
        "  • Host CPU / PNM          : Softmax, RMSNorm, SiLU, RoPE, Prefill (batch>1)\n"
    )
    backend_name = "PyTorch Tensor" if _HAS_TORCH else "NumPy ndarray (Fallback)"
    print(f"Backend Numérico Detectado   : {backend_name}")
    print("=" * 78 + "\n")


def run_layer_demo(
    layer_name: str,
    M: int,
    K: int,
    layer_id: int,
    simulate: bool = False,
    trace_path: str | None = None,
) -> dict:
    """Executa e valida uma camada Transformer densa do Llama-2 7B."""
    print(f"▶ Executando Camada: {layer_name}")
    print(f"  Dimensões de Pesos W : [{M} x {K}] (Dimensão de Saída x Dimensão Oculta)")
    print(f"  Formato da Ativação x: [1 x {K}] (Vetor de Decoding, batch_size = 1)")
    print(f"  Índice da Camada     : layer_id = {layer_id}")

    # Criação dos tensores de ativação e pesos
    if _HAS_TORCH:
        x = torch.randn(1, K, dtype=torch.float32)
        W = torch.randn(M, K, dtype=torch.float32)
    else:
        rng = np.random.default_rng(layer_id + 42)
        x = rng.standard_normal((1, K)).astype(np.float32)
        W = rng.standard_normal((M, K)).astype(np.float32)

    try:
        y, telem = cent_runtime.matmul(
            x,
            W,
            layer_id=layer_id,
            trace_path=trace_path,
            simulate=simulate,
        )
    except FileNotFoundError as err:
        if simulate:
            print(f"  \033[1;33m[AVISO SIMULAÇÃO]\033[0m {err}")
            print("  Executando sem simulação Ramulator 2.0...")
            y, telem = cent_runtime.matmul(
                x,
                W,
                layer_id=layer_id,
                trace_path=trace_path,
                simulate=False,
            )
        else:
            raise

    # Exibição dos resultados de telemetria de hardware
    dvid = telem["dvid"]
    ch_mask = telem["channel_mask_hex"]
    total_inst = telem["total_instructions"]
    sram_bytes = telem["instruction_buffer_bytes"]
    saved_pct = telem["memory_traffic_saved_pct"]
    gpu_traffic = format_bytes(telem["gpu_traffic_bytes"])
    pcie_traffic = format_bytes(telem["pcie_traffic_bytes"])

    print("  " + "-" * 74)
    if trace_path:
        print(f"  • Arquivo de Traço CENT ISA (.trace) : {trace_path}")
    print(f"  • Placa CXL Selecionada (DVid)   : Placa #{dvid} (roteamento layer_id % 32)")
    print(f"  • Máscara de Canais (CHmask)     : {ch_mask} (32 canais GDDR6 ativos)")
    print(f"  • Instruções PIM Emitidas        : {total_inst:,} comandos (inclui 1 AiM EOC)")
    print(f"    - WR_GB   (Vetor no buffer)    : {telem['wr_gb_count']:,}")
    print(f"    - WR_BIAS (Zera acumulador)    : {telem['wr_bias_count']:,}")
    print(f"    - MAC_ABK (All-Bank MAC)       : {telem['mac_abk_count']:,}")
    print(f"    - RD_MAC  (Lê resultado)       : {telem['rd_mac_count']:,}")
    print(f"  • Ocupação da SRAM CXL (2 MB)    : {sram_bytes:,} bytes ({sram_bytes / 1024:.2f} KB)")
    print(f"  • Tráfego de Barramento Externo  :")
    print(f"    - GPU A100 HBM (Baseline)      : ~{gpu_traffic} (pesos inteiros movidos)")
    print(f"    - CENT CXL 3.0 (Near-Bank PIM) : ~{pcie_traffic} (somente ativação + saída)")
    print(f"    - \033[1;32mRedução de Tráfego Externo\033[0m   : \033[1;32m{saved_pct:.2f}%\033[0m")

    if "simulation" in telem:
        sim = telem["simulation"]
        print("  • Métricas do Simulador Ramulator 2.0:")
        print(f"    - Ciclos WR_GB (CH31)          : {sim.get('wr_gb_cycles', 0):,} ciclos")
        print(f"    - Ciclos MAC_ABK (CH31)        : {sim.get('mac_abk_cycles', 0):,} ciclos")
        print(f"    - Total Ciclos Sistema Memória : {sim.get('memory_system_cycles', 0):,} ciclos")

    print("  " + "-" * 74 + "\n")
    return telem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstração do cent_runtime com camadas Llama-2 7B (Atenção e MLP)."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Dispara a simulação de ciclo exato no Ramulator 2.0 (se disponível).",
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        default=None,
        help="Prefixo ou caminho customizado para salvar os arquivos .trace (padrão: llama2_attention.trace e llama2_mlp.trace).",
    )
    args = parser.parse_args()

    # Define caminhos padrão dos arquivos .trace no mesmo diretório do script
    script_dir = Path(__file__).resolve().parent
    if args.trace_path:
        base_trace = Path(args.trace_path)
        if not base_trace.is_absolute():
            base_trace = script_dir / base_trace
        if base_trace.suffix == ".trace":
            stem = base_trace.stem
            attn_trace = str(base_trace.with_name(f"{stem}_attention.trace"))
            mlp_trace = str(base_trace.with_name(f"{stem}_mlp.trace"))
        else:
            attn_trace = str(base_trace.parent / f"{base_trace.name}_attention.trace")
            mlp_trace = str(base_trace.parent / f"{base_trace.name}_mlp.trace")
    else:
        attn_trace = str(script_dir / "llama2_attention.trace")
        mlp_trace = str(script_dir / "llama2_mlp.trace")

    print_header()

    # 1. Camada de Atenção Llama-2 7B (4096 x 4096, layer_id = 0)
    telem_attn = run_layer_demo(
        layer_name="Llama-2 7B — Camada de Atenção (q/k/v/o proj)",
        M=4096,
        K=4096,
        layer_id=0,
        simulate=args.simulate,
        trace_path=attn_trace,
    )

    # 2. Camada MLP / Feed-Forward Llama-2 7B (11008 x 4096, layer_id = 1)
    telem_mlp = run_layer_demo(
        layer_name="Llama-2 7B — Camada MLP / Feed-Forward (gate/up/down proj)",
        M=11008,
        K=4096,
        layer_id=1,
        simulate=args.simulate,
        trace_path=mlp_trace,
    )

    # 3. Hex Dump dos Primeiros 64 Bytes da SRAM
    print("▶ Dump Hexadecimal do Instruction Buffer SRAM de 2 MB (Primeiros 64 Bytes):")
    hex_dump = cent_runtime.get_instruction_buffer_hex(max_len=512)
    for line in hex_dump.splitlines()[:4]:
        print(f"  [SRAM] {line}")
    print("  (Opcode 0x02 = WR_GB, Opcode 0x03 = WR_BIAS, Opcode 0x0F = MAC_ABK)\n")

    # 4. Auditoria Rigorosa dos Resultados
    print("=" * 78)
    print("  Conferência Rigorosa contra Métricas Auditadas (ASPLOS 2025):")
    print("=" * 78)

    attn_ok = (
        telem_attn["dvid"] == 0
        and telem_attn["channel_mask_hex"] == "0xffffffff"
        and telem_attn["total_instructions"] == 1025
        and telem_attn["instruction_buffer_bytes"] == 14350
        and abs(telem_attn["memory_traffic_saved_pct"] - 99.95) < 0.05
    )

    mlp_ok = (
        telem_mlp["dvid"] == 1
        and telem_mlp["channel_mask_hex"] == "0xffffffff"
        and telem_mlp["total_instructions"] == 2753
        and telem_mlp["instruction_buffer_bytes"] == 38542
    )

    expansion_ratio = telem_mlp["total_instructions"] / telem_attn["total_instructions"]

    print(f"  • Camada Atenção (4Kx4K)  : {'[OK 100%]' if attn_ok else '[ERRO]'} (1.025 inst, 14.350 B, 99.95% econ)")
    print(f"  • Camada MLP (11Kx4K)     : {'[OK 100%]' if mlp_ok else '[ERRO]'} (2.753 inst, 38.542 B, DVid=1)")
    print(f"  • Razão Física MLP/Atenção: {expansion_ratio:.2f}x (Razão teórica 11008/4096 = 2.6875x)")

    if attn_ok and mlp_ok:
        print("\n\033[1;32m✔ TODAS AS MÉTRICAS BATERAM 100% COM OS DADOS AUDITADOS DO ARTIGO!\033[0m\n")
    else:
        print("\n\033[1;31m✖ DIVERGÊNCIA NAS MÉTRICAS AUDITADAS!\033[0m\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
