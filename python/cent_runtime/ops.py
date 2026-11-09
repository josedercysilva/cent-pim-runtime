"""
Operador de multiplicação Vetor-Matriz (GEMV) acelerada para arquitetura CENT.
Conecta tensores do PyTorch e NumPy à C ABI nativa (libcent_runtime.so) com despacho
para bancos GDDR6-AiM via CXL 3.0 e integração com o simulador Ramulator 2.0.

Divisão Arquitetural (Gu et al., ASPLOS 2025):
  - Near-Bank PIM (Bancos DRAM GDDR6-AiM):
      Executa exclusivamente multiplicação Vetor-Matriz (GEMV, batch_size=1) durante
      a fase de Decoding de LLMs (projeções densas q_proj, k_proj, v_proj, o_proj,
      gate_proj, up_proj, down_proj). Responsável por mais de 99% do tráfego de pesos.
  - PNM / Host CPU (Dispatcher do PyTorch):
      Executa operações não-lineares, normalizações e reduções (Softmax, RMSNorm,
      LayerNorm, SiLU, RoPE), além da fase de Prefill (GEMM, batch > 1) e autograd.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

from ._bindings import CentDispatchTelemetry, get_lib
from .simulator import simulate_on_ramulator


def matmul(
    x: Any,
    W: Any,
    layer_id: int = 0,
    trace_path: Optional[str] = None,
    save_trace: bool = False,
    simulate: bool = False,
    ramulator_bin: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Executa a multiplicação de ativação por pesos estacionários y = x * W^T
    acelerada pela arquitetura CENT (GDDR6-AiM via CXL 3.0).

    Args:
        x: Vetor de ativação de entrada como torch.Tensor ou np.ndarray.
           Deve possuir formato 1D (K,) ou 2D (1, K) com batch_size = 1.
        W: Matriz de pesos estacionários como torch.Tensor ou np.ndarray
           de formato 2D (M, K) em ordem row-major.
        layer_id: Índice da camada Transformer (utilizado para rotear o DVid
                  no switch CXL entre as 32 placas disponíveis).
        trace_path: Caminho customizado opcional para salvar o arquivo de traço (.trace).
        save_trace: Se True (ou se CENT_SAVE_TRACE=1 no ambiente), salva automaticamente
                    o arquivo de traço como 'cent_layer_{layer_id}.trace' no diretório atual.
        simulate: Se True, sintetiza o traço CENT ISA e dispara o simulador
                  Ramulator 2.0 de ciclo exato para extrair estatísticas de hardware.
        ramulator_bin: Caminho opcional para o executável do Ramulator 2.0.
        config_path: Caminho opcional para o arquivo de configuração YAML GDDR6-AiM.

    Returns:
        Uma tupla (y, telemetry):
          - y: Vetor de saída calculado (mesmo tipo e rank do tensor de entrada x).
          - telemetry: Dicionário contendo estatísticas de despacho CXL, ocupação da SRAM,
                       contadores de instrução da ISA e métricas de simulação física (se ativa).

    Raises:
        TypeError: Se as entradas não forem instâncias suportadas de PyTorch ou NumPy.
        ValueError: Se as dimensões forem incompatíveis ou se for fornecido batch_size > 1.
        RuntimeError: Se ocorrer falha na C ABI nativa ou na execução do simulador.
    """
    is_torch = False
    orig_device = None
    orig_dtype = None

    if _HAS_TORCH and isinstance(x, torch.Tensor):
        is_torch = True
        orig_device = x.device
        orig_dtype = x.dtype
        x_np = x.detach().cpu().to(torch.float32).numpy()
    elif isinstance(x, np.ndarray):
        x_np = x
    else:
        raise TypeError(
            f"Tipo de entrada 'x' não suportado: {type(x)}. Esperado torch.Tensor ou numpy.ndarray."
        )

    if _HAS_TORCH and isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().to(torch.float32).numpy()
    elif isinstance(W, np.ndarray):
        W_np = W
    else:
        raise TypeError(
            f"Tipo de matriz de pesos 'W' não suportado: {type(W)}. Esperado torch.Tensor ou numpy.ndarray."
        )

    # -------------------------------------------------------------------------
    # Validação estrita de Escopo GEMV (Near-Bank PIM)
    # -------------------------------------------------------------------------
    if x_np.ndim == 2:
        if x_np.shape[0] != 1:
            raise ValueError(
                f"A arquitetura CENT Near-Bank PIM opera exclusivamente em multiplicação "
                f"Vetor-Matriz (GEMV, batch_size=1 para a fase de Decoding). "
                f"Recebido batch_size={x_np.shape[0]}. A fase de Prefill (GEMM com batch > 1) "
                f"deve ser executada no Host CPU/PNM."
            )
        is_2d = True
    elif x_np.ndim == 1:
        is_2d = False
    else:
        raise ValueError(
            f"Vetor de entrada 'x' deve ter rank 1 (K,) ou rank 2 com batch 1 (1, K). "
            f"Recebido rank {x_np.ndim} com shape {x_np.shape}."
        )

    if W_np.ndim != 2:
        raise ValueError(
            f"Matriz de pesos 'W' deve ter rank 2 (M, K). "
            f"Recebido rank {W_np.ndim} com shape {W_np.shape}."
        )

    k_in = x_np.shape[-1]
    m_dim = W_np.shape[0]
    k_dim = W_np.shape[1]

    if k_in != k_dim:
        raise ValueError(
            f"Dimensão oculta incompatível: vetor de entrada possui K={k_in}, "
            f"mas matriz de pesos espera K={k_dim}."
        )

    # Garante armazenamento contíguo em float32
    x_contig = np.ascontiguousarray(x_np.ravel(), dtype=np.float32)
    W_contig = np.ascontiguousarray(W_np, dtype=np.float32)
    y_np = np.empty((m_dim,), dtype=np.float32)

    # -------------------------------------------------------------------------
    # Gerenciamento de arquivo .trace (automático via save_trace ou temporário para simulate)
    # -------------------------------------------------------------------------
    env_save_trace = os.environ.get("CENT_SAVE_TRACE", "").lower() in ("1", "true", "yes")
    should_save_trace = save_trace or env_save_trace

    target_trace_path = trace_path
    if should_save_trace and not target_trace_path:
        target_trace_path = f"cent_layer_{layer_id}.trace"

    temp_trace_path: Optional[str] = None
    if simulate and not target_trace_path:
        with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as tmp:
            temp_trace_path = tmp.name
        target_trace_path = temp_trace_path

    try:
        lib = get_lib()
        telem_c = CentDispatchTelemetry()

        x_ptr = x_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = W_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        trace_c_arg = target_trace_path.encode("utf-8") if target_trace_path else None

        status = lib.cent_dispatch_gemv(
            x_ptr,
            ctypes.c_int(m_dim),
            ctypes.c_int(k_dim),
            w_ptr,
            y_ptr,
            ctypes.c_int(layer_id),
            trace_c_arg,
            ctypes.byref(telem_c),
        )

        if status != 0:
            raise RuntimeError(f"cent_dispatch_gemv falhou na C ABI com código de erro {status}")

        telemetry = telem_c.to_dict()

        if target_trace_path and not temp_trace_path:
            telemetry["trace_path"] = target_trace_path

        if simulate and target_trace_path:
            sim_stats = simulate_on_ramulator(
                trace_path=target_trace_path,
                config_path=config_path,
                ramulator_bin=ramulator_bin,
            )
            telemetry["simulation"] = sim_stats
            telemetry["wr_gb_cycles"] = sim_stats.get("wr_gb_cycles", 0)
            telemetry["mac_abk_cycles"] = sim_stats.get("mac_abk_cycles", 0)
            telemetry["memory_system_cycles"] = sim_stats.get("memory_system_cycles", 0)

    finally:
        if temp_trace_path and os.path.exists(temp_trace_path):
            try:
                os.remove(temp_trace_path)
            except OSError:
                pass

    if is_2d:
        y_np = y_np.reshape((1, m_dim))

    if is_torch:
        y_out = torch.from_numpy(y_np).to(dtype=orig_dtype, device=orig_device)
    else:
        y_out = y_np

    return y_out, telemetry
