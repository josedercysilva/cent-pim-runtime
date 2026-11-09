"""
Bindings C ABI de baixo nível via ctypes para a libcent_runtime.so.
Carrega a biblioteca nativa (auto-compilando via 'make shared' se necessário)
e mapeia as assinaturas de funções e estruturas de telemetria.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Optional


class CentDispatchTelemetry(ctypes.Structure):
    """
    Estrutura de telemetria correspondente a struct CentDispatchTelemetry (C ABI).
    Alinhamento padrão x86_64 (56 bytes no total).
    """
    _fields_ = [
        ("dvid", ctypes.c_uint8),
        ("channel_mask", ctypes.c_uint32),
        ("total_instructions", ctypes.c_uint32),
        ("instruction_buffer_bytes", ctypes.c_uint32),
        ("wr_gb_count", ctypes.c_uint32),
        ("wr_bias_count", ctypes.c_uint32),
        ("mac_abk_count", ctypes.c_uint32),
        ("rd_mac_count", ctypes.c_uint32),
        ("memory_traffic_saved_pct", ctypes.c_double),
        ("pcie_traffic_bytes", ctypes.c_double),
        ("gpu_traffic_bytes", ctypes.c_double),
    ]

    def to_dict(self) -> dict:
        """Converte a telemetria em um dicionário Python nativo e formatado."""
        return {
            "dvid": int(self.dvid),
            "channel_mask": int(self.channel_mask),
            "channel_mask_hex": f"0x{self.channel_mask:08x}",
            "total_instructions": int(self.total_instructions),
            "instruction_buffer_bytes": int(self.instruction_buffer_bytes),
            "wr_gb_count": int(self.wr_gb_count),
            "wr_bias_count": int(self.wr_bias_count),
            "mac_abk_count": int(self.mac_abk_count),
            "rd_mac_count": int(self.rd_mac_count),
            "memory_traffic_saved_pct": float(self.memory_traffic_saved_pct),
            "pcie_traffic_bytes": float(self.pcie_traffic_bytes),
            "gpu_traffic_bytes": float(self.gpu_traffic_bytes),
        }


def _find_repo_root() -> Path:
    """Localiza a raiz do repositório cent_runtime."""
    current = Path(__file__).resolve().parent
    for p in [current, current.parent, current.parent.parent]:
        if (p / "Makefile").exists() and (p / "include" / "cent").exists():
            return p
    return current.parent.parent


def _load_native_library() -> ctypes.CDLL:
    """
    Localiza e carrega a libcent_runtime.so.
    Se o arquivo .so não existir, aciona automaticamente 'make shared' na raiz.
    """
    repo_root = _find_repo_root()
    lib_path = repo_root / "build" / "libcent_runtime.so"

    if not lib_path.exists():
        try:
            subprocess.run(
                ["make", "shared"],
                cwd=str(repo_root),
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise RuntimeError(
                f"Falha ao compilar automaticamente libcent_runtime.so via 'make shared' em {repo_root}: {err}"
            ) from err

    if not lib_path.exists():
        raise FileNotFoundError(
            f"Biblioteca compartilhada não encontrada após tentativa de compilação: {lib_path}"
        )

    lib = ctypes.CDLL(str(lib_path))

    # int cent_init_runtime(void);
    lib.cent_init_runtime.argtypes = []
    lib.cent_init_runtime.restype = ctypes.c_int

    # int cent_dispatch_gemv(...)
    lib.cent_dispatch_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_float),            # const float* x
        ctypes.c_int,                              # int M
        ctypes.c_int,                              # int K
        ctypes.POINTER(ctypes.c_float),            # const float* W
        ctypes.POINTER(ctypes.c_float),            # float* y
        ctypes.c_int,                              # int layer_id
        ctypes.c_char_p,                           # const char* trace_path
        ctypes.POINTER(CentDispatchTelemetry),     # CentDispatchTelemetry* out_telem
    ]
    lib.cent_dispatch_gemv.restype = ctypes.c_int

    # int cent_get_instruction_buffer_hex(char* out_hex_str, size_t max_len);
    lib.cent_get_instruction_buffer_hex.argtypes = [
        ctypes.c_char_p,                           # char* out_hex_str
        ctypes.c_size_t,                           # size_t max_len
    ]
    lib.cent_get_instruction_buffer_hex.restype = ctypes.c_int

    return lib


_LIB: Optional[ctypes.CDLL] = None


def get_lib() -> ctypes.CDLL:
    """Retorna a instância singleton da biblioteca compartilhada C carregada."""
    global _LIB
    if _LIB is None:
        _LIB = _load_native_library()
    return _LIB


def get_instruction_buffer_hex(max_len: int = 512) -> str:
    """
    Obtém o dump hexadecimal formatado dos primeiros bytes presentes na SRAM (2 MB).
    """
    lib = get_lib()
    buf = ctypes.create_string_buffer(max_len)
    status = lib.cent_get_instruction_buffer_hex(buf, ctypes.c_size_t(max_len))
    if status < 0:
        raise RuntimeError(f"Erro ao obter dump da SRAM: código de erro {status}")
    return buf.value.decode("utf-8")
