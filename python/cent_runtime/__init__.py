"""
cent_runtime: Biblioteca de Runtime e Hardware Dispatcher para a Arquitetura CENT.
Conecta modelos em PyTorch / NumPy aos aceleradores Near-Bank PIM (GDDR6-AiM via CXL 3.0)
conforme proposto em Gu et al. (ASPLOS 2025).
"""

from ._bindings import CentDispatchTelemetry, get_instruction_buffer_hex
from .ops import matmul
from .simulator import simulate_on_ramulator

__version__ = "0.1.0"

__all__ = [
    "matmul",
    "simulate_on_ramulator",
    "get_instruction_buffer_hex",
    "CentDispatchTelemetry",
    "__version__",
]
