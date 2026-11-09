#!/usr/bin/env python3
"""
Exemplo 02: Integração com PyTorch (Camada Linear PIM + Não-Linearidade no Host)
Demonstra a divisão arquitetural do CENT (Gu et al., ASPLOS 2025):
  - Multiplicação linear densa (GEMV): Descarregada para o Near-Bank PIM (cent_runtime)
  - Ativação não-linear (SiLU / ReLU): Executada no Host CPU (PyTorch nativo)
"""

from pathlib import Path
import sys
import numpy as np

# Permite executar diretamente a partir do repositório
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "python"))

import cent_runtime

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

print("=== cent_runtime: Integração PyTorch + Near-Bank PIM ===\n")

if HAS_TORCH:
    # 1. Cria tensores PyTorch nativos (fase de Decoding, batch_size = 1)
    x = torch.randn(1, 2048)            # Vetor de ativação do token
    W = torch.randn(2048, 2048)         # Matriz de pesos da projeção

    # 2. GEMV descarregada para o Near-Bank PIM via cent_runtime salvando o traço
    y, telem = cent_runtime.matmul(x, W, layer_id=0, save_trace=True)

    # 3. Operação não-linear continua transparentemente no PyTorch (Host CPU / PNM)
    out = F.silu(y)

    print(f"  • Arquivo .trace gerado     : {telem.get('trace_path')}")
    print(f"  • Entrada PyTorch (x)       : {x.shape} (tipo: {type(x).__name__})")
    print(f"  • Saída PIM Linear (y)      : {y.shape} (tipo: {type(y).__name__})")
    print(f"  • Saída após SiLU Host (out): {out.shape}")
    print(f"  • Economia de Tráfego CXL   : {telem['memory_traffic_saved_pct']:.2f}%")

else:
    # Fallback transparente com NumPy caso PyTorch não esteja instalado
    x = np.random.randn(1, 2048).astype(np.float32)
    W = np.random.randn(2048, 2048).astype(np.float32)

    y, telem = cent_runtime.matmul(x, W, layer_id=0, save_trace=True)
    out = np.maximum(y, 0)  # ReLU executada na CPU

    print("PyTorch não detectado no ambiente global. Executando via NumPy (Fallback):")
    print(f"  • Arquivo .trace gerado     : {telem.get('trace_path')}")
    print(f"  • Entrada (x)               : {x.shape}")
    print(f"  • Saída PIM Linear (y)      : {y.shape}")
    print(f"  • Saída após ReLU Host (out): {out.shape}")
    print(f"  • Economia de Tráfego CXL   : {telem['memory_traffic_saved_pct']:.2f}%")
