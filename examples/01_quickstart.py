#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "python"))

import cent_runtime

x = np.random.randn(1, 1024).astype(np.float32)
W = np.random.randn(512, 1024).astype(np.float32)

y, telem = cent_runtime.matmul(x, W, layer_id=0, save_trace=True)

print("=== cent_runtime: Multiplicação Vetor-Matriz PIM ===")
print(f"  • Arquivo .trace gerado      : {telem.get('trace_path')}")
print(f"  • Shape da Ativação (x)      : {x.shape}")
print(f"  • Shape dos Pesos (W)        : {W.shape}")
print(f"  • Shape da Saída (y)         : {y.shape}")
print(f"  • Placa CXL Selecionada      : #{telem['dvid']}")
print(f"  • Canais GDDR6-AiM Ativos    : {telem['channel_mask_hex']}")
print(f"  • Economia de Tráfego CXL/GPU: {telem['memory_traffic_saved_pct']:.2f}%")
