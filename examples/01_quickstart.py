#!/usr/bin/env python3
"""
Exemplo 01: Início Rápido (Quickstart) com cent_runtime
Demonstração minimalista de multiplicação Vetor x Matriz (GEMV) acelerada em Near-Bank PIM.
"""

from pathlib import Path
import sys
import numpy as np

# Permite executar diretamente a partir do repositório
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "python"))

import cent_runtime

# 1. Cria vetor de ativação de entrada (1x1024) e matriz de pesos (512x1024)
x = np.random.randn(1, 1024).astype(np.float32)
W = np.random.randn(512, 1024).astype(np.float32)

# 2. Executa a multiplicação acelerada em PIM e salva o traço com uma simples flag
y, telem = cent_runtime.matmul(x, W, layer_id=0, save_trace=True)

# 3. Exibe o resultado e as métricas de hardware
print("=== cent_runtime: Multiplicação Vetor-Matriz PIM ===")
print(f"  • Arquivo .trace gerado      : {telem.get('trace_path')}")
print(f"  • Shape da Ativação (x)      : {x.shape}")
print(f"  • Shape dos Pesos (W)        : {W.shape}")
print(f"  • Shape da Saída (y)         : {y.shape}")
print(f"  • Placa CXL Selecionada      : #{telem['dvid']}")
print(f"  • Canais GDDR6-AiM Ativos    : {telem['channel_mask_hex']}")
print(f"  • Economia de Tráfego CXL/GPU: {telem['memory_traffic_saved_pct']:.2f}%")
