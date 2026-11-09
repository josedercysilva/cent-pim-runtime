# cent_runtime

Biblioteca de despacho em C/C++ e interface Python para geração de instruções e *traces* da arquitetura PIM/CXL **CENT** (*ASPLOS 2025*).

## Estrutura do Projeto

```text
cent_runtime/
├── include/cent/      # Cabeçalhos C/C++ (ISA, buffer e API)
├── src/               # Implementação do núcleo nativo (libcent_runtime.so)
├── python/            # Pacote Python (bindings ctypes e integração com simulador)
├── examples/          # Script de demonstração (camadas Llama-2 7B)
├── tests/             # Testes automatizados (C++ e Python)
└── external/          # Diretório opcional para binários do Ramulator 2
```

## Compilação e Instalação

Compilar a biblioteca compartilhada (`build/libcent_runtime.so`):

```bash
make
```

Instalar o pacote Python em modo editável (opcional):

```bash
pip install -e .
```

## Como Executar

Executar a demonstração das camadas do Llama-2 7B (geração de instruções e telemetria):

```bash
make demo
```

Executar a demonstração integrada ao simulador de ciclos Ramulator 2:

```bash
make simulate
```

Executar os testes:

```bash
make test-core   # Testes em C++
make test        # Testes em Python
```

## Exemplo Rápido de Uso

```python
import torch
import cent_runtime

x = torch.randn(4096)
W = torch.randn(4096, 4096)

y, stats = cent_runtime.matmul(x, W, device_id=0)
print(stats)
```

## Referência

Baseado no artigo:
> *PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference* (ASPLOS 2025).
