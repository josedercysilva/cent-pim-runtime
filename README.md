# cent_runtime

Runtime e hardware dispatcher de alto desempenho desenvolvido em **C++17 e C ABI nativa** para a arquitetura **CENT** (*PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, **ASPLOS 2025**).

O `cent_runtime` conecta modelos em **PyTorch e NumPy** à arquitetura CENT, acelerando projeções lineares densas (GEMV) diretamente nos bancos de memória **GDDR6-AiM** através de um switch **CXL 3.0**, eliminando o gargalo do *Memory Wall*.

> 📄 **Artigo Técnico e Slides da Apresentação:**  
> Para entender em detalhes a física do *Memory Wall*, a microarquitetura CENT (ASPLOS '25), o funcionamento interno do *hardware dispatcher* em C++17 e a validação no Llama-2 7B, consulte:  
> 👉 **[Artigo Completo: Arquitetura CENT e Anatomia do `cent_runtime` (josedercysilva.github.io)](https://josedercysilva.github.io/post.html?slug=cent-runtime)**  
> 📊 **[Slides da Apresentação do Seminário (`apresentacao_cent.pdf`)](apresentacao_cent.pdf)**

---

## Destaques da Arquitetura

* **Redução de Tráfego de 99.95%:** Na fase de *decoding* do Llama-2 7B, move apenas ativações (~16 KB) em vez de transferir matrizes inteiras de pesos (~32 MB) pelo barramento.
* **Motor C++17 de Baixo Nível (`libcent_runtime.so`):** Sem overhead de exceções, buffer de instruções SRAM de 2 MB pré-alocado e interface C ABI estável (`extern "C"`).
* **Divisão de Responsabilidades (PIM + PNM/Host):**
  * **Near-Bank PIM (Hardware CENT):** Projeções lineares densas (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc., com `batch = 1`).
  * **Host CPU / PyTorch Nativo:** Operações não-lineares e reduções (`Softmax`, `RMSNorm`, `SiLU`, `RoPE`) e fases com `batch > 1`.

---

## Início Rápido

### 1. Criar e configurar o ambiente virtual
Execute o script de automação (configura Python 3.11, instala PyTorch CPU e compila o pacote local):

```bash
./setup_env.sh
```
*(Ou execute via Make: `make setup-env`)*

### 2. Ativar o ambiente
```bash
source .venv/bin/activate
```

### 3. Executar Demonstração do Llama-2 7B
Executa as camadas reais de Atenção (4K×4K) e MLP (11K×4K) com validação de métricas:
```bash
make demo
```

### 4. Rodar Testes Unitários
```bash
make test        # Suíte de testes em Python (15 testes)
make test-core   # Testbench nativo em C++ (C ABI, memória e limites)
```

---

## Exemplo de Uso com PyTorch

```python
import torch
import torch.nn.functional as F
import cent_runtime

# 1. Ativação do token e pesos estacionários (fase de decoding, batch = 1)
x = torch.randn(1, 4096)        # Vetor de ativação
W = torch.randn(4096, 4096)     # Matriz de pesos da camada de atenção

# 2. Multiplicação linear (GEMV) acelerada diretamente na memória GDDR6-AiM
y, telemetry = cent_runtime.matmul(x, W, layer_id=0)

# 3. Operação não-linear processada na CPU pelo PyTorch nativo
out = F.silu(y)

print(f"Dimensão de saída: {out.shape}")
print(f"Redução de tráfego externo: {telemetry['memory_traffic_saved_pct']:.2f}%")
print(f"Comandos PIM emitidos: {telemetry['total_instructions']:,}")
```

---

## Simulação de Ciclo Exato (Ramulator 2.0)

> [!NOTE]
> **O simulador é 100% opcional!**  
> Para usar o `cent_runtime`, executar a demonstração do Llama-2 (`make demo`), rodar os testes (`make test`) ou integrar com PyTorch, **você NÃO precisa do simulador instalado**. O motor C++ nativo já calcula os resultados numéricos, a telemetria e a economia de 99.95% de memória externa de forma independente.  
> O Ramulator 2.0 é necessário **exclusivamente** para o comando `make simulate` (ou flag `simulate=True`), que audita os ciclos físicos de clock da DRAM.

| Operação / Comando | Precisa do Simulador? | O que é executado |
| :--- | :---: | :--- |
| **`make demo`** (Llama-2 7B) | ❌ Não | Computação numérica, 99.95% economia CXL e Hex Dump SRAM |
| **`examples/01_quickstart.py`** | ❌ Não | Multiplicação Near-Bank PIM pura com NumPy |
| **`examples/02_pytorch_layer.py`** | ❌ Não | Camada linear Near-Bank PIM com PyTorch |
| **`make test` / `make test-core`** | ❌ Não | Suíte completa de testes unitários |
| **`make simulate`** | ✅ **Sim** | Simulação física de ciclo exato nos chips GDDR6-AiM |

### Como rodar a simulação física:
```bash
make simulate
```

> 📖 **Configuração e Compilação do Simulador:**  
> Por padrão, o runtime busca os binários no diretório `external/`. Para instruções de como clonar e compilar o `ramulator2` e `libramulator.so` do repositório upstream, consulte:  
> 👉 **[Instruções do Simulador em external/README.md](external/README.md)**

---

## Comandos do `Makefile`

| Comando | Descrição |
| :--- | :--- |
| `make` / `make shared` | Compila o núcleo C++ nativo (`build/libcent_runtime.so`) |
| `make setup-env` | Cria o ambiente `.venv` com PyTorch CPU e dependências |
| `make demo` | Executa a demonstração com camadas do Llama-2 7B |
| `make simulate` | Executa a demonstração com simulação física no Ramulator 2.0 |
| `make test` | Roda a suíte completa de testes unitários Python |
| `make test-core` | Roda o testbench nativo em C++ |
| `make clean` | Limpa binários de build, arquivos `.trace` e caches |
| `make venv-clean` | Remove o ambiente virtual `.venv` |
| `make help` | Exibe o menu de ajuda com todos os alvos |

---

## Licença e Autoria

* **Autor:** [José Dercy Vieira da Silva Filho](https://josedercysilva.github.io/)
* **Publicação Técnica:** [https://josedercysilva.github.io/post.html?slug=cent-runtime](https://josedercysilva.github.io/post.html?slug=cent-runtime)
* **Slides da Apresentação:** [`apresentacao_cent.pdf`](apresentacao_cent.pdf)
* **Instituição:** Universidade Federal do Amazonas (UFAM) — Faculdade de Engenharia Elétrica e Computação (FEEC)
* **Licença:** [MIT](LICENSE)
