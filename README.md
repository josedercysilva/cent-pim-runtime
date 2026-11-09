# cent_runtime

Runtime e dispatcher de hardware para a arquitetura PIM/CXL **CENT** (*ASPLOS 2025*), conectando tensores do PyTorch e NumPy a aceleradores Near-Bank PIM (GDDR6-AiM via CXL 3.0).

---

## Passo a Passo: Configuração Rápida

### 1. Criar e configurar o ambiente virtual

Execute o script de automação (ele detecta/instala o `uv`, cria o `.venv` com Python 3.11, instala o PyTorch CPU leve e o pacote local):

```bash
./setup_env.sh
```

*(Ou via Make: `make setup-env`)*

### 2. Ativar (e Desativar) o ambiente virtual

Para ativar na sua sessão:

```bash
source .venv/bin/activate
```

Para desativar da sessão atual:

```bash
deactivate
```

### 3. Executar

Execute a demonstração do Llama-2 7B (atenção e MLP):

```bash
make demo
```

Ou execute a suíte de testes:

```bash
make test        # Testes Python
make test-core   # Testes C++ nativos
```

---

## Exemplo de Uso com PyTorch

```python
import torch
import torch.nn.functional as F
import cent_runtime

# 1. Tensores de ativação e pesos (fase de decoding de LLM, batch = 1)
x = torch.randn(1, 2048)         # Vetor de ativação do token
W = torch.randn(2048, 2048)      # Matriz de projeção linear

# 2. Multiplicação linear (GEMV) acelerada no Near-Bank PIM
y, telemetry = cent_runtime.matmul(x, W, layer_id=0)

# 3. Operação não-linear executada no Host CPU (PyTorch nativo)
out = F.silu(y)

print(f"Saída PIM: {y.shape}")
print(f"Economia de tráfego CXL: {telemetry['memory_traffic_saved_pct']:.2f}%")
print(f"Instruções PIM (MAC_ABK): {telemetry['mac_abk_count']:,}")
```

---

## Comandos Úteis do `Makefile`

| Comando | Descrição |
| :--- | :--- |
| `make setup-env` | Cria o `.venv` e instala PyTorch CPU + `cent_runtime` |
| `make venv-clean` | Remove o ambiente virtual `.venv` e `.egg-info` |
| `make` / `make shared` | Compila o núcleo C++ (`build/libcent_runtime.so`) |
| `make demo` | Roda a demo microarquitetural do Llama-2 7B |
| `make simulate` | Executa a demo com simulação de ciclos no Ramulator 2 |
| `make test` | Roda os testes unitários em Python |
| `make clean` | Remove arquivos de build, traces `.trace` e caches |

> **Nota sobre `make venv-clean`**:  
> Como comandos `make` executam em um subshell (processo filho), eles removem os arquivos do `.venv` em disco, mas não podem alterar variáveis da sua sessão ativa de terminal. Se o ambiente estiver ativo, execute primeiro o comando de desativação:
> ```bash
> deactivate && make venv-clean
> ```
