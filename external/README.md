# Instruções de Configuração do Simulador Ramulator 2.0 (AiM Simulator)

Este diretório (`external/`) é o **caminho padrão** onde o `cent_runtime` busca o simulador de ciclo exato da arquitetura **CENT** (*ASPLOS 2025*).

Por boas práticas de engenharia e controle de versão, **os arquivos binários pesados de compilação não são versionados no Git**.

---

## 1. Arquivos Esperados Nesta Pasta

Para que as simulações físicas (`simulate=True` ou `make simulate`) funcionem, esta pasta deve conter os seguintes arquivos:

| Nome do Arquivo | Descrição |
| :--- | :--- |
| **`ramulator2`** | Executável principal do simulador de ciclo exato AiM. |
| **`libramulator.so`** | Biblioteca compartilhada que contém o controlador GDDR6-AiM e modelos de DRAM. |

---

## 2. Passo a Passo para Obter e Compilar o CENT

Repositório Oficial Upstream: [https://github.com/Yufeng98/CENT.git](https://github.com/Yufeng98/CENT.git)

> [!WARNING]
> ### ⚠️ Aviso Importante sobre a Compilação do CENT Original
> O repositório original do CENT foi desenvolvido como um artefato acadêmico pontual e **os autores originais tiveram zero preocupação com a experiência do usuário**. 
> 
> O código original é frágil, não possui scripts de instalação amigáveis, tem dependências implícitas e costuma quebrar com facilidade na compilação dependendo da versão do seu compilador (`gcc`/`g++`), `cmake` ou bibliotecas do sistema. Prepare-se para enfrentar erros de build no repositório deles até conseguir gerar os binários. Tenha paciência ao compilar.

### Passo 2.1. Clonar e Compilar no repositório do CENT
Em qualquer diretório do seu computador (fora deste projeto):

```bash
# Clone o repositório original
git clone https://github.com/Yufeng98/CENT.git
cd CENT/aim_simulator

# Prepare o diretório de build
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

*(Se o `make` falhar com erros de versão de C++, flags estritas ou bibliotecas ausentes, você precisará ajustar as dependências do ambiente deles antes de prosseguir).*

### Passo 2.2. Copiar os binários gerados para esta pasta
Assim que a compilação do CENT terminar e gerar o executável e a biblioteca compartilhada, copie ambos para cá:

```bash
# Estando na raiz do repositório cent_runtime:
cp /caminho/para/CENT/aim_simulator/build/ramulator2 external/
cp /caminho/para/CENT/aim_simulator/libramulator.so external/

# Garanta permissão de execução no binário
chmod +x external/ramulator2
```

*(Uma vez copiados para cá, você nunca mais precisará mexer no código do CENT original — o `cent_runtime` assume o controle e roda de forma limpa e automática).*

---

## 3. Como Validar a Instalação

Na raiz do repositório `cent_runtime`, execute:
```bash
make simulate
```
Ou diretamente via Python:
```bash
python3 examples/llama2_7b_demo.py --simulate
```

Se os binários estiverem corretos, a simulação exibirá os ciclos físicos extraídos do GDDR6-AiM (ex.: **16.416 ciclos** na Atenção do Llama-2 7B e **44.064 ciclos** no MLP).

---

## 4. Alternativa (Sem Copiar Arquivos)

Caso prefira manter os arquivos no local original de compilação sem copiá-los para cá, basta apontar a variável de ambiente:

```bash
export CENT_RAMULATOR_BIN=/caminho/para/CENT/aim_simulator/build/ramulator2
```
O `cent_runtime` detectará essa variável automaticamente e resolverá a biblioteca `libramulator.so` correspondente.
