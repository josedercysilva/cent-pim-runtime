#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "============================================================"
echo "    Configurando Ambiente Virtual (CENT PIM Runtime)        "
echo "============================================================"

if ! command -v uv &> /dev/null; then
    echo "[-] 'uv' não encontrado no sistema. Instalando uv automaticamente..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "[+] Usando: $(uv --version)"

echo "[+] Criando .venv com Python 3.11..."
uv venv --python 3.11 .venv

echo "[+] Instalando PyTorch CPU-only..."
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu

echo "[+] Instalando o cent_runtime em modo editável (-e .)..."
uv pip install --python .venv -e .

echo ""
echo "============================================================"
echo " [OK] Ambiente configurado com sucesso!"
echo " Para começar a usar no seu terminal, execute:"
echo ""
echo "     source .venv/bin/activate"
echo "============================================================"
