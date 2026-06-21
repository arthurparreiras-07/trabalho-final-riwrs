#!/usr/bin/env bash
# Cria o ambiente virtual e instala as dependências do requirements.txt
set -e

cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Setup concluído. Ative o ambiente com: source .venv/bin/activate"
