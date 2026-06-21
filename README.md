# Detecção de Fake News em Português (Fake.Br Corpus)

Implementação prática referente ao artigo "Detecção automática de fake news em
língua portuguesa: uma abordagem baseada em recuperação de informação e
aprendizado de máquina" (Trabalho final — Recuperação de Informação na Web e
Redes Sociais, PUC Minas).

Para a fundamentação teórica, trabalhos relacionados e discussão dos
resultados, consulte o artigo (`artigo_fakenews.pdf`). Este README cobre
apenas como reproduzir o experimento.

## Estrutura do projeto

```
experimento_fakenews.py     # script principal do experimento
requirements.txt               # dependências Python (equivalente ao package.json)
setup.sh                          # instala o ambiente virtual e as dependências
Fake.br-Corpus/              # corpus de dados (não versionado neste repo)
resultados/                     # saídas geradas pela execução do script
    resultados.csv                # métricas por modelo
    metricas.json                  # métricas + matriz de confusão (JSON)
    grafico_comparacao.png    # gráfico de acurácia/F1 por modelo
    grafico_matriz.png            # matriz de confusão do melhor modelo
```

## Requisitos

- Python 3.12+
- Pacotes: `scikit-learn`, `pandas`, `numpy`, `matplotlib`

## Como obter o corpus

Baixe o corpus Fake.Br (Monteiro et al., 2018) e extraia-o na raiz do
projeto, de forma que o arquivo pré-processado fique em:

```
Fake.br-Corpus/preprocessed/pre-processed.csv
```

Repositório oficial: https://github.com/roneysco/Fake.br-Corpus

## Como executar

```bash
./setup.sh                 # cria o ambiente virtual e instala requirements.txt
source .venv/bin/activate
python experimento_fakenews.py
```

O script imprime as métricas no terminal e gera/sobrescreve os arquivos
`resultados.csv`, `metricas.json`, `grafico_comparacao.png` e
`grafico_matriz.png` dentro da pasta `resultados/`.
