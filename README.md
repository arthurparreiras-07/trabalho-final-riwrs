# Detecção de Fake News em Português (Fake.Br Corpus)

Implementação prática referente ao artigo "Detecção automática de fake news em
língua portuguesa: uma abordagem baseada em recuperação de informação e
aprendizado de máquina" (Trabalho final — Recuperação de Informação na Web e
Redes Sociais, PUC Minas).

## Grupo

- Disciplina: Recuperação de Informação na Web e Redes Sociais
- Professora: Jeanne Louize Emygdio
- Grupo 3:
  - Ana Julia Ferreira Soares
  - Arthur Felipe Parreiras
  - Bruno Maciel dos Santos
  - Guilherme Saliba Moreira de Oliveira
  - Tiago Rafael Martins Cardoso
  - Victoria Gonçalves da Silva

## Estrutura do projeto

```
experimento_fakenews.py     # script principal do experimento
requirements.txt               # dependências Python (equivalente ao package.json)
requirements-dev.txt          # dependências de desenvolvimento (lint)
pyproject.toml                  # configuração do Ruff (linter)
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

## Lint

O projeto usa o [Ruff](https://docs.astral.sh/ruff/) como linter.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check .
```

## Como o script funciona

O `experimento_fakenews.py` executa o pipeline do experimento em sequência:

1. **Carregamento dos dados**: lê `Fake.br-Corpus/preprocessed/pre-processed.csv`
   e descarta linhas sem texto.
2. **Divisão treino/teste**: separa 70% para treino e 30% para teste, de
   forma estratificada (mantendo a proporção de notícias verdadeiras/falsas).
3. **Vetorização TF-IDF**: transforma os textos em vetores numéricos
   (unigramas e bigramas), ajustando o vetorizador apenas no conjunto de
   treino e aplicando-o ao teste.
4. **Treino e avaliação**: treina quatro classificadores (Naive Bayes,
   Regressão Logística, SVM Linear e Random Forest) sobre a mesma
   representação TF-IDF, calculando acurácia, precisão, recall, F1-score e
   validação cruzada (5-fold) para cada um.
5. **Seleção do melhor modelo**: ordena os classificadores pelo F1-score e
   gera a matriz de confusão do melhor.
6. **Geração de saídas**: salva métricas e gráficos na pasta `resultados/`.

## Output gerado

Durante a execução, o script imprime no terminal: contagem de notícias por
classe, estatísticas de tamanho dos textos, dimensão da matriz TF-IDF,
métricas de cada classificador e o relatório de classificação do melhor
modelo.

Ao final, são gerados/sobrescritos os seguintes arquivos em `resultados/`:

- **`resultados.csv`** — tabela com as métricas (acurácia, precisão, recall,
  F1, média/desvio da validação cruzada e tempo de treino) de cada
  classificador, ordenada pelo F1-score.
- **`metricas.json`** — as mesmas métricas em formato JSON, acrescidas do
  nome do melhor modelo, da matriz de confusão, do tamanho do vocabulário e
  do tamanho dos conjuntos de treino/teste.
- **`grafico_comparacao.png`** — gráfico de barras comparando acurácia e
  F1-score entre os quatro classificadores.
- **`grafico_matriz.png`** — matriz de confusão (verdadeira x falsa) do
  melhor modelo.
