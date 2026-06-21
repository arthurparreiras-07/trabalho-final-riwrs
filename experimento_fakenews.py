# -*- coding: utf-8 -*-
"""
Detecção de Fake News em português (corpus Fake.Br)
Pipeline: TF-IDF + classificadores supervisionados.
Disciplina: Recuperação de Informação na Web e Redes Sociais - PUC Minas
"""
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

RND = 42
DIR_RESULTADOS = "resultados"
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Carregamento dos dados
# ----------------------------------------------------------------------
df = pd.read_csv("Fake.br-Corpus/preprocessed/pre-processed.csv")
df = df.dropna(subset=["preprocessed_news"])
print(f"Total de notícias: {len(df)}")
print(df["label"].value_counts(), "\n")

X = df["preprocessed_news"].values
y = (df["label"] == "fake").astype(int).values   # 1 = fake, 0 = true

# estatística simples de comprimento
df["n_palavras"] = df["preprocessed_news"].str.split().apply(len)
print("Média de palavras (após pré-proc.) por classe:")
print(df.groupby("label")["n_palavras"].mean().round(1), "\n")

# ----------------------------------------------------------------------
# 2. Divisão treino/teste (estratificada 70/30)
# ----------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, random_state=RND, stratify=y)
print(f"Treino: {len(X_tr)}  |  Teste: {len(X_te)}\n")

# ----------------------------------------------------------------------
# 3. Vetorização TF-IDF (unigramas + bigramas)
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9,
                             sublinear_tf=True)
Xtr = vectorizer.fit_transform(X_tr)
Xte = vectorizer.transform(X_te)
print(f"Dimensão da matriz TF-IDF de treino: {Xtr.shape}")
print(f"Tamanho do vocabulário: {len(vectorizer.vocabulary_)}\n")

# ----------------------------------------------------------------------
# 4. Treino e avaliação dos classificadores
# ----------------------------------------------------------------------
modelos = {
    "Naive Bayes":          MultinomialNB(),
    "Regressão Logística":  LogisticRegression(max_iter=1000, random_state=RND),
    "SVM Linear":           LinearSVC(random_state=RND),
    "Random Forest":        RandomForestClassifier(n_estimators=200,
                                                   n_jobs=-1, random_state=RND),
}

resultados = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)

for nome, modelo in modelos.items():
    t0 = time.time()
    modelo.fit(Xtr, y_tr)
    pred = modelo.predict(Xte)
    dt = time.time() - t0

    acc = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred)
    rec = recall_score(y_te, pred)
    f1 = f1_score(y_te, pred)

    # validação cruzada (acurácia) no conjunto de treino
    cv_scores = cross_val_score(modelo, Xtr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)

    resultados.append({
        "Modelo": nome,
        "Acuracia": round(acc, 4),
        "Precisao": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "CV_media": round(cv_scores.mean(), 4),
        "CV_desvio": round(cv_scores.std(), 4),
        "Tempo_s": round(dt, 2),
    })
    print(f"--- {nome} ---")
    print(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  "
          f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  ({dt:.1f}s)")

res_df = pd.DataFrame(resultados).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n===== TABELA DE RESULTADOS =====")
print(res_df.to_string(index=False))
res_df.to_csv(os.path.join(DIR_RESULTADOS, "resultados.csv"), index=False)

# melhor modelo
melhor_nome = res_df.iloc[0]["Modelo"]
melhor = modelos[melhor_nome]
pred_melhor = melhor.predict(Xte)
cm = confusion_matrix(y_te, pred_melhor)
print(f"\nMelhor modelo: {melhor_nome}")
print("Matriz de confusão [linhas=real, cols=previsto] (0=true,1=fake):")
print(cm)
print("\n", classification_report(y_te, pred_melhor,
      target_names=["Verdadeira", "Falsa"]))

with open(os.path.join(DIR_RESULTADOS, "metricas.json"), "w") as f:
    json.dump({"resultados": resultados, "melhor": melhor_nome,
               "matriz_confusao": cm.tolist(),
               "vocab": len(vectorizer.vocabulary_),
               "treino": len(X_tr), "teste": len(X_te)}, f,
              ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 5. Gráficos
# ----------------------------------------------------------------------
# 5a. comparação de F1 / acurácia
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(res_df))
w = 0.38
ax.bar(x - w/2, res_df["Acuracia"], w, label="Acurácia", color="#1f5c8b")
ax.bar(x + w/2, res_df["F1"], w, label="F1-score", color="#7bb0d6")
ax.set_xticks(x)
ax.set_xticklabels(res_df["Modelo"], rotation=12)
ax.set_ylim(0.6, 1.02)
ax.set_ylabel("Pontuação")
ax.set_title("Desempenho dos classificadores (conjunto de teste)")
ax.legend(loc="lower left")
ax.grid(axis="y", alpha=0.3)
for i, (a, f1) in enumerate(zip(res_df["Acuracia"], res_df["F1"])):
    ax.text(i - w/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i + w/2, f1 + 0.005, f"{f1:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "grafico_comparacao.png"), dpi=150)

# 5b. matriz de confusão do melhor
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Verdadeira", "Falsa"])
ax.set_yticklabels(["Verdadeira", "Falsa"])
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusão – {melhor_nome}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "grafico_matriz.png"), dpi=150)

print(f"\nArquivos gerados em '{DIR_RESULTADOS}/': resultados.csv, metricas.json, "
      "grafico_comparacao.png, grafico_matriz.png")
