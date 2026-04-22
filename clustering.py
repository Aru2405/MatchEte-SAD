"""
Clustering de topicos -- Boo vs Hinge
Sistemas de Ayuda a la Decision

Uso:
  python clustering.py -i Boo.csv -a boo
  python clustering.py -i Hinge.csv -a hinge

Outputs:
  clusters_boo_negative.csv   <- reviews negativas con cluster asignado (para Tableau)
  clusters_boo_positive.csv   <- reviews positivas con cluster asignado (para Tableau)
  topwords_boo.json           <- palabras significativas por cluster
  codo_boo.png                <- grafico del codo (obligatorio en el poster)
  (mismo para hinge)
"""

import argparse
import json
import os
import re
import warnings
warnings.filterwarnings("ignore")

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # sin ventana grafica, guarda directo a fichero

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD  # para reducir dimensiones antes de silhouette

for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Palabras genericas de reviews de apps que no aportan informacion de topico
EXTRA_STOPWORDS = {
    "app", "application", "use", "used", "using", "would", "could",
    "get", "got", "one", "really", "like", "just", "make", "good",
    "great", "love", "hate", "bad", "time", "need", "want", "know",
    "people", "thing", "way", "give", "go", "see", "even", "much",
    "also", "still", "well", "im", "ive", "dont", "doesnt", "cant",
    "update", "version", "star", "rating", "review", "please"
}


# ── Preprocesado ──────────────────────────────────────────────────────────────

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        lemmatizer.lemmatize(t) for t in text.split()
        if t not in stop_words
        and t not in EXTRA_STOPWORDS
        and len(t) > 3
    ]
    return " ".join(tokens)


# ── Carga y separacion por sentimiento ───────────────────────────────────────

def load_and_split(filepath):
    """
    Carga el CSV y separa en positivas y negativas.
    Las neutras se descartan para clustering (son demasiado ambiguas).
    """
    df = pd.read_csv(filepath)

    # Detectar columnas
    text_col  = next((c for c in df.columns if c.lower() in ["text","review","comment","content"]), None)
    label_col = next((c for c in df.columns if c.lower() in ["label","sentiment","class","score"]), None)

    if not text_col or not label_col:
        raise ValueError(
            f"No se encontraron columnas de texto/etiqueta.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text", "label"])
    df["text_clean"] = df["text"].apply(preprocess)
    df = df[df["text_clean"].str.strip() != ""]  # eliminar textos vacios tras preprocesar

    # Mapear etiquetas a nombres legibles
    # Soporta tanto numerico (0,1,2) como texto (negative, neutral, positive)
    label_map_num  = {0: "negative", 1: "neutral", 2: "positive"}
    label_map_text = {"negative": "negative", "neutral": "neutral", "positive": "positive"}

    if df["label"].dtype in [np.int64, np.float64]:
        df["sentiment"] = df["label"].map(label_map_num)
    else:
        df["sentiment"] = df["label"].str.lower().map(label_map_text)

    neg_df = df[df["sentiment"] == "negative"].copy().reset_index(drop=True)
    pos_df = df[df["sentiment"] == "positive"].copy().reset_index(drop=True)

    print(f"  Reviews negativas: {len(neg_df)}")
    print(f"  Reviews positivas: {len(pos_df)}")
    print(f"  Reviews neutras:   {len(df[df['sentiment']=='neutral'])} (descartadas para clustering)\n")

    return neg_df, pos_df


# ── Vectorizacion para clustering ────────────────────────────────────────────

def vectorize_for_clustering(texts):
    """
    TF-IDF especifico para clustering.
    - Sin sublinear_tf: queremos frecuencias reales para detectar topicos
    - max_df=0.85: ignorar palabras que aparecen en >85% de docs (demasiado genericas)
    - min_df=3: ignorar palabras muy raras (ruido)
    """
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=False,   # frecuencias reales para clustering
    )
    X = tfidf.fit_transform(texts)
    return X, tfidf


# ── Grafico del codo ─────────────────────────────────────────────────────────

def plot_elbow(X, app_name, sentiment, k_range=range(2, 11), output_dir="."):
    """
    Genera el grafico del codo con inercia y silhouette.
    Este grafico es OBLIGATORIO en el poster segun el enunciado.
    """
    inertias    = []
    silhouettes = []

    # Reducir dimensiones para silhouette (muy costoso en alta dimension)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)

    print(f"  Calculando inercia y silhouette para K=2..{max(k_range)}...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_reduced, labels, sample_size=min(2000, X.shape[0]))
        silhouettes.append(sil)
        print(f"    K={k} | Inercia: {km.inertia_:.0f} | Silhouette: {sil:.4f}")

    # Plot con dos ejes Y
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(list(k_range), inertias,    "bo-", linewidth=2, markersize=7, label="Inercia")
    ax2.plot(list(k_range), silhouettes, "rs--", linewidth=2, markersize=7, label="Silhouette")

    ax1.set_xlabel("Numero de clusters (K)", fontsize=12)
    ax1.set_ylabel("Inercia", color="blue", fontsize=11)
    ax2.set_ylabel("Silhouette Score", color="red", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(f"Grafico del Codo — {app_name.upper()} ({sentiment})", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    fname = os.path.join(output_dir, f"codo_{app_name}_{sentiment}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Grafico guardado: {fname}\n")

    # Devolver el K optimo segun silhouette
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"  K optimo segun Silhouette: {best_k}")
    return best_k, inertias, silhouettes


# ── Clustering y palabras significativas ─────────────────────────────────────

def cluster_and_extract_words(X, tfidf, texts_clean, k, app_name, sentiment, output_dir="."):
    """
    Aplica KMeans con K elegido y extrae las palabras MAS SIGNIFICATIVAS
    por cluster usando TF-IDF within-cluster (NO frecuencia bruta).

    La diferencia clave:
    - Frecuencia bruta: "app", "good", "bad" aparecen en todos -> inutil
    - TF-IDF within-cluster: palabras que son frecuentes en ESE cluster
      pero raras en los demas -> verdaderamente discriminativas
    """
    print(f"  Aplicando KMeans con K={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    feature_names = np.array(tfidf.get_feature_names_out())
    results = {}

    print(f"\n  TOP PALABRAS SIGNIFICATIVAS POR CLUSTER ({sentiment.upper()}):")
    print(f"  {'─'*55}")

    for cluster_id in range(k):
        # Seleccionamos solo los documentos de este cluster
        mask = labels == cluster_id
        n_docs = mask.sum()

        if n_docs < 3:
            continue

        # TF-IDF within-cluster: media de los vectores TF-IDF del cluster
        # Esto da el peso promedio de cada palabra EN ese cluster
        cluster_tfidf_mean = np.asarray(X[mask].mean(axis=0)).flatten()

        # Top 15 palabras por peso TF-IDF medio en el cluster
        top_idx   = cluster_tfidf_mean.argsort()[-15:][::-1]
        top_words = feature_names[top_idx].tolist()
        top_scores = cluster_tfidf_mean[top_idx].tolist()

        results[f"cluster_{cluster_id}"] = {
            "n_docs": int(n_docs),
            "top_words": top_words,
            "top_scores": [round(s, 4) for s in top_scores],
        }

        print(f"  Cluster {cluster_id} ({n_docs} reviews):")
        print(f"    {', '.join(top_words[:10])}")

    print()

    # Guardar CSV con cluster asignado (para Tableau)
    df_out = pd.DataFrame({"text": texts_clean, "cluster": labels, "sentiment": sentiment})
    csv_path = os.path.join(output_dir, f"clusters_{app_name}_{sentiment}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"  CSV guardado: {csv_path}")

    return results, labels


# ── Visualizacion de palabras por cluster ────────────────────────────────────

def plot_top_words(results, app_name, sentiment, output_dir="."):
    """Grafico de barras horizontales con las top palabras de cada cluster."""
    n_clusters = len(results)
    if n_clusters == 0:
        return

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))
    if n_clusters == 1:
        axes = [axes]

    color = "#E24B4A" if sentiment == "negative" else "#2E86AB"

    for ax, (cname, cdata) in zip(axes, results.items()):
        words  = cdata["top_words"][:10][::-1]
        scores = cdata["top_scores"][:10][::-1]
        ax.barh(words, scores, color=color, alpha=0.8)
        ax.set_title(f"{cname}\n({cdata['n_docs']} reviews)", fontsize=10)
        ax.set_xlabel("TF-IDF medio", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle(
        f"Palabras significativas por cluster — {app_name.upper()} ({sentiment})",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    fname = os.path.join(output_dir, f"topwords_{app_name}_{sentiment}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafico palabras guardado: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clustering Boo vs Hinge")
    parser.add_argument("-i", "--input",  required=True, help="CSV de entrada (ej. Boo.csv)")
    parser.add_argument("-a", "--app",    required=True, help="Nombre de la app (boo o hinge)")
    parser.add_argument("-k", "--k_neg",  type=int, default=0,
                        help="K para negativas (0=auto por silhouette)")
    parser.add_argument("-K", "--k_pos",  type=int, default=0,
                        help="K para positivas (0=auto por silhouette)")
    parser.add_argument("-o", "--output", default=".", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    app = args.app.lower()

    print(f"\n{'='*60}")
    print(f"  CLUSTERING — {app.upper()}")
    print(f"{'='*60}\n")

    # 1. Carga y separacion
    print("[PASO 1] Cargando y separando por sentimiento...")
    neg_df, pos_df = load_and_split(args.input)

    all_topwords = {}

    # 2. Clustering NEGATIVAS
    print("[PASO 2] Clustering de reviews NEGATIVAS")
    print(f"  {'─'*55}")
    X_neg, tfidf_neg = vectorize_for_clustering(neg_df["text_clean"])

    if args.k_neg > 0:
        k_neg = args.k_neg
        print(f"  K fijado manualmente: {k_neg}")
    else:
        k_neg, _, _ = plot_elbow(X_neg, app, "negative", range(2, 9), args.output)

    neg_results, neg_labels = cluster_and_extract_words(
        X_neg, tfidf_neg, neg_df["text_clean"].values, k_neg, app, "negative", args.output
    )
    plot_top_words(neg_results, app, "negative", args.output)
    all_topwords["negative"] = neg_results

    # 3. Clustering POSITIVAS
    print("\n[PASO 3] Clustering de reviews POSITIVAS")
    print(f"  {'─'*55}")
    X_pos, tfidf_pos = vectorize_for_clustering(pos_df["text_clean"])

    if args.k_pos > 0:
        k_pos = args.k_pos
        print(f"  K fijado manualmente: {k_pos}")
    else:
        k_pos, _, _ = plot_elbow(X_pos, app, "positive", range(2, 9), args.output)

    pos_results, pos_labels = cluster_and_extract_words(
        X_pos, tfidf_pos, pos_df["text_clean"].values, k_pos, app, "positive", args.output
    )
    plot_top_words(pos_results, app, "positive", args.output)
    all_topwords["positive"] = pos_results

    # 4. Guardar JSON de topwords (para el poster y Tableau)
    json_path = os.path.join(args.output, f"topwords_{app}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_topwords, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] JSON de palabras clave guardado: {json_path}")

    print(f"\n{'='*60}")
    print(f"  CLUSTERING COMPLETADO — {app.upper()}")
    print(f"  Outputs en: {args.output}/")
    print(f"    codo_{app}_negative.png      <- para el poster")
    print(f"    codo_{app}_positive.png      <- para el poster")
    print(f"    topwords_{app}_negative.png  <- para el poster")
    print(f"    topwords_{app}_positive.png  <- para el poster")
    print(f"    clusters_{app}_negative.csv  <- para Tableau")
    print(f"    clusters_{app}_positive.csv  <- para Tableau")
    print(f"    topwords_{app}.json          <- resumen completo")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
