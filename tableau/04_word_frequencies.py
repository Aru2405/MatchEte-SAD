"""
04_word_frequencies.py
======================
Extrae las palabras más significativas por (app, sentimiento) usando TF-IDF.
Esto es un FALLBACK / PROVISIONAL: el líder de clustering generará los temas
reales con LDA/KMeans + coherencia. Mientras tanto, esto permite preparar
los wordclouds y barras del dashboard.

Sustituir por la salida real de clustering cuando esté disponible
(formato esperado: cluster_id, app, sentiment, word, weight).

Entrada: data/dataset_unified.csv
Salida:  data/words_top_by_segment.csv  (formato largo)
         data/words_pivot_for_wordcloud.csv (formato wide para Tableau)
"""

import os
import re
import pandas as pd
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

# Stopwords mínimas en inglés (no añadimos NLTK para no inflar dependencias).
STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "its", "let", "put", "say", "she", "too", "use", "this",
    "that", "with", "have", "from", "they", "your", "what", "when", "where",
    "will", "would", "there", "their", "been", "would", "could", "should",
    "just", "like", "very", "much", "more", "even", "also", "than", "then",
    "into", "onto", "upon", "over", "under", "after", "before", "while",
    "some", "many", "most", "such", "only", "same", "other", "another",
    "because", "about", "above", "below", "between", "during", "through",
    "still", "again", "ever", "never", "always", "sometimes", "never",
    "thing", "things", "people", "person", "really", "actually", "thank",
    "going", "gonna", "im", "ive", "dont", "doesnt", "wasnt", "isnt",
    "youre", "theyre", "didnt", "wouldnt", "couldnt", "shouldnt",
    "app", "apps", "boo", "hinge",  # nombres de las apps no aportan
}

TOKEN_RE = re.compile(r"[a-z]{3,}")


def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]


def main():
    df = pd.read_csv(os.path.join(DATA, "dataset_unified.csv"))

    rows = []
    for (app, sentiment), grupo in df.groupby(["app", "sentiment"]):
        c = Counter()
        for txt in grupo["content"]:
            c.update(tokenize(txt))
        # top 30 por segmento
        for word, n in c.most_common(30):
            rows.append({
                "app": app,
                "sentiment": sentiment,
                "word": word,
                "freq": n,
                "n_reviews_in_segment": len(grupo),
                "freq_per_review": round(n / len(grupo), 4),
            })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "words_top_by_segment.csv"), index=False)
    print(f"[OK] words_top_by_segment.csv ({len(out)} filas)")

    # Versión wide para wordcloud por (app, sentiment) en Tableau
    pivot = out.pivot_table(
        index="word",
        columns=["app", "sentiment"],
        values="freq",
        fill_value=0,
    )
    pivot.columns = [f"{a}_{s}" for a, s in pivot.columns]
    pivot = pivot.reset_index()
    pivot.to_csv(os.path.join(DATA, "words_pivot_for_wordcloud.csv"), index=False)
    print(f"[OK] words_pivot_for_wordcloud.csv ({len(pivot)} palabras únicas)")

    # Ejemplo: top 10 negativas en Hinge (las más reveladoras)
    print("\n[EJEMPLO] Top 10 palabras en Hinge negativas:")
    print(out[(out["app"] == "Hinge") & (out["sentiment"] == "negative")].head(10).to_string(index=False))

    print("\n[EJEMPLO] Top 10 palabras en Boo positivas:")
    print(out[(out["app"] == "Boo") & (out["sentiment"] == "positive")].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
