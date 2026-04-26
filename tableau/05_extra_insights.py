"""
05_extra_insights.py
====================
Análisis exploratorios adicionales que sirven como "preguntas propias" en
la Story de Tableau. El enunciado da 0.2 puntos extra por proponer
preguntas nuevas no planteadas por la profesora.

Genera CSVs y los imprime para inspirar las slides.

Entrada: data/dataset_geo.csv
Salida:  data/extra_*.csv
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")


def main():
    df = pd.read_csv(os.path.join(DATA, "dataset_geo.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    print("=" * 70)
    print("PREGUNTAS PROPIAS — material para slides extra")
    print("=" * 70)

    # ---------- P1: ¿Hay sesgo de género en la valoración? ----------
    print("\n[P1] ¿Las mujeres puntúan distinto que los hombres?")
    g = df.groupby(["app", "gender"])["score"].agg(["mean", "count"]).round(3)
    print(g)
    g.to_csv(os.path.join(DATA, "extra_gender_bias.csv"))

    # ---------- P2: ¿Hay estacionalidad (mes del año)? ----------
    print("\n[P2] ¿Existe estacionalidad mensual en las quejas?")
    s = df.groupby(["app", "month"]).agg(
        n=("score", "count"),
        avg_score=("score", "mean"),
        pct_negative=("sentiment", lambda x: round(100 * (x == "negative").sum() / len(x), 2)),
    ).round(3).reset_index()
    s.to_csv(os.path.join(DATA, "extra_seasonality.csv"), index=False)
    print(s.head(12))

    # ---------- P3: ¿Los críticos escriben más largo que los contentos? ----------
    print("\n[P3] Longitud media de comentario por sentimiento:")
    l = df.groupby(["app", "sentiment"]).agg(
        avg_words=("word_count", "mean"),
        median_words=("word_count", "median"),
        n=("word_count", "count"),
    ).round(1)
    print(l)
    # Ya guardado en agg_length_by_sentiment.csv

    # ---------- P4: ¿En qué continente Hinge se nos acerca más? ----------
    print("\n[P4] Score medio por continente y app (Hinge se acerca donde la diff es menor):")
    c = df.groupby(["continent", "app"])["score"].mean().unstack("app").round(3)
    c["delta"] = (c["Boo"] - c["Hinge"]).round(3)
    c = c.sort_values("delta")
    print(c)
    c.to_csv(os.path.join(DATA, "extra_continent_comparison.csv"))

    # ---------- P5: ¿Hay países donde NO tenemos volumen suficiente? ----------
    print("\n[P5] Países con <30 reviews de Boo (mercados emergentes / sub-representados):")
    low = df[df["app"] == "Boo"].groupby("country").size()
    low = low[low < 30].sort_values()
    print(low)
    low.reset_index(name="n_reviews_boo").to_csv(
        os.path.join(DATA, "extra_underrepresented_markets.csv"), index=False
    )

    # ---------- P6: ¿Cuáles son los días/meses pico de quejas? ----------
    print("\n[P6] Mes-año con más volumen de reviews negativas (cualquier app):")
    neg = df[df["sentiment"] == "negative"]
    pico = neg.groupby(["app", "year_month"]).size().reset_index(name="n_negative")
    pico = pico.sort_values("n_negative", ascending=False).head(10)
    print(pico.to_string(index=False))
    pico.to_csv(os.path.join(DATA, "extra_negative_spikes.csv"), index=False)

    # ---------- P7: % de "extremos" (1 estrella vs 5 estrellas) ----------
    print("\n[P7] % de reviews extremas (1-star y 5-star) por app:")
    extr = df.groupby("app")["score"].apply(
        lambda x: pd.Series({
            "pct_1_star": round(100 * (x == 1).sum() / len(x), 2),
            "pct_5_star": round(100 * (x == 5).sum() / len(x), 2),
            "polarization": round(100 * ((x == 1) | (x == 5)).sum() / len(x), 2),
        })
    ).unstack()
    print(extr)
    extr.to_csv(os.path.join(DATA, "extra_polarization.csv"))

    print("\n" + "=" * 70)
    print("[OK] Inspiración para slides extra guardada en data/extra_*.csv")
    print("=" * 70)
    print("""
    Sugerencias de slides extra para la Story (0.2 pts):
      1. "¿Hay sesgo de género?" -> P1 (extra_gender_bias)
      2. "¿Cuándo se quejan más?" -> P2 + P6 (seasonality + spikes)
      3. "Mercados a explorar" -> P5 (underrepresented_markets)
      4. "¿Estamos polarizando?" -> P7 (extra_polarization)
    """)


if __name__ == "__main__":
    main()
