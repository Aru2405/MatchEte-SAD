"""
03_aggregations.py
==================
Genera CSVs con agregaciones pre-calculadas que responden a las preguntas
mínimas del enunciado. Tableau puede agregar solo, pero tenerlos en disco:
  - Acelera el armado de dashboards.
  - Sirven como tablas para el póster.
  - Ayudan a verificar números antes de presentarlos al "jefe".

Entrada: data/dataset_geo.csv
Salida:  data/agg_*.csv
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")


def main():
    df = pd.read_csv(os.path.join(DATA, "dataset_geo.csv"))
    df["date"] = pd.to_datetime(df["date"])

    # ---------- 1. Distribución global por app y sentimiento ----------
    g = df.groupby(["app", "sentiment"]).size().reset_index(name="n")
    g["pct_within_app"] = g.groupby("app")["n"].transform(lambda x: round(100 * x / x.sum(), 2))
    g.to_csv(os.path.join(DATA, "agg_app_sentiment.csv"), index=False)

    # ---------- 2. Score medio por app ----------
    score_app = df.groupby("app")["score"].agg(["mean", "median", "std", "count"]).round(3)
    score_app.to_csv(os.path.join(DATA, "agg_score_summary.csv"))

    # ---------- 3. País + app: número de reviews y score medio ----------
    pa = df.groupby(["country", "iso3", "continent", "app"]).agg(
        n=("score", "count"),
        avg_score=("score", "mean"),
        pct_negative=("sentiment", lambda x: round(100 * (x == "negative").sum() / len(x), 2)),
        pct_positive=("sentiment", lambda x: round(100 * (x == "positive").sum() / len(x), 2)),
    ).reset_index()
    pa["avg_score"] = pa["avg_score"].round(3)
    pa.to_csv(os.path.join(DATA, "agg_country_app.csv"), index=False)

    # ---------- 4. Evolución mensual por app y sentimiento ----------
    ev = df.groupby(["year_month", "app", "sentiment"]).size().reset_index(name="n")
    ev = ev.sort_values(["year_month", "app", "sentiment"])
    ev.to_csv(os.path.join(DATA, "agg_evolution_monthly.csv"), index=False)

    # ---------- 5. Evolución anual con score medio ----------
    ea = df.groupby(["year", "app"]).agg(
        n=("score", "count"),
        avg_score=("score", "mean"),
        pct_negative=("sentiment", lambda x: round(100 * (x == "negative").sum() / len(x), 2)),
    ).reset_index()
    ea["avg_score"] = ea["avg_score"].round(3)
    ea.to_csv(os.path.join(DATA, "agg_evolution_yearly.csv"), index=False)

    # ---------- 6. Género por app y sentimiento ----------
    gen = df.groupby(["app", "gender", "sentiment"]).size().reset_index(name="n")
    gen["pct_within_app_gender"] = gen.groupby(["app", "gender"])["n"].transform(
        lambda x: round(100 * x / x.sum(), 2)
    )
    gen.to_csv(os.path.join(DATA, "agg_gender.csv"), index=False)

    # ---------- 7. Top 20 países por volumen ----------
    top = df.groupby("country").size().reset_index(name="n").sort_values("n", ascending=False).head(20)
    top.to_csv(os.path.join(DATA, "agg_top20_countries.csv"), index=False)

    # ---------- 8. Comparativa Boo vs Hinge por país (formato wide) ----------
    pivot = df.groupby(["country", "iso3", "continent", "app"])["score"].mean().unstack("app").round(3)
    pivot.columns = [f"avg_score_{c}" for c in pivot.columns]
    pivot["delta_Boo_vs_Hinge"] = (pivot.get("avg_score_Boo", 0) - pivot.get("avg_score_Hinge", 0)).round(3)
    pivot.to_csv(os.path.join(DATA, "agg_country_comparison.csv"))

    # ---------- 9. Longitud de comentario por sentimiento (insight extra) ----------
    longitud = df.groupby(["app", "sentiment"]).agg(
        avg_words=("word_count", "mean"),
        avg_chars=("content_length", "mean"),
        n=("word_count", "count"),
    ).round(1).reset_index()
    longitud.to_csv(os.path.join(DATA, "agg_length_by_sentiment.csv"), index=False)

    print("=" * 60)
    print("[OK] Agregaciones generadas en data/:")
    for f in sorted(os.listdir(DATA)):
        if f.startswith("agg_"):
            path = os.path.join(DATA, f)
            print(f"  - {f:40s} ({os.path.getsize(path):>7} bytes)")
    print("=" * 60)

    # Pequeño resumen de insights clave
    print("\n[INSIGHTS DE PARTIDA — útiles para la historia]:")
    print(f"\n  Score medio:")
    print(score_app[["mean", "count"]])

    print(f"\n  % reviews negativas por app:")
    pct_neg = df.groupby("app")["sentiment"].apply(lambda x: round(100 * (x == "negative").sum() / len(x), 2))
    print(pct_neg)

    print(f"\n  Top 5 países donde Boo gana a Hinge (mayor delta):")
    comp = pd.read_csv(os.path.join(DATA, "agg_country_comparison.csv"))
    print(comp.dropna(subset=["delta_Boo_vs_Hinge"]).nlargest(5, "delta_Boo_vs_Hinge")[
        ["country", "avg_score_Boo", "avg_score_Hinge", "delta_Boo_vs_Hinge"]
    ].to_string(index=False))

    print(f"\n  Top 5 países donde Hinge gana a Boo (delta más negativa):")
    print(comp.dropna(subset=["delta_Boo_vs_Hinge"]).nsmallest(5, "delta_Boo_vs_Hinge")[
        ["country", "avg_score_Boo", "avg_score_Hinge", "delta_Boo_vs_Hinge"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
