"""
01_prepare_data.py
==================
Limpia, valida y unifica los CSV de Boo y Hinge en un único dataset
listo para visualización en Tableau.

Pasos:
  1. Lee Boo.csv y Hinge.csv con parser tolerante.
  2. Filtra filas inválidas (score fuera de 1-5, fechas/géneros corruptos).
  3. Deriva columna `sentiment` (negative / neutral / positive) desde score.
  4. Separa `location` en `city` y `country`.
  5. Añade columna `app` ("Boo" / "Hinge") y normaliza nombres de columnas.
  6. Exporta `data/dataset_unified.csv`.

Salida: data/dataset_unified.csv
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT_DIR = os.path.join(BASE, "data")
os.makedirs(OUT_DIR, exist_ok=True)


def score_to_sentiment(score):
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def split_location(loc):
    """'Monterrey, Mexico' -> ('Monterrey', 'Mexico'). Si no hay coma, todo va a country."""
    if not isinstance(loc, str) or not loc.strip():
        return pd.Series([None, None])
    parts = [p.strip() for p in loc.split(",")]
    if len(parts) == 1:
        return pd.Series([None, parts[0]])
    return pd.Series([parts[0], parts[-1]])


def load_and_clean(path, app_name):
    print(f"[INFO] Cargando {app_name} desde {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="warn")

    # Eliminar columnas basura "Unnamed: N"
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    n_inicial = len(df)

    # Validar score numérico 1-5
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["score"].between(1, 5, inclusive="both")]
    df["score"] = df["score"].astype(int)

    # Validar gender
    df = df[df["gender"].isin(["male", "female"])]

    # Validar fecha
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Eliminar contenido vacío
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip() != ""]

    print(f"[INFO]   {n_inicial} -> {len(df)} filas válidas ({n_inicial - len(df)} descartadas)")

    # Derivar columnas
    df["sentiment"] = df["score"].apply(score_to_sentiment)
    df[["city", "country"]] = df["location"].apply(split_location)
    df["app"] = app_name

    # Columnas temporales útiles para Tableau
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    # Longitud del comentario (proxy de "esfuerzo" del usuario)
    df["content_length"] = df["content"].str.len()
    df["word_count"] = df["content"].str.split().str.len()

    return df


def main():
    boo = load_and_clean(os.path.join(ROOT, "Boo.csv"), "Boo")
    hinge = load_and_clean(os.path.join(ROOT, "Hinge.csv"), "Hinge")

    unified = pd.concat([boo, hinge], ignore_index=True)

    cols = [
        "reviewId", "app", "date", "year", "quarter", "year_month",
        "score", "sentiment", "gender",
        "city", "country", "location",
        "content", "content_length", "word_count",
    ]
    unified = unified[cols]

    out = os.path.join(OUT_DIR, "dataset_unified.csv")
    unified.to_csv(out, index=False)

    print()
    print("=" * 60)
    print(f"[OK] Dataset unificado: {out}")
    print(f"     Total filas: {len(unified)}")
    print(f"     Boo:   {(unified['app'] == 'Boo').sum()}")
    print(f"     Hinge: {(unified['app'] == 'Hinge').sum()}")
    print()
    print("Distribución por app y sentimiento:")
    print(unified.groupby(["app", "sentiment"]).size().unstack(fill_value=0))
    print()
    print(f"Rango temporal: {unified['date'].min().date()} -> {unified['date'].max().date()}")
    print(f"Países distintos: {unified['country'].nunique()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
