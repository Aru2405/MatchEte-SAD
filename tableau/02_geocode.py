"""
02_geocode.py
=============
Añade columnas geográficas (lat, lon, iso3, continent) al dataset unificado
usando un mapping estático de centroides de país (country_centroids.py).

Esto permite a Tableau:
  - Pintar un mapa coroplético sin depender de su geocoder online (más rápido).
  - Agrupar por continente (filtros y storytelling).
  - Calcular distancias entre países si se quisiera (lat/lon numéricos).

Entrada: data/dataset_unified.csv
Salida:  data/dataset_geo.csv
"""

import os
import pandas as pd
from country_centroids import enrich_country

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")


def main():
    src = os.path.join(DATA, "dataset_unified.csv")
    df = pd.read_csv(src)

    enriched = df["country"].apply(enrich_country).apply(pd.Series)
    df = pd.concat([df, enriched], axis=1)

    n_sin_geo = df["lat"].isna().sum()
    print(f"[INFO] Filas sin geocoding: {n_sin_geo} ({100*n_sin_geo/len(df):.2f}%)")
    if n_sin_geo > 0:
        faltan = df[df["lat"].isna()]["country"].value_counts()
        print("[WARN] Países sin centroide:")
        print(faltan)

    out = os.path.join(DATA, "dataset_geo.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {out} ({len(df)} filas, {df.shape[1]} columnas)")
    print(f"     Continentes: {df['continent'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
