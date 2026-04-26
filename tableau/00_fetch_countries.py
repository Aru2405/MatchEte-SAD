"""
00_fetch_countries.py
=====================
Descarga datos VERIFICABLES de países (códigos ISO-3, centroides geográficos,
continentes) desde la API pública REST Countries y regenera el módulo
`country_centroids.py` con esos datos.

Fuente:
    https://restcountries.com/  (API REST gratuita, sin autenticación)
    Endpoint: https://restcountries.com/v3.1/all?fields=name,cca3,latlng,region

Devuelve, para cada país:
    - name.common  -> nombre común en inglés (p.ej. "Spain")
    - cca3         -> código ISO 3166-1 alpha-3 oficial (p.ej. "ESP")
    - latlng       -> [lat, lon] del centroide geográfico
    - region       -> continente según ONU (Africa/Americas/Asia/Europe/Oceania)

Uso:
    python 00_fetch_countries.py
    # Esto sobrescribe country_centroids.py con datos descargados.

Solo dependencias de la stdlib (urllib + json) — no necesita pip install.
"""

import json
import os
import sys
import urllib.request
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
API_URL = "https://restcountries.com/v3.1/all?fields=name,cca3,latlng,region"

# 70 países presentes en Boo.csv + Hinge.csv (los que necesitamos resolver).
NEEDED_COUNTRIES = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Bolivia",
    "Brazil", "Canada", "Chile", "China", "Colombia", "Czech Republic",
    "Denmark", "Ecuador", "Egypt", "England", "Ethiopia", "Fiji",
    "Finland", "France", "Germany", "Ghana", "Greece", "Hungary",
    "India", "Indonesia", "Ireland", "Israel", "Italy", "Japan",
    "Kenya", "Libya", "Malaysia", "Mexico", "Morocco", "Namibia",
    "Nepal", "Netherlands", "New Zealand", "Nigeria", "Norway",
    "Pakistan", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Qatar", "Saudi Arabia", "Senegal", "Singapore",
    "South Africa", "South Korea", "Spain", "Sweden", "Switzerland",
    "Taiwan", "Tanzania", "Thailand", "Tunisia", "Uganda",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
    "Venezuela", "Vietnam", "Zambia", "Zimbabwe",
]

# Aliases para resolver nombres que difieren entre nuestros datos y la API:
#  - "Czech Republic" en los datasets vs "Czechia" en ISO 3166-1 (oficial desde 2016).
#  - "England" no es un país soberano; se mapea a "United Kingdom".
NAME_ALIASES = {
    "Czech Republic": "Czechia",
    "England": "United Kingdom",
}


def fetch_data():
    """Descarga el JSON de la API. Lanza excepción si falla."""
    print(f"[INFO] GET {API_URL}")
    req = urllib.request.Request(API_URL, headers={"User-Agent": "MatchEte-SAD/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    print(f"[INFO] Descargados {len(data)} países desde la API")
    return data


def build_index(api_data):
    """Indexa la respuesta por nombre común y oficial para búsqueda rápida."""
    index = {}
    for entry in api_data:
        for key in ("common", "official"):
            name = entry.get("name", {}).get(key)
            if name:
                index[name] = entry
    return index


def resolve(name, index):
    """Devuelve el registro de la API correspondiente a `name`, aplicando aliases."""
    candidates = [name, NAME_ALIASES.get(name)]
    for c in filter(None, candidates):
        if c in index:
            return index[c]
    return None


def render_module(records, fetched_at):
    """Genera el contenido del nuevo country_centroids.py."""
    lines = [
        '"""',
        "country_centroids.py",
        "====================",
        "Centroides geográficos (lat, lon), códigos ISO 3166-1 alpha-3 y continentes",
        "para los 70 países presentes en Boo.csv y Hinge.csv.",
        "",
        f"Datos descargados el {fetched_at} desde:",
        f"    {API_URL}",
        "",
        "Para regenerar este archivo, ejecuta:",
        "    python 00_fetch_countries.py",
        "",
        "Notas:",
        '  - "Czech Republic" en los datasets se mapea a "Czechia" (nombre ISO oficial).',
        '  - "England" se trata como parte de "United Kingdom" (no es país soberano).',
        '"""',
        "",
        "# country_name -> (lat, lon, iso3)",
        "COUNTRY_CENTROIDS = {",
    ]
    # Calcula el ancho máximo del nombre para alinear visualmente
    width = max(len(repr(r["our_name"])) for r in records)
    for r in records:
        lines.append(
            f'    {r["our_name"]!r:<{width}}: ({r["lat"]:>9.4f}, {r["lon"]:>9.4f}, {r["iso3"]!r}),'
        )
    lines.append("}")
    lines.append("")
    lines.append("# Continente por país (campo `region` de la API REST Countries).")
    lines.append("COUNTRY_TO_CONTINENT = {")
    width2 = max(len(repr(r["our_name"])) for r in records)
    for r in records:
        lines.append(f'    {r["our_name"]!r:<{width2}}: {r["continent"]!r},')
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def enrich_country(country):")
    lines.append('    """Devuelve dict con lat, lon, iso3, continent. None si no está mapeado."""')
    lines.append("    if country not in COUNTRY_CENTROIDS:")
    lines.append('        return {"lat": None, "lon": None, "iso3": None, "continent": None}')
    lines.append("    lat, lon, iso3 = COUNTRY_CENTROIDS[country]")
    lines.append("    return {")
    lines.append('        "lat": lat,')
    lines.append('        "lon": lon,')
    lines.append('        "iso3": iso3,')
    lines.append('        "continent": COUNTRY_TO_CONTINENT.get(country),')
    lines.append("    }")
    return "\n".join(lines) + "\n"


def main():
    try:
        api_data = fetch_data()
    except Exception as e:
        print(f"[ERROR] No se pudo descargar la API: {e}", file=sys.stderr)
        print("[INFO] Revisa tu conexión y vuelve a intentarlo.", file=sys.stderr)
        sys.exit(1)

    index = build_index(api_data)

    records = []
    missing = []
    for name in NEEDED_COUNTRIES:
        entry = resolve(name, index)
        if not entry:
            missing.append(name)
            continue
        latlng = entry.get("latlng") or [None, None]
        records.append({
            "our_name":  name,
            "iso3":      entry["cca3"],
            "lat":       latlng[0],
            "lon":       latlng[1],
            "continent": entry.get("region"),
        })

    if missing:
        print(f"[ERROR] No resueltos: {missing}", file=sys.stderr)
        sys.exit(2)

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_path = os.path.join(BASE, "country_centroids.py")
    new_content = render_module(records, fetched_at)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[OK] {out_path} regenerado con {len(records)} países verificados.")
    print(f"     Continentes: ", end="")
    from collections import Counter
    print(dict(Counter(r["continent"] for r in records)))


if __name__ == "__main__":
    main()
