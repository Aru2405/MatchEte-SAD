"""
run_all.py
==========
Ejecuta los 5 scripts en orden para regenerar TODOS los CSV desde cero.
Útil cuando lleguen los train.csv/dev.csv oficiales o cambien los datos
fuente.

Uso:
    cd tableau
    python run_all.py
"""

import subprocess
import sys
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "01_prepare_data.py",
    "02_geocode.py",
    "03_aggregations.py",
    "04_word_frequencies.py",
    "05_extra_insights.py",
]


def main():
    t0 = time.time()
    for script in SCRIPTS:
        path = os.path.join(BASE, script)
        print(f"\n{'='*70}\n>> Ejecutando {script}\n{'='*70}")
        rc = subprocess.call([sys.executable, path], cwd=BASE)
        if rc != 0:
            print(f"\n[ERROR] {script} salió con código {rc}. Abortando.")
            sys.exit(rc)
    print(f"\n{'='*70}")
    print(f"[OK] Todo regenerado en {time.time() - t0:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
