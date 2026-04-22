import pandas as pd
import numpy as np
import json
import pickle
from sys import argv
from getopt import getopt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Configuración de NLTK ---
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def main():
    input_file, algo, config_file = "", "LR", "configuration.json"
    opts, _ = getopt(argv[1:], 'i:a:c:')
    for opt, arg in opts:
        if opt == '-i': input_file = arg
        if opt == '-a': algo = arg.upper()
        if opt == '-c': config_file = arg

    # 1. Cargar configuración y CSV
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo de configuración: {config_file}")
        return

    df = pd.read_csv(input_file)

    # 2. Forzar nombres de columnas correctos para Boo.csv
    # Ignoramos lo que diga el JSON en estos campos para evitar errores de clave
    target = 'label'      # La que vamos a crear
    text_col = 'content'  # La que viene en tu CSV

    # 3. Limpieza y creación de etiquetas (Negativo, Neutro, Positivo)
    # Convertimos score a numérico por si viene como texto
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    # Eliminamos filas que no tengan score o contenido de texto
    df = df.dropna(subset=['score', text_col])

    condiciones = [
        (df['score'] <= 2),          # 0: Negativo
        (df['score'] == 3),          # 1: Neutro
        (df['score'] >= 4)           # 2: Positivo
    ]
    valores = [0, 1, 2]
    df[target] = np.select(condiciones, valores)

    lista_resultados = []

    print(f"\n[INFO] Fichero: {input_file} | Algoritmo: {algo}")
    print(f"[INFO] Total muestras válidas: {len(df)}")
    print(f"[INFO] Distribución de clases (0:Neg, 1:Neu, 2:Pos):\n{df[target].value_counts().sort_index().to_string()}\n")

    # ── Limpieza de columnas según configuración ──────────────────────────────
    df = df.drop(columns=config.get('eliminar', []), errors='ignore')

    # ── Preprocesado de texto ─────────────────────────────────────────────────
    print("[INFO] Preprocesando texto... (esto puede tardar)")
    df['text_clean'] = df[text_col].apply(preprocess)
    
    # Eliminar posibles textos que quedaron vacíos tras la limpieza
    df = df[df['text_clean'] != ""]

    # ── SPLIT 70 / 15 / 15 ───────────────────────────────────────────────────
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        df['text_clean'].values,
        df[target].values,
        test_size=0.30,
        random_state=42,
        stratify=df[target].values
    )
    
    X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    n_total = len(df)
    print(f"[INFO] Split 70/15/15 completado.")
    print(f"  Train: {len(X_train_raw)} | Dev: {len(X_dev_raw)} | Test: {len(X_test_raw)}")

    # Guardamos el test split
    pd.DataFrame({'text': X_test_raw, target: y_test}).to_csv("test_split.csv", index=False)

    # ── Vectorizacion TF-IDF ──────────────────────────────────────────────────
    tfidf_cfg = config['tfidf_params']
    tfidf = TfidfVectorizer(
        max_features=tfidf_cfg['max_features'],
        ngram_range=(tfidf_cfg['ngram_min'], tfidf_cfg['ngram_max']),
        min_df=tfidf_cfg['min_df'],
        sublinear_tf=tfidf_cfg['sublinear_tf'],
    )
    
    X_train = tfidf.fit_transform(X_train_raw)
    X_dev   = tfidf.transform(X_dev_raw)
    y_train = y_train_raw

    # ── Balanceo ──────────────────────────────────────────────────────────────
    sampling = config.get('sampling', 'none')
    if sampling == "smote":
        X_train, y_train = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train, y_train)
        print(f"[INFO] SMOTE aplicado. Muestras train: {X_train.shape[0]}")
    elif sampling == "undersampling":
        X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
        print(f"[INFO] Undersampling aplicado. Muestras train: {X_train.shape[0]}")

    # ── Pipeline y Evaluación ─────────────────────────────────────────────────
    def crear_pipeline(modelo):
        steps = [('scaler', MaxAbsScaler()), ('model', modelo)]
        return Pipeline(steps)

    mejor_f1, mejor_pipeline = -1, None
    avg = config.get('metrics_average', 'macro')

    def evaluar(modelo, params_str):
        nonlocal mejor_f1, mejor_pipeline
        p = crear_pipeline(modelo)
        p.fit(X_train, y_train)
        y_pred = p.predict(X_dev)
        score  = f1_score(y_dev, y_pred, average=avg)
        report = classification_report(y_dev, y_pred)
        print(f"  [{algo}] {params_str} -> {avg} F1 dev: {score:.4f}")
        
        lista_resultados.append({
            'Algoritmo': algo, 'Parametros': params_str, 'F1_dev': round(score, 4)
        })
        
        if score > mejor_f1:
            mejor_f1, mejor_pipeline = score, p
            mejor_pipeline._report = report

    # ── Barrido de hiperparametros ────────────────────────────────────────────
    print(f"\n[BARRIDO] Iniciando entrenamiento de {algo}...\n")

    if algo == "LR":
        for C in config['lr_params']['C_values']:
            for solver in config['lr_params']['solvers']:
                evaluar(LogisticRegression(
                    C=C, solver=solver, max_iter=1000,
                    class_weight="balanced", random_state=42), f"C={C}, solver={solver}")

    elif algo == "KNN":
        kp = config['knn_params']
        for k in range(kp['k_min'], kp['k_max'] + 1, 2):
            for p in range(kp['p_min'], kp['p_max'] + 1):
                evaluar(KNeighborsClassifier(n_neighbors=k, p=p), f"k={k}, p={p}")

    elif algo == "RANDOM_F":
        for n in config['rf_params']['n_estimators']:
            evaluar(RandomForestClassifier(
                n_estimators=n, class_weight="balanced", random_state=42), f"n={n}")

    elif algo == "DEC_TREE":
        for d in config['tree_params']['depths']:
            evaluar(DecisionTreeClassifier(
                max_depth=d, class_weight="balanced", random_state=42), f"depth={d}")

    elif algo == "NAIVE_BAYES":
        evaluar(MultinomialNB(), "alpha=1.0")

    elif algo == "SVM":
        from sklearn.svm import LinearSVC
        for C in config['lr_params']['C_values']:
            evaluar(LinearSVC(
                C=C, max_iter=2000,
                class_weight="balanced",
                random_state=42), f"C={C}")

    else:
        print(f"[ERROR] Algoritmo '{algo}' no reconocido.")
        print("Opciones: LR, KNN, RANDOM_F, DEC_TREE, NAIVE_BAYES, SVM")
        
    # ── Finalización y Guardado ───────────────────────────────────────────────
    X_test_vec = tfidf.transform(X_test_raw)
    y_test_pred = mejor_pipeline.predict(X_test_vec)
    test_f1 = f1_score(y_test, y_test_pred, average=avg)
    test_report = classification_report(y_test, y_test_pred)

    pd.DataFrame(lista_resultados).to_csv("RESULTADOS_BARRIDO.csv", index=False)
    pickle.dump({'pipeline': mejor_pipeline, 'tfidf': tfidf}, open("mejorModelo.sav", 'wb'))

    print(f"\n{'='*55}")
    print(f"RESULTADO FINAL:")
    print(f"  Mejor F1 en Dev: {mejor_f1:.4f}")
    print(f"  F1 en Test final: {test_f1:.4f}")
    print(f"{'='*55}")
    print("\nReporte detallado en TEST:")
    print(test_report)

if __name__ == "__main__":
    main()
