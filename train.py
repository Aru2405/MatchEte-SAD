import pandas as pd
import json
import joblib
import re
import string
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

def load_config():
    try:
        with open('configuration.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ ERROR: No se encuentra configuration.json.")
        sys.exit(1)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\d+\s*stars?', '', text)
    return " ".join(text.split())

def run_training():
    config = load_config()
    p = config['project_params']
    a = config['algorithm_params']
    
    estrategias = ['None', 'IA', 'SMOTE'] if p['balancing_strategy'] == 'all' else [p['balancing_strategy']]
    modelos_nombres = ['KNN', 'LR'] if p['model_to_run'] == 'all' else [p['model_to_run']]
    
    mejor_f1 = 0
    ganador = None

    print(f"🚀 Iniciando entrenamiento para: {p['industry']}")
    print("-" * 60)

    for est in estrategias:
        actual_path = p['train_ia_path'] if est == 'IA' else p['train_path']
        if not os.path.exists(actual_path):
            print(f"⚠️ Saltando {est}: Archivo {actual_path} no encontrado.")
            continue
            
        df_train = pd.read_csv(actual_path)
        df_dev = pd.read_csv(p['dev_path'])

        df_train['clean'] = df_train[p['text_column']].apply(preprocess_text)
        df_dev['clean'] = df_dev[p['text_column']].apply(preprocess_text)

        tfidf = TfidfVectorizer(
            max_features=a['tfidf']['max_features'],
            ngram_range=tuple(a['tfidf']['ngram_range']),
            stop_words=a['tfidf']['stop_words'],
            sublinear_tf=True,       # Suaviza el impacto de palabras repetitivas
        )
        
        X_train = tfidf.fit_transform(df_train['clean'])
        y_train = df_train[p['target_column']]
        X_dev = tfidf.transform(df_dev['clean'])
        y_dev = df_dev[p['target_column']]

        if est == 'SMOTE':
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

        for m_type in modelos_nombres:
            print(f"🧪 Probando {m_type} + {est}...")
            if m_type == 'KNN':
                model = KNeighborsClassifier(**a['knn'])
            else:
                # --- LÓGICA DE PESOS DINÁMICOS ---
                lr_params = a['logistic_regression']
                # Si no están en el JSON, usamos los pesos del 0.61 por defecto
                raw_weights = lr_params.get('class_weights', {"0": 1.8, "1": 2.3, "2": 0.5})
                
                # Convertir llaves de string a int (el JSON siempre trae strings)
                if isinstance(raw_weights, dict):
                    custom_weights = {int(k): v for k, v in raw_weights.items()}
                else:
                    custom_weights = raw_weights
                
                model = LogisticRegression(
                    max_iter=lr_params['max_iter'],
                    C=lr_params['C'],
                    solver=lr_params['solver'],
                    class_weight=custom_weights
                )

            model.fit(X_train, y_train)
            f1 = f1_score(y_dev, model.predict(X_dev), average='macro')
            print(f"  📈 F1: {f1:.4f}")

            if f1 > mejor_f1:
                mejor_f1 = f1
                ganador = f"{m_type}_{est}"
                # CAMBIO CLAVE: Guardado dinámico
                joblib.dump(model, p['model_name'])
                joblib.dump(tfidf, p['vectorizer_name'])
                print(f"  🌟 Guardado como: {p['model_name']}")

    print("-" * 60)
    print(f"🏆 GANADOR FINAL: {ganador} (F1: {mejor_f1:.4f})")

if __name__ == "__main__":
    run_training()
