import pandas as pd
import joblib
import json
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

def preprocesar(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    # Limpieza estándar que te dio el 0.55
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def main():
    # 1. CARGAR CONFIGURACIÓN
    with open('configuration.json', 'r') as f:
        config = json.load(f)
    
    p = config['project_params']
    a = config['algorithm_params']

    # 2. CARGA DE DATOS
    train_file = p['train_ia_path'] if p['balancing_strategy'] == "IA" else p['train_path']
    print(f"Cargando entrenamiento desde: {train_file}")
    
    df_train = pd.read_csv(train_file)
    df_dev = pd.read_csv(p['dev_path'])

    # 3. PREPROCESAMIENTO
    print(f"Preprocesando columna: {p['text_column']}...")
    df_train[p['text_column']] = df_train[p['text_column']].apply(preprocesar)
    df_dev[p['text_column']] = df_dev[p['text_column']].apply(preprocesar)

    # 4. VECTORIZACIÓN (TF-IDF)
    print("Vectorizando con TF-IDF (Volviendo a Bigramas)...")
    tfidf_args = a['tfidf']
    vectorizer = TfidfVectorizer(
        max_features=tfidf_args['max_features'],
        ngram_range=tuple(tfidf_args['ngram_range']),
        stop_words=None,
        sublinear_tf=True 
    )
    
    X_train_raw = vectorizer.fit_transform(df_train[p['text_column']])
    X_dev = vectorizer.transform(df_dev[p['text_column']])
    y_train = df_train[p['target_column']]
    y_dev = df_dev[p['target_column']]

    # 5. BALANCEO (SMOTE)
    if p['balancing_strategy'] == "SMOTE":
        print("Aplicando SMOTE...")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train_raw, y_train)
    else:
        X_train = X_train_raw

    # 6. COMPETICIÓN DE MODELOS
    lr_params = a['logistic_regression']
    knn_params = a['knn']

    models = {
        "LR": LogisticRegression(
            max_iter=5000, 
            C=lr_params['C'], 
            solver='lbfgs',
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=knn_params['n_neighbors'],
            weights=knn_params['weights'],
            metric=knn_params['metric']
        )
    }

    best_f1 = 0
    best_model = None
    best_name = ""

    print("\n--- Competición de la IA ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)
        f1 = f1_score(y_dev, preds, average='macro')
        print(f"Modelo {name}: F1-Score Dev = {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\n🏆 GANADOR FINAL: {best_name} con F1: {best_f1:.4f}")
    joblib.dump(best_model, p['model_name'])
    joblib.dump(vectorizer, p['vectorizer_name'])
    
    print(f"Archivos {p['model_name']} y {p['vectorizer_name']} listos. ✅")

if __name__ == "__main__":
    main()
