import pandas as pd
import joblib
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

def preprocesar(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    # Limpieza estándar
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def main():
    # 1. CARGAR CONFIGURACIÓN
    with open('configuration.json', 'r') as f:
        config = json.load(f)
    
    p = config['project_params']
    a = config['algorithm_params']

    # 2. INFORME DE ESTRATEGIA
    strategy = p.get('balancing_strategy', 'NONE').upper()
    print(f"MOTOR UNIFICADO - Industria: {p['industry']}")
    print(f"⚖️ Estrategia de Balanceo: {strategy}")

    if strategy == "IA":
        train_file = p['train_ia_path']
        print(f"Cargando datos aumentados (IA): {train_file}")
    else:
        train_file = p['train_path']
        print(f"Cargando datos originales: {train_file}")
    
    df_train = pd.read_csv(train_file)
    df_dev = pd.read_csv(p['dev_path'])

    # 3. PREPROCESAMIENTO
    print(f"Preprocesando texto...")
    df_train[p['text_column']] = df_train[p['text_column']].apply(preprocesar)
    df_dev[p['text_column']] = df_dev[p['text_column']].apply(preprocesar)

    # 4. VECTORIZACIÓN (TF-IDF)
    tfidf_args = a['tfidf']
    vectorizer = TfidfVectorizer(
        max_features=tfidf_args['max_features'],
        ngram_range=tuple(tfidf_args['ngram_range']),
        min_df=tfidf_args['min_df'],
        stop_words=tfidf_args['stop_words'],
        sublinear_tf=True 
    )
    
    X_train_raw = vectorizer.fit_transform(df_train[p['text_column']])
    X_dev = vectorizer.transform(df_dev[p['text_column']])
    y_train = df_train[p['target_column']]
    y_dev = df_dev[p['target_column']]

    # 5. BALANCEO SMOTE (Solo si se activa en el JSON)
    if strategy == "SMOTE":
        print("Aplicando SMOTE...")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train_raw, y_train)
    else:
        X_train = X_train_raw

    # 6. COMPETICIÓN DE MODELOS
    lr_params = a['logistic_regression']
    knn_params = a['knn']

    # Carga de pesos para LR si existen
    pesos = {int(k): v for k, v in lr_params['class_weights'].items()} if 'class_weights' in lr_params else None

    models = {
        "LR": LogisticRegression(
            max_iter=lr_params['max_iter'], 
            C=lr_params['C'], 
            solver=lr_params['solver'],
            class_weight=pesos,
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

    print("\n--- Competición LR vs KNN ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)
        f1 = f1_score(y_dev, preds, average='macro')
        print(f"Modelo {name}: F1-Score Dev = {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\nGANADOR: {best_name} con F1: {best_f1:.4f}")
    
    # 7. GUARDAR RESULTADOS
    joblib.dump(best_model, p['model_name'])
    joblib.dump(vectorizer, p['vectorizer_name'])
    print(f"Archivos {p['model_name']} y {p['vectorizer_name']} listos para test.py")

if __name__ == "__main__":
    main()
