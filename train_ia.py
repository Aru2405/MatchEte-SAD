import pandas as pd
import pickle
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

def entrenar_modelo():
    # 1. Validar argumentos de la terminal
    if len(sys.argv) < 3:
        print("\nUso: python entrenar_ia.py <archivo_train.csv> <archivo_dev.csv>")
        print("Ejemplo: python entrenar_ia.py train_balanceado.csv dev.csv")
        return

    file_train = sys.argv[1]
    file_dev = sys.argv[2]

    if not os.path.exists(file_train) or not os.path.exists(file_dev):
        print("Error: Uno de los archivos no existe.")
        return

    # 2. Cargar los datos (usando las columnas que descubrimos antes: content y score)
    print(f"Cargando datos de {file_train} y {file_dev}...")
    df_train = pd.read_csv(file_train).dropna(subset=['content', 'score'])
    df_dev = pd.read_csv(file_dev).dropna(subset=['content', 'score'])

    X_train, y_train = df_train['content'], df_train['score'].astype(str)
    X_dev, y_dev = df_dev['content'], df_dev['score'].astype(str)

    # 3. Crear el Pipeline de la IA
    # Usamos TF-IDF con n-gramas (detecta frases de 1 y 2 palabras)
    print("Entrenando el modelo ganador...")
    modelo_final = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ('clf', LinearSVC(C=0.8, class_weight='balanced', max_iter=2000))
    ])

    modelo_final.fit(X_train, y_train)

    # 4. Evaluación con el set de DEV (Validación)
    y_pred = modelo_final.predict(X_dev)
    score_f1 = f1_score(y_dev, y_pred, average='macro')

    print("\n" + "="*40)
    print(f"RESULTADO EN DEV - F1-SCORE: {score_f1:.4f}")
    print("="*40)
    print(classification_report(y_dev, y_pred))

    # 5. Guardar el "Cerebro" (.pkl)
    nombre_modelo = "mejor_modelo_ia.pkl"
    with open(nombre_modelo, 'wb') as f:
        pickle.dump(modelo_final, f)
   

if __name__ == "__main__":
    entrenar_modelo()
