import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================
datasets = [
    "boo_positive.csv", "boo_negative.csv", "boo_neutral.csv",
    "hinge_positive.csv", "hinge_negative.csv", "hinge_neutral.csv"
]

def run_kmeans_pipeline(file_path):
    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return

    print(f"\n" + "="*50)
    print(f" EJECUTANDO K-MEANS PARA: {file_path}")
    print("="*50)

    df = pd.read_csv(file_path)
    
    # Nos aseguramos de tener texto para procesar
    df = df.dropna(subset=['text_final'])
    
    # 2. VECTORIZACIÓN TF-IDF
    # Convertimos el texto en una matriz numérica ponderada
    vectorizer = TfidfVectorizer(
        max_features=1500, 
        stop_words='english', # Por si se coló alguna stopword
        ngram_range=(1, 2)    # Para capturar bigramas como "not_work"
    )
    X = vectorizer.fit_transform(df['text_final'])

    # 3. MÉTODO DEL CODO (ELBOW METHOD)
    # Calculamos la inercia para distintos valores de K
    inercias = []
    rango_k = range(2, 11)
    
    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inercias.append(km.inertia_)

    # Guardar gráfico del codo para el póster
    plt.figure(figsize=(8, 5))
    plt.plot(rango_k, inercias, marker='o', color='royalblue')
    plt.title(f'Método del Codo - {file_path}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia (Suma de cuadrados)')
    plt.grid(True)
    plot_name = f"elbow_{file_path.replace('.csv', '.png')}"
    plt.savefig(plot_name)
    print(f"-> Gráfico del codo guardado como: {plot_name}")

    # 4. APLICAR K-MEANS FINAL
    # Usamos un K basado en tus resultados de LDA (ej: 6) o el "codo" detectado
    # Para automatizar, usaremos 6, pero puedes ajustarlo según el gráfico
    k_final = 6 
    kmeans_model = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    df['cluster_id'] = kmeans_model.fit_predict(X)

    # 5. EXTRAER PALABRAS CLAVE POR CLUSTER
    # Esto te sirve para saber de qué trata cada grupo en Tableau
    print("\nPalabras clave por Cluster:")
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    for i in range(k_final):
        top_terms = [terms[ind] for ind in order_centroids[i, :8]]
        print(f"  Cluster {i}: {', '.join(top_terms)}")

    # 6. GUARDAR RESULTADO PARA TABLEAU
    output_name = f"clustered_{file_path}"
    df.to_csv(output_name, index=False)
    print(f"\n-> CSV final guardado: {output_name}")

# Ejecutar para todos los archivos
for dataset in datasets:
    run_kmeans_pipeline(dataset)

print("\nPROCESO DE CLUSTERING FINALIZADO.")
