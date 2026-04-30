import pandas as pd
import numpy as np
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# =========================================================
# 1. DICCIONARIO DE TEMAS (Esto tengo que revisarlo, lo he hecho rapido )
# =========================================================
TOPIC_NAMES = {
    0: "Análisis de Producto y Modelo Premium",
    1: "Satisfacción y Feedback Rápido",
    2: "Seguridad, Moderación y Baneos",
    3: "Frustración, Críticas y Desinstalación",
    4: "Comunidad, Conexión y Dating",
    5: "Usabilidad e Interacción Social"
}

UMBRAL_LONGITUD = 10  # Reseñas > 10 palabras -> LDA, resto -> K-Means

# =========================================================
# 2. PROCESO HÍBRIDO
# =========================================================
datasets = [
    "boo_positive.csv", "boo_negative.csv", "boo_neutral.csv",
    "hinge_positive.csv", "hinge_negative.csv", "hinge_neutral.csv"
]

def process_hybrid():
    all_dfs = []

    for file in datasets:
        if not os.path.exists(file):
            continue
        
        print(f"--- Analizando: {file} ---")
        df = pd.read_csv(file)
        
        # Asegurar que text_final existe y no es nulo
        df = df.dropna(subset=['text_final'])
        
        # Metadatos del archivo
        df['app'] = 'Boo' if 'boo' in file else 'Hinge'
        df['sentimiento_analisis'] = 'positive' if 'positive' in file else ('negative' if 'negative' in file else 'neutral')
        df['num_palabras'] = df['text_final'].apply(lambda x: len(str(x).split()))

        # --- MOTOR A: K-MEANS (Para reseñas cortas) ---
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_tfidf = vectorizer.fit_transform(df['text_final'])
        km = KMeans(n_clusters=6, random_state=42, n_init=10)
        df['kmeans_cluster'] = km.fit_predict(X_tfidf)

        # --- MOTOR B: LDA (Para reseñas largas) ---
        tokens_list = df['text_final'].apply(lambda x: str(x).split()).tolist()
        dictionary = Dictionary(tokens_list)
        corpus = [dictionary.doc2bow(t) for t in tokens_list]
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=6, passes=10, random_state=42)

        def get_lda_data(text):
            bow = dictionary.doc2bow(str(text).split())
            probs = lda.get_document_topics(bow, minimum_probability=0)
            res = {f"Prob_Topico_{i}": 0.0 for i in range(6)}
            for p in probs:
                if isinstance(p, (tuple, list)):
                    res[f"Prob_Topico_{p[0]}"] = p[1]
            
            # Determinamos el cluster ganador del LDA aquí mismo
            res['lda_cluster'] = int(max(res, key=res.get).replace('Prob_Topico_', ''))
            return pd.Series(res)

        # Aplicamos LDA y concatenamos resultados
        lda_info = df['text_final'].apply(get_lda_data)
        df = pd.concat([df, lda_info], axis=1)

        # --- LÓGICA DE DECISIÓN HÍBRIDA ---
        def apply_hybrid_logic(row):
            if row['num_palabras'] > UMBRAL_LONGITUD:
                return row['lda_cluster'], 'LDA'
            else:
                return row['kmeans_cluster'], 'K-Means'

        df[['Cluster_Dominante', 'Modelo']] = df.apply(
            lambda x: pd.Series(apply_hybrid_logic(x)), axis=1
        )

        # Asignar nombres bonitos y extraer palabras clave visuales
        df['Tema_Nombre'] = df['Cluster_Dominante'].map(TOPIC_NAMES)
        df['Palabras_Clave'] = df['text_final'].apply(lambda x: ", ".join(str(x).split()[:8]))

        all_dfs.append(df)

    if not all_dfs:
        print("Error: No se encontraron archivos para procesar.")
        return

    # UNIFICACIÓN Y EXPORTACIÓN
    final_unified = pd.concat(all_dfs, ignore_index=True)
    
    # Lista de columnas definitiva para Tableau (incluyendo tus metadatos)
    columnas_finales = [
        'reviewId', 'content', 'score', 'gender', 'location', 'date', 
        'sentimiento_analisis', 'text_final', 'num_palabras', 'Cluster_Dominante', 
        'Tema_Nombre', 'Modelo', 'Palabras_Clave'
    ] + [f"Prob_Topico_{i}" for i in range(6)]
    
    # Creamos columnas vacías si alguna falta para evitar errores de exportación
    for col in columnas_finales:
        if col not in final_unified.columns:
            final_unified[col] = np.nan

    # Guardar CSV final
    final_unified[columnas_finales].to_csv("AnalisisClustering.csv", index=False)
    print(f"\n ¡PROCESO COMPLETADO!")
    print(f"Archivo generado: AnalisisClustering.csv con {len(final_unified)} registros.")

if __name__ == "__main__":
    process_hybrid()
