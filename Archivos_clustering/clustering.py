import pandas as pd
import numpy as np
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. DICCIONARIOS DE TEMAS (Tus nombres reales)
NOMBRES_CLUSTERS = {
    'positive': {
        0: "Calidad de la Comunidad y Perfiles Auténticos",
        1: "Optimización del Tiempo y Éxito en Matches",
        2: "Buena Experiencia de Usuario",
        3: "Eficacia en el Descubrimiento de Personas"
    },
    'negative': {
        0: "Problemas Críticos de Baneos y Gestión de Cuenta",
        1: "Barreras de Pago y Limitaciones de Monetización",
        2: "Inseguridad: Perfiles Falsos y Estafas",
        3: "Mala Comunicación y Mensajería"
    },
    'neutral': {
        0: "Primeros Pasos e Introducción al Sistema",
        1: "Valoración del Modelo de Suscripción",
        2: "Opinión del Funcionamiento de la App",
        3: "Novedades y Actualizaciones en el Sistema de Pago"
    }
}

UMBRAL_LONGITUD = 10
BEST_K = 4

datasets = [
    "boo_positive.csv", "boo_negative.csv", "boo_neutral.csv",
    "hinge_positive.csv", "hinge_negative.csv", "hinge_neutral.csv"
]

def process_clustering_final_K4():
    all_dfs = []

    for file in datasets:
        if not os.path.exists(file): continue
        
        print(f"Procesando: {file}")
        df = pd.read_csv(file)
        
        # Limpieza de columnas Unnamed y rename de texto
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'text_final' in df.columns:
            df = df.rename(columns={'text_final': 'texto_limpio'})
        
        df = df.dropna(subset=['texto_limpio'])
        
        sentiment = 'positive' if 'positive' in file else ('negative' if 'negative' in file else 'neutral')
        df['sentimiento_analisis'] = sentiment
        df['num_palabras'] = df['texto_limpio'].apply(lambda x: len(str(x).split()))
        df['Longitud'] = df['num_palabras']

        # --- K-MEANS ---
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_tfidf = vectorizer.fit_transform(df['texto_limpio'])
        df['kmeans_cluster'] = KMeans(n_clusters=BEST_K, random_state=42, n_init=10).fit_predict(X_tfidf)

        # --- LDA ---
        tokens = df['texto_limpio'].apply(lambda x: str(x).split()).tolist()
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(t) for t in tokens]
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=BEST_K, passes=10, random_state=42)

        # Probabilidades 
        def get_lda_probs(text):
            bow = dictionary.doc2bow(str(text).split())
            topics = dict(lda.get_document_topics(bow, minimum_probability=0))
            return pd.Series([topics.get(i, 0.0) for i in range(4)])

        lda_res = df['texto_limpio'].apply(get_lda_probs)
        lda_res.columns = ['Prob_Topico_0', 'Prob_Topico_1', 'Prob_Topico_2', 'Prob_Topico_3']
        df = pd.concat([df, lda_res], axis=1)
        df['lda_cluster'] = lda_res.idxmax(axis=1).str.replace('Prob_Topico_', '').astype(int)

        # --- LÓGICA HÍBRIDA ---
        def apply_hybrid(row):
            if row['num_palabras'] > UMBRAL_LONGITUD:
                return row['lda_cluster'], 'LDA'
            else:
                return row['kmeans_cluster'], 'K-Means'

        df[['Cluster_Dominante', 'Modelo']] = df.apply(lambda x: pd.Series(apply_hybrid(x)), axis=1)
        df['Tema_Nombre'] = df['Cluster_Dominante'].map(NOMBRES_CLUSTERS[sentiment])
        df['Palabras_Clave'] = df['texto_limpio'].apply(lambda x: ", ".join(str(x).split()[:8]))
        
        all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        columnas_finales = [
            'reviewId', 'content', 'score', 'gender', 'location', 'date', 
            'sentimiento_analisis', 'texto_limpio', 'num_palabras', 'Cluster_Dominante', 
            'Tema_Nombre', 'Prob_Topico_0', 'Prob_Topico_1', 'Prob_Topico_2', 'Prob_Topico_3', 
            'Palabras_Clave', 'Modelo', 'Longitud'
        ]
  
        final_df = final_df[[c for c in columnas_finales if c in final_df.columns]]
        
        final_df.to_csv("AnalisisClustering.csv", index=False)
        print("\nCSV generado")

if __name__ == "__main__":
    process_clustering_final_K4()
