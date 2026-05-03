import pandas as pd
import numpy as np
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. DICCIONARIOS DE TEMAS 
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

grupos_datasets = {
    'positive': ["boo_positive.csv", "hinge_positive.csv"],
    'negative': ["boo_negative.csv", "hinge_negative.csv"],
    'neutral': ["boo_neutral.csv", "hinge_neutral.csv"]
}

def process_clustering_final():
    all_dfs = []

    for sentiment, files in grupos_datasets.items():
        temp_dfs = []
        for f in files:
            if os.path.exists(f):
                df_temp = pd.read_csv(f)
                df_temp['app'] = 'Boo' if 'boo' in f.lower() else 'Hinge'
                temp_dfs.append(df_temp)
        
        if not temp_dfs: continue
            
        # Unificamos apps para entrenar el modelo una sola vez por sentimiento
        df = pd.concat(temp_dfs).dropna(subset=['text_final'])
        df['sentimiento_analisis'] = sentiment
        df['num_palabras'] = df['text_final'].apply(lambda x: len(str(x).split()))

        print(f"\n>>> Entrenando modelos para clase: {sentiment.upper()}")

        # --- VALIDACIÓN K-MEANS (Inercia) ---
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=['app', 'people', 'match'])
        X_tfidf = vectorizer.fit_transform(df['text_final'])
        
        inertia = []
        for k in range(2, 7):
            km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_temp.fit(X_tfidf)
            inertia.append(km_temp.inertia_)
        
        plt.figure()
        plt.plot(range(2, 7), inertia, marker='o')
        plt.title(f"Codo KMeans - {sentiment}")
        plt.savefig(f"codo_kmeans_{sentiment}.png")
        plt.close()

        # Ejecución final KMeans
        km = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
        df['kmeans_cluster'] = km.fit_predict(X_tfidf)

        # --- VALIDACIÓN LDA (Coherencia) ---
        tokens = df['text_final'].apply(lambda x: str(x).split()).tolist()
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(t) for t in tokens]

        coherencias = []
        for k in range(2, 6):
            lda_temp = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=5, random_state=42)
            cm = CoherenceModel(model=lda_temp, texts=tokens, dictionary=dictionary, coherence='c_v')
            coherencias.append(cm.get_coherence())
        
        plt.figure()
        plt.plot(range(2, 6), coherencias, marker='o')
        plt.title(f"Coherencia LDA - {sentiment}")
        plt.savefig(f"coherencia_lda_{sentiment}.png")
        plt.close()

        # Ejecución final LDA
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=BEST_K, passes=10, random_state=42)
        
        def get_lda_cluster(text):
            bow = dictionary.doc2bow(str(text).split())
            return max(lda.get_document_topics(bow), key=lambda x: x[1])[0]

        df['lda_cluster'] = df['text_final'].apply(get_lda_cluster)

        # --- LÓGICA HÍBRIDA Y MAPEO ---
        def apply_hybrid(row):
            if row['num_palabras'] > UMBRAL_LONGITUD:
                return row['lda_cluster']
            else:
                return row['kmeans_cluster']

        df['Cluster_Dominante'] = df.apply(apply_hybrid, axis=1)
        df['Tema_Nombre'] = df['Cluster_Dominante'].map(NOMBRES_CLUSTERS[sentiment])
        
        # Guardamos palabras clave (Top 8)
        df['Palabras_Clave'] = df['text_final'].apply(lambda x: ", ".join(str(x).split()[:8]))
        
        all_dfs.append(df)

    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv("AnalisisClustering.csv", index=False)
        print("\nTODO LISTO: CSV Unificado, Codos y Coherencias guardados.")

if __name__ == "__main__":
    process_clustering_final()
