import pandas as pd
import numpy as np
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. TEMAS 
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
        if not os.path.exists(file):
            print(f"⚠️ Archivo no encontrado: {file}")
            continue
        
        print(f"\n--- Procesando: {file} ---")
        df = pd.read_csv(file)
        
        # Limpieza inicial
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'text_final' in df.columns:
            df = df.rename(columns={'text_final': 'texto_limpio'})
        
        df = df.dropna(subset=['texto_limpio']).reset_index(drop=True)
        
        # Metadata
        sentiment = 'positive' if 'positive' in file else ('negative' if 'negative' in file else 'neutral')
        df['sentimiento_analisis'] = sentiment
        df['app'] = 'Boo' if 'boo' in file.lower() else 'Hinge'
        df['num_palabras'] = df['texto_limpio'].apply(lambda x: len(str(x).split()))
        df['Longitud'] = df['num_palabras']

        # --- 1. GENERAR  GRÁFICOS DE CODO (K-MEANS) ---
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_tfidf = vectorizer.fit_transform(df['texto_limpio'])
        
        inertia = []
        ks_kmeans = range(2, 8)
        for k in ks_kmeans:
            km_temp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_tfidf)
            inertia.append(km_temp.inertia_)
        
        plt.figure(figsize=(8, 5))
        plt.plot(ks_kmeans, inertia, marker='o', color='blue')
        plt.title(f"Codo KMeans - {file}")
        plt.xlabel("Número de Clusters (k)")
        plt.ylabel("Inercia")
        plt.savefig(f"codo_kmeans_{file}.png")
        plt.close()

        # Ejecución final KMeans K=4
        df['kmeans_cluster'] = KMeans(n_clusters=BEST_K, random_state=42, n_init=10).fit_predict(X_tfidf)

        # --- 2. GRÁFICOS DE COHERENCIA (LDA) ---
        tokens = df['texto_limpio'].apply(lambda x: str(x).split()).tolist()
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(t) for t in tokens]

        coherencias = []
        ks_lda = range(2, 7)
        for k in ks_lda:
            lda_temp = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=5, random_state=42)
            cm = CoherenceModel(model=lda_temp, texts=tokens, dictionary=dictionary, coherence='c_v')
            coherencias.append(cm.get_coherence())
        
        plt.figure(figsize=(8, 5))
        plt.plot(ks_lda, coherencias, marker='o', color='green')
        plt.title(f"Coherencia LDA - {file}")
        plt.xlabel("Número de Tópicos")
        plt.ylabel("Puntuación Coherencia (c_v)")
        plt.savefig(f"coherencia_lda_{file}.png") 
        plt.close()

        # Ejecución final LDA K=4
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=BEST_K, passes=10, random_state=42)

        # Probabilidades y Cluster LDA
        def get_lda_data(text):
            bow = dictionary.doc2bow(str(text).split())
            topics = dict(lda.get_document_topics(bow, minimum_probability=0))
            return pd.Series([topics.get(i, 0.0) for i in range(4)])

        lda_res = df['texto_limpio'].apply(get_lda_data)
        lda_res.columns = ['Prob_Topico_0', 'Prob_Topico_1', 'Prob_Topico_2', 'Prob_Topico_3']
        df = pd.concat([df, lda_res], axis=1)
        df['lda_cluster'] = lda_res.idxmax(axis=1).str.replace('Prob_Topico_', '').astype(int)

        # --- 3. LÓGICA HÍBRIDA  ---
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
            'reviewId', 'content', 'score', 'gender', 'location', 'date', 'app',
            'sentimiento_analisis', 'texto_limpio', 'num_palabras', 'Cluster_Dominante', 
            'Tema_Nombre', 'Prob_Topico_0', 'Prob_Topico_1', 'Prob_Topico_2', 'Prob_Topico_3', 
            'Palabras_Clave', 'Modelo', 'Longitud'
        ]

        final_df = final_df[[c for c in columnas_finales if c in final_df.columns]]
        
        final_df.to_csv("AnalisisClustering.csv", index=False)
        print("\nTODO GENERADO:")
        print("-  Gráficos PNG CREADOS")
        print("- AnalisisClustering.csv ")

if __name__ == "__main__":
    process_clustering_final_K4()
