import pandas as pd
import numpy as np
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 1. DICCIONARIO DE TEMAS 
TOPIC_NAMES = {
    0: "Personalidad y experiencia ",
    1: "Crítica al Modelo de Negocio y Accesibilidad",
    2: "Conexiones y Éxito Social",
    3: "Feedback Estético y Reacción Visual (Emojis)"
}

UMBRAL_LONGITUD = 10


# 2. PROCESO HÍBRIDO
datasets = [
    "boo_positive.csv", "boo_negative.csv", "boo_neutral.csv",
    "hinge_positive.csv", "hinge_negative.csv", "hinge_neutral.csv"
]

def process_hybrid():
    all_dfs = []

    for file in datasets:
        if not os.path.exists(file):
            continue
        
        print(f"\n--- Analizando: {file} ---")
        df = pd.read_csv(file)
        df = df.dropna(subset=['text_final'])
        
        df['app'] = 'Boo' if 'boo' in file else 'Hinge'
        df['sentimiento_analisis'] = 'positive' if 'positive' in file else ('negative' if 'negative' in file else 'neutral')
        df['num_palabras'] = df['text_final'].apply(lambda x: len(str(x).split()))

        # K-MEANS 
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_tfidf = vectorizer.fit_transform(df['text_final'])
        

        # CODO KMEANS 
        inertia = []
        ks_kmeans = range(2, 8)

        print("\n--- Buscando mejor número de clusters (KMeans) ---")

        for k in ks_kmeans:
            km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_temp.fit(X_tfidf)
            inertia.append(km_temp.inertia_)
            print(f"K={k} | Inertia={km_temp.inertia_:.2f}")

        # Gráfico Codo KMeans
        plt.figure()
        plt.plot(ks_kmeans, inertia, marker='o')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Inercia")
        plt.title(f"Codo KMeans - {file}")
        plt.grid()
        plt.savefig(f"codo_kmeans_{file}.png")
        plt.close()

        # Usamos K=4 para KMeans
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['kmeans_cluster'] = km.fit_predict(X_tfidf)

        #  LDA MODEL
        tokens_list = df['text_final'].apply(lambda x: str(x).split()).tolist()
        dictionary = Dictionary(tokens_list)
        corpus = [dictionary.doc2bow(t) for t in tokens_list]


        # COHERENCIA LDA
        coherencias = []
        ks_lda = range(2, 11)

        print("\n--- Buscando mejor número de clusters (LDA) ---")

        for k in ks_lda:
            lda_temp = LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=k,
                                passes=10,
                                random_state=42)
            
            coherence_model = CoherenceModel(model=lda_temp,
                                             texts=tokens_list,
                                             dictionary=dictionary,
                                             coherence='c_v')
            
            score = coherence_model.get_coherence()
            coherencias.append(score)
            print(f"K={k} | Coherencia={score:.4f}")

        # Gráfico Coherencia LDA
        plt.figure()
        plt.plot(ks_lda, coherencias, marker='o')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Coherencia (c_v)")
        plt.title(f"Codo LDA - {file}")
        plt.grid()
        plt.savefig(f"codo_{file}.png")
        plt.close()

        BEST_K = 4  # Ajustado a 4 según  pruebas

        lda = LdaModel(corpus=corpus,
                       id2word=dictionary,
                       num_topics=BEST_K,
                       passes=10,
                       random_state=42)

        # --- EVALUACIÓN FINAL ---
        coherence_model_cv = CoherenceModel(model=lda, texts=tokens_list, dictionary=dictionary, coherence='c_v')
        score_cv = coherence_model_cv.get_coherence()

        coherence_model_umass = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        score_umass = coherence_model_umass.get_coherence()

        print(f"\n--- RESULTADO FINAL (K={BEST_K}) ---")
        print(f"Score Cv: {score_cv:.4f}")
        print(f"Score UMass: {score_umass:.4f}")

        def get_lda_data(text):
            bow = dictionary.doc2bow(str(text).split())
            probs = lda.get_document_topics(bow, minimum_probability=0)
            res = {f"Prob_Topico_{i}": 0.0 for i in range(BEST_K)}
            for p in probs:
                res[f"Prob_Topico_{p[0]}"] = p[1]
            res['lda_cluster'] = int(max(res, key=res.get).replace('Prob_Topico_', ''))
            return pd.Series(res)

        lda_info = df['text_final'].apply(get_lda_data)
        df = pd.concat([df, lda_info], axis=1)

        def apply_hybrid_logic(row):
            if row['num_palabras'] > UMBRAL_LONGITUD:
                return row['lda_cluster'], 'LDA'
            else:
                return row['kmeans_cluster'], 'K-Means'

        df[['Cluster_Dominante', 'Modelo']] = df.apply(lambda x: pd.Series(apply_hybrid_logic(x)), axis=1)
        df['Tema_Nombre'] = df['Cluster_Dominante'].map(TOPIC_NAMES)
        df['Palabras_Clave'] = df['text_final'].apply(lambda x: ", ".join(str(x).split()[:8]))

        all_dfs.append(df)

    if all_dfs:
        final_unified = pd.concat(all_dfs, ignore_index=True)
        final_unified.to_csv("AnalisisClustering.csv", index=False)
        print("\nROCESO COMPLETADO")
    else:
        print("No se encontraron archivos para procesar.")

if __name__ == "__main__":
    process_hybrid()
