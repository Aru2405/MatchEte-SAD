import pandas as pd
import numpy as np
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel


NOMBRES_CLUSTERS = {
    'positive': {
        0: "Calidad de la Comunidad y Perfiles Autenticos",
        1: "Optimizacion del Tiempo y Éxito en Matches",
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

datasets = {
    'positive': ["boo_positive.csv", "hinge_positive.csv"],
    'negative': ["boo_negative.csv", "hinge_negative.csv"],
    'neutral': ["boo_neutral.csv", "hinge_neutral.csv"]
}

def process_and_save_final():
    all_dfs = []
    print("Iniciando clustering")

    for sentiment, files in datasets.items():
        temp_dfs = []
        for f in files:
            if os.path.exists(f):
                df_temp = pd.read_csv(f)
                df_temp['app'] = 'Boo' if 'boo' in f else 'Hinge'
                temp_dfs.append(df_temp)
        
        if not temp_dfs: continue
            
        df = pd.concat(temp_dfs).dropna(subset=['text_final'])
        df['sentimiento_analisis'] = sentiment

        # LDA
        tokens = df['text_final'].apply(lambda x: str(x).split()).tolist()
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(t) for t in tokens]
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, passes=10, random_state=42)

        # Función para asignar el tema
        def get_topic(text):
            bow = dictionary.doc2bow(str(text).split())
            topic_id = max(lda.get_document_topics(bow), key=lambda x: x[1])[0]
            return topic_id

       
        df['Cluster_Dominante'] = df['text_final'].apply(get_topic)
        df['Tema_Nombre'] = df['Cluster_Dominante'].map(NOMBRES_CLUSTERS[sentiment])
        
        # Palabras clave 
        df['Palabras_Clave'] = df['text_final'].apply(lambda x: ", ".join(str(x).split()[:5]))
        
        all_dfs.append(df)
        print(f"Clase {sentiment} completada.")

    if all_dfs:
        pd.concat(all_dfs).to_csv("AnalisisClustering.csv", index=False)
        print("Archivo csv creado")

if __name__ == "__main__":
    process_and_save_final()
