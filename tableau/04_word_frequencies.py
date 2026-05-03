import pandas as pd
from collections import Counter
import re

def generate_word_frequencies():
    print("Generando frecuencias de palabras por Tópico...")
    
    # 1. Leer el output del clustering que acabas de crear
    if not pd.io.common.file_exists("AnalisisClustering.csv"):
        print("Error: No encuentro AnalisisClustering.csv. Ejecuta primero el script de clustering.")
        return

    df = pd.read_csv("AnalisisClustering.csv")
    
    # 2. Lista de palabras a ignorar 
    stopwords_custom = {'app', 'people', 'match', 'dating', 'hinge', 'boo', 'get', 'like', 'good', 'bad'}

    results = []

    # 3. Agrupar por App, Sentimiento y TEMA
    groups = df.groupby(['app', 'sentimiento_analisis', 'Tema_Nombre'])

    for (app, sentiment, topic), group_df in groups:
        # Juntar todo el texto del grupo
        text = " ".join(group_df['text_final'].astype(str))
        # Limpieza rápida y split
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Contar frecuencias filtrando stopwords
        counts = Counter(w for w in words if w not in stopwords_custom and len(w) > 2)
        
        # Guardar las 50 palabras más frecuentes de cada tema
        for word, freq in counts.most_common(50):
            results.append({
                'app': app,
                'sentimiento': sentiment,
                'tema': topic,
                'word': word,
                'n': freq
            })

    # 4. Guardar el archivo que leerá Tableau
    pd.DataFrame(results).to_csv("word_frequencies_BY_TOPIC.csv", index=False)
    print("Archivo 'word_frequencies_BY_TOPIC.csv' generado con éxito.")

if __name__ == "__main__":
    generate_word_frequencies()
