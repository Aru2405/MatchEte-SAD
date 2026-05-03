import pandas as pd
from collections import Counter
import re
import os

def generate_word_frequencies():
    print("Generando frecuencias de palabras por Tópico...")
    
    # 1. Leer el output del clustering que acabas de crear
    file_path = "AnalisisClustering.csv"
    if not os.path.exists(file_path):
        print(f"Error: No encuentro {file_path}. ¡Ejecuta primero el script de clustering!")
        return

    df = pd.read_csv(file_path)
    
    # 2. Lista de palabras a ignorar (Stopwords de dominio para que no salgan en la nube)
    stopwords_custom = {
        'app', 'people', 'match', 'dating', 'hinge', 'boo', 'get', 'like', 
        'good', 'bad', 'really', 'would', 'even', 'make', 'time'
    }

    results = []

    # 3. Agrupar por App, Sentimiento y TEMA (Nombres largos)
    columnas_necesarias = ['app', 'sentimiento_analisis', 'Tema_Nombre', 'texto_limpio']
    if not all(col in df.columns for col in columnas_necesarias):
        print("Error: El CSV no tiene las columnas esperadas. Revisa el script de clustering.")
        return

    groups = df.groupby(['app', 'sentimiento_analisis', 'Tema_Nombre'])

    for (app, sentiment, topic), group_df in groups:
        text = " ".join(group_df['texto_limpio'].astype(str))
        
        # Limpieza rápida y split 
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Contar frecuencias filtrando stopwords
        counts = Counter(w for w in words if w not in stopwords_custom and len(w) > 2)
        
        # Guardar las 50 palabras más frecuentes de cada tema para Tableau
        for word, freq in counts.most_common(50):
            results.append({
                'app': app,
                'sentimiento': sentiment,
                'tema': topic,
                'word': word,
                'n': freq
            })

    # 4. Guardar el archivo que leerá Tableau para las nubes de palabras
    output_file = "word_frequencies_BY_TOPIC.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"¡Éxito! Archivo '{output_file}' generado.")

if __name__ == "__main__":
    generate_word_frequencies()
