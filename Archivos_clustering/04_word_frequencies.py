import pandas as pd
from collections import Counter
import re
import os

def generate_word_frequencies():
    df = pd.read_csv("AnalisisClustering.csv")
    stopwords_custom = {'app', 'people', 'match', 'dating', 'hinge', 'boo', 'get', 'like', 'good', 'bad'}
    results = []

    # Agrupamos por los nombres de columna del clustering final
    groups = df.groupby(['app', 'sentimiento_analisis', 'Tema_Nombre'])

    for (app, sentiment, topic), group_df in groups:
        text = " ".join(group_df['texto_limpio'].astype(str))
        words = re.findall(r'\b\w+\b', text.lower())
        counts = Counter(w for w in words if w not in stopwords_custom and len(w) > 2)
        
        for word, freq in counts.most_common(50):
            results.append({
                'App': app,        # <--- Con mayúscula como parece querer tu Tableau
                'Sentiment': sentiment,
                'Tema': topic,
                'Word': word,
                'Frequency': freq
            })

    pd.DataFrame(results).to_csv("word_frequencies_BY_TOPIC.csv", index=False)
    print("Archivo regenerado con nombres estándar.")

if __name__ == "__main__":
    generate_word_frequencies()
