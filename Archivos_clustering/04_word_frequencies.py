import pandas as pd
from collections import Counter

# Cargamos el archivo que tiene los nombres de los temas
df = pd.read_csv('AnalisisClustering.csv')

def get_word_counts_by_topic(data):
    results = []
    # Ahora agrupamos por App, Sentimiento Y Tema_Nombre
    grupos = data.groupby(['app', 'sentimiento_analisis', 'Tema_Nombre'])
    
    for (app, sentiment, tema), group in grupos:
        # Unimos el texto de este tema específico
        words = " ".join(group['text_final'].astype(str)).split()
        counts = Counter(words)
        
        # Tomamos las 30 palabras más frecuentes de este tema
        for word, freq in counts.most_common(30):
            results.append({
                'App': app,
                'Sentiment': sentiment,
                'Tema': tema,
                'Word': word,
                'Frequency': freq
            })
    return pd.DataFrame(results)

# Generamos el nuevo CSV
final_word_freq = get_word_counts_by_topic(df)
final_word_freq.to_csv('word_frequencies_BY_TOPIC.csv', index=False)
print("¡Archivo word_frequencies_BY_TOPIC.csv listo!")
