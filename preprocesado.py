import pandas as pd
import re
import spacy
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary

# ==========================================
# 1. CARGA
# ==========================================
try:
    # Desactivamos lo que no necesitamos para ir más rápido
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ==========================================
# 2. FUNCIONES DE APOYO
# ==========================================
def clean_and_lemmatize(text):
    """
    Usa Gensim para tokenizar y spaCy para limpiar/lematizar.
    """
    # simple_preprocess de Gensim: quita puntuación y pone minúsculas
    tokens_base = simple_preprocess(str(text), deacc=True)
    
    # Convertimos a string para que spaCy lo analice
    doc = nlp(" ".join(tokens_base))
    
    # Filtro PRO: No stopword, solo letras, longitud > 2
    # El lemma convierte 'running' en 'run'
    return [t.lemma_ for t in doc if not t.is_stop and t.is_alpha and len(t.lemma_) > 2]

# ==========================================
# 3. PIPELINE DE PREPROCESADO (Para LDA / K-Means)
# ==========================================
def preprocess_pipeline(path, text_column="content"):
    print(f"Cargando {path}...")
    df = pd.read_csv(path)
    
    # Asegurar que la columna existe
    if text_column not in df.columns:
        text_column = df.columns[0]
        
    df = df.dropna(subset=[text_column])

    print("Limpiando y lematizando (paso lento pero necesario)...")
    # Aplicamos la limpieza a cada fila
    data_words = df[text_column].apply(clean_and_lemmatize).tolist()

    print("Generando Bigramas con Gensim (ej: 'social_media')...")
    # Esto es lo que le gustará a tu profe: detectar conceptos de 2 palabras
    phrases = Phrases(data_words, min_count=5, threshold=10)
    bigram_mod = Phraser(phrases)
    df["tokens"] = [bigram_mod[doc] for doc in data_words]

    print("Creando Diccionario de Gensim...")
    dictionary = Dictionary(df["tokens"])
    
    # Limpieza de extremos: quitamos palabras que salen en menos de 5 filas
    # o que salen en más del 50% de las filas (palabras genéricas)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    # Creamos la columna final para que la veas en el CSV
    df["text_final"] = df["tokens"].apply(lambda x: " ".join(x))

    return df, dictionary



# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Ejecutar el proceso
    df_resultado, dictionary = preprocess_pipeline("Boo.csv")

    # 2. Guardar el CSV limpio (¡Añadimos 'score' aquí!)
    # Asegúrate de que en tu archivo original 'Boo.csv' la columna se llame exactamente 'score'
    columnas_finales = ["content", "text_final", "score"] 
    
    # Por si acaso la columna se llama diferente en el CSV original (ej: 'rating')
    # puedes verificarlo con: print(df_resultado.columns)
    
    df_resultado[columnas_finales].to_csv("Boo_limpio.csv", index=False)
    
    # 3. Guardar el diccionario de Gensim
    dictionary.save("boo_gensim.dict")

    print(f"Archivo generado: Boo_limpio.csv")
    
    
    
    # Ejemplo de control
    print(f"\nOriginal: {df_resultado['content'].iloc[0][:60]}...")
    print(f"Limpio:   {df_resultado['text_final'].iloc[0]}")
