import pandas as pd
import re
import spacy
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary

# ==========================================
# 1. CARGA Y CONFIGURACIÓN
# ==========================================
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Añadimos "app" a las stopwords de spaCy
nlp.Defaults.stop_words.add("app")

# ==========================================
# 2. FUNCIONES DE APOYO
# ==========================================
def clean_and_lemmatize(text):
    """
    Usa Gensim para tokenizar y spaCy para limpiar/lematizar.
    """
    tokens_base = simple_preprocess(str(text), deacc=True)
    
    doc = nlp(" ".join(tokens_base))
    
    # Filtro: No stopword (incluye "app"), solo letras, longitud > 2
    return [t.lemma_ for t in doc if not t.is_stop and t.is_alpha and len(t.lemma_) > 2]

# ==========================================
# 3. PIPELINE DE PREPROCESADO
# ==========================================
def preprocess_pipeline(path, text_column="content"):
    print(f"\n--- Procesando {path} ---")
    df = pd.read_csv(path)
    
    if text_column not in df.columns:
        text_column = df.columns[0]
        
    df = df.dropna(subset=[text_column])

    print("Limpiando y lematizando...")
    data_words = df[text_column].apply(clean_and_lemmatize).tolist()

    print("Generando Bigramas...")
    phrases = Phrases(data_words, min_count=5, threshold=10)
    bigram_mod = Phraser(phrases)
    df["tokens"] = [bigram_mod[doc] for doc in data_words]

    print("Creando Diccionario de Gensim...")
    dictionary = Dictionary(df["tokens"])
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    df["text_final"] = df["tokens"].apply(lambda x: " ".join(x))

    return df, dictionary

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # Lista de archivos a procesar
    archivos = ["Boo.csv", "Hinge.csv"]

    for archivo in archivos:
        try:
            # 1. Ejecutar el proceso
            df_res, dico = preprocess_pipeline(archivo)

            # 2. Guardar el CSV limpio
            # Usamos "score" o lo que esté disponible
            col_original = "content" if "content" in df_res.columns else df_res.columns[0]
            col_score = "score" if "score" in df_res.columns else df_res.columns[-1]
            
            nombre_salida = archivo.replace(".csv", "_limpio.csv")
            df_res[[col_original, "text_final", col_score]].to_csv(nombre_salida, index=False)
            
            # 3. Guardar el diccionario
            dico.save(archivo.replace(".csv", "_gensim.dict"))

            print(f"¡Hecho! Generado: {nombre_salida}")
            
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

    print("\nProceso finalizado para todos los archivos.")
