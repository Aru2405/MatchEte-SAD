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
    # simple_preprocess quita puntuación (don't -> don, t)
    tokens_base = simple_preprocess(str(text), deacc=True)
    doc = nlp(" ".join(tokens_base))
    
    resultado = []
    # Lista de palabras que indican negación tras el preproceso
    negaciones = ["not", "no", "don", "dont", "doesnt", "didnt", "isnt", "arent"]
    
    for t in doc:
        token_text = t.text.lower()
        
        # Si es cualquier tipo de negación, lo normalizamos a "not"
        if token_text in negaciones:
            resultado.append("not")
            
        # Filtro normal: No stopword, solo letras, largo > 2
        # (Asegúrate de que 'not' NO sea filtrado aquí)
        elif not t.is_stop and t.is_alpha and len(t.lemma_) > 2:
            resultado.append(t.lemma_)
            
    return resultado

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

 
    print("Generando Bigramas (ej: 'not_good', 'fake_profile')...")
    # Bajamos min_count a 3 y threshold a 5 para que sea más sensible
    phrases = Phrases(data_words, min_count=3, threshold=5) 
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
    archivos = ["Boo.csv", "Hinge.csv"]

    for archivo in archivos:
        try:
            # 1. Ejecutar el proceso
            df_res, dico = preprocess_pipeline(archivo)

            # 2. Guardar el CSV limpio
            col_original = "content" if "content" in df_res.columns else df_res.columns[0]
            col_score = "score" if "score" in df_res.columns else df_res.columns[-1]
            
            nombre_salida = archivo.replace(".csv", "_limpio.csv")
            df_res[[col_original, "text_final", col_score]].to_csv(nombre_salida, index=False)
            
            # 3. Guardar el diccionario
            dico.save(archivo.replace(".csv", "_gensim.dict"))

            # --- COMPROBACIÓN DE BIGRAMAS (EL TEST) ---
            con_not = df_res[df_res['text_final'].str.contains('not_', na=False)]
            print(f">>> Análisis de {archivo}:")
            print(f"    - Bigramas de negación ('not_...') encontrados: {len(con_not)}")
            if len(con_not) > 0:
                print(f"    - Ejemplo de texto con bigrama: {con_not['text_final'].iloc[0][:100]}")
            
            print(f"¡Hecho! Generado: {nombre_salida}\n")
            
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

    print("Proceso finalizado para todos los archivos.")
    
    
    
