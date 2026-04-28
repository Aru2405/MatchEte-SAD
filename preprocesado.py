import pandas as pd
import re
import emoji
import spacy
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from collections import Counter

# ============================================================
# 1. CARGA DEL MODELO
# ============================================================
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ============================================================
# 2. STOPWORDS PERSONALIZADAS
#
# Solo eliminamos palabras que:
#   a) Son ruido estructural del dominio (nombre de la app, etc.)
#   b) Son tan genéricas que no aportan a ningún tema
#
# NOTA: Siguiendo la indicación de la profesora, NO eliminamos
# palabras de valoración (good, great, bad...) como stopwords.
# En su lugar, se preservan para que los bigramas capturen su
# contexto: "not_good", "not_great", "really_good", etc.
# ============================================================
CUSTOM_STOPWORDS = {
    # Nombres de apps (no distinguen entre apps en este corpus)
    "app", "boo", "hinge", "tinder", "bumble",
    # Palabras funcionales que spaCy no filtra siempre
    "dating", "use", "like", "get", "also", "even",
    "make", "one", "would", "could", "really", "much",
    "well", "way", "thing", "go", "said", "say",
    # Palabras muy cortas que pasan el filtro len > 2
    "got", "lot", "let", "put", "see", "try", "good", "nice",
    "great", "love", "amazing", "bad", "worst", "horrible",
}

for word in CUSTOM_STOPWORDS:
    nlp.vocab[word].is_stop = True

# ============================================================
# 3. NORMALIZACIÓN DE NEGACIONES
#
# PROBLEMA CRÍTICO del pipeline original:
#   simple_preprocess() destruye las contracciones ANTES de
#   que spaCy pueda procesarlas:
#       "don't" → ["don", "t"]   ← spaCy nunca ve "n't"
#
# SOLUCIÓN: Normalizar contracciones con regex ANTES de
#   simple_preprocess, para que spaCy reciba texto coherente.
#   Luego capturamos el lemma "not" que spaCy asigna a "n't".
# ============================================================
CONTRACCIONES = {
    r"don't|dont":     "do not",
    r"doesn't|doesnt": "does not",
    r"didn't|didnt":   "did not",
    r"isn't|isnt":     "is not",
    r"aren't|arent":   "are not",
    r"wasn't|wasnt":   "was not",
    r"weren't|werent": "were not",
    r"can't|cant":     "can not",
    r"won't|wont":     "will not",
    r"wouldn't":       "would not",
    r"shouldn't":      "should not",
    r"couldn't":       "could not",
    r"haven't":        "have not",
    r"hasn't":         "has not",
    r"hadn't":         "had not",
    r"it's|its(?=\s)": "it is",
    r"i'm":            "i am",
    r"i've":           "i have",
    r"i'll":           "i will",
    r"i'd":            "i would",
    r"you're":         "you are",
    r"they're":        "they are",
    r"we're":          "we are",
}

NEGATION_LEMMAS = {"not", "no", "never"}

def expand_contractions(text):
    """Expande contracciones antes de tokenizar para no perder negaciones."""
    text = text.lower()
    for pattern, replacement in CONTRACCIONES.items():
        text = re.sub(pattern, replacement, text)
    return text

# ============================================================
# 4. FUNCIÓN PRINCIPAL DE LIMPIEZA
# ============================================================
def clean_and_lemmatize(text):
    if not isinstance(text, str) or text.strip() == "":
        return []

    # A. Emojis (Ya lo tienes)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace(":", " ").replace("_", " ")
    text = re.sub(r'\s+', ' ', text).strip()

    # B. Expandir contracciones (Ya lo tienes)
    text = expand_contractions(text)

    # C. PROCESADO CON SPACY
    doc = nlp(text)
    resultado = []
    
    # Definimos negaciones (puedes usar tu NEGATION_LEMMAS de la línea 84)
    negaciones = {"not", "no", "never"}
    
    # Usamos un iterador para poder saltar tokens si los unimos
    skip_next = False
    for i, token in enumerate(doc):
        if skip_next:
            skip_next = False
            continue
            
        t_lemma = token.lemma_.lower()
        
        # LÓGICA DE NEGACIÓN PRO
        if t_lemma in negaciones and i + 1 < len(doc):
            next_token = doc[i+1]
            # Si lo siguiente es adjetivo o verbo, los unimos (ej: not_good, not_work)
            if next_token.pos_ in ["ADJ", "VERB"]:
                resultado.append(f"not_{next_token.lemma_.lower()}")
                skip_next = True # Saltamos el siguiente para no duplicar
                continue

        # FILTRO NORMAL (Sustantivos, Adjetivos, Verbos y Emojis)
        # Solo añadimos si no es stopword y es palabra real
        if not token.is_stop and token.is_alpha and len(t_lemma) > 1:
            if token.pos_ in ["NOUN", "ADJ", "VERB", "PROPN"]:
                resultado.append(t_lemma)
                
    return resultado

# ============================================================
# 5. PIPELINE COMPLETO
# ============================================================
def preprocess_pipeline(path, text_column="content"):
    print(f"\n{'='*50}")
    print(f"  Procesando: {path}")
    print(f"{'='*50}")

    df = pd.read_csv(path)

    if text_column not in df.columns:
        text_column = df.columns[0]
        print(f"  [INFO] Columna de texto detectada automáticamente: '{text_column}'")

    n_original = len(df)
    df = df.dropna(subset=[text_column])
    print(f"  Registros originales : {n_original}")
    print(f"  Registros tras dropna: {len(df)}")

    # --- Paso 1: Limpieza y lematización ---
    print("\n  [1/4] Limpiando y lematizando...")
    data_words = df[text_column].apply(clean_and_lemmatize).tolist()

    # --- Diagnóstico previo a bigramas ---
    all_tokens = [t for doc in data_words for t in doc]
    freq = Counter(all_tokens)
    print(f"  Vocabulario único (pre-bigramas): {len(freq)} tokens")
    print(f"  Top 15 tokens más frecuentes:")
    for word, count in freq.most_common(15):
        print(f"      '{word}': {count}")

    # --- Paso 2: Bigramas ---
    # min_count=5  → el par debe aparecer al menos 5 veces en el corpus
    # threshold=10 → umbral de puntuación para considerar bigrama válido
    #                (más alto = más conservador = menos bigramas espurios)
    print("\n  [2/4] Generando bigramas...")
    phrases = Phrases(data_words, min_count=5, threshold=10)
    bigram_mod = Phraser(phrases)
    data_bigrams = [bigram_mod[doc] for doc in data_words]

    # Diagnóstico: ¿cuántos bigramas se generaron?
    bigramas_encontrados = set(
        t for doc in data_bigrams for t in doc if '_' in t and not t.startswith('not')
    )
    not_bigramas = set(
        t for doc in data_bigrams for t in doc if t.startswith('not_')
    )
    print(f"  Bigramas de contenido detectados: {len(bigramas_encontrados)}")
    print(f"  Bigramas de negación (not_*) detectados: {len(not_bigramas)}")
    if not_bigramas:
        print(f"  Ejemplos de bigramas de negación: {list(not_bigramas)[:10]}")

    # --- Paso 3: Diccionario Gensim con filtrado de extremos ---
    # no_below=5  → token debe aparecer en al menos 5 documentos
    # no_above=0.5 → token no puede estar en más del 50% de documentos
    #                (palabras demasiado frecuentes = no discriminan temas)
    print("\n  [3/4] Construyendo diccionario Gensim...")
    df["tokens"] = data_bigrams
    dictionary = Dictionary(df["tokens"])
    n_before = len(dictionary)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    print(f"  Tokens en diccionario antes de filtrar: {n_before}")
    print(f"  Tokens en diccionario después de filtrar: {len(dictionary)}")

    # --- Paso 4: Exportar ---
    print("\n  [4/4] Exportando resultados...")
    df["text_final"] = df["tokens"].apply(lambda x: " ".join(x))
    df = df[df["text_final"].str.strip() != ""]

    avg_len = df["tokens"].apply(len).mean()
    empty_pct = (df["text_final"].str.strip() == "").sum() / len(df) * 100
    print(f"  Longitud media de tokens por reseña: {avg_len:.1f}")
    print(f"  Reseñas vacías tras preprocesado   : {empty_pct:.1f}%")

    return df, dictionary


# ============================================================
# 6. EJECUCIÓN PRINCIPAL
# ============================================================
if __name__ == "__main__":
    archivos = ["Boo.csv", "Hinge.csv"]

    for archivo in archivos:
        try:
            df_res, dico = preprocess_pipeline(archivo)

            # Columnas a guardar
            cols_guardar = []
            for col in ["content", "score", "text_final", "tokens"]:
                if col in df_res.columns:
                    cols_guardar.append(col)

            nombre_salida = archivo.replace(".csv", "_limpio.csv")
            df_res[cols_guardar].to_csv(nombre_salida, index=False)
            dico.save(archivo.replace(".csv", "_gensim.dict"))

            # Verificación de bigramas de negación en el output final
            con_not = df_res[df_res["text_final"].str.contains(r'\bnot_\w+', na=False, regex=True)]
            print(f"\n  >>> Verificación final de {archivo}:")
            print(f"      Reseñas con bigramas 'not_*': {len(con_not)}")
            if len(con_not) > 0:
                print(f"      Ejemplo: {con_not['text_final'].iloc[0][:120]}")

            print(f"\n  ✓ Guardado: {nombre_salida}")
            print(f"  ✓ Diccionario: {archivo.replace('.csv', '_gensim.dict')}")

        except FileNotFoundError:
            print(f"\n  [ERROR] No se encontró el archivo: {archivo}")
        except Exception as e:
            print(f"\n  [ERROR] procesando {archivo}: {e}")
            raise  # Muestra el traceback completo durante desarrollo

    print("\n" + "="*50)
    print("  Pipeline finalizado para todos los archivos.")
    print("="*50)
