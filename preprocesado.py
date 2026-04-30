import pandas as pd
import re
import emoji
import spacy
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from collections import Counter
import os

# ============================================================
# 1. CARGA DEL MODELO
# ============================================================
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ============================================================
# 2. STOPWORDS PERSONALIZADAS
# ============================================================
CUSTOM_STOPWORDS = {
    "app", "boo", "hinge", "tinder", "bumble", "dating", "use", "like", 
    "get", "also", "even", "make", "one", "would", "could", "really", 
    "much", "well", "way", "thing", "go", "said", "say", "got", "lot", 
    "let", "put", "see", "try", "good", "nice", "great", "love", 
    "amazing", "bad", "worst", "horrible",
}

for word in CUSTOM_STOPWORDS:
    nlp.vocab[word].is_stop = True

# ============================================================
# 3. NORMALIZACIÓN DE NEGACIONES
# ============================================================
CONTRACCIONES = {
    r"don't|dont": "do not", r"doesn't|doesnt": "does not",
    r"didn't|didnt": "did not", r"isn't|isnt": "is not",
    r"aren't|arent": "are not", r"wasn't|wasnt": "was not",
    r"weren't|werent": "were not", r"can't|cant": "can not",
    r"won't|wont": "will not", r"wouldn't": "would not",
    r"shouldn't": "should not", r"couldn't": "could not",
    r"haven't": "have not", r"hasn't": "has not",
    r"hadn't": "had not", r"it's|its(?=\s)": "it is",
    r"i'm": "i am", r"i've": "i have", r"i'll": "i will",
    r"i'd": "i would", r"you're": "you are",
    r"they're": "they are", r"we're": "we are",
}

def expand_contractions(text):
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

    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace(":", " ").replace("_", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    text = expand_contractions(text)

    doc = nlp(text)
    resultado = []
    negaciones = {"not", "no", "never"}
    skip_next = False

    for i, token in enumerate(doc):
        if skip_next:
            skip_next = False
            continue
        t_lemma = token.lemma_.lower()
        if t_lemma in negaciones and i + 1 < len(doc):
            next_token = doc[i+1]
            if next_token.pos_ in ["ADJ", "VERB"]:
                resultado.append(f"not_{next_token.lemma_.lower()}")
                skip_next = True
                continue
        if not token.is_stop and token.is_alpha and len(t_lemma) > 1:
            if token.pos_ in ["NOUN", "ADJ", "VERB", "PROPN"]:
                resultado.append(t_lemma)
    return resultado

# ============================================================
# 5. PIPELINE COMPLETO
# ============================================================
def preprocess_pipeline(path, text_column="content"):
    print(f"\n{'='*50}\nProcesando: {path}\n{'='*50}")
    df = pd.read_csv(path)

    if text_column not in df.columns:
        text_column = df.columns[0]
    
    df = df.dropna(subset=[text_column])
    data_words = df[text_column].apply(clean_and_lemmatize).tolist()

    phrases = Phrases(data_words, min_count=5, threshold=10)
    bigram_mod = Phraser(phrases)
    data_bigrams = [bigram_mod[doc] for doc in data_words]

    df["tokens"] = data_bigrams
    dictionary = Dictionary(df["tokens"])
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    df["text_final"] = df["tokens"].apply(lambda x: " ".join(x))
    df = df[df["text_final"].str.strip() != ""]

    return df, dictionary

# ============================================================
# 6. EJECUCIÓN PRINCIPAL 
# ============================================================
if __name__ == "__main__":
    archivos = ["Boo.csv", "Hinge.csv"]

    for archivo in archivos:
        try:
            df_res, dico = preprocess_pipeline(archivo)

            # Esto mantiene reviewId, score, gender, location, date, etc.
            nombre_salida = archivo.replace(".csv", "_limpio.csv")
            df_res.to_csv(nombre_salida, index=False)
            
            dico.save(archivo.replace(".csv", "_gensim.dict"))
            print(f"✓ Guardado con éxito: {nombre_salida}")

        except Exception as e:
            print(f"[ERROR] en {archivo}: {e}")

    print("\nPROCESO FINALIZADO.")
