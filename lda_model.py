import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

# =========================
# 1. PREPARACIÓN DE DATOS 
# =========================
def prepare_data(df, text_col="text_final"):
    df = df.dropna(subset=[text_col])
    tokens = df[text_col].apply(lambda x: str(x).split()).tolist()
    dictionary = Dictionary(tokens)
    
    # El filtro de 0.5 es lo que quita el "good" y "app" que nos molestaban
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    dictionary.compactify() 
    
    corpus = [dictionary.doc2bow(t) for t in tokens]
    return tokens, dictionary, corpus

# =========================
# 2. LDA + COHERENCIA
# =========================
def run_lda_pipeline(tokens, dictionary, corpus, nombre_grupo):
    print(f"\n" + "="*40)
    print(f" ANALIZANDO: {nombre_grupo}")
    print("="*40)
    
    if len(corpus) == 0:
        print(f"Aviso: No hay datos para {nombre_grupo}. Saltando...")
        return None

    best_k = 0
    best_score = -1
    best_model = None

    for k in range(2, 9):
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=10, random_state=42)
        coherence_model = CoherenceModel(model=lda, texts=tokens, dictionary=dictionary, coherence='c_v')
        score = coherence_model.get_coherence()
        print(f"Temas: {k} | Coherencia: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = lda

    print(f"\n GANADOR para {nombre_grupo}: {best_k} temas (Coherencia: {best_score:.4f})")
    for idx, topic in best_model.print_topics(-1):
        print(f"  Tema {idx}: {topic}")
    return best_model

# =========================
# 3. EJECUCIÓN PRINCIPAL (CON LOS 6 ARCHIVOS)
# =========================

# Esta lista es la que le dice al script que busque los NEUTROS también
datasets = [
    ("boo_positive.csv", "BOO POSITIVOS"),
    ("boo_negative.csv", "BOO NEGATIVOS"),
    ("boo_neutral.csv", "BOO NEUTROS"),
    ("hinge_positive.csv", "HINGE POSITIVOS"),
    ("hinge_negative.csv", "HINGE NEGATIVOS"),
    ("hinge_neutral.csv", "HINGE NEUTROS")
]

for archivo, etiqueta in datasets:
    try:
        df = pd.read_csv(archivo)
        t, d, c = prepare_data(df)
        run_lda_pipeline(t, d, c, etiqueta)
    except FileNotFoundError:
        print(f"\n[!] Saltando {archivo}: No existe.")
    except Exception as e:
        print(f"\n[!] Error en {etiqueta}: {e}")

print("\n" + "="*40)
print(" ANÁLISIS COMPLETO FINALIZADO ")
print("="*40)
