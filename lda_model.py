import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

# =========================
# 1. PREPARACIÓN DE DATOS 
# =========================
def prepare_data(df, text_col="text_final"):
    # 1. Aseguramos que no haya nulos y convertimos a string por seguridad
    df = df.dropna(subset=[text_col])
    
    # 2. Convertimos la cadena de texto en lista de tokens para Gensim
    tokens = df[text_col].apply(lambda x: str(x).split()).tolist()
    
    # 3. Crear Diccionario
    dictionary = Dictionary(tokens)
    
    # 4. FILTRADO DE EXTREMOS (Clave para el sobresaliente)
    # no_below=5: elimina palabras que salen en menos de 5 reseñas (ruido)
    # no_above=0.5: elimina palabras que salen en más del 50% (como "app", "good", "nice")
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # IMPORTANTE: Limpiar huecos tras el filtrado
    dictionary.compactify() 
    
    # 5. Crear Corpus (Bag of Words)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    
    return tokens, dictionary, corpus

# =========================
# 2. LDA + COHERENCIA
# =========================
def run_lda_pipeline(tokens, dictionary, corpus, nombre_grupo):
    print(f"\n--- Buscando el mejor número de temas para {nombre_grupo} ---")
    best_k = 0
    best_score = -1
    best_model = None

    # Probamos de 2 a 8 temas para ver cuál es más coherente
    for k in range(2, 9):
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=10, random_state=42)
        
        coherence_model = CoherenceModel(
            model=lda, 
            texts=tokens, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        score = coherence_model.get_coherence()
        
        print(f"Temas: {k} | Coherencia: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = lda

    print(f"\n GANADOR para {nombre_grupo}: {best_k} temas (Coherencia: {best_score:.4f})")
    
    # Mostramos los temas del mejor modelo
    print(f"\nInterpretación de Temas ({nombre_grupo}):")
    for idx, topic in best_model.print_topics(-1):
        print(f"  Tema {idx}: {topic}")
    
    return best_model

# =========================
# 3. EJECUCIÓN PRINCIPAL
# =========================

# --- PROCESAR BOO POSITIVOS ---
try:
    print("\n" + "="*30 + "\n ANALIZANDO BOO POSITIVOS \n" + "="*30)
    df_pos = pd.read_csv("boo_positive.csv")
    t_pos, d_pos, c_pos = prepare_data(df_pos)
    model_pos = run_lda_pipeline(t_pos, d_pos, c_pos, "BOO POSITIVOS")
except FileNotFoundError:
    print("Error: No se encuentra boo_positive.csv")

# --- PROCESAR BOO NEGATIVOS ---
try:
    print("\n" + "="*30 + "\n ANALIZANDO BOO NEGATIVOS \n" + "="*30)
    df_neg = pd.read_csv("boo_negative.csv")
    t_neg, d_neg, c_neg = prepare_data(df_neg)
    model_neg = run_lda_pipeline(t_neg, d_neg, c_neg, "BOO NEGATIVOS")
except FileNotFoundError:
    print("Error: No se encuentra boo_negative.csv")

# --- PROCESAR HINGE POSITIVOS ---
try:
    print("\n" + "="*30 + "\n ANALIZANDO HINGE POSITIVOS \n" + "="*30)
    df_h_pos = pd.read_csv("hinge_positive.csv")
    t_h_pos, d_h_pos, c_h_pos = prepare_data(df_h_pos)
    model_h_pos = run_lda_pipeline(t_h_pos, d_h_pos, c_h_pos, "HINGE POSITIVOS")
except FileNotFoundError:
    print("Error: No se encuentra hinge_positive.csv")

# --- PROCESAR HINGE NEGATIVOS ---
try:
    print("\n" + "="*30 + "\n ANALIZANDO HINGE NEGATIVOS \n" + "="*30)
    df_h_neg = pd.read_csv("hinge_negative.csv")
    t_h_neg, d_h_neg, c_h_neg = prepare_data(df_h_neg)
    model_h_neg = run_lda_pipeline(t_h_neg, d_h_neg, c_h_neg, "HINGE NEGATIVOS")
except FileNotFoundError:
    print("Error: No se encuentra hinge_negative.csv")
