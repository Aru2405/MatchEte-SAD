import pandas as pd
import ollama
import json
import os
import sys

def cargar_config():
    try:
        with open('configuration.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Error: No se encuentra 'configuration.json'.")
        sys.exit(1)

def balancear_dataset():
    if len(sys.argv) < 2:
        print("\n❌ Error: Indica el archivo. Uso: python balanceo_pro.py <archivo.csv>")
        return

    archivo_entrada = sys.argv[1]
    if not os.path.exists(archivo_entrada):
        print(f"❌ Error: El archivo '{archivo_entrada}' no existe.")
        return

    config = cargar_config()
    TEXT_COL = config['project_params']['text_column']
    TARGET_COL = config['project_params']['target_column']
    # --- AQUÍ ESTÁ LA MAGIA PROFESIONAL ---
    INDUSTRIA = config['project_params'].get('industry', 'general products')
    CLASES_OBJETIVO = config['project_params'].get('balancing_classes', 'all')
    
    MODELO = "llama3:8b"
    archivo_salida = archivo_entrada.replace(".csv", "_IA.csv")

    df = pd.read_csv(archivo_entrada)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    ultimo_guardado = 0
    
    conteos = df[TARGET_COL].value_counts()
    objetivo = conteos.max() 
    
    print(f"📊 Sector detectado: {INDUSTRIA}")
    print(f"📈 Objetivo: {objetivo} filas por clase.")

    categorias = sorted(df[TARGET_COL].unique()) if CLASES_OBJETIVO == "all" else CLASES_OBJETIVO

    for clase in categorias:
        total_actual = conteos.get(clase, 0)
        faltantes = objetivo - total_actual
        if faltantes <= 0: continue

        print(f"🚀 Generando {faltantes} reseñas de {INDUSTRIA} para Clase {clase}...")
        
        # Tomamos ejemplos para estilo, pero le diremos que no los copie
        ejemplos = df[df[TARGET_COL] == clase][TEXT_COL].head(3).tolist()
        contexto = " | ".join(ejemplos)

        generadas = 0
        while generadas < faltantes:
            batch_size = min(5, faltantes - generadas)
            
            # PROMPT AGREGADO: Instrucción de no repetir ejemplos
            prompt = (
                f"You are a synthetic data generator for the {INDUSTRIA} sector. "
                f"TASK: Generate exactly {batch_size} unique user reviews. "
                f"SENTIMENT: Class {clase}. "
                f"STYLE REFERENCE (DO NOT COPY THESE WORDS): {contexto}. "
                f"STRICT RULES: Output ONLY the reviews. NO introductions like 'Here are'. NO rules acknowledgement. NO quotes. "
                f"SEPARATOR: Use '##' between reviews."
            )
            
            try:
                response = ollama.generate(model=MODELO, prompt=prompt)
                raw_text = response['response']

                # --- LIMPIEZA QUIRÚRGICA ---
                # Borramos cualquier frase de cortesía de la IA
                bad_starts = ["Here", "Sure", "I'll", "OK", "Certainly", "Certainly!"]
                lines_to_check = raw_text.split("\n")
                clean_lines = [l for l in lines_to_check if not any(l.strip().startswith(x) for x in bad_starts)]
                raw_text = "\n".join(clean_lines)

                # Separamos y limpiamos basura
                reviews = [r.strip().replace('"', '') for r in raw_text.split('##') if len(r.strip()) > 20]
                reviews = reviews[:batch_size]

                if reviews:
                    nuevas_filas = pd.DataFrame({TEXT_COL: reviews, TARGET_COL: [clase] * len(reviews)})
                    df = pd.concat([df, nuevas_filas], ignore_index=True)
                    generadas += len(reviews)
                    print(f"   [+] {generadas}/{faltantes}...")

                    if generadas - ultimo_guardado >= 50:
                        df.to_csv(archivo_salida, index=False)
                        ultimo_guardado = generadas
                        print(f"   💾 Checkpoint guardado.")
                
            except Exception as e:
                print(f"⚠️ Reintentando por error: {e}")
                continue

        df.to_csv(archivo_salida, index=False)
    print(f"✨ Archivo final listo para cualquier empresa: {archivo_salida}")

if __name__ == "__main__":
    balancear_dataset()
