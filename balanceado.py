import pandas as pd
import os

def llamar_ai_generativa(sentimiento, app_name):
    # Simulación de la IA (Llama 3 + Few-shot)
    return f"New {sentimiento} review generated for {app_name} by IA"

def ejecutar_balanceo():
    archivos_objetivo = ["Boo.csv", "Hinge.csv"]
    
    for archivo in archivos_objetivo:
        if not os.path.exists(archivo):
            print(f"⚠️ No encuentro {archivo}, me lo salto.")
            continue
            
        print(f"\n--- Procesando archivo: {archivo} ---")
        df = pd.read_csv(archivo)
        
        # Convertimos la columna 'score' a números
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df = df.dropna(subset=['score'])
        
        app_name = os.path.splitext(archivo)[0].capitalize()
        
        # Contamos las tres categorías
        positivas = len(df[df['score'] >= 4])
        neutras = len(df[df['score'] == 3])
        negativas = len(df[df['score'] <= 2])
        
        # CAMBIO 1: El objetivo ahora es el número más alto de los tres
        objetivo = max(positivas, neutras, negativas)
        nuevas_filas = []

        # CAMBIO 2: Incluimos las 3 categorías en el bucle para que genere de lo que falte
        categorias = [
            ("POSITIVA", positivas, 'positive', 5),
            ("NEGATIVA", negativas, 'negative', 1),
            ("NEUTRA", neutras, 'neutral', 3)
        ]

        for tipo, actual, label, score_val in categorias:
            faltan = objetivo - actual
            if faltan > 0:
                print(f"-> Generando {faltan} reseñas de tipo {tipo} para {app_name}...")
                for _ in range(faltan):
                    texto = llamar_ai_generativa(tipo, app_name)
                    nuevas_filas.append({'content': texto, 'score': score_val, 'sentiment': label})

        # Unimos todo y guardamos
        df_final = pd.concat([df, pd.DataFrame(nuevas_filas)], ignore_index=True)
        nombre_salida = f"{app_name}_Balanceado_Final.csv"
        
        df_final.to_csv(nombre_salida, index=False)
        print(f"✅ ¡Éxito! Creado: {nombre_salida}")

if __name__ == "__main__":
    ejecutar_balanceo()
