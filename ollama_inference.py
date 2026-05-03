import pandas as pd
import ollama
import os

# --- CONFIGURACIÓN DE RUTAS ---
INPUT_FILE = 'Boo.csv' 
OUTPUT_FILE = 'predicciones_generativas.csv'

def clasificar_comentario(texto):
    """
    Función de inferencia que aplica Prompt Engineering (Few-Shot + Rol)
    para clasificar el sentimiento de un comentario.
    """
    prompt = f"""
    Act as a sentiment analyst for a dating app company. 
    Analyze the following comment and classify it strictly as:
    0: Negative (bugs, bad experience, complaints)
    1: Neutral (indifference, average experience, comparison to others)
    2: Positive (loves the app, good experience, praises)

    Examples:
    'I love this app' -> 2
    'This app is terrible' -> 0
    'It is just like the others' -> 1

    Respond ONLY with the number (0, 1, or 2). Do not write anything else.

    Comment: "{texto}"
    Prediction:"""

    try:
        # LLAMADA A OLLAMA
        response = ollama.generate(model='llama3', prompt=prompt)
        
        # Limpieza de la respuesta
        resultado = response['response'].strip()
        
        # Validación: nos aseguramos de que devuelva un solo dígito válido
        if resultado in ['0', '1', '2']:
            return int(resultado)
        else:
            # Si la IA responde algo raro, devolvemos el valor por defecto (ej: neutro)
            return 1
    except Exception as e:
        print(f"Error en el comentario: {texto[:30]}... -> {e}")
        return 1

def main():
    # 1. CARGAR DATOS
    if not os.path.exists(INPUT_FILE):
        print(f"Error: No se encuentra el archivo {INPUT_FILE}")
        return

    print(f"Cargando datos de {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE).sample(100).reset_index(drop=True)

    # Verificamos que la columna de texto exista
    column_text = 'content' 
    if column_text not in df.columns:
        print(f"Error: La columna '{column_text}' no existe en el CSV.")
        return

    # 2. PROCESO DE INFERENCIA
    print(f"Iniciando clasificación con Ollama ...")
    
    # Aplicamos la función a cada fila del dataset
    df['prediction'] = df[column_text].apply(clasificar_comentario)

    # 3. GUARDAR RESULTADOS
    # El profesor pide: texto del comentario y predicción
    df_final = df[[column_text, 'prediction']]
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print(f"¡Proceso completado!")
    print(f"Archivo guardado: {OUTPUT_FILE}")
    print(f"Total procesado: {len(df_final)} filas.")

if __name__ == "__main__":
    main()
