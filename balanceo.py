import pandas as pd
import ollama
import os
import sys

# === CONFIGURACIÓN GENERAL ===
MODELO = "llama3:8b"
ARCHIVO_SALIDA = "train_balanceado.csv"
COL_SENTIMIENTO = 'score'
COL_CONTENIDO = 'content'

def balancear_dataset():
    # 1. Validar argumentos de entrada
    if len(sys.argv) < 2:
        print("\nUso: python balanceo.py <tu_archivo.csv>")
        print("Ejemplo: python balanceo.py Boo_limpio.csv")
        return

    archivo_entrada = sys.argv[1]

    if not os.path.exists(archivo_entrada):
        print(f"❌ Error: No se encuentra el archivo '{archivo_entrada}'")
        return

    # 2. Cargar datos y calcular objetivo
    df = pd.read_csv(archivo_entrada)
    df[COL_SENTIMIENTO] = df[COL_SENTIMIENTO].astype(str)
    
    conteos = df[COL_SENTIMIENTO].value_counts()
    objetivo = conteos.max()  # Se balancea al número de la clase mayoritaria
    
    print(f"📊 Dataset cargado: {len(df)} filas.")
    print(f"📈 Objetivo de balanceo: {objetivo} filas por categoría.")
    print(f"🧠 Usando modelo: {MODELO}\n")

    # 3. Identificar qué clases necesitan más datos
    categorias = sorted(df[COL_SENTIMIENTO].unique())

    for clase in categorias:
        total_actual = conteos.get(clase, 0)
        faltantes = objetivo - total_actual

        if faltantes <= 0:
            print(f"✅ Categoría '{clase}' ya está balanceada ({total_actual} filas).")
            continue

        print(f"🚀 Generando {faltantes} nuevas reseñas para Score {clase}...")
        
        # Tomar ejemplos reales para que la IA mantenga el estilo
        ejemplos = df[df[COL_SENTIMIENTO] == clase][COL_CONTENIDO].head(2).tolist()
        contexto = " | ".join(ejemplos)

        generadas_esta_clase = 0
        while generadas_esta_clase < faltantes:
            # Lotes pequeños (batch) para no saturar la RAM de Llama 3
            batch_size = min(4, faltantes - generadas_esta_clase)
            
            prompt = (
                f"Actúa como un usuario de una app de citas. Escribe exactamente {batch_size} "
                f"reseñas muy breves con una puntuación de {clase} sobre 5. "
                f"Separa cada reseña con el símbolo '##'. No uses listas numeradas.\n"
                f"Estilo de referencia: {contexto}"
            )
            
            try:
                response = ollama.generate(model=MODELO, prompt=prompt)
                respuestas_limpias = [r.strip() for r in response['response'].split('##') if len(r.strip()) > 5]
                
                # Crear DataFrame temporal y añadir al principal
                if respuestas_limpias:
                    nuevas_filas = pd.DataFrame({
                        COL_CONTENIDO: respuestas_limpias,
                        COL_SENTIMIENTO: [clase] * len(respuestas_limpias)
                    })
                    df = pd.concat([df, nuevas_filas], ignore_index=True)
                    generadas_esta_clase += len(respuestas_limpias)
                    
                    # Feedback visual de progreso
                    if generadas_esta_clase % 12 < 4:
                        print(f"   ... Progreso Score {clase}: {generadas_esta_clase}/{faltantes}")
                
            except Exception as e:
                print(f"⚠️ Error en la conexión con Ollama: {e}. Reintentando...")
                continue

        # 4. Guardar progreso al terminar cada categoría (Checkpoint)
        df.to_csv(ARCHIVO_SALIDA, index=False)
        print(f"💾 ¡Categoría {clase} completada y guardada en {ARCHIVO_SALIDA}!")

    print(f"\n✨ ¡PROCESO FINALIZADO! El archivo '{ARCHIVO_SALIDA}' está listo para el entrenamiento.")

if __name__ == "__main__":
    balancear_dataset()
