import pandas as pd

# Lista de archivos limpios a procesar
archivos_limpios = ["Boo_limpio.csv", "Hinge_limpio.csv"]

def label_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

for archivo in archivos_limpios:
    try:
        # 1. Cargar el csv
        df = pd.read_csv(archivo)
        print(f"Procesando sentimientos para: {archivo}")

        # 2. Convertir score a numérico
        df["score"] = pd.to_numeric(df["score"], errors='coerce')
        df = df.dropna(subset=["score"]) # Eliminamos filas sin score válido

        # 3. Etiquetar
        df["label"] = df["score"].apply(label_sentiment)

        # 4. Separar
        df_pos = df[df["label"] == "positive"]
        df_neg = df[df["label"] == "negative"]

        # 5. Guardar con nombres dinámicos (basados en el nombre original)
        prefix = archivo.split("_")[0].lower() # Extrae "boo" o "hinge"
        
        df_pos.to_csv(f"{prefix}_positive.csv", index=False)
        df_neg.to_csv(f"{prefix}_negative.csv", index=False)

        print(f"-> Generados: {prefix}_positive.csv y {prefix}_negative.csv")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo}. Asegúrate de haber ejecutado el script de limpieza antes.")
    except Exception as e:
        print(f"Error procesando {archivo}: {e}")

print("\n¡Todo listo!")
