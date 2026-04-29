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
        df = df.dropna(subset=["score"]) 

        # 3. Etiquetar
        df["label"] = df["score"].apply(label_sentiment)

        # 4. Separar (AQUÍ FALTABA EL NEUTRAL)
        df_pos = df[df["label"] == "positive"]
        df_neg = df[df["label"] == "negative"]
        df_neu = df[df["label"] == "neutral"] # <--- ESTA LÍNEA FALTABA

        # 5. Guardar con nombres dinámicos
        prefix = archivo.split("_")[0].lower() 
        
        df_pos.to_csv(f"{prefix}_positive.csv", index=False)
        df_neg.to_csv(f"{prefix}_negative.csv", index=False)
        df_neu.to_csv(f"{prefix}_neutral.csv", index=False) # <--- ESTA LÍNEA FALTABA

        print(f"-> Generados: {prefix}_positive.csv, {prefix}_negative.csv y {prefix}_neutral.csv") # <--- Y ESTE AVISO

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo}.")
    except Exception as e:
        print(f"Error procesando {archivo}: {e}")

print("\n¡Todo listo!")
