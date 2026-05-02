import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

def split_dataset(input_file):
    if not os.path.exists(input_file):
        print(f"Error: El archivo {input_file} no existe.")
        return

    df = pd.read_csv(input_file)
    
    # 1. Crear la columna 'label' si no existe a partir de 'score'
    if 'label' not in df.columns and 'score' in df.columns:
        print("--- Creando etiquetas (0, 1, 2) a partir de 'score' ---")
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df = df.dropna(subset=['score'])
        
        condiciones = [(df['score'] <= 2), (df['score'] == 3), (df['score'] >= 4)]
        df['label'] = np.select(condiciones, [0, 1, 2])
    elif 'label' not in df.columns:
        print("Error: El archivo no tiene columna 'label' ni 'score'.")
        return

    print(f"--- Dataset original: {len(df)} filas ---")

    # 2. Split aleatorio y ESTRATIFICADO (70% Train, 30% Resto)
    # Stratify asegura que el % de cada clase sea igual en todos los archivos
    df_train, df_temp = train_test_split(
        df, 
        test_size=0.30, 
        random_state=42, 
        shuffle=True, 
        stratify=df['label']
    )

    # 3. Dividir el 30% restante a la mitad (15% Dev, 15% Test)
    df_dev, df_test = train_test_split(
        df_temp, 
        test_size=0.50, 
        random_state=42, 
        shuffle=True, 
        stratify=df_temp['label']
    )

    # 4. Guardar archivos
    df_train.to_csv("Dataset_Train.csv", index=False)
    df_dev.to_csv("Dataset_Dev.csv", index=False)
    df_test.to_csv("Dataset_Test.csv", index=False)

    print(f"\n¡Todo listo! Archivos generados en carpeta actual:")
    print(f" - Dataset_Train.csv: {len(df_train)} filas (Usa este para balancear/entrenar)")
    print(f" - Dataset_Dev.csv:   {len(df_dev)} filas (Usa este para validar/ajustar)")
    print(f" - Dataset_Test.csv:  {len(df_test)} filas (Examen final - NO TOCAR)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python split.py Boo_limpio.csv")
    else:
        split_dataset(sys.argv[1])
