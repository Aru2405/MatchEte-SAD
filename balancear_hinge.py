import pandas as pd
from sklearn.utils import resample

# 1. Cargar el dataset de Hinge
df = pd.read_csv('Hinge.csv')

# 2. Asegurarnos de que el score es un número (limpieza)
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df = df.dropna(subset=['score'])

# 3. Definir el objetivo (la clase mayoritaria de Hinge)
target_count = df['score'].value_counts().max()
print(f"La clase mayoritaria tiene {target_count} muestras.")

# 4. Función para balancear
def balance_dataset(df):
    classes = df['score'].unique()
    resampled_list = []
    for cls in classes:
        df_class = df[df['score'] == cls]
        # Oversample
        df_class_resampled = resample(df_class, replace=True, n_samples=target_count, random_state=42)
        resampled_list.append(df_class_resampled)
    return pd.concat(resampled_list)

# 5. Ejecutar y guardar
df_balanced = balance_dataset(df)
df_balanced.to_csv('Hinge_Balanceado.csv', index=False)
print("¡Éxito! Archivo 'Hinge_Balanceado.csv' creado correctamente.")
