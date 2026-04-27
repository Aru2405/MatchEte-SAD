import pandas as pd
from sklearn.utils import resample

# 1. Cargar el dataset
df = pd.read_csv('Boo.csv')

# 2. Asegurarnos de que el score es un número
df['score'] = pd.to_numeric(df['score'], errors='coerce')

# 3. Definir el objetivo (la clase mayoritaria)
target_count = df['score'].value_counts().max()

# 4. Función para balancear
def balance_dataset(df):
    classes = df['score'].dropna().unique()
    resampled_list = []
    for cls in classes:
        df_class = df[df['score'] == cls]
        df_class_resampled = resample(df_class, replace=True, n_samples=target_count, random_state=42)
        resampled_list.append(df_class_resampled)
    return pd.concat(resampled_list)

# 5. Ejecutar y guardar
df_balanced = balance_dataset(df)
df_balanced.to_csv('Boo_Balanceado.csv', index=False)
print("¡Éxito! Archivo 'Boo_Balanceado.csv' creado correctamente.")
