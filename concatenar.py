import pandas as pd

# 1. Cargar los dos archivos balanceados que acabamos de crear
df_boo = pd.read_csv('Boo_Balanceado_Final.csv')
df_hinge = pd.read_csv('Hinge_Balanceado_Final.csv')

# 2. Concatenarlos (poner uno debajo del otro)
# Usamos ignore_index=True para que el contador de filas sea continuo
df_total = pd.concat([df_boo, df_hinge], ignore_index=True)

# 3. BARAJAR (Shuffle) - ¡Este paso es fundamental!
# Si no barajas, el modelo aprenderá primero todo lo de Boo y luego todo lo de Hinge,
# lo que puede crear un sesgo en el aprendizaje.
df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Guardar el dataset de entrenamiento final o usarlo directamente
df_total.to_csv('Dataset_Train_DatingApps.csv', index=False)

print(f"✅ Unión completada. Total de filas para entrenar: {len(df_total)}")
