import pandas as pd

# cargar el csv limpio
df = pd.read_csv("Boo_limpio.csv")

# ---- IMPORTANTE ----
def label_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

# Convertimos la columna score a numérico por si acaso viene como texto
df["score"] = pd.to_numeric(df["score"], errors='coerce')

df["label"] = df["score"].apply(label_sentiment)

# separar
df_pos = df[df["label"] == "positive"]
df_neg = df[df["label"] == "negative"]

# guardar
df_pos.to_csv("boo_positive.csv", index=False)
df_neg.to_csv("boo_negative.csv", index=False)

print("Hecho")
