import pandas as pd
import json
import joblib
import re
import string
import os
import sys
from sklearn.metrics import f1_score, classification_report

def load_config():
    try:
        with open('configuration.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ ERROR: No se encuentra configuration.json.")
        sys.exit(1)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    return " ".join(text.split())

def run_test():
    config = load_config()
    p = config['project_params']
    
    print(f"🧐 Evaluación Final: {p['industry']}")
    print("-" * 60)

    # CAMBIO CLAVE: Carga dinámica desde JSON
    if not os.path.exists(p['model_name']) or not os.path.exists(p['vectorizer_name']):
        print(f"❌ ERROR: No encuentro {p['model_name']} o {p['vectorizer_name']}.")
        return

    model = joblib.load(p['model_name'])
    vectorizer = joblib.load(p['vectorizer_name'])
    print(f"✅ Cargados correctamente: {p['model_name']} y {p['vectorizer_name']}")

    df_test = pd.read_csv(p['test_path'])
    df_test['clean'] = df_test[p['text_column']].apply(preprocess_text)
    
    X_test = vectorizer.transform(df_test['clean'])
    y_test = df_test[p['target_column']]

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    
    print(f"\n📊 RESULTADO EN TEST:")
    print(f"🏆 F1-Score: {f1:.4f}")
    print("\n📝 Detalle por clases:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    run_test()
