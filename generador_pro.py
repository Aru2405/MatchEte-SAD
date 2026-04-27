import requests
import csv
import random

# CONFIGURACIÓN
num_comentarios = 100
url = "http://localhost:11434/api/generate"
modelo = "gemma2:2b-text-q4_K_S" # El que tienes instalado y es rápido

generos = ['male', 'female']
ubicaciones = ['New York, United States', 'London, UK', 'Madrid, Spain', 'Sydney, Australia', 'Berlin, Germany']

print(f"Iniciando generación con {modelo}...")

with open('sinteticos_neutros_v2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['reviewId', 'content', 'score', 'gender', 'location', 'date'])

    for i in range(1, num_comentarios + 1):
        # Prompt muy directo para ir rápido
        prompt = "Write a short, realistic, neutral dating app review (max 20 words). Just the text."
        payload = {"model": modelo, "prompt": prompt, "stream": False}
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() # Lanza error si la web falla
            data = response.json()
            
            if 'response' in data:
                texto = data['response'].strip()
                writer.writerow([f"auto-{i}", texto, 2, random.choice(generos), random.choice(ubicaciones), "2026-04-27"])
                print(f"Generado {i}/{num_comentarios}: {texto[:40]}...")
            elif 'error' in data:
                print(f"Ollama devolvió un error: {data['error']}")
                break
            else:
                print(f"Respuesta inesperada: {data}")
                break
                
        except Exception as e:
            print(f"Error crítico: {e}")
            break

print("¡Proceso terminado!")
