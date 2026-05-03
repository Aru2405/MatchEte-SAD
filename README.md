# MatchEte-SAD: Análisis de Sentimiento y Aumento de Datos en Dating Apps (Boo vs Hinge)

Este proyecto desarrolla un ecosistema analítico para reseñas de las plataformas **Boo** y **Hinge**. El objetivo principal es comparar la efectividad de un entrenamiento tradicional frente a uno potenciado por **IA Generativa**, integrando además segmentación por clustering y visualización analítica.

## 👥 Equipo y Responsabilidades

- **María Fernández (IA Generativa)**: Diseño de la estrategia de aumento de datos sintéticos, balanceo de clases mediante `balanceo_pro.py` y ejecución de `train_ia.py` para generar los modelos exportables (`.pkl`) entrenados con IA.
- **Aitzol Rivera (Clasificación)**: Responsable del modelado de clasificación tradicional, optimización de hiperparámetros y la validación final de los rendimientos (F1-Score) utilizando el script de test.
- **Paula Tapias (Clustering)**: Análisis de segmentación temática y cálculo de coherencia de grupos mediante los algoritmos **LDA** (reseñas largas) y **K-Means** (reseñas cortas).
- **Iván Salazar (Tableau)**: Responsable de la visualización analítica de mercado y la creación de dashboards interactivos para la comparativa de las aplicaciones.

## 📂 Componentes del Proyecto

### 1. Parte Generativa y Entrenamiento IA (María)

Es el motor del proyecto, encargado de equilibrar el dataset y producir los modelos finales:

- **Aumento de Datos**: Uso de `balanceo_pro.py` con **Ollama (Llama 3:8b)** para generar reseñas sintéticas realistas de las clases minoritarias (Negativa y Neutra).
- **Generación de Modelos**: El script `train_ia.py` procesa el dataset aumentado (`Dataset_Train_IA.csv`) y exporta dos archivos críticos:
  - **`model.pkl`**: El mejor modelo de clasificación guardado (Logistic Regression o KNN).
  - **`vectorizer.pkl`**: El transformador TF-IDF ajustado a los datos de la IA.

### 2. Clasificación y Validación (Aitzol)

Fase dedicada a la optimización de los algoritmos de Machine Learning y su evaluación final:

- **Modelado**: Comparativa de rendimiento entre modelos entrenados con métodos tradicionales frente a los generados por IA.
- **Validación**: Uso del script de test para obtener métricas detalladas (F1-Score, Precision y Recall) sobre el conjunto de test independiente.

### 3. Clustering Temático (Paula)

Para profundizar en el análisis más allá del sentimiento, implementamos una estrategia de procesamiento de lenguaje natural (NLP) para agrupar las reseñas por temática.

- Análisis reseñas y Segmentación por Sentimiento: El modelo después de preprocesar los datos, procesa de forma independiente los datasets de Positive, Negative y Neutral. Esto evita que los tópicos se mezclen y permite identificar problemas específicos.


- Modelado Híbrido: Se utilizaron dos enfoques según la longitud de los textos:
1. *K-Means*: Aplicado sobre las reseñas de texto corto para una segmentación rápida basada en frecuencia de términos.
2. *LDA* (Latent Dirichlet Allocation): Utilizado en reseñas extensas para descubrir estructuras de tópicos más profundas y latentes.

- Optimización (K=4): Tras analizar la curva del codo (Inercia) y los niveles de coherencia
- TEMAS ENCONTRADOS:
    - 1. Negativos: Centrados en Gestión de Cuentas (Baneos), Monetización (Barreras de Pago), Inseguridad (Estafas) y Mensajería.
    - 2. Positivos: Enfocados en Autenticidad de Perfiles, Optimización de Tiempo, UX y Descubrimiento de Personas.
- OUTPUT: El script genera AnalisisClustering.csv, unificando Boo y Hinge con sus respectivos "Tema_Nombre"


### 4. Visualización y Storytelling - Tableau (Iván)

La parte de visualización vive en la subcarpeta [`tableau/`](tableau/). Consta de un **pipeline de Python** que prepara y enriquece los datos, más una **Story de Tableau** publicada en Tableau Online que responde a las preguntas del enunciado.

#### 4.1 Pipeline de preparación de datos

Scripts ejecutados en orden por [`tableau/run_all.py`](tableau/run_all.py):

| #   | Script                             | Genera                                                                                                                          |
| --- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 0   | `00_fetch_countries.py` (one-shot) | `country_centroids.py` desde la API REST Countries (códigos ISO-3 oficiales + centroides geográficos)                           |
| 1   | `01_prepare_data.py`               | `data/dataset_unified.csv` — Boo+Hinge unificados, validados, con `sentiment` derivado y campos temporales/demográficos         |
| 2   | `02_geocode.py`                    | `data/dataset_geo.csv` — añade `lat`, `lon`, `iso3`, `continent`                                                                |
| 3   | `03_aggregations.py`               | 9 ficheros `data/agg_*.csv` con agregaciones que responden las 4 preguntas obligatorias                                         |
| 4   | `04_word_frequencies.py`           | `data/word_frequencies_BY_TOPIC.csv — Frecuencias segmentadas por App, Sentimiento y Tema (Cluster) para análisis profundo.      |
| 5   | `05_extra_insights.py`             | 6 ficheros `data/extra_*.csv` para preguntas propias (sesgo de género, polarización, mercados sub-representados)                |


Para regenerar todo:

```bash
cd tableau
pip install pandas
python run_all.py
```

> ℹ️ El script `00_fetch_countries.py` solo se ejecuta una vez (al clonar el repo o cuando aparece un país nuevo). No está en `run_all.py` porque requiere conexión a internet y los datos resultantes son estables.

#### 4.2 Decisión de diseño: dos pipelines complementarios

Existen **dos preprocesados** en el proyecto con propósitos distintos:

| Pipeline             | Propósito                                       | Salida                                    |
| -------------------- | ----------------------------------------------- | ----------------------------------------- |
| `preprocesado.py`    | NLP: lematización + bigramas para LDA/KMeans    | `*_limpio.csv` con `text_final`           |
| `01_prepare_data.py` | Validación + enriquecimiento para visualización | `dataset_geo.csv` con metadatos completos |

El primero **descarta** columnas como `gender`, `date`, `location` (no las necesita para clustering). El segundo las **preserva y enriquece** (geocoding, parsing de fechas, derivación de `sentiment`). No es duplicación — son complementarios.

#### 4.3 Story de Tableau (6 dashboards)

Se ha desarrollado una **Story de Tableau de 6 niveles** diseñada para transformar los datos técnicos en insights estratégicos de negocio:

- **D1 — Diagnóstico Global de Mercado**: Comparativa de valoraciones donde **Boo (4,10)** supera ampliamente a **Hinge (1,93)**. Se destaca una distribución de sentimientos inversa: Boo cuenta con un **75% de reseñas positivas**, mientras que Hinge registra un **74% de negativas**.
- **D2 — Análisis Geográfico**: Mapeo de la satisfacción por países identificando el liderazgo máximo en Malasia (+2,91★) y Chequia (+2,83★). Boo lidera en el 100% de países con datos.
- **D3 — Evolución Temporal**: Hinge tardó 4,5 años en superar la fase inicial de 1★; Boo solo 1,5 años — el competidor lleva una década intentando recuperar a sus usuarios.
- **D4 — Segmentación Demográfica**: Las mujeres son la base más leal de Boo (66% les dan 5★, 3% 1★). Los hombres son los más críticos de Hinge (66% les dan 1★, 9% 5★). Confirmamos que los usuarios insatisfechos escriben textos más extensos.
- **D5 — La Voz del Usuario e Indagación de Tópicos** :Es la representación visual de los clusters. Implementamos una cuadrícula 2x2 que cruza App vs. Sentimiento, permitiendo filtrar por el Tema detectado en la fase de Clustering.Además cuenta con un  gráfico de barras anexo revela que la insatisfacción en Hinge es estructural y económica. Las palabras más frecuentes en las quejas son pay,money y month (monetización agresiva), seguidas de ban y account (problemas de moderación). A Diferencia de Boo que se centran en la experiencia de usuario y personalidad, validando su ventaja competitiva.
- **D6 — Resumen Ejecutivo y Recomendaciones**: Dashboard final con KPIs críticos y tres acciones propuestas: mantener inversión en mercados líderes, reforzar marketing en zonas de competencia estrecha (UAE/Paraguay) y comunicar la ventaja diferencial de Boo frente a las restricciones de Hinge.

#### 4.4 Notas metodológicas

- **Ordenamiento del dataset**: Tras análisis de los CSVs originales se detectó que el dataset proporcionado presenta ordenamiento perfecto entre `score` y `date`: cero overlap entre extremos (ningún 5★ en periodos tempranos, ningún 1★ en tardíos). El 45-48% de los meses tienen todas sus reviews con el mismo score. El Dashboard 3 cuenta la historia que los datos muestran, pero esta limitación se considera en la interpretación.
- **Derivación canónica de `sentiment`**: la conversión `score → sentiment` (1-2 = negative, 3 = neutral, 4-5 = positive) se hace **una sola vez** en `01_prepare_data.py`, alineada con el criterio de `separar_sentimiento.py` y `trainBoo.py` para garantizar consistencia entre los módulos del equipo.

## 🚀 Guía de Ejecución

1.  **Generar Dataset Aumentado (IA)**:
    ```bash
    python balanceo_pro.py Dataset_Train.csv
    ```
2.  **Entrenar y Exportar Modelos (`.pkl`)**:
    ```bash
    python train_ia.py
    ```
3.  **Evaluación Final en Test**:
    ```bash
    python test.py --test Dataset_Test.csv --model model.pkl --vectorizer vectorizer.pkl
    ```

## 🛠️ Requisitos e Instalación

Para probar el proyecto, sigue estos pasos en orden:

1.  **Activar el entorno virtual**:
    Asegúrate de tener tu entorno (`venv`) creado y activo antes de continuar.
2.  **Instalar dependencias**:
    Ejecuta el siguiente comando para instalar todas las librerías necesarias desde el archivo de requisitos:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Infraestructura de IA**:
    - Tener instalado [Ollama].
    - Disponer del modelo `llama3:8b` descargado (`ollama run llama3`).
4.  **Versión de Python**: Se recomienda Python 3.10+.
