# MatchEte-SAD: Análisis de Sentimiento y Aumento de Datos en Dating Apps (Boo vs Hinge)

Este proyecto desarrolla un ecosistema analítico para reseñas de las plataformas **Boo** y **Hinge**. El objetivo principal es comparar la efectividad de un entrenamiento tradicional frente a uno potenciado por **IA Generativa**, integrando además segmentación por clustering y visualización analítica.

## 👥 Equipo y Responsabilidades

*   **María Fernández (IA Generativa)**: Diseño de la estrategia de aumento de datos sintéticos, balanceo de clases mediante `balanceo_pro.py` y ejecución de `train_ia.py` para generar los modelos exportables (`.pkl`) entrenados con IA.
*   **Aitzol Rivera (Clasificación)**: Responsable del modelado de clasificación tradicional, optimización de hiperparámetros y la validación final de los rendimientos (F1-Score) utilizando el script de test.
*   **Paula Tapias (Clustering)**: Análisis de segmentación temática y cálculo de coherencia de grupos mediante los algoritmos **LDA** (reseñas largas) y **K-Means** (reseñas cortas).
*   **Iván Salazar (Tableau)**: Responsable de la visualización analítica de mercado y la creación de dashboards interactivos para la comparativa de las aplicaciones.

## 📂 Componentes del Proyecto

### 1. Parte Generativa y Entrenamiento IA (María)
Es el motor del proyecto, encargado de equilibrar el dataset y producir los modelos finales:
*   **Aumento de Datos**: Uso de `balanceo_pro.py` con **Ollama (Llama 3:8b)** para generar reseñas sintéticas realistas de las clases minoritarias (Negativa y Neutra).
*   **Generación de Modelos**: El script `train_ia.py` procesa el dataset aumentado (`Dataset_Train_IA.csv`) y exporta dos archivos críticos:
    *   **`model.pkl`**: El mejor modelo de clasificación guardado (Logistic Regression o KNN).
    *   **`vectorizer.pkl`**: El transformador TF-IDF ajustado a los datos de la IA.

### 2. Clasificación y Validación (Aitzol)
Fase dedicada a la optimización de los algoritmos de Machine Learning y su evaluación final:
*   **Modelado**: Comparativa de rendimiento entre modelos entrenados con métodos tradicionales frente a los generados por IA.
*   **Validación**: Uso del script de test para obtener métricas detalladas (F1-Score, Precision y Recall) sobre el conjunto de test independiente.

### 3. Clustering Temático (Paula)
Identificación de los temas clave que preocupan o satisfacen a los usuarios:
*   **Optimización**: Selección de **k=4** clusters basándose en el análisis de la curva del codo (inercia) y la coherencia de los temas.
*   **Resultados**: Segmentación en categorías como Personalidad, Modelo de Negocio, Conexiones y Feedback Estético.

### 4. Visualización y Storytelling - Tableau (Iván)
Se ha desarrollado una **Story de Tableau de 6 niveles** diseñada para transformar los datos técnicos en insights estratégicos de negocio:

*   **Diagnóstico Global de Mercado**: Comparativa de valoraciones donde **Boo (4,10)** supera ampliamente a **Hinge (1,93)**. Se destaca una distribución de sentimientos inversa: Boo cuenta con un **74% de reseñas positivas**, mientras que Hinge registra un **74% de negativas**.
*   **Análisis Geográfico y Temporal**: Mapeo de la satisfacción por países identificando el liderazgo máximo en Malasia (+2,91) y seguimiento de la evolución mensual de reseñas para detectar picos de insatisfacción relacionados con cambios en el producto.
*   **Segmentación Demográfica**: Análisis detallado de scores por género y estudio de la longitud de las reseñas, confirmando que los usuarios insatisfechos tienden a escribir textos más extensos.
*   **La Voz del Usuario (Wordclouds)**: Visualización de términos clave extraídos de los clusters, revelando que las críticas hacia la competencia se centran en problemas de monetización (`pay`, `money`) y bloqueos de cuenta (`banned`).
*   **Resumen Ejecutivo y Recomendaciones**: Dashboard final con KPIs críticos y tres acciones propuestas: mantener inversión en mercados líderes, reforzar marketing en zonas de competencia estrecha (UAE/Paraguay) y comunicar la ventaja diferencial de Boo frente a las restricciones de Hinge.

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
    *   Tener instalado [Ollama].
    *   Disponer del modelo `llama3:8b` descargado (`ollama run llama3`).
4.  **Versión de Python**: Se recomienda Python 3.10+.
