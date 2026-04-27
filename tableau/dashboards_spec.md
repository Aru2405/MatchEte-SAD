# Especificación de dashboards y Story de Tableau

> Este documento describe **paso a paso** los dashboards a construir en Tableau,
> qué CSV usa cada uno, qué gráfico, qué filtros y qué pregunta del enunciado
> contestan. Sigue este doc en orden y tendrás la Story montada.

## Resumen de la "historia" que vamos a contar

**Mensaje al jefe (1 frase):**
> *Boo lidera ampliamente a Hinge en valoraciones globales (4.10 vs 1.93), pero la
> ventaja se estrecha en Oriente Medio y África subsahariana — y el feedback negativo
> de Hinge ("banned", "pay", "money") señala oportunidades concretas de marketing
> diferencial.*

**Estructura de la Story (5 slides):**
1. **Contexto** — quiénes somos y a qué nos enfrentamos.
2. **Diagnóstico global** — score medio Boo vs Hinge.
3. **Mapa mundial** — dónde estamos fuertes y dónde no.
4. **Voz del usuario** — clusters/palabras de cada sentimiento.
5. **Conclusiones y recomendación** al jefe.

---

## Antes de empezar en Tableau

1. Abrir Tableau Online: <https://sso.online.tableau.com/> (usuario = email UPV, pass inicial `sad1`).
2. **Configuración regional**: España/Spanish (importante por el separador decimal — el enunciado lo subraya).
3. Conectar la fuente principal: `tableau/data/dataset_geo.csv` (19.984 filas).
4. Verificar tipos:
   - `date` → Fecha
   - `lat`, `lon`, `score`, `content_length`, `word_count` → Numérico
   - `country` → Geográfico (Country/Region)
   - `iso3` → Geográfico (ISO 3166-1 alpha-3)

> **Tip:** carga también las `agg_*.csv` como fuentes secundarias para las
> tablas que vayan al póster (más rápido que recalcular cada vez).

---

## Dashboard 1 — "El termómetro" (Pregunta 1: estado general)

**Pregunta:** ¿Cómo está siendo valorada nuestra app vs el competidor?

**Hojas:**

| Hoja | Tipo | Filas | Columnas | Color | Filtros |
|---|---|---|---|---|---|
| `H1.1 Score medio` | Barras horizontales | `app` | `AVG(score)` | `app` | — |
| `H1.2 Distribución de sentimiento` | 100% stacked bar | `app` | `COUNT` | `sentiment` (rojo/gris/verde) | — |
| `H1.3 Volumen de reviews` | KPI grande | — | `COUNT(reviewId)` por app | — | — |

**Disposición:** KPI arriba (3 tarjetas: total reviews / score medio Boo / score medio Hinge), barras al medio, stacked bar abajo.

**Insight a destacar:** Boo tiene **74% positivas**, Hinge **74% negativas**. Es invertido.

---

## Dashboard 2 — "El mapa" (Pregunta 3: distribución geográfica)

**Pregunta:** ¿Cómo se correlaciona zona con satisfacción?

**Hojas:**

| Hoja | Tipo | Filas/Columnas | Color | Tamaño | Filtros |
|---|---|---|---|---|---|
| `H2.1 Mapa Boo` | Map symbol/filled | `country` | `AVG(score)` (verde→rojo) | `COUNT` | `app=Boo` |
| `H2.2 Mapa Hinge` | Map symbol/filled | `country` | `AVG(score)` | `COUNT` | `app=Hinge` |
| `H2.3 Delta Boo-Hinge` | Map filled | `country` | `delta_Boo_vs_Hinge` (azul=ganamos, naranja=pierde) | — | usa `agg_country_comparison.csv` |
| `H2.4 Top/Bottom 10` | Tabla | `country`, `delta` | — | — | top 10 deltas más positivas y más negativas |

**Disposición:** Mapas Boo + Hinge en paralelo arriba, mapa de delta abajo, tabla a la derecha.

**Filtros globales:** continente (multi-select).

**Insight:** Boo gana en TODOS los países. Mejor delta: **Malaysia (+2.91)**, **Czech Republic (+2.83)**. Peor delta (donde más cerca está Hinge): **UAE (+1.71)**, **Paraguay (+1.67)**.

---

## Dashboard 3 — "La evolución" (Pregunta 1.b: temporal)

**Pregunta:** ¿Cómo evolucionan las opiniones en el tiempo?

**Hojas:**

| Hoja | Tipo | Eje X | Eje Y | Color | Filtros |
|---|---|---|---|---|---|
| `H3.1 Score medio mensual` | Líneas + área | `MONTH(date)` | `AVG(score)` | `app` | rango fechas |
| `H3.2 Volumen de reviews mensual` | Barras apiladas | `MONTH(date)` | `COUNT` | `sentiment` | `app` (parámetro) |
| `H3.3 % negativas trimestral` | Líneas | `QUARTER(date)` | `% negative` | `app` | — |

**Disposición:** Las 3 hojas verticalmente alineadas comparten eje X de tiempo.

**Insight:** mirar si hay picos de quejas tras cambios de monetización (`pay`, `money` son top words en Hinge).

---

## Dashboard 4 — "Demografía" (Pregunta 1.a: género/perfil)

**Pregunta:** ¿Hay diferencias por género?

**Hojas:**

| Hoja | Tipo | Filas | Columnas | Color | Filtros |
|---|---|---|---|---|---|
| `H4.1 Score por género` | Barras agrupadas | `app` | `AVG(score)` | `gender` | — |
| `H4.2 % sentimiento × género × app` | Heatmap | `app+gender` | `sentiment` | `% del total` | — |
| `H4.3 Distribución de longitud` | Box plot | `sentiment` | `word_count` | `app` | — |

**Insight extra (para preguntas propias):** ¿comentarios negativos son más largos? Suele ser sí (clientes enfadados escriben más).

---

## Dashboard 5 — "La voz del usuario" (Pregunta 2: clusters)

**Pregunta:** ¿Qué palabras significativas dominan opiniones + y −?

**Datos a usar:**
- **Provisional:** `data/words_top_by_segment.csv` (frecuencias propias).
- **Final:** sustituir por output del líder de clustering (LDA/KMeans con palabras significativas por tópico).

**Hojas:**

| Hoja | Tipo | Filas | Tamaño | Color | Filtros |
|---|---|---|---|---|---|
| `H5.1 Wordcloud Boo positivas` | Packed bubble / Words | `word` | `freq` | `freq` (verde) | `app=Boo, sentiment=positive` |
| `H5.2 Wordcloud Boo negativas` | Packed bubble | `word` | `freq` | rojo | `app=Boo, sentiment=negative` |
| `H5.3 Wordcloud Hinge positivas` | Packed bubble | `word` | `freq` | verde | `app=Hinge, sentiment=positive` |
| `H5.4 Wordcloud Hinge negativas` | Packed bubble | `word` | `freq` | rojo | `app=Hinge, sentiment=negative` |
| `H5.5 Top 10 palabras crítica Hinge` | Barras horizontales | `word` | `freq` | gris | `app=Hinge, sentiment=negative` |

**Disposición:** los 4 wordclouds en cuadrícula 2×2 (eje vertical = sentimiento, eje horizontal = app), barras a la derecha.

**Insight clave:** En Hinge negativo aparecen `banned`, `pay`, `money`, `account`, `matches` — claras quejas de monetización + restricciones. Esto es lo que el jefe necesita saber.

---

## Dashboard 6 — "Resumen ejecutivo" (slide final de la Story)

Una sola página con:
- **3 KPIs grandes** (score medio Boo, score medio Hinge, delta).
- **Mini-mapa delta** simplificado.
- **Top 5 países donde brillamos** (tabla pequeña).
- **3 acciones recomendadas al jefe** (texto):
  1. Mantener inversión en mercados líderes (México, Canadá, USA).
  2. Reforzar marketing en UAE/Paraguay/Qatar (donde Hinge se nos acerca más).
  3. Comunicar nuestra ventaja "no monetizamos como Hinge" (las palabras `pay`, `money`, `banned` no aparecen como queja en nuestras reviews).

---

## Story de Tableau (montaje final)

1. **Slide 1 — "Quiénes somos"**: dashboard 1 (KPIs y distribución).
2. **Slide 2 — "Dónde brillamos"**: dashboard 2 (mapas).
3. **Slide 3 — "Cómo evolucionamos"**: dashboard 3 (líneas temporales).
4. **Slide 4 — "Quiénes nos quieren (y quiénes no)"**: dashboard 4 (demografía).
5. **Slide 5 — "Qué dicen de nosotros (y de Hinge)"**: dashboard 5 (clusters/palabras).
6. **Slide 6 — "Recomendaciones al jefe"**: dashboard 6.

**Tip de storytelling (libro de Cole Knaflic):** cada slide debe tener **un solo
mensaje** subrayado en el título. No "Mapa de scores" sino **"Boo lidera en
América con score 4.2"**.

---

## Checklist antes de la entrega (3-mayo-2026)

- [ ] Configuración regional revisada (separador decimal correcto)
- [ ] Todos los mapas reconocen los 70 países (verde = OK, gris = error de geocoding)
- [ ] Filtros aplicados a nivel dashboard (no por hoja, evita confusión)
- [ ] Tooltips personalizados con métricas clave
- [ ] Títulos de cada slide formulados como "insight", no como descripción
- [ ] Story exportada a PDF/imagen para incluir en el póster
- [ ] Workbook publicado en Tableau Online compartido con el equipo y profesora
- [ ] Slides 1 y 6 ensayadas para presentar en clase
