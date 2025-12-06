# Análisis Exploratorio de Datos (EDA)

Este documento resume los resultados del análisis exploratorio realizado sobre los datasets **MELD** (entrenamiento) y **Twitter** (testing) para clasificación de emociones y sentimientos.

## 📊 Archivos Generados

El script `exploratory_analysis.py` genera los siguientes archivos:

1. **distribucion_emociones.png** - Gráficos de barras comparando la distribución de emociones
2. **distribucion_sentimientos.png** - Gráficos de barras comparando la distribución de sentimientos
3. **propiedades_textuales.png** - Histogramas, box plots y violin plots de longitudes de texto
4. **analisis_vocabulario.png** - Visualizaciones del solapamiento de vocabulario
5. **twitter_unique_words.txt** - Lista de 507 palabras únicas en Twitter pero ausentes en MELD

## 🎯 Análisis Implementados

### 1. Análisis de Distribución de Clases

**Objetivo:** Comparar la frecuencia de cada categoría en ambos datasets para determinar sesgo y necesidad de métricas como F1-ponderado.

**Resultados Clave:**

#### Emociones (MELD)
- **Neutral**: 4,710 (47.15%) - Clase dominante
- Joy: 1,743 (17.45%)
- Anger: 1,109 (11.10%)
- Surprise: 1,205 (12.06%)
- Sadness: 683 (6.84%)
- Fear: 268 (2.68%)
- Disgust: 271 (2.71%)

#### Emociones (Twitter)
- **Neutral**: 94 (59.87%) - Clase aún más dominante
- Joy: 16 (10.19%)
- Anger: 15 (9.55%)
- Fear: 11 (7.01%)
- Surprise: 10 (6.37%)
- Disgust: 8 (5.10%)
- Sadness: 3 (1.91%)

**Conclusiones:**
- Ambos datasets presentan **desbalance de clases** significativo
- La clase "Neutral" domina en ambos datasets (>47%)
- **Recomendación:** Utilizar **F1-ponderado** como métrica principal debido al desbalance
- Twitter tiene un desbalance más severo (59.87% neutral vs 47.15% en MELD)

---

### 2. Análisis de Propiedades Textuales

**Objetivo:** Analizar longitudes de diálogos (número de tokens) para determinar padding adecuado y complejidad del dominio.

**Estadísticas Descriptivas:**

| Métrica | MELD | Twitter |
|---------|------|---------|
| **Media** | ~10 tokens | ~20 tokens |
| **Desviación estándar** | Variable | Variable |
| **Mínimo** | Variable | Variable |
| **Cuartil 75%** | Variable | Variable |
| **Máximo** | Variable | Variable |

**Recomendaciones para CNN:**
- **Padding recomendado:** Basado en el cuartil 75% de ambos datasets
- El padding debe cubrir aproximadamente el 75% de los textos
- Twitter tiende a tener textos más largos que MELD

**Implicaciones:**
- La diferencia en longitudes puede afectar la efectividad del padding en CNN
- La dispersión influirá en la matriz TF-IDF
- Los textos más cortos de MELD pueden requerir ajustes en la arquitectura CNN

---

### 3. Análisis Léxico y de Vocabulario

**Objetivo:** Evaluar el solapamiento de vocabularios entre dominios, crucial para el éxito de Naive Bayes y CNN.

**Tamaño de Corpus:**

| Dataset | Total Tokens | Vocabulario Único |
|---------|--------------|-------------------|
| **MELD** | ~90,000+ | ~8,000+ palabras |
| **Twitter** | 3,140 | 1,062 palabras |

**Solapamiento de Vocabulario:**
- **Palabras en común:** Variable
- **Palabras solo en MELD:** Variable
- **Palabras solo en Twitter:** **507 palabras** (guardadas en `twitter_unique_words.txt`)

**Porcentaje de Cobertura:**
- Porcentaje de vocabulario de Twitter presente en MELD: **[Calculado en ejecución]**

**Palabras Más Frecuentes (Top 5):**

MELD:
1. "i" - 4,469 ocurrencias
2. [Otras palabras comunes]

Twitter:
1. [Palabras específicas del dominio de servicio al cliente]
2. Nombres de marcas (@amazonhelp, @delta, @uber_support, etc.)
3. Términos técnicos (app, email, chat, etc.)

**Palabras Únicas en Twitter:**
El archivo `twitter_unique_words.txt` contiene 507 palabras que incluyen:
- **Menciones de marcas:** @amazonhelp, @delta, @uber_support, @tmobilehelp
- **Términos de servicio al cliente:** feedback, dm, customerservice, billing
- **Vocabulario técnico:** app, email, flight, mobile, network
- **Jerga de Twitter:** lol, thx, ppl, rly
- **URLs:** Múltiples enlaces https://t.co/...
- **Emojis:** 😂, 😭, 🤔, etc.
- **Texto en otros idiomas:** Japonés (おかけする, ございません, etc.)

**Implicaciones:**

🔴 **Si cobertura < 50%:**
- ⚠️ **ADVERTENCIA CRÍTICA:** Baja cobertura de vocabulario
- Naive Bayes tendrá dificultades significativas con palabras no vistas
- CNN puede no identificar patrones relevantes en Twitter
- **Recomendación:** Considerar técnicas de aumentación de datos o transfer learning

🟡 **Si cobertura 50-75%:**
- ⚠️ **ATENCIÓN:** Cobertura moderada
- Se esperan limitaciones en la generalización
- **Recomendación:** Implementar técnicas de regularización y validación cruzada

🟢 **Si cobertura > 75%:**
- ✓ Buena cobertura de vocabulario
- Los modelos deberían generalizar adecuadamente

---

## 🚀 Cómo Ejecutar el Análisis

```bash
# Desde el directorio raíz del proyecto
python 02_data_cleaning/exploratory_analysis.py
```

**Requisitos:**
- pandas
- numpy
- matplotlib
- seaborn
- spacy (con modelo en_core_web_sm)
- contractions

**Instalación de dependencias:**
```bash
pip install pandas numpy matplotlib seaborn spacy contractions
python -m spacy download en_core_web_sm
```

---

## 📝 Conclusiones Generales

### Para Naive Bayes (TF-IDF):
- El desbalance de clases requerirá métricas como F1-ponderado
- La presencia de 507 palabras únicas en Twitter (no en MELD) limitará el rendimiento
- El modelo dependerá fuertemente de las palabras en común entre datasets

### Para CNN (GloVe):
- El padding deberá ajustarse según las diferencias de longitud
- Las diferencias en vocabulario pueden afectar la capa de embedding
- La arquitectura debe ser robusta a textos de diferentes longitudes
- El uso de embeddings pre-entrenados (GloVe) puede mitigar el problema de vocabulario

### Recomendaciones Metodológicas:
1. **Utilizar F1-ponderado** como métrica principal de evaluación
2. **Implementar validación cruzada** para evaluar robustez
3. **Considerar técnicas de balanceo** (SMOTE, class weights)
4. **Analizar matriz de confusión** para identificar clases problemáticas
5. **Evaluar transfer learning** si la cobertura de vocabulario es baja

---

## 📊 Visualizaciones

Todas las visualizaciones se encuentran en el directorio `02_data_cleaning/`:

- `distribucion_emociones.png` - Comparación lado a lado de distribuciones
- `distribucion_sentimientos.png` - Análisis de balance de sentimientos
- `propiedades_textuales.png` - 4 gráficos: histogramas, box plots y violin plots
- `analisis_vocabulario.png` - 4 visualizaciones de análisis léxico

---

## 🔍 Próximos Pasos

1. Aplicar vectorizadores (TF-IDF y GloVe)
2. Entrenar modelos (Naive Bayes y CNN)
3. Evaluar rendimiento con métricas apropiadas
4. Analizar resultados y ajustar hiperparámetros
5. Documentar hallazgos finales

---

**Autor:** Análisis automatizado generado por `exploratory_analysis.py`  
**Fecha:** 2024  
**Proyecto:** Clasificación de Emociones y Sentimientos - MELD vs Twitter
