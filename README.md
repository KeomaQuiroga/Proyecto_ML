# Proyecto ML - Clasificación de Emociones y Sentimientos
(Generado usando Gemini)

Proyecto de Machine Learning para clasificación de emociones y sentimientos usando datasets **MELD** (entrenamiento) y **Twitter** (prueba).

## 📁 Estructura del Proyecto

```
Proyecto_ML/
├── 01_datasets/        # Datasets MELD y Twitter
├── 02_data_analysis/   # Análisis exploratorio y visualizaciones
├── 03_models/          # Implementaciones de modelos ML
└── requirements.txt    # Dependencias del proyecto
```

## 🎯 Modelos Implementados

### 1. Naive Bayes (TF-IDF)
- Vectorización con **TF-IDF**
- Clasificación de emociones (7 clases) y sentimientos (3 clases)
- Optimización de hiperparámetros con GridSearchCV
- Archivo: `03_models/bayes.py`

### 2. CNN (GloVe Embeddings)
- Arquitectura CNN personalizada para NLP
- Embeddings pre-entrenados **GloVe 300d** obtenido de "The Standford Natural Language Processing Group" 
- Modelos separados para emociones y sentimientos
- Archivos: `03_models/cnn_emo.py`, `03_models/cnn_sent.py`

**Emociones:** Neutral, Joy, Sadness, Anger, Surprise, Fear, Disgust  
**Sentimientos:** Neutral, Positive, Negative

## 🚀 Instalación y Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo spaCy
python -m spacy download en_core_web_sm

# Ejecutar análisis exploratorio
python 02_data_analysis/exploratory_analysis.py

# Entrenar modelos
python 03_models/bayes.py
python 03_models/cnn_emo.py
python 03_models/cnn_sent.py
```

## 📊 Análisis Exploratorio

El análisis genera visualizaciones de:
- Distribución de clases (desbalance significativo ~47% neutral en MELD)
- Propiedades textuales (longitud de textos)
- Solapamiento de vocabulario entre datasets
- Ver detalles en: `02_data_analysis/README_analisis.md`

## 📝 Notas

- Ambos modelos evaluados con y sin clase *Neutral*
- Uso de Accuracy para evaluación del rendimiento general de los modelos
- GloVe: `glove.6B.300d.txt` necesario en `03_models/`
