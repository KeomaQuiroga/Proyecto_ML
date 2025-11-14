from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, BernoulliNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import spacy
import contractions

# preprocesamiento de texto
nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    text = contractions.fix(text, slang=False)       # expandimos abreviaciones
    doc = nlp(text)
    filtered = []

    for token in doc:       # tokenizamos
        if token.is_punct:      # quitamos signos de puntuacion
            continue
        filtered.append(token.lemma_)       # lemanizamos palabras
    
    return " ".join(filtered)

df = pd.read_csv("train_sent_emo.csv")
df = df.drop(columns=["Speaker", "Season", "Episode","StartTime", "EndTime"])       # eliminamos las columnas innescesarias

# mapeamos las categorias a numeros
df["label_emotion"] = df.Emotion.map(
    {
        "neutral" : 0,
        "joy" : 1,
        "sadness" : 2,
        "anger" : 3,
        "surprise" : 4,
        "fear" : 5,
        "disgust" : 6
    }
)

df["label_sentiment"] = df.Sentiment.map(
    {
        "neutral" : 0,
        "positive" : 1,
        "negative" : 2
    }
)

# balance de clases
print("Emociones")
print(df.label_emotion.value_counts(), "\n")

print("Sentimientos")
print(df.label_sentiment.value_counts(), "\n")

print(df.shape, "\n")
print(df.head(), "\n")

# A PARTIR DE AQUI ES PRUEBA
# cargamos datos prueba
prueba = pd.read_csv("dev_sent_emo.csv")
prueba = prueba.drop(columns=["Speaker", "Season", "Episode","StartTime", "EndTime"])       # eliminamos las columnas innescesarias
prueba["label_emotion"] = prueba.Emotion.map(
    {
        "neutral" : 0,
        "joy" : 1,
        "sadness" : 2,
        "anger" : 3,
        "surprise" : 4,
        "fear" : 5,
        "disgust" : 6
    }
)

prueba["label_sentiment"] = prueba.Sentiment.map(
    {
        "neutral" : 0,
        "positive" : 1,
        "negative" : 2
    }
)

# balance de clases
print("Emociones")
print(prueba.label_emotion.value_counts(), "\n")

print("Sentimientos")
print(prueba.label_sentiment.value_counts(), "\n")

# preprocesamos el texto para lemanizar y quitar palabras stop
df["prepro_txt"]= df["Utterance"].apply(preprocess)
prueba["prepro_txt"]= prueba["Utterance"].apply(preprocess)

print(df.head())
print(prueba.head())

#preparamos el set
# X = df.Utterance
X = df.prepro_txt
y1 = df.label_emotion
y2 = df.label_sentiment

# X_test = prueba.Utterance
X_test = prueba.prepro_txt
y1_test = prueba.label_emotion
y2_test = prueba.label_sentiment

# parametros = {
#     "vectorizer__ngram_range" : [(1, 1), (1, 2), (2, 2)],
#     "vectorizer__max_df" : [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
#     "vectorizer__min_df" : [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
#     "nb__alpha" : [0.01, 0.1, 0.25, 0.5, 0.99, 1.5, 3, 5]
# }

modelo = Pipeline ([
    ("vectorizer", TfidfVectorizer()),      # vectorizamos las palabras
    ("nb", MultinomialNB())     # modelo
])

# grid = GridSearchCV(modelo, param_grid=parametros, n_jobs=-1)

# entrenamos para emociones
modelo.fit(X, y1)
y_pred = modelo.predict(X_test)
print(classification_report(y1_test, y_pred, zero_division=0), "\n")

# entrenamos para sentimientos
modelo.fit(X, y2)
y_pred = modelo.predict(X_test)
print(classification_report(y2_test, y_pred), "\n")