import pandas as pd
import numpy as np
import contractions
import spacy
import torch.nn as nn
import torch.nn.functional as F

# preprocesamiento de texto
nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    text = contractions.fix(text, slang=False)       # expandimos abreviaciones y evitamos uso de jergas
    doc = nlp(text)

    tokens = []
    for token in doc:       # tokenizamos
        if token.is_punct:      # quitamos signos de puntuacion
            continue
        tokens.append(token.lower_)     # minusculas las palabras
    
    return tokens

# cargar archivo glove
def load_glove(archivo):
    embeder = {}
    with open(archivo, "r") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype="float32")
            embeder[word] = embedding
    return embeder

# usar glove en las oraciones
def sentence_glove(tokens, glove):
    vector = []

    for tok in tokens:
        if tok in glove:
            vector.append(glove[tok])

    if len(vector) == 0:
        return np.zeros(300)

    return np.mean(vector, axis=0)

glove = load_glove("03_models/glove.6B.300d.txt")     # cargamos glove

# DATASET ENTRENAMIENTO
df = pd.read_csv("01_datasets/MELD/train_sent_emo.csv")
df = df.drop(columns=["Speaker", "Season", "Episode","StartTime", "EndTime"])       # eliminamos las columnas innescesarias

df["prepro_txt"]= df["Utterance"].apply(preprocess)
print(df.shape, "\n")
print(df.head(), "\n")

df["glove_txt"]= df["prepro_txt"].apply(lambda x: sentence_glove(x, glove))
print(df.head())

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

# DATASET PRUEBA
prueba = pd.read_csv("01_datasets/twitter/Data/conversations_sent&emo.csv")
prueba = prueba.drop(columns=["author_id", "inbound","created_at", "response_tweet_id", "in_response_to_tweet_id"])       # eliminamos las columnas innescesarias

prueba["prepro_txt"]= prueba["text"].apply(preprocess)
print(prueba.shape, "\n")
print(prueba.head(), "\n")

prueba["glove_txt"]= prueba["prepro_txt"].apply(lambda x: sentence_glove(x, glove))
print(prueba.head())

# mapeamos las categorias a numeros
prueba["label_emotion"] = prueba.emotion.map(
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

prueba["label_sentiment"] = prueba.sentiment.map(
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

# APLICACION MODELO
class CNN(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):