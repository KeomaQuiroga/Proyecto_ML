import pandas as pd
import numpy as np
import contractions
import spacy
import torch

# preprocesamiento de texto
nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    text = contractions.fix(text, slang=False)       # expandimos abreviaciones
    doc = nlp(text)
    filtered = []

    tokens = []
    for token in doc:       # tokenizamos
        if token.is_punct:      # quitamos signos de puntuacion
            continue
        tokens.append(token.lower_)
    
    return tokens

def load_glove(archivo):
    embeder = {}
    with open(archivo, "r") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype="float32")
            embeder[word] = embedding
    return embeder

def sentence_glove(tokens, glove):
    vector = []

    for tok in tokens:
        if tok in glove:
            vector.append(glove[tok])

    if len(vector) == 0:
        return np.zeros(300)

    return np.mean(vector, axis=0)


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

df["prepro_txt"]= df["Utterance"].apply(preprocess)
print(df.shape, "\n")
print(df.head(), "\n")

glove = load_glove("dolma_300_2024_1.2M.100_combined.txt")

# modelo
df["glove_txt"]= df["prepro_txt"].apply(lambda x: sentence_glove(x, glove))
print(df.head())