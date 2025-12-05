from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import contractions
import spacy
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import dataseto

emociones = ["Neutral", "Joy", "Sadness", "Anger", "Surprise", "Fear", "Disgust"]
sentimientos = ["Neutral", "Positve", "Negative"]

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
df = dataseto.prepo(df)     # mapeamos las categorias a numeros
dataseto.clases(df, "MELD")

# DATASET PRUEBA
prueba = pd.read_csv("01_datasets/twitter/Data/conversations_sent&emo.csv")
prueba = prueba.drop(columns=["author_id", "inbound","created_at", "response_tweet_id", "in_response_to_tweet_id"])       # eliminamos las columnas innescesarias

prueba["prepro_txt"]= prueba["text"].apply(preprocess)
print(prueba.shape, "\n")
print(prueba.head(), "\n")

prueba["glove_txt"]= prueba["prepro_txt"].apply(lambda x: sentence_glove(x, glove))
print(prueba.head())

prueba = dataseto.prepo(prueba)     # mapeamos las categorias a numeros
dataseto.clases(prueba, "Twitter")      # balance de clases

# APLICACION MODELO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Ocupando {device}")

class CNN_sentiments(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(32, 64, 5)
        self.pool3 = nn.MaxPool1d(3)
        self.flat = nn.Flatten()
        self.fc = nn.LazyLinear(out_features=128, bias=True)
        self.fc2 = nn.Linear(128, 3, bias=True)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flat(x)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# preparamos dataset para prueba y evaluacion  
# datos entrenamiento
X_train = np.array(df["glove_txt"].tolist())
X_train = torch.tensor(X_train, dtype=torch.float32)
X_train = X_train.unsqueeze(1)

X_test = np.array(prueba["glove_txt"].tolist())
X_test = torch.tensor(X_test, dtype=torch.float32)
X_test = X_test.unsqueeze(1)

# modelo de sentimientos
criterio = nn.CrossEntropyLoss()
model_sentimientos = CNN_sentiments().to(device)
optimizer = optim.Adam(model_sentimientos.parameters(), lr=0.001)

y_sent_train = torch.tensor(df["label_sentiment"].tolist())
y_sent_test = torch.tensor(prueba["label_sentiment"].tolist())

train_data = TensorDataset(X_train, y_sent_train)
train_load_sent = DataLoader(train_data, batch_size=16, shuffle=True)

test_data = TensorDataset(X_test, y_sent_test)
test_load_sent = DataLoader(test_data, batch_size=16, shuffle=True)

acc_sent = []
epocas = 10
for epoca in range(epocas):
    model_sentimientos.train()

    for X, Y in train_load_sent:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        scores = model_sentimientos(X)
        loss = criterio(scores, Y)
        loss.backward()
        optimizer.step()

    model_sentimientos.eval()

    total_sent = 0
    correct_sent = 0
    total_pred_sent = {classname: 0 for classname in sentimientos}
    correct_pred_sent = {classname: 0 for classname in sentimientos}

    y_pred_sent = []
    y_true_sent = []

    with torch.no_grad():
        for X, Y in test_load_sent:
            X = X.to(device)
            Y = Y.to(device)

            out = model_sentimientos(X)
            _, predicted = torch.max(out, 1)
            y_pred_sent.extend(predicted.cpu().numpy())
            y_true_sent.extend(Y.cpu().numpy())
            
            for label, prediction in zip(Y, predicted):
                if label == prediction:
                    correct_pred_sent[sentimientos[label]] += 1
                total_pred_sent[sentimientos[label]] += 1

            total_sent += Y.size(0) 
            correct_sent += (predicted == Y).sum().item()
        acc_sent.append(100 * (correct_sent / total_sent))

# resultados finales
print("")
print("----- Estadisticas sentimientos -----")
print(f"Exactitud general: {np.mean(acc_sent)}")
print(f"--- Estadisticas por clase ---")
for classname, correct_count in correct_pred_sent.items():
    acc = 100 * float(correct_count) / total_pred_sent[classname]
    print(f"Exactitud {classname}: {acc}")

cm = confusion_matrix(y_true_sent, y_pred_sent)
dataseto.matriz(cm, sentimientos, "Sentimientos")