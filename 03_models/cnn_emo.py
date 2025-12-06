from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
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
        if not token.is_punct and not token.is_space:      # quitamos signos de puntuacion
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
class CNN_emotions(nn.Module):
    def __init__(self):
        super().__init__()
        # Pooling reducido a tamaño 2
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)  # Macrodetalles
        self.pool1 = nn.MaxPool1d(2)  

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)  # Detalles medianos
        self.pool2 = nn.MaxPool1d(2)  

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # Detalles finos
        self.pool3 = nn.MaxPool1d(2)  
        
        self.flat = nn.Flatten()
        self.fc = nn.LazyLinear(out_features=128, bias=True)
        self.fc2 = nn.Linear(128, 7, bias=True)

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
# modelo de emociones
X_train = torch.tensor(np.array(df["glove_txt"].tolist()), dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(np.array(prueba["glove_txt"].tolist()), dtype=torch.float32).unsqueeze(1)

y_emo_train = torch.tensor(df["label_emotion"].tolist())
y_emo_test = torch.tensor(prueba["label_emotion"].tolist())


train_data = TensorDataset(X_train, y_emo_train)
train_load_emo = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TensorDataset(X_test, y_emo_test)
test_load_emo = DataLoader(test_data, batch_size=32, shuffle=True)

# predictor de emociones
criterio = nn.CrossEntropyLoss()
model_emociones = CNN_emotions().to(device)
optimizer = optim.Adam(model_emociones.parameters(), lr=0.001)

acc_emo = []
epocas = 25
for epoca in range(epocas):
    print("")
    model_emociones.train()

    # entrenamiento
    for X, Y in tqdm(train_load_emo, desc=f"Epoca {epoca+1}/{epocas}"):
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        scores = model_emociones(X)
        loss = criterio(scores, Y)
        loss.backward()
        optimizer.step()

    model_emociones.eval()

    # testeo
    total_emo = 0
    correct_emo = 0
    total_pred_emo = {classname: 0 for classname in emociones}
    correct_pred_emo = {classname: 0 for classname in emociones}

    y_pred_emo = []
    y_true_emo = []

    with torch.no_grad():
        for X, Y in test_load_emo:
            X = X.to(device)
            Y = Y.to(device)

            out = model_emociones(X)
            _, predicted = torch.max(out, 1)
            y_pred_emo.extend(predicted.cpu() .numpy())
            y_true_emo.extend(Y.cpu().numpy())
            
            for label, prediction in zip(Y, predicted):
                if label == prediction:
                    correct_pred_emo[emociones[label.item()]] += 1
                total_pred_emo[emociones[label.item()]] += 1

            total_emo += Y.size(0) 
            correct_emo += (predicted == Y).sum().item()
    acc_emo.append(100 * (correct_emo / total_emo))

# resultados finales
print("")
print("----- Estadisticas emociones -----")
print(f"Exactitud general: {np.mean(acc_emo)}")
print(f"--- Estadisticas por clase ---")
for classname, correct_count in correct_pred_emo.items():
    acc = 100 * float(correct_count) / total_pred_emo[classname]
    print(f"Exactitud {classname}: {acc}")

print(classification_report(y_true_emo, y_pred_emo, zero_division=1), "\n")
cm = confusion_matrix(y_true_emo, y_pred_emo)
dataseto.matriz(cm, emociones, "Emociones")