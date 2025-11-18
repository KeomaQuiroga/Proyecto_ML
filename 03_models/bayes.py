from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
        filtered.append(token.lower_)       # lemanizamos palabras
    
    return " ".join(filtered)

# matriz de confusion
def matrizConfusion(cm, clase, nombre):
    disp = ConfusionMatrixDisplay(cm, display_labels=clase)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de confusion de {nombre}")
    plt.show()

df = pd.read_csv("01_datasets/MELD/train_sent_emo.csv")
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
prueba = pd.read_csv("01_datasets/twitter/Data/conversations_sent&emo.csv")
prueba = prueba.drop(columns=["author_id", "inbound","created_at", "response_tweet_id", "in_response_to_tweet_id"])       # eliminamos las columnas innescesarias
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

# preprocesamos el texto para lemanizar y quitar palabras stop
df["prepro_txt"]= df["Utterance"].apply(preprocess)
prueba["prepro_txt"]= prueba["text"].apply(preprocess)

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

modelo = Pipeline ([
    ("vectorizer", TfidfVectorizer()),      # vectorizamos las palabras
    ("nb", MultinomialNB())     # modelo
])

# entrenamos para emociones
modelo.fit(X, y1)
y_pred = modelo.predict(X_test)
print(classification_report(y1_test, y_pred), "\n")
emo = ["Neutral", "Joy", "Sadness", "Anger", "Surprise", "Fear", "Disgust"]
cm = confusion_matrix(y1_test, y_pred)
matrizConfusion(cm, emo, "Emociones")

# entrenamos para sentimientos
modelo.fit(X, y2)
y_pred = modelo.predict(X_test)
print(classification_report(y2_test, y_pred), "\n")
sent = ["Nuetral", "Positive", "Negative"]
cm = confusion_matrix(y2_test, y_pred)
matrizConfusion(cm, sent, "Sentimientos")