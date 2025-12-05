from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import spacy
import contractions
import dataseto

emociones = ["Neutral", "Joy", "Sadness", "Anger", "Surprise", "Fear", "Disgust"]
sentimientos = ["Neutral", "Positve", "Negative"]

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

df = pd.read_csv("01_datasets/MELD/train_sent_emo.csv")
df = df.drop(columns=["Speaker", "Season", "Episode","StartTime", "EndTime"])       # eliminamos las columnas innescesarias
df = dataseto.prepo(df)     # mapeamos las categorias a numeros

# balance de clases
dataseto.clases(df, "MELD")

# A PARTIR DE AQUI ES PRUEBA
# cargamos datos prueba
prueba = pd.read_csv("01_datasets/twitter/Data/conversations_sent&emo.csv")
prueba = prueba.drop(columns=["author_id", "inbound","created_at", "response_tweet_id", "in_response_to_tweet_id"])       # eliminamos las columnas innescesarias
prueba = dataseto.prepo(prueba)
dataseto.clases(prueba, "Twitter")

# preprocesamos el texto para lemanizar y quitar palabras stop
df["prepro_txt"]= df["Utterance"].apply(preprocess)
prueba["prepro_txt"]= prueba["text"].apply(preprocess)

print(df.head(), "\n")
print(prueba.head(), "\n")

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
print(classification_report(y1_test, y_pred, zero_division=1), "\n")
cm = confusion_matrix(y1_test, y_pred)
dataseto.matriz(cm, emociones, "Emociones")

# entrenamos para sentimientos
modelo.fit(X, y2)
y_pred = modelo.predict(X_test)
print(classification_report(y2_test, y_pred, zero_division=1), "\n")
cm = confusion_matrix(y2_test, y_pred)
dataseto.matriz(cm, sentimientos, "Sentimientos")