from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


emociones = ["Neutral", "Joy", "Sadness", "Anger", "Surprise", "Fear", "Disgust"]
sentimientos = ["Neutral", "Positve", "Negative"]

def prepo(dt):
    dt["label_emotion"] = dt.Emotion.map(
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

    dt["label_sentiment"] = dt.Sentiment.map(
        {
            "neutral" : 0,
            "positive" : 1,
            "negative" : 2
        }
    )

    return dt

def prepo_sin_neutral(dt):
    dt["label_emotion"] = dt.Emotion.map(
        {
            "joy" : 0,
            "sadness" : 1,
            "anger" : 2,
            "surprise" : 3,
            "fear" : 4,
            "disgust" : 5
        }
    )

    dt["label_sentiment"] = dt.Sentiment.map(
        {
            "positive" : 0,
            "negative" : 1
        }
    )

    return dt

def balance(clases, labels, titulo):
    count = clases.value_counts().sort_index()
    count.index = [labels[i] for i in count.index]
    count.plot(kind="bar")
    plt.title(titulo)
    plt.xlabel("Clase")
    plt.ylabel("Frecuencia")
    plt.show()

def clases(dt, nombre):
    # balance de clases
    print("Emociones")
    print(dt.label_emotion.value_counts(), "\n")
    balance(dt.label_emotion, emociones, nombre)

    print("Sentimientos")
    print(dt.label_sentiment.value_counts(), "\n")
    balance(dt.label_sentiment, sentimientos, nombre)

def matriz(cm, clase, nombre):
    disp = ConfusionMatrixDisplay(cm, display_labels=clase)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de confusion de {nombre}")
    plt.show()