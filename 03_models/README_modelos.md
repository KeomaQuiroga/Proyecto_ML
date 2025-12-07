# Modelos usados

Este documento resume los archivos quese encuentran en la carpeta **03_models**.

## Archivos 
- **glove.6b.300.txt:** Conjunto de palabras necesarias para aplicar Glove.

- **bayes.py:** Aplicación de Multinomial Naive Bayes, se construye el modelo con y sin el tipo *Neutral* en sentimientos y emociones. Ocupa la tecnica **TD-IDF** como vectorizador, perteneciente a *bag of words*.

- **cnn_emo.py:** Uso de una red neuronal convolucional, específicamente NLP para la detección de emociones. Se prueba y testea el modelo con y sin el tipo *Neutral*.

- **cnn_sent.py:** Uso de una red neuronal convolucional, específicamente NLP para la detección de sentimientos. Se prueba y testea el modelo con y sin el tipo *Neutral*.

- **dataseto.py:** Archivo con funciones necesarias para ambos modelos.
    - **prepo:** Mapea las emociones y sentimientos a números. Necesita al dataset.
    - **prepo_sin_neutral:** Mapea las emociones y sentimientos a números, excepto *Neutral* que se considera eliminado previamente. Necesita al dataset.
    - **matriz:** Muestra la matriz de confusión en gráfico haciendo uso de matplotlib. Necesita la matriz de confusión, el conjunto de clases y el nombre del dataset.
    - **clases:** Muestra la distribución de clases de emociones y sentimientos, llama a la función **balance** para graficar. Necesita al dataset y el nombre del mismo.
    - **balance:** Función que gráfica la distribución de clases. Necesitla columna del dataset, el conjunto de clases y el nombre del dataset.
