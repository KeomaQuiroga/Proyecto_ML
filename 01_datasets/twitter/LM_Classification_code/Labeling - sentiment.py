import pandas as pd

# Cargar el archivo CSV con las emociones
df = pd.read_csv('01_datasets/twitter/conversations_with_emotions.csv')

# Lista de sentimientos clasificados manualmente para cada fila
# La clasificación es 'positive' o 'negative' según el contexto del tuit.
sentiments = [
    # Conv 0
    'negative', 'positive',
    # Conv 1
    'positive', 'positive',
    # Conv 2
    'negative', 'positive',
    # Conv 3
    'negative', 'positive',
    # Conv 4
    'negative', 'positive',
    # Conv 5
    'negative', 'positive',
    # Conv 6
    'positive', 'positive',
    # Conv 7
    'negative', 'positive',
    # Conv 8
    'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
    # Conv 9
    'positive', 'positive', 'positive', 'positive',
    # Conv 10
    'negative', 'positive', 'negative', 'positive', 'negative', 'positive',
    # Conv 11
    'negative', 'negative', 'positive',
    # Conv 12
    'negative', 'positive',
    # Conv 13
    'negative', 'positive', 'negative',
    # Conv 14
    'negative', 'positive', 'positive', 'positive', 'negative',
    # Conv 15
    'negative', 'positive',
    # Conv 16
    'negative', 'positive', 'negative',
    # Conv 17
    'negative', 'positive',
    # Conv 18
    'negative', 'positive',
    # Conv 19
    'negative', 'positive', 'negative', 'negative',
    # Conv 20
    'negative', 'positive', 'negative', 'positive', 'negative',
    # Conv 21
    'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
    # Conv 22
    'negative', 'positive',
    # Conv 23
    'negative', 'positive',
    # Conv 24
    'negative', 'positive', 'negative', 'positive',
    # Conv 25
    'negative', 'positive',
    # Conv 26
    'negative', 'positive',
    # Conv 27
    'positive', 'positive',
    # Conv 28
    'negative', 'positive',
    # Conv 29
    'negative', 'positive', 'negative', 'positive', 'negative', 'positive',
    # Conv 30
    'negative', 'positive',
    # Conv 31
    'positive', 'positive',
    # Conv 32
    'negative', 'negative', 'positive', 'positive', 'positive',
    # Conv 33
    'positive', 'positive',
    # Conv 34
    'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'positive',
    # Conv 35
    'negative', 'positive', 'positive', 'positive',
    # Conv 36
    'negative', 'positive',
    # Conv 37
    'negative', 'positive',
    # Conv 38
    'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
    # Conv 39
    'negative', 'positive', 'positive',
    # Conv 40
    'negative', 'positive',
    # Conv 41
    'negative', 'positive', 'positive',
    # Conv 42
    'negative', 'positive', 'negative',
    # Conv 43
    'negative', 'positive',
    # Conv 44
    'negative', 'positive',
    # Conv 45
    'negative', 'positive', 'negative',
    # Conv 46
    'negative', 'positive',
    # Conv 47
    'positive', 'positive',
    # Conv 48
    'positive', 'positive', 'positive', 'positive',
    # Conv 49
    'negative', 'positive'
]


# Añadir la nueva columna de sentimiento
df['sentiment'] = sentiments

# Guardar el DataFrame final a un nuevo archivo CSV
output_filename = 'conversations_with_sentiments.csv'
df.to_csv(output_filename, index=False)

print(f"Archivo '{output_filename}' creado exitosamente con las columnas de emoción y sentimiento.")