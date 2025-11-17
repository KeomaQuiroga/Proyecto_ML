import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('01_datasets/twitter/customer_support_conversations.csv')

# Lista de emociones corregida y verificada para cada fila del CSV.
# La longitud de esta lista (157) ahora coincide con el número de filas en el DataFrame.
emotions = [
    # Conv 0
    'Anger', 'Neutral',
    # Conv 1
    'Neutral', 'Joy',
    # Conv 2
    'Anger', 'Neutral',
    # Conv 3
    'Anger', 'Neutral',
    # Conv 4
    'Neutral', 'Neutral',
    # Conv 5
    'Sadness', 'Neutral',
    # Conv 6
    'Joy', 'Neutral',
    # Conv 7
    'Anger', 'Neutral',
    # Conv 8
    'Neutral', 'Joy', 'Joy', 'Neutral', 'Neutral', 'Joy',
    # Conv 9
    'Neutral', 'Neutral', 'Neutral', 'Neutral',
    # Conv 10
    'Anger', 'Neutral', 'Fear', 'Neutral', 'Neutral', 'Neutral',
    # Conv 11
    'Anger', 'Anger', 'Neutral',
    # Conv 12
    'Neutral', 'Neutral',
    # Conv 13
    'Fear', 'Neutral', 'Anger',
    # Conv 14
    'Anger', 'Neutral', 'Joy', 'Neutral', 'Disgust',
    # Conv 15
    'Disgust', 'Neutral',
    # Conv 16
    'Anger', 'Neutral', 'Anger',
    # Conv 17
    'Anger', 'Neutral',
    # Conv 18
    'Neutral', 'Neutral',
    # Conv 19
    'Anger', 'Neutral', 'Anger', 'Sadness',
    # Conv 20
    'Surprise', 'Surprise', 'Neutral', 'Joy', 'Anger',
    # Conv 21
    'Neutral', 'Neutral', 'Joy', 'Neutral', 'Joy', 'Neutral', 'Neutral',
    # Conv 22
    'Anger', 'Neutral',
    # Conv 23
    'Disgust', 'Neutral',
    # Conv 24
    'Neutral', 'Neutral', 'Neutral', 'Neutral',
    # Conv 25
    'Neutral', 'Neutral',
    # Conv 26
    'Anger', 'Neutral',
    # Conv 27
    'Joy', 'Neutral',
    # Conv 28
    'Neutral', 'Neutral',
    # Conv 29
    'Anger', 'Neutral', 'Neutral', 'Neutral', 'Anger', 'Neutral',
    # Conv 30
    'Anger', 'Neutral',
    # Conv 31
    'Joy', 'Joy',
    # Conv 32
    'Sadness', 'Sadness', 'Joy', 'Joy', 'Joy',
    # Conv 33
    'Neutral', 'Neutral',
    # Conv 34
    'Neutral', 'Neutral', 'Fear', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Joy',
    # Conv 35
    'Neutral', 'Neutral', 'Neutral', 'Neutral',
    # Conv 36
    'Anger', 'Neutral',
    # Conv 37
    'Neutral', 'Neutral',
    # Conv 38
    'Joy', 'Joy', 'Joy', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral',
    # Conv 39
    'Neutral', 'Neutral', 'Neutral',
    # Conv 40
    'Anger', 'Neutral',
    # Conv 41
    'Anger', 'Neutral', 'Neutral',
    # Conv 42
    'Sadness', 'Neutral', 'Anger',
    # Conv 43
    'Neutral', 'Neutral',
    # Conv 44
    'Anger', 'Neutral',
    # Conv 45
    'Disgust', 'Neutral', 'Anger',
    # Conv 46
    'Anger', 'Neutral',
    # Conv 47
    'Joy', 'Joy',
    # Conv 48
    'Neutral', 'Neutral', 'Joy', 'Joy',
    # Conv 49
    'Anger', 'Neutral'
]


# Añadir la nueva columna al DataFrame
df['emotion'] = emotions

# Guardar el DataFrame actualizado a un nuevo archivo CSV
output_filename = 'conversations_with_emotions.csv'
df.to_csv(output_filename, index=False)

print(f"Archivo '{output_filename}' creado exitosamente con la columna de emociones.")