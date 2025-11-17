import pandas as pd
import random
import os

def extract_conversations(csv_path, num_conversations_to_sample):
    """
    Extrae una muestra aleatoria de conversaciones del conjunto de datos de soporte al cliente de Twitter.

    Args:
        csv_path (str): La ruta al archivo CSV de entrada (por ejemplo, 'twcs.csv').
        num_conversations_to_sample (int): El número de conversaciones a muestrear.

    Returns:
        str: La ruta al archivo CSV de salida.
    """
    # Cargar el conjunto de datos
    df = pd.read_csv(csv_path)

    # Convertir created_at a objetos datetime
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')

    # Crear un diccionario para acceder fácilmente a los tuits por su ID
    tweets_dict = {tweet['tweet_id']: tweet for index, tweet in df.iterrows()}

    # Obtener todos los primeros tuits (tuits que no son respuestas)
    first_tweets = df[df['in_response_to_tweet_id'].isnull()]

    # Crear una lista de todos los iniciadores de conversación (tweet_ids de los primeros tuits)
    all_conversation_starters = first_tweets['tweet_id'].tolist()

    # Muestrear aleatoriamente los iniciadores de conversación
    sampled_starters = random.sample(all_conversation_starters, num_conversations_to_sample)

    # Extraer conversaciones completas para los iniciadores muestreados
    all_conversations = []
    for conv_id, start_tweet_id in enumerate(sampled_starters):
        conversation = []
        
        # Usar una cola para la búsqueda en anchura para obtener todas las respuestas en una conversación
        queue = [start_tweet_id]
        
        processed_tweets = set()

        while queue:
            current_tweet_id = queue.pop(0)
            
            if current_tweet_id in tweets_dict and current_tweet_id not in processed_tweets:
                processed_tweets.add(current_tweet_id)
                tweet = tweets_dict[current_tweet_id]
                tweet_data = tweet.to_dict()
                tweet_data['conversation_id'] = conv_id
                conversation.append(tweet_data)

                # Encontrar respuestas al tuit actual
                response_ids = tweet['response_tweet_id']
                if isinstance(response_ids, str):
                    response_ids = [int(i) for i in response_ids.split(',')]
                    queue.extend(response_ids)
        
        all_conversations.extend(conversation)

    # Crear un nuevo DataFrame con las conversaciones
    conversations_df = pd.DataFrame(all_conversations)

    # Ordenar el DataFrame por conversation_id y luego por created_at
    conversations_df = conversations_df.sort_values(by=['conversation_id', 'created_at'])

    # Guardar las conversaciones en un nuevo archivo CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'customer_support_conversations.csv')
    conversations_df.to_csv(output_path, index=False)
    
    return output_path

# --- Instrucciones para el usuario ---
# 1. Descarga el conjunto de datos desde la URL: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
# 2. Descomprime el archivo y encuentra 'twcs.csv'.
# 3. Coloca 'twcs.csv' en el mismo directorio que este script: c:\Users\NW\Documents\!Personales\_git\Proyecto_ML\01_datasets\twitter\
# 4. Establece el número de conversaciones que deseas muestrear en la variable a continuación.

# Configuración
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, 'twcs.csv')
num_conversations = 50  # Cambia esto al número deseado de conversaciones

# Ejemplo de uso:
output_file = extract_conversations(csv_file_path, num_conversations)
print(f"Se extrajeron con éxito {num_conversations} conversaciones en '{output_file}'")