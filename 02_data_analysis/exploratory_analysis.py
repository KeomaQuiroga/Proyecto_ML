"""
Análisis Exploratorio de Datos (EDA)
=====================================
Este script realiza un análisis estadístico comparativo entre los datasets MELD y Twitter
para clasificación de emociones y sentimientos.

Análisis implementados:
1. Distribución de clases en ambos datasets
2. Propiedades textuales (longitud de diálogos/tokens)
3. Análisis léxico y de vocabulario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import contractions

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Carga del modelo de spaCy para procesamiento de texto
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """
    Preprocesa el texto aplicando:
    - Expansión de contracciones
    - Tokenización
    - Eliminación de puntuación
    - Conversión a minúsculas
    """
    text = contractions.fix(text, slang=False)
    doc = nlp(text)
    filtered = []
    
    for token in doc:
        if token.is_punct:
            continue
        filtered.append(token.lower_)
    
    return " ".join(filtered)


def cargar_datasets():
    """
    Carga los datasets MELD y Twitter con el preprocesamiento necesario
    """
    print("=" * 80)
    print("CARGANDO DATASETS")
    print("=" * 80)
    
    # Cargar MELD (entrenamiento)
    meld = pd.read_csv("01_datasets/MELD/train_sent_emo.csv")
    meld = meld.drop(columns=["Speaker", "Season", "Episode", "StartTime", "EndTime"])
    
    # Mapeo de emociones MELD
    meld["label_emotion"] = meld.Emotion.map({
        "neutral": 0, "joy": 1, "sadness": 2, "anger": 3,
        "surprise": 4, "fear": 5, "disgust": 6
    })
    
    # Mapeo de sentimientos MELD
    meld["label_sentiment"] = meld.Sentiment.map({
        "neutral": 0, "positive": 1, "negative": 2
    })
    
    # Cargar Twitter (testing)
    twitter = pd.read_csv("01_datasets/twitter/Data/conversations_sent&emo.csv")
    twitter = twitter.drop(columns=["author_id", "inbound", "created_at", 
                                     "response_tweet_id", "in_response_to_tweet_id"])
    
    # Mapeo de emociones Twitter
    twitter["label_emotion"] = twitter.Emotion.map({
        "neutral": 0, "joy": 1, "sadness": 2, "anger": 3,
        "surprise": 4, "fear": 5, "disgust": 6
    })
    
    # Mapeo de sentimientos Twitter
    twitter["label_sentiment"] = twitter.Sentiment.map({
        "neutral": 0, "positive": 1, "negative": 2
    })
    
    # Preprocesar texto
    print("\nPreprocesando texto MELD...")
    meld["prepro_txt"] = meld["Utterance"].apply(preprocess)
    
    print("Preprocesando texto Twitter...")
    twitter["prepro_txt"] = twitter["text"].apply(preprocess)
    
    # Calcular longitud en tokens
    meld["num_tokens"] = meld["prepro_txt"].apply(lambda x: len(x.split()))
    twitter["num_tokens"] = twitter["prepro_txt"].apply(lambda x: len(x.split()))
    
    print(f"\nMELD shape: {meld.shape}")
    print(f"Twitter shape: {twitter.shape}")
    
    return meld, twitter


def analisis_distribucion_clases(meld, twitter):
    """
    Análisis 1: Distribución de clases
    Compara la frecuencia de cada categoría en ambos datasets mediante gráficos de barras
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS 1: DISTRIBUCIÓN DE CLASES")
    print("=" * 80)
    
    # Mapeos inversos para etiquetas
    emotion_labels = {0: "Neutral", 1: "Joy", 2: "Sadness", 3: "Anger", 
                     4: "Surprise", 5: "Fear", 6: "Disgust"}
    sentiment_labels = {0: "Neutral", 1: "Positive", 2: "Negative"}
    
    # === EMOCIONES ===
    print("\n--- DISTRIBUCIÓN DE EMOCIONES ---")
    
    # Conteos
    meld_emo_counts = meld.label_emotion.value_counts().sort_index()
    twitter_emo_counts = twitter.label_emotion.value_counts().sort_index()
    
    # Porcentajes
    meld_emo_pct = (meld_emo_counts / len(meld) * 100).round(2)
    twitter_emo_pct = (twitter_emo_counts / len(twitter) * 100).round(2)
    
    print("\nMELD - Distribución de emociones:")
    for idx, count in meld_emo_counts.items():
        print(f"  {emotion_labels[idx]:10s}: {count:5d} ({meld_emo_pct[idx]:5.2f}%)")
    
    print("\nTwitter - Distribución de emociones:")
    for idx, count in twitter_emo_counts.items():
        print(f"  {emotion_labels[idx]:10s}: {count:5d} ({twitter_emo_pct[idx]:5.2f}%)")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MELD
    axes[0].bar(range(len(meld_emo_counts)), meld_emo_counts.values, 
                color='skyblue', edgecolor='black')
    axes[0].set_xticks(range(len(meld_emo_counts)))
    axes[0].set_xticklabels([emotion_labels[i] for i in meld_emo_counts.index], 
                            rotation=45, ha='right')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Emociones - MELD (Entrenamiento)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Twitter
    axes[1].bar(range(len(twitter_emo_counts)), twitter_emo_counts.values, 
                color='lightcoral', edgecolor='black')
    axes[1].set_xticks(range(len(twitter_emo_counts)))
    axes[1].set_xticklabels([emotion_labels[i] for i in twitter_emo_counts.index], 
                            rotation=45, ha='right')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Emociones - Twitter (Testing)')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_data_analysis/distribucion_emociones.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # === SENTIMIENTOS ===
    print("\n--- DISTRIBUCIÓN DE SENTIMIENTOS ---")
    
    # Conteos
    meld_sent_counts = meld.label_sentiment.value_counts().sort_index()
    twitter_sent_counts = twitter.label_sentiment.value_counts().sort_index()
    
    # Porcentajes
    meld_sent_pct = (meld_sent_counts / len(meld) * 100).round(2)
    twitter_sent_pct = (twitter_sent_counts / len(twitter) * 100).round(2)
    
    print("\nMELD - Distribución de sentimientos:")
    for idx, count in meld_sent_counts.items():
        print(f"  {sentiment_labels[idx]:10s}: {count:5d} ({meld_sent_pct[idx]:5.2f}%)")
    
    print("\nTwitter - Distribución de sentimientos:")
    for idx, count in twitter_sent_counts.items():
        print(f"  {sentiment_labels[idx]:10s}: {count:5d} ({twitter_sent_pct[idx]:5.2f}%)")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MELD
    axes[0].bar(range(len(meld_sent_counts)), meld_sent_counts.values, 
                color='lightgreen', edgecolor='black')
    axes[0].set_xticks(range(len(meld_sent_counts)))
    axes[0].set_xticklabels([sentiment_labels[i] for i in meld_sent_counts.index], 
                            rotation=45, ha='right')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Sentimientos - MELD (Entrenamiento)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Twitter
    axes[1].bar(range(len(twitter_sent_counts)), twitter_sent_counts.values, 
                color='plum', edgecolor='black')
    axes[1].set_xticks(range(len(twitter_sent_counts)))
    axes[1].set_xticklabels([sentiment_labels[i] for i in twitter_sent_counts.index], 
                            rotation=45, ha='right')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Sentimientos - Twitter (Testing)')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_data_analysis/distribucion_sentimientos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Análisis de sesgo
    print("\n--- ANÁLISIS DE SESGO ---")
    print("\nNota: Se recomienda utilizar F1-ponderado como métrica de evaluación")
    print("debido al desbalance de clases en ambos datasets.")


def analisis_propiedades_textuales(meld, twitter):
    """
    Análisis 2: Propiedades textuales
    Analiza las longitudes de los diálogos (número de tokens) en cada dataset
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS 2: PROPIEDADES TEXTUALES")
    print("=" * 80)
    
    # Estadísticas descriptivas
    print("\n--- ESTADÍSTICAS DE LONGITUD (NÚMERO DE TOKENS) ---")
    
    meld_stats = meld.num_tokens.describe()
    twitter_stats = twitter.num_tokens.describe()
    
    print("\nMELD:")
    print(f"  Media:             {meld_stats['mean']:.2f} tokens")
    print(f"  Desviación estándar: {meld_stats['std']:.2f} tokens")
    print(f"  Mínimo:            {meld_stats['min']:.0f} tokens")
    print(f"  Cuartil 25%:       {meld_stats['25%']:.0f} tokens")
    print(f"  Mediana:           {meld_stats['50%']:.0f} tokens")
    print(f"  Cuartil 75%:       {meld_stats['75%']:.0f} tokens")
    print(f"  Máximo:            {meld_stats['max']:.0f} tokens")
    
    print("\nTwitter:")
    print(f"  Media:             {twitter_stats['mean']:.2f} tokens")
    print(f"  Desviación estándar: {twitter_stats['std']:.2f} tokens")
    print(f"  Mínimo:            {twitter_stats['min']:.0f} tokens")
    print(f"  Cuartil 25%:       {twitter_stats['25%']:.0f} tokens")
    print(f"  Mediana:           {twitter_stats['50%']:.0f} tokens")
    print(f"  Cuartil 75%:       {twitter_stats['75%']:.0f} tokens")
    print(f"  Máximo:            {twitter_stats['max']:.0f} tokens")
    
    # Recomendación de padding
    max_length = max(meld_stats['max'], twitter_stats['max'])
    recommended_padding = int(meld_stats['75%'] + twitter_stats['75%']) // 2
    
    print(f"\n--- RECOMENDACIONES PARA CNN ---")
    print(f"  Longitud máxima observada: {max_length:.0f} tokens")
    print(f"  Padding recomendado (basado en Q3): {recommended_padding} tokens")
    print(f"  Nota: El padding cubre aproximadamente el 75% de ambos datasets")
    
    # Visualización - Histogramas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MELD - Histograma
    axes[0, 0].hist(meld.num_tokens, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(meld_stats['mean'], color='red', linestyle='--', 
                       linewidth=2, label=f'Media: {meld_stats["mean"]:.1f}')
    axes[0, 0].axvline(meld_stats['50%'], color='green', linestyle='--', 
                       linewidth=2, label=f'Mediana: {meld_stats["50%"]:.1f}')
    axes[0, 0].set_xlabel('Número de Tokens')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Longitud - MELD')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Twitter - Histograma
    axes[0, 1].hist(twitter.num_tokens, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(twitter_stats['mean'], color='red', linestyle='--', 
                       linewidth=2, label=f'Media: {twitter_stats["mean"]:.1f}')
    axes[0, 1].axvline(twitter_stats['50%'], color='green', linestyle='--', 
                       linewidth=2, label=f'Mediana: {twitter_stats["50%"]:.1f}')
    axes[0, 1].set_xlabel('Número de Tokens')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Longitud - Twitter')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Box plots comparativos
    data_to_plot = [meld.num_tokens, twitter.num_tokens]
    axes[1, 0].boxplot(data_to_plot, labels=['MELD', 'Twitter'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue'),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('Número de Tokens')
    axes[1, 0].set_title('Comparación de Longitudes (Box Plot)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Violin plot
    positions = [1, 2]
    parts = axes[1, 1].violinplot([meld.num_tokens, twitter.num_tokens], 
                                   positions=positions, showmeans=True, showmedians=True)
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(['MELD', 'Twitter'])
    axes[1, 1].set_ylabel('Número de Tokens')
    axes[1, 1].set_title('Comparación de Longitudes (Violin Plot)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_data_analysis/propiedades_textuales.png', dpi=300, bbox_inches='tight')
    plt.show()


def analisis_lexico_vocabulario(meld, twitter):
    """
    Análisis 3: Análisis léxico y de vocabulario
    Evalúa la coincidencia o divergencia entre vocabularios
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS 3: ANÁLISIS LÉXICO Y DE VOCABULARIO")
    print("=" * 80)
    
    # Crear vocabularios
    meld_words = []
    for text in meld.prepro_txt:
        meld_words.extend(text.split())
    
    twitter_words = []
    for text in twitter.prepro_txt:
        twitter_words.extend(text.split())
    
    # Conjuntos de palabras únicas
    meld_vocab = set(meld_words)
    twitter_vocab = set(twitter_words)
    
    # Tamaño de corpus
    print("\n--- TAMAÑO DE CORPUS ---")
    print(f"  MELD:")
    print(f"    Total de palabras (tokens): {len(meld_words):,}")
    print(f"    Vocabulario (palabras únicas): {len(meld_vocab):,}")
    print(f"\n  Twitter:")
    print(f"    Total de palabras (tokens): {len(twitter_words):,}")
    print(f"    Vocabulario (palabras únicas): {len(twitter_vocab):,}")
    
    # Solapamiento de vocabulario
    overlap = meld_vocab.intersection(twitter_vocab)
    twitter_only = twitter_vocab - meld_vocab
    meld_only = meld_vocab - twitter_vocab
    
    print("\n--- SOLAPAMIENTO DE VOCABULARIO ---")
    print(f"  Palabras en común: {len(overlap):,}")
    print(f"  Palabras solo en MELD: {len(meld_only):,}")
    print(f"  Palabras solo en Twitter: {len(twitter_only):,}")
    
    # Porcentaje de cobertura
    coverage_pct = (len(overlap) / len(twitter_vocab)) * 100
    print(f"\n  Porcentaje de vocabulario de Twitter presente en MELD: {coverage_pct:.2f}%")
    
    print("\n--- IMPLICACIONES ---")
    if coverage_pct < 50:
        print("  ⚠ ADVERTENCIA: Baja cobertura de vocabulario (<50%)")
        print("  - Naive Bayes puede tener dificultades con palabras no vistas")
        print("  - CNN puede no identificar patrones relevantes en Twitter")
    elif coverage_pct < 75:
        print("  ⚠ ATENCIÓN: Cobertura moderada de vocabulario (50-75%)")
        print("  - Se esperan limitaciones en la generalización")
        print("  - Considerar técnicas de transfer learning")
    else:
        print("  ✓ Buena cobertura de vocabulario (>75%)")
        print("  - Los modelos deberían generalizar adecuadamente")
    
    # Palabras más frecuentes (excluyendo stop words)
    print("\n--- 30 PALABRAS MÁS FRECUENTES (sin stop words) ---")
    
    # Filtrar stop words
    meld_words_filtered = [word for word in meld_words if word.lower() not in STOP_WORDS]
    twitter_words_filtered = [word for word in twitter_words if word.lower() not in STOP_WORDS]
    
    meld_counter = Counter(meld_words_filtered)
    twitter_counter = Counter(twitter_words_filtered)
    
    meld_top30 = meld_counter.most_common(30)
    twitter_top30 = twitter_counter.most_common(30)
    
    print("\nMELD (Top 30):")
    for i, (word, count) in enumerate(meld_top30, 1):
        print(f"  {i:2d}. {word:15s} ({count:6,} ocurrencias)")
    
    print("\nTwitter (Top 30):")
    for i, (word, count) in enumerate(twitter_top30, 1):
        print(f"  {i:2d}. {word:15s} ({count:6,} ocurrencias)")
    
    # Visualización - Gráfico de Venn conceptual con barras
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Solapamiento de vocabulario
    categories = ['Solo MELD', 'En Común', 'Solo Twitter']
    values = [len(meld_only), len(overlap), len(twitter_only)]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    axes[0, 0].bar(categories, values, color=colors, edgecolor='black')
    axes[0, 0].set_ylabel('Número de Palabras Únicas')
    axes[0, 0].set_title('Solapamiento de Vocabulario entre Datasets')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        axes[0, 0].text(i, v + max(values)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Top 15 palabras MELD (sin stop words)
    top15_meld = meld_counter.most_common(15)
    words_meld = [w[0] for w in top15_meld]
    counts_meld = [w[1] for w in top15_meld]
    
    axes[0, 1].barh(range(len(words_meld)), counts_meld, color='skyblue', edgecolor='black')
    axes[0, 1].set_yticks(range(len(words_meld)))
    axes[0, 1].set_yticklabels(words_meld)
    axes[0, 1].set_xlabel('Frecuencia')
    axes[0, 1].set_title('Top 15 Palabras Más Frecuentes - MELD')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Top 15 palabras Twitter (sin stop words)
    top15_twitter = twitter_counter.most_common(15)
    words_twitter = [w[0] for w in top15_twitter]
    counts_twitter = [w[1] for w in top15_twitter]
    
    axes[1, 0].barh(range(len(words_twitter)), counts_twitter, color='lightcoral', edgecolor='black')
    axes[1, 0].set_yticks(range(len(words_twitter)))
    axes[1, 0].set_yticklabels(words_twitter)
    axes[1, 0].set_xlabel('Frecuencia')
    axes[1, 0].set_title('Top 15 Palabras Más Frecuentes - Twitter')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Comparación de tamaños de corpus
    corpus_sizes = [len(meld_words), len(twitter_words)]
    vocab_sizes = [len(meld_vocab), len(twitter_vocab)]
    
    x = np.arange(2)
    width = 0.35
    
    axes[1, 1].bar(x - width/2, corpus_sizes, width, label='Total Tokens', 
                   color='steelblue', edgecolor='black')
    axes[1, 1].bar(x + width/2, vocab_sizes, width, label='Vocabulario Único', 
                   color='orange', edgecolor='black')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['MELD', 'Twitter'])
    axes[1, 1].set_ylabel('Número de Palabras')
    axes[1, 1].set_title('Comparación de Tamaños de Corpus')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, (corpus, vocab) in enumerate(zip(corpus_sizes, vocab_sizes)):
        axes[1, 1].text(i - width/2, corpus + max(corpus_sizes)*0.01, 
                       f'{corpus:,}', ha='center', va='bottom', fontsize=9)
        axes[1, 1].text(i + width/2, vocab + max(corpus_sizes)*0.01, 
                       f'{vocab:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('02_data_analysis/analisis_vocabulario.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Guardar palabras únicas solo en Twitter (útil para análisis posterior)
    print(f"\nGuardando {len(twitter_only)} palabras únicas de Twitter...")
    with open('02_data_analysis/twitter_unique_words.txt', 'w', encoding='utf-8') as f:
        f.write("Palabras presentes en Twitter pero NO en MELD:\n")
        f.write("=" * 50 + "\n\n")
        for word in sorted(twitter_only):
            f.write(f"{word}\n")
    print("  Archivo guardado: 02_data_analysis/twitter_unique_words.txt")


def main():
    """
    Función principal que ejecuta todos los análisis
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("Comparación MELD vs Twitter para Clasificación de Emociones")
    print("=" * 80)
    
    # Cargar datasets
    meld, twitter = cargar_datasets()
    
    # Ejecutar análisis
    analisis_distribucion_clases(meld, twitter)
    analisis_propiedades_textuales(meld, twitter)
    analisis_lexico_vocabulario(meld, twitter)
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  - 02_data_analysis/distribucion_emociones.png")
    print("  - 02_data_analysis/distribucion_sentimientos.png")
    print("  - 02_data_analysis/propiedades_textuales.png")
    print("  - 02_data_analysis/analisis_vocabulario.png")
    print("  - 02_data_analysis/twitter_unique_words.txt")
    print("\n")


if __name__ == "__main__":
    main()
