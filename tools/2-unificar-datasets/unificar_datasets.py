import pandas as pd
import requests
import zipfile
import io
import os
from datasets import load_dataset
import numpy as np

print("--- INICIANDO PROCESO DE UNIFICACIÓN DE DATASETS ---")

# 1. STANFORD SENTIMENT TREEBANK (SST)
print("Descargando y preparando Stanford Sentiment Treebank (SST)...")
url = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
respuesta = requests.get(url)

with zipfile.ZipFile(io.BytesIO(respuesta.content)) as z:
    with z.open('stanfordSentimentTreebank/datasetSentences.txt') as f:
        df_frases = pd.read_csv(f, sep='\t')
    with z.open('stanfordSentimentTreebank/dictionary.txt') as f:
        df_diccionario = pd.read_csv(f, sep='|', names=['frase', 'id_frase'])
    with z.open('stanfordSentimentTreebank/sentiment_labels.txt') as f:
        df_etiquetas = pd.read_csv(f, sep='|')
        df_etiquetas.rename(columns={'phrase ids': 'id_frase', 'sentiment values': 'sentimiento_continuo'}, inplace=True)
    with z.open('stanfordSentimentTreebank/datasetSplit.txt') as f:
        df_splits = pd.read_csv(f, sep=',')
        df_splits['split_nombre'] = df_splits['splitset_label'].map({1: 'train', 2: 'test', 3: 'dev'})

# Ajustar para cruce: hay diferencias de espacios y caracteres especiales
df_frases['frase_limpia'] = df_frases['sentence'].str.replace('-LRB-', '(').str.replace('-RRB-', ')').str.replace(r'\s+', '', regex=True)
df_diccionario['frase_limpia'] = df_diccionario['frase'].str.replace(r'\s+', '', regex=True)

df_sst = pd.merge(df_frases, df_diccionario, on='frase_limpia', how='inner')
df_sst = pd.merge(df_sst, df_etiquetas, on='id_frase', how='inner')
df_sst = pd.merge(df_sst, df_splits, on='sentence_index', how='inner')

# Conversión a binario
registros_antes_stanford = len(df_sst)
df_sst = df_sst[(df_sst['sentimiento_continuo'] >= 0.6) | (df_sst['sentimiento_continuo'] <= 0.4)].copy()
neutrales_descartados = registros_antes_stanford - len(df_sst)
df_sst['sentimiento'] = np.where(df_sst['sentimiento_continuo'] >= 0.6, 'pos', 'neg')

df_final_sst = pd.DataFrame({
    'texto': df_sst['sentence'],
    'sentimiento': df_sst['sentimiento'],
    'dataset': 'stanfordSentimentTreebank',
    'split': df_sst['split_nombre']
})
print(f"SST procesado: {len(df_final_sst)} registros finales.")

# 2. IMDB
print("Descargando y preparando IMDB (aclImdb)...")
dataset_imdb = load_dataset('imdb')
dfs_imdb = []
for split_key in ['train', 'test']:
    df_split = dataset_imdb[split_key].to_pandas()
    # En IMDB de HuggingFace, 0 es negativo y 1 es positivo
    df_split['sentimiento'] = df_split['label'].map({0: 'neg', 1: 'pos'})
    df_split['dataset'] = 'aclImdb'
    df_split['split'] = split_key
    df_split = df_split[['text', 'sentimiento', 'dataset', 'split']].rename(columns={'text': 'texto'})
    dfs_imdb.append(df_split)
df_final_imdb = pd.concat(dfs_imdb, ignore_index=True)
print(f"IMDB procesado: {len(df_final_imdb)} registros finales.")

# 3. ROTTEN TOMATOES
print("Descargando y preparando Rotten Tomatoes (review_polarity)...")
dataset_rt = load_dataset('rotten_tomatoes')
dfs_rt = []
for split_key in ['train', 'validation', 'test']:
    df_split = dataset_rt[split_key].to_pandas()
    # En Rotten Tomatoes, 0 es negativo y 1 es positivo
    df_split['sentimiento'] = df_split['label'].map({0: 'neg', 1: 'pos'})
    df_split['dataset'] = 'review_polarity'
    nombre_split = 'dev' if split_key == 'validation' else split_key
    df_split['split'] = nombre_split
    df_split = df_split[['text', 'sentimiento', 'dataset', 'split']].rename(columns={'text': 'texto'})
    dfs_rt.append(df_split)
df_final_rt = pd.concat(dfs_rt, ignore_index=True)
print(f"Rotten Tomatoes procesado: {len(df_final_rt)} registros finales.")

# 4. UNIFICACIÓN
print("Unificando los tres datasets...")
df_unificado = pd.concat([df_final_sst, df_final_imdb, df_final_rt], ignore_index=True)
df_unificado['texto'] = df_unificado['texto'].astype(str)

registros_originales = len(df_unificado) + neutrales_descartados

# Eliminar vacías
registros_antes_vacias = len(df_unificado)
df_unificado = df_unificado[df_unificado['texto'].str.strip() != '']
vacias_eliminadas = registros_antes_vacias - len(df_unificado)

# Eliminar duplicados
registros_antes_duplicados = len(df_unificado)
df_unificado = df_unificado.drop_duplicates(subset=['texto'])
duplicados_eliminados = registros_antes_duplicados - len(df_unificado)

registros_etiquetados = registros_originales - neutrales_descartados
dataset_final_unificado = len(df_unificado)

print("\n--- RESULTADOS DE LA PREPARACIÓN ---")
print(f"- Registros originales:                   {registros_originales:,}")
print(f"- Registros etiquetados:                  {registros_etiquetados:,}")
print(f"- Reseñas vacías eliminadas:              {vacias_eliminadas:,}")
print(f"- Duplicados eliminados:                  {duplicados_eliminados:,}")
print(f"- Stanford neutrales descartados:         {neutrales_descartados:,}")
print(f"- Dataset final unificado:                {dataset_final_unificado:,}\n")

# Determinar la ruta de salida relativa al propio script para evitar errores según desde donde se ejecute
script_dir = os.path.dirname(os.path.abspath(__file__))
directorio_salida = os.path.normpath(os.path.join(script_dir, '../../datasets'))
os.makedirs(directorio_salida, exist_ok=True)
# Guardar CSV sin análisis
ruta_bruto = os.path.join(directorio_salida, 'dataset_unificado.csv')
df_unificado.to_csv(ruta_bruto, index=False)
print(f"Dataset guardado en: {os.path.abspath(ruta_bruto)}")

# 5. ANÁLISIS
print("Calculando variables derivadas de texto (longitud, vocabulario)...")
df_unificado['longitud'] = df_unificado['texto'].apply(lambda x: len(x.split()))
df_unificado['vocabulario'] = df_unificado['texto'].apply(lambda x: len(set(x.lower().split())))
df_unificado['num_caracteres'] = df_unificado['texto'].apply(len)

# Guardar CSV con análisis
ruta_analisis = os.path.join(directorio_salida, 'dataset_unificado_analisis.csv')
df_unificado.to_csv(ruta_analisis, index=False)
print(f"Dataset con análisis guardado en: {os.path.abspath(ruta_analisis)}")

print("--- PROCESO COMPLETADO ---")