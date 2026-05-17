import string
import pandas as pd
import re
import html
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
import pickle



import sys
from pathlib import Path

# Asegurar que el directorio raíz está en el PYTHONPATH para importar dsr
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dsr.text_clean import CleanText


def clean_docs(df: pd.DataFrame) -> list:
    """Aplica la limpieza de texto centralizada a una columna de DataFrame."""
    tokenized_docs_clean = []
    for text in df["texto"]:
        tokenized_docs_clean.append(CleanText.clean_text(text))
    return tokenized_docs_clean


# Carga de dataset
df = pd.read_csv("data/raw/dataset_unificado_corregido.csv")

# Division de registros. Separando 10000 registros clases balanceadas y con etiqueta de entrenamiento
pos_train = df[(df["sentimiento"]=="pos") & (df["split"]=="train")].sample(n=5000, random_state=42)
neg_train = df[(df["sentimiento"]=="neg") & (df["split"]=="train")].sample(n=5000, random_state=42)
df_balanced = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42).reset_index(drop=True)
tokenized_docs_train_clean = clean_docs(df_balanced)


# Separar datos de test
pos_test = df[(df["sentimiento"]=="pos") & (df["split"]=="test")].sample(n=2500, random_state=42)
neg_test = df[(df["sentimiento"]=="neg") & (df["split"]=="test")].sample(n=2500, random_state=42)
df_test = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=42).reset_index(drop=True)
tokenized_docs_test_clean = clean_docs(df_test)


# Creacion de y_test y y_train
df_balanced["tag"] = df_balanced["sentimiento"].map({"pos":1, "neg":0})
y_train  = [int(df_balanced["tag"][x]) for x in range(len(df_balanced["texto"]))]

df_test["tag"] = df_test["sentimiento"].map({"pos":1, "neg":0})
y_test = [int(df_test["tag"][x]) for x in range(len(df_test["texto"]))]


# Guardardo de documentos tokenizados y listas de y_train e y_test
with open('data/processed/tokenized_docs_train_clean.pkl', 'wb') as f:
    pickle.dump(tokenized_docs_train_clean, f)
print("Variable tokenized_docs_train_clean guardada correctamente")

with open('data/processed/tokenized_docs_test_clean.pkl', 'wb') as f:
    pickle.dump(tokenized_docs_test_clean, f)
    print("Variable tokenized_docs_test_clean guardada correctamente")

with open('data/processed/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
    print("Variable y_train guardada correctamente")

with open('data/processed/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
    print("Variable y_test guardada correctamente")