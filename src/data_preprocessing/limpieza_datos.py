from IPython.utils.text import string
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
import pickle


# Carga de dataset
df = pd.read_csv("datasets/dataset_unificado_corregido.csv")

# Division de registros. Separando 10000 registros clases balanceadas y con etiqueta de entrenamiento
pos_train = df[(df["sentimiento"]=="pos") & (df["split"]=="train")].sample(n=5000, random_state=42)
neg_train = df[(df["sentimiento"]=="neg") & (df["split"]=="train")].sample(n=5000, random_state=42)
df_balanced = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42).reset_index(drop=True)


lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag): # Devuelve el significado del diccionario de wordnet
    """Convierte el POS tag de NLTK al formato de WordNet."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 



def clean_docs(df: pd.DataFrame):

  # conversion de documentos a minusculas
  raw_docs = [df["texto"][x].lower() for x in range(len(df["texto"]))]
  
  # Eliminar etiquetas HTML (<br/>, etc.), URLs y números
  cleaned_docs = []
  for text in raw_docs:
    text = re.sub(r'<[^>]+>', ' ', text)        
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  
    text = re.sub(r'\d+', ' ', text)            
    cleaned_docs.append(text)
  
  # Tokenizacion
  tokenize_docs = [word_tokenize(doc) for doc in cleaned_docs]
  
  # Remover puntuacion
  regex = re.compile('[%s]' % re.escape(string.punctuation))
  tokenize_docs_no_punctuation = []
  for review in tokenize_docs:
    new_review = []
    for token in review:
      new_token = regex.sub(u'', token)
      if not new_token == u'':
        new_review.append(new_token)
    tokenize_docs_no_punctuation.append(new_review)

  # Remover stopwords
  stop_words = set(stopwords.words('english'))
  tokenized_docs_no_stopwords = []
  for doc in tokenize_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
      if not word in stop_words:
        new_term_vector.append(word)

    tokenized_docs_no_stopwords.append(new_term_vector)
  
  # Lemmatizacion
  tokenized_docs_clean = []
  for doc in tokenized_docs_no_stopwords:
    pos_tags = pos_tag(doc)  # [('running', 'VBG'), ('good', 'JJ'), ...]
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    tokenized_docs_clean.append(lemmatized)

  return tokenized_docs_clean


# Carga de dataset
df = pd.read_csv("datasets/dataset_unificado_corregido.csv")

# Division de registros. Separando 10000 registros clases balanceadas y con etiqueta de entrenamiento
pos_train = df[(df["sentimiento"]=="pos") & (df["split"]=="train")].sample(n=5000, random_state=42)
neg_train = df[(df["sentimiento"]=="neg") & (df["split"]=="train")].sample(n=5000, random_state=42)
df_balanced = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42).reset_index(drop=True)


# Separar datos de test
pos_test = df[(df["sentimiento"]=="pos") & (df["split"]=="test")].sample(n=2500, random_state=42)
neg_test = df[(df["sentimiento"]=="neg") & (df["split"]=="test")].sample(n=2500, random_state=42)
df_test = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=42).reset_index(drop=True)



tokenized_docs_train_clean = clean_docs(df_balanced)
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


