import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np


with open('data/processed/tokenized_docs_test_clean.pkl', 'rb') as f:
    tokenized_docs_test_clean = pickle.load(f)

with open('data/processed/tokenized_docs_train_clean.pkl', 'rb') as f:
    tokenized_docs_train_clean = pickle.load(f)

# Entrenamiento y vectorizacion de documentos usando Doc2Vec
labeled_data = [TaggedDocument(words=tokenized_docs_train_clean[x], tags=[x]) for x in range(len(tokenized_docs_train_clean))]
model = Doc2Vec(vector_size=384, window=5, min_count=2, workers=1, seed=42, epochs=150, dm=1) # Ajustes de entrenamiento de Doc2Vec
model.build_vocab(labeled_data)
model.train(labeled_data, total_examples=model.corpus_count, epochs=model.epochs)
print("Entrenamiento Doc2Vec finalizado correctamente")

# Guardar modelo para no repetir este proceso
model.save("dsr/doc2vec_model")
print("Modelo Doc2Vec guardado en dsr/doc2vec_model")

# Vectorizacion usando el modelo ya entrenado
# infer_vector se aplica a cada documento individualmente
X_test = np.array([model.infer_vector(doc) for doc in tokenized_docs_test_clean])
X_train = np.array([model.dv[i] for i in range(len(tokenized_docs_train_clean))])

print("Variable X_test creada correctamente")

with open('data/processed/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
print("X_train guaradado correctamente")
with open('data/processed/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
print("X_test guardado correctamente")


