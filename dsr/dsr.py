from .text_clean import CleanText
from deep_translator import GoogleTranslator
from gensim.models.doc2vec import Doc2Vec
import pickle
import os

_DIR = os.path.dirname(os.path.abspath(__file__))

class Dsr:
    def __init__(self):
        # Cargar modelos en la instancia (no al importar la clase)
        self.__doc2vec_model = Doc2Vec.load(os.path.join(_DIR, "doc2vec_model"))
        
        with open(os.path.join(_DIR, "knn_model.pkl"), "rb") as f:
            self.__knn_model = pickle.load(f)
        
        with open(os.path.join(_DIR, "lr_model.pkl"), "rb") as f:
            self.__lr_model = pickle.load(f)
        
        self.__translator = GoogleTranslator(source="auto", target="en")
    
    
    def __model_prediction(self, vectorized_text, modelo):
        if modelo.lower() == "knn":
            return self.__knn_model.predict([vectorized_text]), self.__knn_model.predict_proba([vectorized_text]) 
        else:
            return self.__lr_model.predict([vectorized_text]), self.__lr_model.predict_proba([vectorized_text])
    
    
    def __clean_text(self, text:str) -> list: # Limpieza de texto
        return CleanText.clean_text(text)
    
    
    def __vectorize(self, text:str) -> list: # Vectorizacion
        return self.__doc2vec_model.infer_vector(text)

    
    def predict(self, text:str, modelo="knn"):
        # Traducir texto 
        translated_text = self.__translator.translate(text)
        # Limpiar texto
        cleaned_text = self.__clean_text(translated_text)
        # Vectorizar
        vectorized_text = self.__vectorize(cleaned_text)
        # Prediccion
        prediction = self.__model_prediction(vectorized_text, modelo)
        return prediction                                            
        

if __name__ == '__main__':
    # Codigo de Prueba
    texto = "Pues a decir verdad esta pelicula es una pesima idea y tonta no me gusta"
    print(Dsr().predict(texto))
