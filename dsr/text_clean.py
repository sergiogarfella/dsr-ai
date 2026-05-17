import string
import html
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk

class CleanText:
    __EMOJI_PATTERN = re.compile(
        "["
        u"\U0001F600-\U0001F64F"   # emoticons
        u"\U0001F300-\U0001F5FF"   # símbolos y pictogramas
        u"\U0001F680-\U0001F9FF"   # transport, misc symbols
        u"\U00002600-\U000027BF"   # símbolos varios
        "]+",
        flags=re.UNICODE
    )


    def __get_wordnet_pos(treebank_tag): # Devuelve el significado del diccionario de wordnet
        """Convierte el POS tag de NLTK al formato de WordNet."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN 

    @staticmethod
    def clean_text(text: str) -> list:
        lemmatizer = WordNetLemmatizer()
        
        # Eliminacion caracteres especiales. urls, htmls, emojis, etc.
        text = html.unescape(text) 
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = CleanText.__EMOJI_PATTERN.sub(' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Cambiar texto a todo minusculas
        text = text.lower()
    
        # Eliminar numeros
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()  # limpiar espacios generados
    
        # Tokenización -
        tokens = word_tokenize(text)
    
        # Eliminar puntuación
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [regex.sub('', token) for token in tokens]
    
        # Filtrar tokens vacíos y tokens de 1-2 caracteres
        tokens = [t for t in tokens if len(t) > 2]
    
        # Eliminar stopwords (conservando palabras clave para sentimiento)
        stop_words = set(stopwords.words('english'))
        keep = {"no", "not", "nor", "never", "neither", "very", "too", "but", "however", "although"}
        stop_words -= keep
        tokens = [t for t in tokens if t not in stop_words]
    
        #Lematización con POS tagging
        pos_tags = pos_tag(tokens)
        tokens = [
            lemmatizer.lemmatize(word, CleanText.__get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]
    
        return tokens

    

if __name__ == '__main__':
    frase = "- Emojis 🎉🔥💀👀 y HTML: <b>negrita</b> &amp; &lt;p&gt;párrafo&lt;/p&gt;"
    texto_limpio = CleanText.clean_text(frase)
    print(texto_limpio)
