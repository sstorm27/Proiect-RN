import pandas as pd
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br />', ' ', text) # Elimină tag-uri HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Doar litere
    return text

def prepare_data(vocab_size=10000, max_length=200):
    # Încarcă datele (presupunem că ai descărcat și datasetul Kaggle în data/raw)
    # df = pd.read_csv('data/raw/IMDB_Dataset.csv') 
    # df_orig = pd.read_csv('data/generated/original_reviews.csv')
    # df = pd.concat([df, df_orig])

    # Pentru testare rapidă creăm un df dummy dacă nu ai fișierele:
    df = pd.DataFrame({'review': ["Great movie!", "Bad film."], 'sentiment': [1, 0]})

    df['review'] = df['review'].apply(clean_text)
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['review'])
    
    sequences = tokenizer.texts_to_sequences(df['review'])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Salvează tokenizer-ul pentru a-l folosi în UI
    with open('config/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        
    return train_test_split(padded, df['sentiment'].values, test_size=0.2)