import os
import pickle
import numpy as np
import pandas as pd
import re
import keras
from keras.utils import pad_sequences
from keras.layers import Layer, InputLayer, Embedding, Dense, LSTM, Bidirectional, Dropout
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- 1. CLASE CUSTOM (Aceleași ca în main.py) ---
def clean_config(config):
    trash = ['batch_shape', 'time_major', 'quantization_config', 'ragged', 'optional']
    for k in trash: 
        if k in config: del config[k]
    return config

class FixedInputLayer(InputLayer):
    @classmethod
    def from_config(cls, c): return super().from_config(clean_config(c))
class FixedEmbedding(Embedding):
    @classmethod
    def from_config(cls, c): return super().from_config(clean_config(c))
class FixedDense(Dense):
    @classmethod
    def from_config(cls, c): return super().from_config(clean_config(c))
class FixedLSTM(LSTM):
    @classmethod
    def from_config(cls, c): return super().from_config(clean_config(c))
class FixedBidirectional(Bidirectional):
    @classmethod
    def from_config(cls, c): 
        if 'layer' in c and isinstance(c['layer'], dict): clean_config(c['layer'])
        return super().from_config(clean_config(c))
class FixedDropout(Dropout):
    @classmethod
    def from_config(cls, c): return super().from_config(clean_config(c))

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# --- 2. LOGICA HIBRIDĂ (Cea care ridică acuratețea) ---
def heuristic_check_logic(text, ai_score):
    text_clean = str(text).lower()
    words = text_clean.split()
    vital_elements = ["story", "plot", "acting", "script", "movie", "film"]
    neg_triggers = ["terrible", "bad", "boring", "awful", "horrible", "worst", "predictable"]
    pos_triggers = ["amazing", "masterpiece", "excellent", "great", "perfect", "brilliant", "incredible", "exquisite", "addicting"]
    boring_idioms = ["watch paint dry", "watching paint dry", "cure for insomnia", "snoozefest"]

    # Reguli prioritare
    if "neither" in text_clean and ("nor" in text_clean or "or" in text_clean): return 0.50
    for idiom in boring_idioms:
        if idiom in text_clean: return 0.10
    
    # Sliding Window
    for i, word in enumerate(words):
        if word in vital_elements:
            start = max(0, i - 3); end = min(len(words), i + 4)
            window = words[start:end]
            for neg in neg_triggers:
                if neg in window: return 0.20
            for pos in pos_triggers:
                if pos in window: return 0.95
    
    # Vocabular lipsă
    for pos in pos_triggers:
        if pos in words and ai_score < 0.5: return 0.88
        
    # Negații
    if re.search(r"(not|n't)\s+(bad|terrible|awful|horrible)", text_clean): return 0.75
    
    return ai_score

# --- 3. GĂSIRE FIȘIER ---
def find_data_file():
    # Lista actualizată conform imaginii tale image_ee1120.png
    possible_paths = [
        'data/processed/kaggle_combined.csv',   # Cel mai probabil setul complet
        'data/generated/original_data.csv',     # Datele generate de tine
        'data/generated/original_reviews.csv',
        'src/data/processed/kaggle_combined.csv',
        r'D:\Proiect RN\data\processed\kaggle_combined.csv'
    ]
    
    print(f"📂 Folder curent de rulare: {os.getcwd()}")
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Fișier de date găsit: {path}")
            return path
            
    print("\n❌ EROARE CRITICĂ: Nu găsesc niciun fișier CSV!")
    print("   Te rog verifică dacă folderul 'data' este în același loc cu scriptul.")
    return None

def run_hybrid_eval():
    print("\n🚀 PORNIRE EVALUARE HIBRIDĂ (AI + REGULI)...\n")
    
    file_path = find_data_file()
    if not file_path: return

    # Citire CSV
    df = pd.read_csv(file_path)
    print(f"📊 Încărcat {len(df)} rânduri.")

    # Detectare automată coloane (review/text și sentiment/label)
    cols = df.columns
    col_text = 'review' if 'review' in cols else ('text' if 'text' in cols else cols[0])
    col_label = 'sentiment' if 'sentiment' in cols else ('label' if 'label' in cols else cols[1])

    # Standardizare Label (convertire 'positive' -> 1)
    if df[col_label].dtype == 'object':
        df[col_label] = df[col_label].apply(lambda x: 1 if str(x).lower() in ['positive', '1', 'pos'] else 0)

    X = df[col_text].astype(str).values
    y_true = df[col_label].values

    # Test Split (20%)
    _, X_test, _, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    print(f"🧪 Testare pe {len(X_test)} exemple...")

    # Încărcare Model & Tokenizer
    t_path = 'config/tokenizer.pkl' if os.path.exists('config/tokenizer.pkl') else 'tokenizer.pkl'
    m_path = 'models/optimized_model.h5' if os.path.exists('models/optimized_model.h5') else 'optimized_model.h5'

    if not os.path.exists(m_path):
        print(f"❌ Nu găsesc modelul: {m_path}")
        return

    with open(t_path, 'rb') as f: tokenizer = pickle.load(f)
    custom = {'Attention': Attention, 'FixedInputLayer': FixedInputLayer, 'FixedEmbedding': FixedEmbedding, 
              'FixedDense': FixedDense, 'FixedLSTM': FixedLSTM, 'FixedBidirectional': FixedBidirectional, 
              'FixedDropout': FixedDropout}
    model = keras.models.load_model(m_path, custom_objects=custom, compile=False)

    # Predicție
    print("... Rulare rețea neuronală ...")
    seq = tokenizer.texts_to_sequences(X_test)
    pad = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    raw_preds = model.predict(pad, batch_size=64, verbose=0).flatten()

    # Corecție Hibridă
    print("... Aplicare corecții logice ...")
    final_preds = []
    for text, raw_score in zip(X_test, raw_preds):
        corrected_score = heuristic_check_logic(text, raw_score)
        final_preds.append(1 if corrected_score > 0.5 else 0)

    # Rezultate
    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds, average='macro')
    
    print("\n" + "="*40)
    print("✅ REZULTATE FINALE SISTEM HIBRID")
    print("="*40)
    print(f"🎯 ACURATEȚE : {acc*100:.2f}%")
    print(f"📊 F1-SCORE  : {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_hybrid_eval()