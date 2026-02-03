import streamlit as st
import pickle
import numpy as np
import os
import re
import keras
from keras.utils import pad_sequences
from keras.layers import Layer, InputLayer, Embedding, Dense, LSTM, Bidirectional, Dropout
import tensorflow.keras.backend as K

# =========================================================
# 🛠️ COMPATIBILITATE & ATTENTION (Păstrate pentru stabilitate)
# =========================================================
def clean_config(config):
    trash_keys = ['batch_shape', 'time_major', 'quantization_config', 'ragged', 'optional']
    for key in trash_keys:
        if key in config: del config[key]
    return config

class FixedInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config['batch_shape']
            del config['batch_shape']
        return super().from_config(clean_config(config))

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

# =========================================================
# 🕵️ LOGICA HIBRIDĂ (Sliding Window & Prioritizare)
# =========================================================
def heuristic_check(text, ai_score):
    text_clean = text.lower()
    words = text_clean.split()
    
    # Liste de control
    vital_elements = ["story", "plot", "acting", "script", "movie", "film"]
    neg_triggers = ["terrible", "bad", "boring", "awful", "horrible", "worst", "predictable"]
    pos_triggers = ["amazing", "masterpiece", "excellent", "great", "perfect", "brilliant", "incredible", "exquisite", "addicting", "superb"]
    boring_idioms = ["watch paint dry", "watching paint dry", "cure for insomnia", "snoozefest"]

    # 1. STRUCTURI SPECIALE (REPARĂ image_d34a80.png)
    # Punem această verificare prima pentru a prinde opiniile neutre
    if "neither" in text_clean and ("nor" in text_clean or "or" in text_clean):
        return 0.50, "Opinie neutră detectată (Neither/Nor)"

    # 2. SARCASM
    for idiom in boring_idioms:
        if idiom in text_clean: return 0.10, f"Expresie detectată: {idiom}"

    # 3. LOGICA DE PROXIMITATE (Rezolvă ierarhia Story vs Music)
    for i, word in enumerate(words):
        if word in vital_elements:
            start = max(0, i - 3)
            end = min(len(words), i + 4)
            window = words[start:end]
            for neg in neg_triggers:
                if neg in window: return 0.20, f"Elementul '{word}' descris negativ"
            for pos in pos_triggers:
                if pos in window: return 0.95, f"Elementul '{word}' descris ca excepțional"

    # 4. CORECTOR VOCABULAR (Exquisite/Addicting)
    for pos in pos_triggers:
        if pos in words and ai_score < 0.5:
            return 0.88, f"Cuvânt pozitiv puternic detectat: {pos}"

    # 5. NEGAȚII
    if re.search(r"(not|n't)\s+(bad|terrible|awful|horrible)", text_clean):
        return 0.75, "Negație a negativului ('Not bad')"

    return ai_score, ""

# =========================================================
# ✨ INTERFAȚA MODERNĂ STREAMLIT
# =========================================================
st.set_page_config(page_title="Sentiment AI", page_icon="🎭", layout="centered")

# CSS pentru carduri moderne și animații
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTextArea textarea { border-radius: 20px; border: 2px solid #d1d3e2; padding: 15px; }
    .sentiment-card {
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin-top: 20px;
    }
    .sentiment-card:hover { transform: translateY(-5px); }
    .pos { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); }
    .neg { background: linear-gradient(135deg, #dc3545 0%, #f86384 100%); }
    .neu { background: linear-gradient(135deg, #ffc107 0%, #ffdb6e 100%); color: #212529; }
    .stButton button { 
        width: 100%; border-radius: 50px; height: 3.5em; 
        background-color: #4e73df; color: white; font-size: 1.2em; 
        font-weight: bold; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    m_path = 'models/optimized_model.h5' if os.path.exists('models/optimized_model.h5') else 'optimized_model.h5'
    t_path = 'config/tokenizer.pkl' if os.path.exists('config/tokenizer.pkl') else 'tokenizer.pkl'
    try:
        custom = {'Attention': Attention, 'InputLayer': FixedInputLayer, 'Embedding': FixedEmbedding, 
                  'Dense': FixedDense, 'LSTM': FixedLSTM, 'Bidirectional': FixedBidirectional, 'Dropout': FixedDropout}
        model = keras.models.load_model(m_path, custom_objects=custom, compile=False)
        with open(t_path, 'rb') as f: tokenizer = pickle.load(f)
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

res = load_resources()
if res[2]: st.error(f"Eroare: {res[2]}"); st.stop()
model, tokenizer, _ = res

st.title("🎭 Sentiment Analyzer AI")
st.markdown("##### Decodează emoțiile din recenziile de film cu ajutorul Inteligenței Artificiale")

user_input = st.text_area("", placeholder="Scrie aici recenzia ta...", height=150)

if st.button("🔍 Analizează Sentimentul"):
    if user_input.strip():
        # Predicție
        seq = tokenizer.texts_to_sequences([user_input.lower()])
        pad = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        raw_score = float(model.predict(pad, verbose=0)[0][0])
        
        # Logica Hibridă
        final_score, msg = heuristic_check(user_input, raw_score)
        
        st.divider()
        
        # Afișare rezultate cu carduri moderne
        if final_score > 0.55:
            st.markdown('<div class="sentiment-card pos">😊 POZITIV</div>', unsafe_allow_html=True)
        elif final_score < 0.45:
            st.markdown('<div class="sentiment-card neg">😡 NEGATIV</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sentiment-card neu">😐 NEUTRU</div>', unsafe_allow_html=True)
        
        if msg:
            st.info(f"✨ **Insight:** {msg}")
            
        with st.expander("🛠️ Detalii Tehnice"):
            st.write(f"Scor Procesat: {final_score:.4f}")
            st.write(f"Scor Neural Brut: {raw_score:.4f}")
    else:
        st.warning("Te rugăm să introduci o recenzie.")