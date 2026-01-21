import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import sys

# --- 1. IMPORTAREA CLASEI ATTENTION ---
# Trebuie să îi spunem lui Streamlit unde să găsească 'creierul' nou
# Adăugăm calea către folderul neural_network
sys.path.append(os.path.join(os.getcwd(), 'src', 'neural_network'))

# Încercăm să importăm clasa. Dacă nu o găsește, definim un fallback.
try:
    from attention import Attention
except ImportError:
    # Dacă importul direct eșuează, definim clasa aici (copie de siguranță)
    from tensorflow.keras.layers import Layer
    import tensorflow.keras.backend as K
    
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

# Configurare pagină
st.set_page_config(page_title="Neural Sentiment AI", page_icon="🧠")

@st.cache_resource
def load_resources():
    base_path = os.getcwd()
    model_path = os.path.join(base_path, 'models', 'optimized_model.h5')
    token_path = os.path.join(base_path, 'config', 'tokenizer.pkl')
    
    if not os.path.exists(model_path):
        # Fallback la modelul standard dacă cel optimizat lipsește
        model_path = os.path.join(base_path, 'models', 'trained_model.h5')

    # --- 2. ÎNCĂRCAREA CU CUSTOM OBJECTS ---
    # Aici este cheia! Îi spunem lui Keras: "Când vezi 'Attention', folosește clasa mea."
    model = tf.keras.models.load_model(model_path, custom_objects={'Attention': Attention})
    
    with open(token_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
    return model, tokenizer

# --- LOGICA HIBRIDĂ (PAZNICUL) ---
# Păstrăm regulile simple pentru siguranță maximă
def heuristic_check(text, ai_score):
    text = text.lower()
    
    # Regula 1: Sarcasm evident
    if "cure for insomnia" in text or "fell asleep instantly" in text:
        return 0.10, "Sarcasm detectat (Plictiseală)"
        
    # Regula 2: Opinia Nepopulară explicită (Safety Net)
    # Deși AI-ul știe acum asta, e bine să avem un backup
    if "even though" in text and "not recommend" in text:
        return 0.20, "Structură concesivă negativă"
        
    return ai_score, ""

# --- INTERFAȚA ---
try:
    model, tokenizer = load_resources()
    st.success("✅ Model Bi-LSTM + Attention Încărcat!")
except Exception as e:
    st.error(f"Eroare critică la încărcare: {e}")
    st.stop()

st.title("🧠 Analiză Sentiment (Contextual AI)")
st.write("Acest model folosește mecanismul de **Atenție** pentru a înțelege contextul (ex: 'Start plictisitor, dar final genial').")

user_input = st.text_area("Scrie recenzia:", height=100)

if st.button("Analizează"):
    if not user_input.strip():
        st.warning("Scrie ceva!")
    else:
        # Preprocesare
        seq = tokenizer.texts_to_sequences([user_input])
        pad = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        
        # Predicție Neurală
        raw_score = model.predict(pad)[0][0]
        
        # Ajustare Hibridă (dacă e cazul)
        final_score, msg = heuristic_check(user_input, raw_score)
        
        # Afișare
        st.write("---")
        
        # Interpretare (Praguri ajustate pentru nuanțe)
        if final_score > 0.55:
            st.success(f"😊 POZITIV (Scor: {final_score:.2f})")
        elif final_score < 0.45:
            st.error(f"😡 NEGATIV (Scor: {final_score:.2f})")
        else:
            st.warning(f"😐 NEUTRU / MIXT (Scor: {final_score:.2f})")
            
        with st.expander("Vezi cum a 'gândit' AI-ul"):
            st.metric("Scor Brut", f"{raw_score:.4f}")
            if msg:
                st.info(f"Intervenție Logică: {msg}")
            else:
                st.write("Decizie bazată 100% pe rețeaua neurală și mecanismul de atenție.")