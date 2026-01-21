import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle
import os
import sys
import random
import json

# Import model
sys.path.append(os.path.join(os.getcwd(), 'src', 'neural_network'))
try:
    from model import build_model
except ImportError:
    from src.neural_network.model import build_model

# --- CREARE FOLDERE OBLIGATORII ---
os.makedirs('results', exist_ok=True)
os.makedirs('docs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('config', exist_ok=True)

# --- 1. GENERARE DATE (Păstrăm logica ta excelentă) ---
def generate_robust_data():
    print("🧠 Generare Date Hibride (Logic Injection)...")
    reviews, labels = [], []
    
    # A. LOGICĂ COMPLEXĂ (40% din date)
    # Happy End
    for _ in range(4000):
        reviews.append(f"{random.choice(['boring start', 'slow movie'])} but {random.choice(['amazing ending', 'great film'])}")
        labels.append(1.0)
    # Deception
    for _ in range(4000):
        reviews.append(f"{random.choice(['great visuals', 'nice cast'])} but {random.choice(['boring story', 'bad acting'])}")
        labels.append(0.0)
    # Sarcasm (Critic pentru nota 10)
    sarcasm = ["best cure for insomnia", "watch paint dry", "waste of time", "save your money"]
    for _ in range(5000):
        reviews.append(f"this movie is {random.choice(sarcasm)}")
        labels.append(0.0)
    # Opinii Nepopulare
    for _ in range(3000):
        reviews.append(f"even though everyone likes it i found it boring")
        labels.append(0.0)
    # Zona Neutră
    for _ in range(3000):
        reviews.append("it was an average movie nothing special")
        labels.append(0.5)

    # B. VOCABULAR DIVERS (Simulare date reale daca nu gasim fisierul)
    # Daca ai fisierul kaggle, il incarcam jos, aici punem filler
    for _ in range(2000):
        reviews.append("bad movie"); labels.append(0.0)
        reviews.append("good movie"); labels.append(1.0)
        
    return pd.DataFrame({'review': reviews, 'sentiment': labels})

def load_and_combine():
    df_gen = generate_robust_data()
    
    # Încercăm să încărcăm date reale
    real_path = os.path.join('data', 'processed', 'kaggle_combined.csv')
    if os.path.exists(real_path):
        print("📚 Adăugăm date reale Kaggle...")
        df_real = pd.read_csv(real_path).head(25000) # 25k reale
        # Curățare rapidă
        if 'sentiment' in df_real.columns:
            df_real['sentiment'] = df_real['sentiment'].apply(lambda x: 1 if str(x).lower() in ['1', 'positive'] else 0)
        df = pd.concat([df_real, df_gen, df_gen]) # 1x Real, 2x Sintetic (Mixul ideal)
    else:
        print("⚠️ Folosim doar date generate (Nu am găsit Kaggle).")
        df = pd.concat([df_gen, df_gen, df_gen])
        
    return df.sample(frac=1).reset_index(drop=True)

def run_training():
    # 1. Pregătire
    df = load_and_combine()
    df['review'] = df['review'].astype(str)
    print(f"📊 Total Dataset: {len(df)} mostre")

    # 2. Tokenizare (Vocabular crescut pentru acuratețe)
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['review'])
    sequences = tokenizer.texts_to_sequences(df['review'])
    padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
    
    with open('config/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # 3. Split Stratificat
    # Rotunjim etichetele 0.5 la 0 sau 1 doar pentru stratificare, dar antrenăm cu 0.5
    y_strat = df['sentiment'].apply(lambda x: 0 if x < 0.5 else 1)
    X_train, X_test, y_train, y_test = train_test_split(padded, df['sentiment'], test_size=0.15, stratify=y_strat)

    # 4. Antrenare
    model = build_model(vocab_size=20000)
    
    callbacks = [
        ModelCheckpoint('models/optimized_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        CSVLogger('results/training_history.csv') # OBLIGATORIU pentru README
    ]

    print("🚀 Start Antrenare (10+ Epoci)...")
    history = model.fit(
        X_train, y_train,
        epochs=12,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # 5. GENERARE DOVEZI PENTRU README FINAL
    print("\n📈 Generare Grafice și Metrici...")
    
    # A. Evaluare
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    y_true_class = (y_test > 0.5).astype(int) # Binary classification metrics
    
    acc = accuracy_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class, average='macro')
    
    # Salvare JSON Metrici
    metrics = {
        "model": "optimized_model.h5",
        "test_accuracy": round(acc, 4),
        "test_f1_macro": round(f1, 4),
        "vocab_size": 20000,
        "epochs": len(history.history['loss'])
    }
    with open('results/final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # B. Confusion Matrix Plot
    cm = confusion_matrix(y_true_class, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativ', 'Pozitiv'], yticklabels=['Negativ', 'Pozitiv'])
    plt.title('Confusion Matrix - Model Optimizat')
    plt.ylabel('Real')
    plt.xlabel('Predicție')
    plt.savefig('docs/confusion_matrix_optimized.png') # OBLIGATORIU
    print("✅ Confusion Matrix salvată în docs/")
    
    # C. Loss Curve Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Acuratețe')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.savefig('docs/loss_curve.png') # OBLIGATORIU
    print("✅ Curbe de învățare salvate în docs/")

    print(f"\n🏆 REZULTAT FINAL: Acuratețe = {acc*100:.2f}% | F1 = {f1:.4f}")
    
if __name__ == "__main__":
    run_training()