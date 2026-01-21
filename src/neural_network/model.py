import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, SpatialDropout1D
import sys
import os

# Import dinamic pentru Attention
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from attention import Attention
except ImportError:
    from src.neural_network.attention import Attention

def build_model(vocab_size, embedding_dim=100, max_length=200):
    inputs = Input(shape=(max_length,))
    
    # 1. Embedding + SPATIAL DROPOUT (Truc Pro pentru NLP)
    # Elimină cuvinte întregi, nu doar neuroni, forțând modelul să nu memoreze "cuvinte cheie"
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = SpatialDropout1D(0.3)(x) 
    
    # 2. Bi-LSTM Adânc (2 Straturi)
    # Strat 1: Extrage features complexe
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = BatchNormalization()(x)
    
    # Strat 2: Rafinează contextul pentru Attention
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    
    # 3. ATTENTION LAYER (Creierul)
    x = Attention()(x)
    
    # 4. Clasificator Dens (MLP Head)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Learning rate dinamic (va fi controlat de scheduler)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model