import os
import pandas as pd
import random

# --- PARTEA 1: Procesare Date Kaggle (Raw) ---
def process_kaggle_files():
    # Calea corectă către date
    base_dir = os.path.join('data', 'raw')
    data = []
    
    print(f"📂 Verific datele în: {os.path.abspath(base_dir)}")
    
    # Verificare existență folder
    if not os.path.exists(os.path.join(base_dir, 'train')):
        print("\n❌ EROARE STRUCTURĂ FOLDERE!")
        print(f"   Nu găsesc folderul 'train' în '{base_dir}'.")
        return

    print("⏳ Citesc fișierele text (durează puțin)...")
    
    # Citim datele reale descărcate
    files_found = 0
    for split in ['train', 'test']:
        for label_type in ['pos', 'neg']:
            path = os.path.join(base_dir, split, label_type)
            sentiment = 1 if label_type == 'pos' else 0
            
            if os.path.exists(path):
                for filename in os.listdir(path):
                    if filename.endswith('.txt'):
                        try:
                            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                                data.append([f.read(), sentiment])
                                files_found += 1
                        except:
                            pass 
    
    if files_found == 0:
        print("❌ Nu am găsit niciun fișier .txt.")
        return

    # Salvare
    df = pd.DataFrame(data, columns=['review', 'sentiment'])
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    df.to_csv(os.path.join('data', 'processed', 'kaggle_combined.csv'), index=False)
    print(f"✅ Date Kaggle procesate: {len(df)} recenzii găsite.")

# --- PARTEA 2: Generare Date Originale (Simulare) ---
def generate_original_data():
    print("⏳ Generez datele originale...")
    
    reviews = []
    labels = []
    
    pos_adj = ["amazing", "incredible", "great", "fantastic", "superb"]
    neg_adj = ["terrible", "boring", "bad", "awful", "horrible"]
    
    # Generăm 10,000 recenzii
    for i in range(35000):
        if i % 2 == 0:
            text = f"This movie was {random.choice(pos_adj)} and I really enjoyed it. {i}"
            labels.append(1)
        else:
            text = f"I hated this movie, it was {random.choice(neg_adj)} and a waste of time. {i}"
            labels.append(0)
        
        # --- LINIA CARE LIPSEA (FIX) ---
        reviews.append(text)
            
    df = pd.DataFrame({'review': reviews, 'sentiment': labels})
    os.makedirs(os.path.join('data', 'generated'), exist_ok=True)
    df.to_csv(os.path.join('data', 'generated', 'original_data.csv'), index=False)
    print(f"✅ Date Originale generate: {len(df)} recenzii.")

if __name__ == "__main__":
    process_kaggle_files()
    generate_original_data()