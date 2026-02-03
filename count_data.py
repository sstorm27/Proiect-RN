import pandas as pd
import os

# Căile către fișierele tale (bazat pe structura din poze)
# Ajustează numele dacă diferă puțin
path_original = 'data/generated/original_data.csv'   # Datele tale "pure" (contribuția ta)
path_kaggle = 'data/processed/kaggle_combined.csv'   # Dataset-ul public (IMDB/Kaggle)
path_augmented = 'data/generated/augmented_reviews.csv' # Dacă ai date generate separat

def count_rows(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return len(df)
        except:
            return 0
    return 0

# 1. Numărăm Datele Tale Originale (Generare + Etichetare proprie)
count_orig = count_rows(path_original)
# Dacă ai și augmented separat, adaugă-le aici:
count_aug = count_rows(path_augmented)

total_orig = count_orig + count_aug

# 2. Numărăm Datele Publice (Kaggle/IMDB)
count_public = count_rows(path_kaggle)

# 3. Total
total_final = total_orig + count_public

print("="*40)
print("📊 STATISTICI PENTRU README")
print("="*40)
print(f"1. Date Originale (Contribuție): {total_orig}")
print(f"2. Date Publice (Kaggle):      {count_public}")
print("-" * 40)
print(f"TOTAL OBSERVAȚII (N):          {total_final}")

if total_final > 0:
    percent = (total_orig / total_final) * 100
    print(f"Procent Contribuție Originală: {percent:.2f}%")
    
    if percent >= 40:
        print("✅ Status: OK (Peste 40%)")
    else:
        print("⚠️ Status: ATENȚIE (Sub 40% - mai generează date!)")
print("="*40)