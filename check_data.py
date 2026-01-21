import pandas as pd
import os

print("--- DIAGNOSTIC DATASET ---")

# 1. Verificam Kaggle
if os.path.exists('data/processed/kaggle_combined.csv'):
    df_k = pd.read_csv('data/processed/kaggle_combined.csv')
    counts = df_k['sentiment'].value_counts()
    print(f"\n[Kaggle] Total: {len(df_k)}")
    print(f"Pozitive (1): {counts.get(1, 0)}")
    print(f"Negative (0): {counts.get(0, 0)}")
    
    if counts.get(0, 0) == 0:
        print("⚠️ ALERTA: Nu s-au gasit recenzii NEGATIVE in datele Kaggle!")
else:
    print("❌ Nu gasesc fisierul Kaggle CSV.")

# 2. Verificam Originale
if os.path.exists('data/generated/original_data.csv'):
    df_o = pd.read_csv('data/generated/original_data.csv')
    counts = df_o['sentiment'].value_counts()
    print(f"\n[Originale] Total: {len(df_o)}")
    print(f"Pozitive (1): {counts.get(1, 0)}")
    print(f"Negative (0): {counts.get(0, 0)}")
else:
    print("❌ Nu gasesc fisierul Original CSV.")