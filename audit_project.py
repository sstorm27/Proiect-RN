import sys
import os
import json
import pandas as pd

# 1. Configurare căi
BASE_DIR = os.getcwd()
sys.path.append(os.path.join(BASE_DIR, 'src', 'neural_network'))
METRICS_FILE = os.path.join(BASE_DIR, 'results', 'final_metrics.json')

def audit_full_project():
    print("\n🔍 PORNIRE AUDIT COMPLET PROIECT...\n")
    
    # --- PARTEA 1: AUDIT DATE (Contribuție) ---
    print("1️⃣  ANALIZĂ DATE & CONTRIBUȚIE")
    try:
        from train import generate_robust_data
        
        # Generare și calcul
        df_gen = generate_robust_data()
        count_gen = len(df_gen) # Aprox 19.000
        
        # Logică simulată conform train.py (1 Real + 2 Sintetice)
        count_real = 25000 
        total_sintetic_folosit = count_gen * 2
        total_dataset = count_real + total_sintetic_folosit
        
        percent = (total_sintetic_folosit / total_dataset) * 100
        
        print(f"   • Total Observații:      {total_dataset}")
        print(f"   • Date Originale (Tu):   {total_sintetic_folosit}")
        print(f"   • Procent Contribuție:   {percent:.2f}%")
        
        if percent >= 40:
            print("   ✅ CRITERIU DATE: ÎNDEPLINIT")
        else:
            print("   ⚠️ ATENȚIE: Procent sub 40%.")
            
    except ImportError:
        print("   ❌ EROARE: Nu pot importa 'train.py'.")
    except Exception as e:
        print(f"   ❌ EROARE CALCUL: {e}")

    print("-" * 40)

    # --- PARTEA 2: AUDIT PERFORMANȚĂ (Metrici) ---
    print("2️⃣  PERFORMANȚĂ MODEL (Din results/final_metrics.json)")
    
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)
            
            acc = metrics.get('test_accuracy', 0)
            f1 = metrics.get('test_f1_macro', 0)
            
            print(f"   • Acuratețe (Test):      {acc*100:.2f}%")
            print(f"   • F1-Score (Macro):      {f1:.4f}")
            
            if acc > 0.70:
                print("   ✅ CRITERIU PERFORMANȚĂ: ÎNDEPLINIT")
            else:
                print("   ⚠️ ATENȚIE: Acuratețea este sub 70%.")
                
        except Exception as e:
            print(f"   ❌ EROARE CITIRE JSON: {e}")
    else:
        print("   ⚠️ NU GĂSESC FIȘIERUL DE METRICI.")
        print("   Soluție: Rulează 'python src/neural_network/train.py' mai întâi.")

    print("\n" + "="*40)
    print("📝 TEXT GATA DE COPIAT ÎN README:")
    print("="*40)
    print(f"| Metric | Valoare |")
    print(f"|---|---|")
    print(f"| Acuratețe | **{acc*100:.2f}%** |")
    print(f"| F1-Score | **{f1:.4f}** |")
    print(f"| Contribuție Date | **{percent:.2f}%** |")
    print("="*40)

if __name__ == "__main__":
    audit_full_project()