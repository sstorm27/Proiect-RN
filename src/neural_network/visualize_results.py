import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_visualizations():
    base_path = os.getcwd()
    os.makedirs(os.path.join(base_path, 'docs', 'results'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'docs', 'optimization'), exist_ok=True)

    print("   🎨 Generăm graficele...")

    # Grafic 1: Evoluția
    models = ['V1 (Media)', 'V2 (LSTM Simplu)', 'V3 (Bi-LSTM Logic)']
    scores = [0.86, 0.50, 0.94]
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, scores, color=['gray', 'red', 'green'])
    plt.title('Evoluția Performanței pe Cazuri Logice')
    plt.ylim(0, 1.1)
    plt.ylabel('Scor Acuratețe Logică')
    plt.savefig(os.path.join(base_path, 'docs', 'optimization', 'accuracy_comparison.png'))
    plt.close()

    # Grafic 2: Tabel Rezultate
    data = [
        ["The action was boring but overall amazing", "0.93", "POZITIV"],
        ["Great visuals but story was terrible", "0.15", "NEGATIV"],
        ["The movie was not terrible at all", "0.77", "POZITIV"],
        ["The movie was not good", "0.32", "NEGATIV"]
    ]
    df = pd.DataFrame(data, columns=["Text", "Scor", "Rezultat"])
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    plt.title("Rezultate Finale (Edge Cases)", y=1.1)
    plt.savefig(os.path.join(base_path, 'docs', 'results', 'example_predictions.png'))
    plt.close()
    print("   ✅ Grafice salvate în folderul docs/")

if __name__ == "__main__":
    generate_visualizations()
