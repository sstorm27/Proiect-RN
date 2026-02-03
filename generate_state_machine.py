import os
import matplotlib.pyplot as plt

# Asigurăm existența folderului
os.makedirs("docs", exist_ok=True)

def draw_pipeline_diagram():
    print("🎨 Generare Diagramă State Machine (Pipeline)...")
    
    # Configurare dimensiuni figură
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Stiluri pentru căsuțe și săgeți
    box_props = dict(boxstyle="round,pad=0.5", fc="#e3f2fd", ec="#1565c0", lw=2)
    arrow_props = dict(facecolor='black', width=0.05, headwidth=0.4, headlength=0.5)

    # Definirea pașilor din codul tău main.py
    steps = [
        (2, 2, "Input Utilizator\n(Streamlit UI)", "Text Brut"),
        (5, 2, "Preprocesare AI\n(clean_text_for_ai)", "Fără 'the', 'is'..."),
        (8, 2, "Predicție Neurală\n(Bi-LSTM Model)", "Scor Brut: 0.0 - 1.0"),
        (11, 2, "Logic Check\n(heuristic_check)", "Reguli Context & 'But'"),
        (11, 0.6, "Rezultat Final\n(Card Vizual)", "Pozitiv/Negativ") # Output jos
    ]

    # Desenare Căsuțe
    for i, (x, y, title, subtitle) in enumerate(steps):
        ax.text(x, y, f"{title}\n\nScan: {subtitle}", ha="center", va="center", 
                size=9, bbox=box_props, fontname='Sans Serif', weight='bold')

    # Desenare Săgeți (Fluxul Datelor)
    # Săgeată 1 -> 2
    ax.annotate("", xy=(3.5, 2), xytext=(2.9, 2), arrowprops=arrow_props)
    # Săgeată 2 -> 3
    ax.annotate("", xy=(6.5, 2), xytext=(5.9, 2), arrowprops=arrow_props)
    # Săgeată 3 -> 4
    ax.annotate("", xy=(9.5, 2), xytext=(8.9, 2), arrowprops=arrow_props)
    # Săgeată 4 -> 5 (Verticală în jos către Output)
    ax.annotate("", xy=(11, 1.1), xytext=(11, 1.5), arrowprops=arrow_props)

    # Titlu
    plt.title("Fluxul de Procesare al Datelor (System Pipeline)", fontsize=14, pad=10)
    
    # Salvare
    output_path = 'docs/state_machine_pipeline.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Diagramă salvată cu succes la: {output_path}")

if __name__ == "__main__":
    draw_pipeline_diagram()