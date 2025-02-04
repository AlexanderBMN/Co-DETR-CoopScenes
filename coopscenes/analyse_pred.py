# Erneute Definition der Funktion collect_class_statistics und der Visualisierungslogik

import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def collect_class_statistics(results_dir, score_threshold=0.35):
    """
    Iteriert über alle Unterordner von results_dir, liest alle CSV-Dateien,
    und zählt die absoluten Häufigkeiten der Klassen.

    :param results_dir: Verzeichnis mit den Prediction-Unterordnern
    :param score_threshold: Minimaler Score, um eine Prediction zu berücksichtigen
    :return: Dictionary mit Klassen-IDs als Keys und deren Häufigkeiten als Values
    """
    class_counter = Counter()

    # Iteriere über alle Unterordner
    for root, _, files in os.walk(results_dir):
        for file in files:
            if not file.endswith('.csv'):
                continue

            file_path = os.path.join(root, file)

            # Lade die CSV-Datei
            df = pd.read_csv(file_path)

            # Filtere nur die relevanten Zeilen basierend auf dem Score
            filtered_df = df[df['score'] > score_threshold]

            # Zähle die Häufigkeiten der Klassen
            class_counter.update(filtered_df['class_id'].astype(int).tolist())

    return dict(class_counter)

def plot_class_statistics_with_labels(class_statistics):
    """
    Plottet ein logarithmisches Balkendiagramm basierend auf der Klassenstatistik,
    wobei jede Klasse eine andere Farbe hat und die absoluten Zahlen über den Balken angezeigt werden.

    :param class_statistics: Dictionary mit Klassen-IDs und deren Häufigkeiten.
    """
    # COCO-Notation für Klassen
    coco_classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
        11: "stop sign"
    }

    # Filtern der Klassenstatistik basierend auf COCO-Notation
    filtered_statistics = {coco_classes[k]: v for k, v in class_statistics.items() if k in coco_classes}

    # Sortieren nach Klassen-ID
    sorted_classes = sorted(filtered_statistics.keys(), key=lambda x: list(coco_classes.values()).index(x))
    sorted_counts = [filtered_statistics[cls] for cls in sorted_classes]

    # Erstellen des logarithmischen Balkendiagramms
    plt.figure(figsize=(14, 10))
    colors = plt.cm.Paired(range(len(sorted_classes)))  # Verwende unterschiedliche Farben aus einer Farbpalette
    bars = plt.bar(sorted_classes, sorted_counts, log=True, color=colors, edgecolor='black', width=0.65)

    # Zahlen über den Balken anzeigen
    for bar, count in zip(bars, sorted_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=18  # Größere Schriftgröße für die Zahlen
        )

    # Achsen und Titel mit größerer Schriftgröße
    plt.yscale('log')
    plt.ylim(bottom=1000)
    plt.xlabel("Classes", fontsize=20)
    plt.ylabel("Appearances", fontsize=20)
    plt.xticks(fontsize=18)  # Größere Schriftgröße für die x-Achsen-Ticks
    plt.yticks(fontsize=18)  # Größere Schriftgröße für die y-Achsen-Ticks
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.show()


# Berechnen der Klassenstatistik
results_dir = "results"  # Passe den Pfad an, falls notwendig
score_threshold = 0.35

class_statistics = collect_class_statistics(results_dir, score_threshold)

# Visualisieren der Statistik mit Labels und unterschiedlichen Farben
plot_class_statistics_with_labels(class_statistics)

print(f"Gesamtanzahl der Objekte: {sum(class_statistics.values())}")