import os
from collections import Counter

# Verzeichnis mit den gelabelten Dateien
gt_dir = '/mnt/hot_data/share_dhbw/Label/Continuous/seq_8'

# Initialisiere den Zähler für die Klassen
class_counter = Counter()

# Iteriere durch alle Dateien im Verzeichnis
for file_name in os.listdir(gt_dir):
    file_path = os.path.join(gt_dir, file_name)

    # Überprüfen, ob es sich um eine Datei handelt
    if os.path.isfile(file_path) and file_name.endswith('.txt'):
        try:
            # Datei öffnen und Zeilen lesen
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # YOLO-Format: Die erste Zahl jeder Zeile ist die Klassen-ID
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = parts[0]  # Klassen-ID extrahieren
                    class_counter[class_id] += 1
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {file_name}: {e}")

# Ausgabe der absoluten Anzahl der Klassen
print("Absolute Anzahl der Klassen:")
for class_id, count in class_counter.items():
    print(f"Klasse {class_id}: {count}")