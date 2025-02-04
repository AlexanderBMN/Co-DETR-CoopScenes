import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import aeifdataset as ad
import os

def save_csv_result_image(csv_file, datarecord_file, output_dir, test_frame_id, test_camera_name):
    """Lädt ein Bild aus einem .4mse-File, zeichnet Bounding Boxes und speichert das Ergebnis."""
    # Sicherstellen, dass der Ausgabeordner existiert
    os.makedirs(output_dir, exist_ok=True)

    # CSV-Datei laden
    df = pd.read_csv(csv_file)

    # .4mse-Datei laden
    datarecord = ad.DataRecord(datarecord_file)

    # Frame anhand von frame.frame_id finden
    target_frame = None
    for frame in datarecord:
        if frame.frame_id == test_frame_id:
            target_frame = frame
            break

    if target_frame is None:
        print(f"Frame mit frame_id {test_frame_id} nicht gefunden.")
        return

    # Kamera aus dem Frame laden
    camera = getattr(target_frame.vehicle.cameras, test_camera_name, None)

    if camera is None:
        print(f"Kamera {test_camera_name} nicht gefunden im Frame {test_frame_id}.")
        return

    # Bild als NumPy-Array konvertieren
    image = np.array(camera)

    # Daten für die angegebene Kamera und den Frame filtern
    relevant_data = df[(df['frame_id'] == test_frame_id) & (df['camera_name'] == test_camera_name)]

    if relevant_data.empty:
        print(f"Keine relevanten Daten für frame_id {test_frame_id} und camera_name {test_camera_name} gefunden.")
        return

    # Zeichnen der Bounding Boxes
    for _, row in relevant_data.iterrows():
        x_min, y_min, x_max, y_max = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
        class_id, score = int(row['class_id']), row['score']

        # Zeichne Bounding Box und Label
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)  # Grün für Bounding Box
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

    # Bild speichern
    output_file = os.path.join(output_dir, f"{test_frame_id}_{test_camera_name}.png")
    cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Bild mit Bounding Boxes gespeichert unter: {output_file}")

if __name__ == '__main__':
    # Test-Parameter
    csv_file = '../results/id00021_2024-09-27_10-31-32.csv'  # Beispielpfad zur CSV-Datei
    datarecord_file = '/mnt/hot_data/dataset/aeif/seq_1/id00021_2024-09-27_10-31-32.4mse'  # .4mse-Datei
    output_dir = 'output_images/'  # Ausgabeordner für gespeicherte Bilder
    test_frame_id = 69  # Test-Frame-ID
    test_camera_name = 'STEREO_LEFT'  # Kamera-Name

    # Ergebnis speichern
    save_csv_result_image(csv_file, datarecord_file, output_dir, test_frame_id, test_camera_name)
