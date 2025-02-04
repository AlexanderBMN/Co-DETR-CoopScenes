import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from mmdet.apis import inference_detector, init_detector
import aeifdataset as ad
import torch.multiprocessing as mp
import time

# Relevante Klassen
RELEVANT_CLASSES = {0, 1, 2, 3, 5, 7, 9, 11, 12}


def process_data_on_gpu(dataset_path, start_idx, end_idx, model_config, checkpoint, output_dir, gpu_id, score_thr,
                        total_progress, lock):
    """Verarbeitet einen Teil des Datensatzes auf einer spezifischen GPU."""
    os.makedirs(output_dir, exist_ok=True)  # Sicherstellen, dass der Ausgabeordner existiert

    # Datensatz laden und slicing durchführen
    dataset = ad.Dataloader(dataset_path)
    dataset_part = (data for i, data in enumerate(dataset) if start_idx <= i < end_idx)

    # Modell initialisieren
    model = init_detector(model_config, checkpoint, device=f'cuda:{gpu_id}')

    for datarecord in tqdm(dataset_part, desc=f"GPU {gpu_id} Processing DataRecords", position=gpu_id):
        results_to_save = []

        for frame in datarecord:
            # Iteriere durch alle Agenten (vehicle, tower)
            for agent in frame:
                # Iteriere durch alle Kameras des Agenten
                for camera_name, camera in agent.cameras:
                    rgb_array = np.array(camera)  # Konvertiere Kamera-Bild in ein NumPy-Array

                    # Falls das Bild einen Alpha-Kanal hat, entfernen
                    if rgb_array.shape[-1] == 4:
                        rgb_array = rgb_array[:, :, :3]

                    # Inferenz durchführen
                    result = inference_detector(model, rgb_array)
                    bboxes, _ = result  # Nur Bounding Boxes verwenden

                    # Relevante Ergebnisse extrahieren
                    for class_id, class_bboxes in enumerate(bboxes):
                        if class_id not in RELEVANT_CLASSES:
                            continue

                        for bbox in class_bboxes:
                            x_min, y_min, x_max, y_max, score = bbox
                            if score >= score_thr:
                                # Konvertierung ins YOLO-Format
                                img_width, img_height = rgb_array.shape[1], rgb_array.shape[0]
                                x_center = ((x_min + x_max) / 2) / img_width
                                y_center = ((y_min + y_max) / 2) / img_height
                                width = (x_max - x_min) / img_width
                                height = (y_max - y_min) / img_height

                                results_to_save.append({
                                    "frame_id": frame.frame_id,
                                    "camera_name": camera_name,
                                    "class_id": class_id,
                                    "x_center": x_center,
                                    "y_center": y_center,
                                    "width": width,
                                    "height": height,
                                    "score": score
                                })

        # Ergebnisse pro Datei speichern
        output_file = os.path.join(output_dir, f"{datarecord.name}.csv")
        pd.DataFrame(results_to_save).to_csv(output_file, index=False)
        print(f"Ergebnisse für {datarecord.name} auf GPU {gpu_id} gespeichert unter: {output_file}")

        # Fortschritt aktualisieren
        with lock:
            total_progress.value += 1


def track_progress(total_progress, total_items):
    """Globale Fortschrittsanzeige."""
    with tqdm(total=total_items, desc="Overall Progress", position=0) as pbar:
        last_value = 0
        while last_value < total_items:
            current_value = total_progress.value
            pbar.update(current_value - last_value)
            last_value = current_value
            time.sleep(60)


if __name__ == '__main__':
    # Modellkonfiguration und Checkpoint
    model_config = '../projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_instance.py'
    checkpoint = '../checkpoints/pytorch_model.pth'

    # Pfad zum Hauptordner mit den Datensätzen
    base_dataset_path = '/mnt/hot_data/dataset/aeif/'
    dataset_dirs = [os.path.join(base_dataset_path, d) for d in os.listdir(base_dataset_path) if
                    os.path.isdir(os.path.join(base_dataset_path, d))]

    # Anzahl der GPUs
    num_gpus = 2

    for dataset_path in sorted(dataset_dirs):
        dataset_name = os.path.basename(dataset_path)  # Extrahiere den Ordnernamen (z. B. "seq_1")
        if dataset_name != 'seq_1':
            continue
        output_dir = os.path.join('results',
                                  dataset_name)  # Erstelle einen spezifischen Ergebnisordner für den Datensatz

        print(f"Starte Verarbeitung für Datensatz: {dataset_path}")
        dataset = ad.Dataloader(dataset_path)
        dataset_length = len(dataset)
        chunk_size = dataset_length // num_gpus

        # Startmethodensetzung
        mp.set_start_method('spawn', force=True)

        # Gemeinsamer Fortschrittszähler und Lock
        total_progress = mp.Value('i', 0)
        lock = mp.Lock()

        # Prozesse erstellen
        processes = []
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * chunk_size
            end_idx = dataset_length if gpu_id == num_gpus - 1 else (gpu_id + 1) * chunk_size
            p = mp.Process(target=process_data_on_gpu, args=(
            dataset_path, start_idx, end_idx, model_config, checkpoint, output_dir, gpu_id, 0.20, total_progress, lock))
            processes.append(p)
            p.start()

        # Fortschritt verfolgen
        tracker = mp.Process(target=track_progress, args=(total_progress, dataset_length))
        tracker.start()

        # Auf Prozesse warten
        for p in processes:
            p.join()
        tracker.join()

        print(f"Verarbeitung für Datensatz {dataset_path} abgeschlossen. Ergebnisse gespeichert in {output_dir}\n")
