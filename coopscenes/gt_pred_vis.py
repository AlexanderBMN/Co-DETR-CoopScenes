import os
import numpy as np
import re
import random
import cv2
import aeifdataset as ad

def parse_yolo_file(filepath):
    """
    Liest ein YOLO-Format-File und gibt eine Liste von Bounding Boxes zurück.
    :param filepath: Pfad zur Datei
    :return: Liste von Bounding Boxes [(class_id, x_center, y_center, width, height)]
    """
    bboxes = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                bboxes.append((int(class_id), x_center, y_center, width, height))
    return bboxes

def draw_bboxes(image, bboxes, color, label_prefix):
    """
    Zeichnet Bounding Boxes auf ein Bild.
    :param image: Das Bild als numpy-Array
    :param bboxes: Liste von Bounding Boxes im YOLO-Format [(class_id, x_center, y_center, width, height)]
    :param color: Farbe der Bounding Boxes
    :param label_prefix: Präfix für die Labels (z. B. "GT" oder "Pred")
    """
    height, width, _ = image.shape
    for bbox in bboxes:
        class_id, x_center, y_center, box_width, box_height = bbox
        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(
            image,
            f"{label_prefix} {int(class_id)}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

def iou(box1, box2):
    """Berechnet die IoU zweier Bounding Boxes im YOLO-Format."""
    x1_min, x1_max = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    y1_min, y1_max = box1[2] - box1[4] / 2, box1[2] + box1[4] / 2
    x2_min, x2_max = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    y2_min, y2_max = box2[2] - box2[4] / 2, box2[2] + box2[4] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    return inter_area / (box1_area + box2_area - inter_area)

def save_visualized_images(dataset_path, gt_dir, pred_dir, output_dir, threshold=0.35, num_images=5, iou_threshold=0.5):
    """
    Speichert zufällige Bilder mit Ground Truth und Predictions basierend auf dem Dataset-Zugriff.
    :param dataset_path: Pfad zum Dataset.
    :param gt_dir: Verzeichnis mit Ground-Truth-Dateien
    :param pred_dir: Verzeichnis mit Predictions
    :param output_dir: Verzeichnis, in dem die visualisierten Bilder gespeichert werden
    :param threshold: Schwellenwert, der für die Predictions verwendet wurde
    :param num_images: Anzahl der zufällig ausgewählten Bilder
    :param iou_threshold: IoU-Schwellenwert, um ein Match zu bestimmen
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = ad.Dataloader(dataset_path)
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
    sampled_files = random.sample(gt_files, min(num_images, len(gt_files)))

    for gt_file in sampled_files:
        gt_path = os.path.join(gt_dir, gt_file)
        pred_filename = "_".join(gt_file.split("_")[:-1]) + ".txt"
        pred_path = os.path.join(pred_dir, pred_filename)

        if not os.path.exists(pred_path):
            print(f"Warnung: Keine Predictions für {gt_file} gefunden.")
            continue

        # Extrahiere frame_id und camera_name aus dem Dateinamen
        match = re.match(r"seq_\d+_(\d+)_(\w+)_\d+\.txt", gt_file)
        if not match:
            print(f"Warnung: Konnte Informationen aus {gt_file} nicht extrahieren.")
            continue

        frame_id = int(match.group(1))
        cam_name = match.group(2)

        # Suche das entsprechende Bild im Dataset
        image = None
        for datarecord in dataset:
            if image is not None:
                break
            for frame in datarecord:
                if image is not None:
                    break
                if frame.frame_id == frame_id:
                    for agent in frame:
                        for camera_name, camera in agent.cameras:
                            if camera_name == cam_name:
                                image = np.array(camera)


        if image is None:
            print(f"Warnung: Kein Bild für {gt_file} im Dataset gefunden.")
            continue

        if image.shape[-1] == 4:  # Entferne Alpha-Kanal falls vorhanden
            image = image[:, :, :3]

        # Ground Truth und Predictions laden
        gt_bboxes = parse_yolo_file(gt_path)
        pred_bboxes = parse_yolo_file(pred_path)

        # Matches berechnen
        matches = []
        unmatched_gt = []
        unmatched_pred = []

        for gt in gt_bboxes:
            best_match = None
            best_iou = 0
            for pred in pred_bboxes:
                current_iou = iou(gt, pred)
                if current_iou >= iou_threshold and current_iou > best_iou:
                    best_match = pred
                    best_iou = current_iou

            if best_match:
                matches.append((gt, best_match))
                pred_bboxes.remove(best_match)
            else:
                unmatched_gt.append(gt)

        unmatched_pred.extend(pred_bboxes)

        # Bounding Boxes zeichnen
        draw_bboxes(image, [m[0] for m in matches], (255, 255, 0), "Match")  # Gelb für Matches
        draw_bboxes(image, unmatched_gt, (0, 255, 0), "GT")  # Grün für nicht gematchte GT
        draw_bboxes(image, unmatched_pred, (0, 0, 255), "Pred")  # Rot für nicht gematchte Predictions

        # Bild speichern
        output_file = os.path.join(output_dir, f"{os.path.splitext(gt_file)[0]}.jpg")
        cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    dataset_path = '/mnt/hot_data/dataset/aeif/seq_8/'
    gt_dir = '/mnt/hot_data/share_dhbw/Label/Continuous/seq_8_label'
    pred_dir = 'results/threshold_0.35'  # Ordner mit Schwellenwert 0.35
    output_dir = 'test'  # Verzeichnis, in dem die Bilder gespeichert werden

    save_visualized_images(dataset_path, gt_dir, pred_dir, output_dir, threshold=0.35, num_images=15)
