import os
import re
import pandas as pd
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def parse_yolo_file(filepath):
    bboxes = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                bboxes.append((int(class_id), x_center, y_center, width, height))
    return bboxes

def parse_csv_files(file_list):
    dataframes = [pd.read_csv(filepath) for filepath in file_list]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    Bounding boxes are expected in (x_center, y_center, width, height) format.
    """
    box1_x_min = box1[1] - box1[3] / 2
    box1_y_min = box1[2] - box1[4] / 2
    box1_x_max = box1[1] + box1[3] / 2
    box1_y_max = box1[2] + box1[4] / 2

    box2_x_min = box2[1] - box2[3] / 2
    box2_y_min = box2[2] - box2[4] / 2
    box2_x_max = box2[1] + box2[3] / 2
    box2_y_max = box2[2] + box2[4] / 2

    inter_x_min = max(box1_x_min, box2_x_min)
    inter_y_min = max(box1_y_min, box2_y_min)
    inter_x_max = min(box1_x_max, box2_x_max)
    inter_y_max = min(box1_y_max, box2_y_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (box1_x_max - box1_x_min) * (box1_y_max - box1_y_min)
    box2_area = (box2_x_max - box2_x_min) * (box2_y_max - box2_y_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def compute_marcel_metrics(gt_bboxes, pred_bboxes, iou_threshold=0.5, match_classes=True, min_height=40, image_height=1200):
    """
    Compute precision, recall, and F1 score for predicted and ground truth bounding boxes.
    """
    matched_gt = set()
    matched_pred = set()

    for pred_idx, pred_box in enumerate(pred_bboxes):
        for gt_idx, gt_box in enumerate(gt_bboxes):
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue

            if match_classes and pred_box[0] != gt_box[0]:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                break

    true_positives = len(matched_pred)  # Successfully matched predictions

    unmatched_gt = [gt_bboxes[i] for i in range(len(gt_bboxes)) if i not in matched_gt]
    unmatched_pred = [pred_bboxes[i] for i in range(len(pred_bboxes)) if i not in matched_pred]

    unmatched_gt = [bbox for bbox in unmatched_gt if bbox[4] * image_height >= min_height]
    unmatched_pred = [bbox for bbox in unmatched_pred if bbox[4] * image_height >= min_height]

    false_positives = len(unmatched_pred)
    false_negatives = len(unmatched_gt)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def compute_metrics(gt_bboxes, pred_bboxes, iou_threshold=0.5, match_classes=True, min_height=25, image_height=1200):
    """
    Compute precision, recall, and F1 score for predicted and ground truth bounding boxes.

    Parameters:
        gt_bboxes (list of tuples): Ground truth bounding boxes [(class_id, x_center, y_center, width, height), ...]
        pred_bboxes (list of tuples): Predicted bounding boxes [(class_id, x_center, y_center, width, height), ...]
        iou_threshold (float): IoU threshold to consider a prediction as a match.
        match_classes (bool): If True, only match boxes of the same class.

    Returns:
        tuple: Precision, Recall, F1 score
    """
    gt_bboxes = [bbox for bbox in gt_bboxes if bbox[4] * image_height >= min_height]
    pred_bboxes = [bbox for bbox in pred_bboxes if bbox[4] * image_height >= min_height]

    matched_gt = set()
    matched_pred = set()

    for pred_idx, pred_box in enumerate(pred_bboxes):
        for gt_idx, gt_box in enumerate(gt_bboxes):
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue

            if match_classes and pred_box[0] != gt_box[0]:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                break

    true_positives = len(matched_pred)
    false_positives = len(pred_bboxes) - true_positives
    false_negatives = len(gt_bboxes) - true_positives

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def find_prediction_file(seq_path, frame_id):
    files = sorted(os.listdir(seq_path))
    for i, filename in enumerate(files):
        if not filename.endswith('.csv'):
            continue

        start_frame_id = int(filename.split('_')[0][2:])
        next_start_frame_id = int(files[i + 1].split('_')[0][2:]) if i + 1 < len(files) else float('inf')

        if start_frame_id <= frame_id < next_start_frame_id:
            return os.path.join(seq_path, filename)

    return None

def evaluate_results(gt_dir, results_dir_base, iou_threshold=0.5, target_class_ids=None):
    seq_name = os.path.basename(gt_dir)
    results_dir = os.path.join(results_dir_base, seq_name)

    if not os.path.exists(results_dir):
        print(f"Warnung: Kein Results-Verzeichnis für {seq_name} gefunden.")
        return {}

    required_files = set()
    for gt_file in os.listdir(gt_dir):
        if not gt_file.endswith('.txt'):
            continue

        match = re.match(r"(seq_\d+)_(\d+)_([A-Z0-9_]+).*.txt", gt_file)
        if not match:
            print(f"Ungültiges Dateiformat: {gt_file}")
            continue

        seq, frame_id, camera_name = match.groups()
        frame_id = int(frame_id)

        pred_file = find_prediction_file(results_dir, frame_id)
        if pred_file:
            required_files.add(pred_file)

    csv_data = parse_csv_files(list(required_files))

    metrics_by_threshold = {}
    thresholds = np.arange(0.2, 0.85, 0.05)

    for threshold in thresholds:
        metrics = []
        threshold_start_time = time.time()
        for gt_file in os.listdir(gt_dir):
            if not gt_file.endswith('.txt'):
                continue

            gt_path = os.path.join(gt_dir, gt_file)

            match = re.match(r"(seq_\d+)_(\d+)_([A-Z0-9_]+)_\d+\.txt$", gt_file)
            if not match:
                print(f"Ungültiges Dateiformat: {gt_file}")
                continue

            seq, frame_id, camera_name = match.groups()
            frame_id = int(frame_id)

            pred_bboxes = csv_data[
                (csv_data['frame_id'] == frame_id) &
                (csv_data['camera_name'] == camera_name) &
                (csv_data['score'] >= threshold)
            ][['class_id', 'x_center', 'y_center', 'width', 'height']].values.tolist()

            if target_class_ids is not None:
                pred_bboxes = [bbox for bbox in pred_bboxes if bbox[0] in target_class_ids]

            gt_bboxes = parse_yolo_file(gt_path)

            if target_class_ids is not None:
                gt_bboxes = [bbox for bbox in gt_bboxes if bbox[0] in target_class_ids]

            precision, recall, f1 = compute_marcel_metrics(gt_bboxes, pred_bboxes, iou_threshold, match_classes=False)
            metrics.append((precision, recall, f1))

        if metrics:
            avg_precision = np.mean([m[0] for m in metrics])
            avg_recall = np.mean([m[1] for m in metrics])
            avg_f1 = np.mean([m[2] for m in metrics])
        else:
            avg_precision, avg_recall, avg_f1 = 0.0, 0.0, 0.0

        metrics_by_threshold[threshold] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }

        print(f"Threshold {threshold:.2f} evaluation completed in {time.time() - threshold_start_time:.2f} seconds.")

    return metrics_by_threshold

def plot_metrics(metrics_by_threshold):
    thresholds = list(metrics_by_threshold.keys())
    precisions = [metrics_by_threshold[t]['precision'] for t in thresholds]
    recalls = [metrics_by_threshold[t]['recall'] for t in thresholds]
    f1_scores = [metrics_by_threshold[t]['f1'] for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='x')
    plt.plot(thresholds, f1_scores, label='F1 Score', marker='s')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics by Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    gt_dir = '/mnt/hot_data/share_dhbw/Label/Continuous/seq_8'
    results_dir_base = 'results'
    iou_threshold = 0.5

    metrics_by_threshold = evaluate_results(gt_dir, results_dir_base, iou_threshold, [2, 5, 7])

    print("\nMetriken pro Schwellenwert:")
    for threshold, metrics in metrics_by_threshold.items():
        print(f"Threshold {threshold:.2f}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    plot_metrics(metrics_by_threshold)
