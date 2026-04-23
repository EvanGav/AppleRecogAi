from __future__ import annotations

from typing import List, Dict
import numpy as np

from utils.BoundingBox import BoundingBox


# ── IoU ───────────────────────────────────────────────────────────────────────

def ComputeIoU(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Parameters :
    box_a, box_b : BoundingBox  (pixel coordinates)

    Returns :
    float in [0, 1]
    """
    # Intersection rectangle
    inter_x1 = max(box_a.x_min, box_b.x_min)
    inter_y1 = max(box_a.y_min, box_b.y_min)
    inter_x2 = min(box_a.x_max, box_b.x_max)
    inter_y2 = min(box_a.y_max, box_b.y_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a.x_max - box_a.x_min) * (box_a.y_max - box_a.y_min)
    area_b = (box_b.x_max - box_b.x_min) * (box_b.y_max - box_b.y_min)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ── Average Precision ─────────────────────────────────────────────────────────

def ComputeAveragePrecision(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision (AP) at a given IoU threshold for a single image.

    Parameters :
    predictions   : List[BoundingBox]  — model outputs (must have .score set)
    ground_truths : List[BoundingBox]  — ground-truth boxes for this image
    iou_threshold : float              — minimum IoU to count as a true positive

    Returns :
    float — AP in [0, 1]
    """
    if not ground_truths:
        return 1.0 if not predictions else 0.0
    if not predictions:
        return 0.0

    # Sort predictions by confidence (highest first)
    sorted_preds = sorted(predictions, key=lambda b: b.score, reverse=True)

    matched_gt: set[int] = set()  # track which GT boxes have been matched
    tp_list: List[int] = []
    fp_list: List[int] = []
    # Loop through predictions and match to best IoU GT box
    for pred in sorted_preds:
        best_iou  = 0.0
        best_idx  = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            iou = ComputeIoU(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx

        if best_iou >= iou_threshold and best_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt.add(best_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Cumulative TP / FP
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    recalls    = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Append sentinel values so the curve starts at (0, 1)
    recalls    = np.concatenate([[0.0], recalls,    [recalls[-1]]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Area under P-R curve (trapezoidal rule)
    ap = float(np.trapezoid(precisions, recalls) if hasattr(np, 'trapezoid') else np.trapz(precisions, recalls))
    return ap


def ComputeMeanAveragePrecision(
    per_image_predictions: List[List[BoundingBox]],
    per_image_ground_truths: List[List[BoundingBox]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mAP across all images in a dataset split.

    Parameters :
    per_image_predictions   : list of prediction lists, one per image
    per_image_ground_truths : list of ground-truth lists, one per image
    iou_threshold           : float

    Returns :
    dict with keys 'mAP', 'per_image_AP'
    """
    ap_scores: List[float] = []

    for preds, gts in zip(per_image_predictions, per_image_ground_truths):
        ap = ComputeAveragePrecision(preds, gts, iou_threshold)
        ap_scores.append(ap)

    mean_ap = float(np.mean(ap_scores)) if ap_scores else 0.0
    return {
        "mAP":          mean_ap,
        "per_image_AP": ap_scores,
    }
