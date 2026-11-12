"""Class-aware, one-to-one object detection metrics at a fixed IoU threshold."""

import numpy as np

from efficient_detection.shared.common import CLASSES, box_iou


def detection_counts(annotations, boxes, scores, labels, match_iou=0.5):
    """Return per-class TP/FP/FN/ignored counts, processing predictions by score.

    Each prediction matches the highest-IoU unused normal ground truth of the same
    class. Unmatched predictions covered by ignored/crowd annotations are ignored;
    crowd uses intersection / detection area and permits repeated matches.
    """
    if not 0 < match_iou <= 1:
        raise ValueError("Matching IoU must be in (0,1]")
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if not len(boxes) == len(scores) == len(labels):
        raise ValueError("Box, score and label counts must match")
    counts = np.zeros((len(CLASSES), 4), dtype=np.int64)
    for category in range(len(CLASSES)):
        truth = [ann for ann in annotations if ann["category_id"] == category]
        truth_boxes = np.asarray(
            [ann["bbox"] for ann in truth], dtype=np.float32
        ).reshape(-1, 4)
        truth_boxes[:, 2:] += truth_boxes[:, :2]
        crowd = np.asarray([bool(ann.get("iscrowd", 0)) for ann in truth], dtype=bool)
        ignored = np.asarray(
            [bool(ann.get("ignore", 0) or ann.get("iscrowd", 0)) for ann in truth],
            dtype=bool,
        )
        used = np.zeros(len(truth), dtype=bool)
        indices = np.flatnonzero(labels == category)
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        for index in indices:
            available = np.flatnonzero(~ignored & ~used)
            overlaps = box_iou(boxes[index], truth_boxes[available])
            if len(overlaps) and overlaps.max() >= match_iou:
                used[available[overlaps.argmax()]] = True
                counts[category, 0] += 1
                continue
            candidates = np.flatnonzero(ignored & (~used | crowd))
            overlaps = box_iou(boxes[index], truth_boxes[candidates])
            if len(candidates):
                intersection = np.maximum(
                    0,
                    np.minimum(boxes[index, 2:], truth_boxes[candidates, 2:])
                    - np.maximum(boxes[index, :2], truth_boxes[candidates, :2]),
                ).prod(axis=1)
                area = np.maximum(0, boxes[index, 2:] - boxes[index, :2]).prod()
                overlaps = np.where(
                    crowd[candidates], intersection / max(float(area), 1e-8), overlaps
                )
            if len(overlaps) and overlaps.max() >= match_iou:
                used[candidates[overlaps.argmax()]] = True
                counts[category, 3] += 1
            else:
                counts[category, 1] += 1
        counts[category, 2] = np.count_nonzero(~ignored & ~used)
    return counts


def metrics_from_counts(counts):
    """Compute fixed-threshold metrics; empty denominators have value zero.

    detection_accuracy is an explicitly defined Jaccard-style detection score,
    TP/(TP+FP+FN), rather than classification accuracy (true negatives undefined).
    """
    tp, fp, fn, ignored = map(int, counts)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_detections": ignored,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "detection_accuracy": tp / max(tp + fp + fn, 1),
    }
