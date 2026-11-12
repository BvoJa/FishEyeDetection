"""Standard COCO AP evaluation shared by model selection and full-dataset reports."""

import numpy as np

from efficient_detection.shared.common import CLASSES


def evaluate_predictions(annotation, predictions):
    from faster_coco_eval import COCO, COCOeval_faster

    truth = COCO(str(annotation))
    if predictions:
        predicted = truth.loadRes(predictions)
    else:
        predicted = COCO()
        predicted.dataset = {
            "images": truth.dataset["images"],
            "categories": truth.dataset["categories"],
            "annotations": [],
        }
        predicted.createIndex()
    evaluator = COCOeval_faster(truth, predicted, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = dict(
        zip(
            ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"],
            map(float, evaluator.stats[:6]),
        )
    )
    precision = evaluator.eval["precision"]
    iou50 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0])
    per_class = []
    for index, category in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        values50 = precision[iou50, :, index, 0, -1]
        per_class.append(
            {
                "category_id": int(category),
                "class": CLASSES[category],
                "AP": float(values[values >= 0].mean())
                if np.any(values >= 0)
                else None,
                "AP50": float(values50[values50 >= 0].mean())
                if np.any(values50 >= 0)
                else None,
            }
        )
    return metrics, per_class
