import numpy as np
import pytest
from efficient_detection.shared.metrics import detection_counts, metrics_from_counts


def truth(box=(0, 0, 10, 10), category=0, **extra):
    return {"category_id": category, "bbox": list(box), **extra}


@pytest.mark.parametrize(
    "annotations,boxes,labels,expected",
    [
        ([truth()], [[0, 0, 10, 10]], [0], (1, 0, 0, 0)),
        ([truth()], [[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0], (1, 1, 0, 0)),
        ([truth()], [[0, 0, 10, 10]], [1], (0, 1, 1, 0)),
        ([truth()], [[20, 20, 30, 30]], [0], (0, 1, 1, 0)),
        ([truth()], [], [], (0, 0, 1, 0)),
        ([], [[0, 0, 10, 10]], [0], (0, 1, 0, 0)),
        ([], [], [], (0, 0, 0, 0)),
    ],
)
def test_one_to_one_detection_counts(annotations, boxes, labels, expected):
    scores = np.linspace(0.9, 0.6, len(boxes))
    counts = detection_counts(annotations, boxes, scores, labels)
    assert tuple(counts.sum(axis=0)) == expected


def test_duplicate_detection_metrics_have_expected_denominators():
    counts = detection_counts([truth()], [[0, 0, 10, 10]] * 2, [0.9, 0.8], [0, 0])
    metrics = metrics_from_counts(counts.sum(axis=0))
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["detection_accuracy"] == 0.5


def test_match_uses_best_unused_ground_truth():
    counts = detection_counts(
        [truth(), truth((2, 0, 10, 10))],
        [[0, 0, 10, 10]] * 2,
        [0.9, 0.8],
        [0, 0],
    )
    assert tuple(counts.sum(axis=0)) == (2, 0, 0, 0)


def test_matching_iou_is_distinct_from_nms():
    annotations = [truth((2, 0, 10, 10))]
    low = detection_counts(annotations, [[0, 0, 10, 10]], [0.9], [0], match_iou=0.5)
    high = detection_counts(annotations, [[0, 0, 10, 10]], [0.9], [0], match_iou=0.75)
    assert tuple(low.sum(axis=0)) == (1, 0, 0, 0)
    assert tuple(high.sum(axis=0)) == (0, 1, 1, 0)


def test_crowd_matches_use_detection_coverage_and_are_reusable():
    crowd = truth((0, 0, 100, 100), iscrowd=1)
    counts = detection_counts(
        [crowd], [[10, 10, 20, 20], [30, 30, 40, 40]], [0.9, 0.8], [0, 0]
    )
    assert tuple(counts.sum(axis=0)) == (0, 0, 0, 2)
    regular_first = detection_counts(
        [crowd, truth((10, 10, 10, 10))], [[10, 10, 20, 20]], [0.9], [0]
    )
    assert tuple(regular_first.sum(axis=0)) == (1, 0, 0, 0)


def test_regular_ignored_ground_truth_is_matched_once():
    counts = detection_counts(
        [truth(ignore=1)], [[0, 0, 10, 10]] * 2, [0.9, 0.8], [0, 0]
    )
    assert tuple(counts.sum(axis=0)) == (0, 1, 0, 1)


def test_empty_denominators_are_zero():
    metrics = metrics_from_counts([0, 0, 0, 0])
    assert all(
        metrics[name] == 0
        for name in ["precision", "recall", "f1", "detection_accuracy"]
    )
