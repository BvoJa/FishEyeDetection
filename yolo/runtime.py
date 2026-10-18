"""PyTorch-free YOLO11 letterboxing, decoding and ONNX CPU execution."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from efficient_detection.shared.common import CLASSES, class_nms, sha256


def letterbox(image, image_size):
    image = image.convert("RGB")
    ratio = min(image_size / image.width, image_size / image.height)
    width, height = round(image.width * ratio), round(image.height * ratio)
    left = round((image_size - width) / 2 - 0.1)
    top = round((image_size - height) / 2 - 0.1)
    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    pixels = np.asarray(canvas, dtype=np.float32) / 255
    geometry = {
        "ratio": ratio,
        "left": left,
        "top": top,
        "width": image.width,
        "height": image.height,
    }
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]), geometry


def decode(
    raw, geometry, confidence=0.25, nms_iou=0.7, max_detections=300, multi_label=False
):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[:2] != (1, 9):
        raise ValueError(
            f"Expected raw five-class YOLO11 output [1,9,N], got {raw.shape}"
        )
    rows = raw[0].T
    rows = rows[np.isfinite(rows).all(axis=1)]
    if multi_label:
        anchors, labels = np.where(rows[:, 4:] >= confidence)
        scores = rows[anchors, labels + 4]
        xywh = rows[anchors, :4]
    else:
        labels = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), labels + 4]
        keep = scores >= confidence
        labels, scores, xywh = labels[keep], scores[keep], rows[keep, :4]
    boxes = np.concatenate(
        [xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1
    )
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry["left"]) / geometry["ratio"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry["top"]) / geometry["ratio"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, geometry["width"])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, geometry["height"])
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    # Bound NMS work in very low-confidence evaluation runs.
    order = np.argsort(-scores, kind="stable")[:30000]
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    keep = class_nms(boxes, scores, labels, nms_iou)[:max_detections]
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep].astype(np.int64),
    }


class YoloSession:
    def __init__(self, model, threads=4, optimization="all", profile=None):
        import onnxruntime as ort

        if threads < 1 or optimization not in {"off", "all"}:
            raise ValueError("Use positive threads and graph optimization off/all")
        self.model = Path(model)
        metadata = self.model.with_suffix(".json")
        if not metadata.is_file():
            raise FileNotFoundError(f"Model metadata is required: {metadata}")
        self.metadata = json.loads(metadata.read_text())
        if self.metadata.get("onnx_sha256") and self.metadata["onnx_sha256"] != sha256(
            self.model
        ):
            raise ValueError(
                "ONNX file does not match its metadata checksum; export/copy the matching pair"
            )
        if (
            self.metadata.get("model_family") != "yolo11"
            or self.metadata.get("classes") != CLASSES
            or self.metadata.get("precision") != "float32"
        ):
            raise ValueError("Expected this workflow's five-class FP32 YOLO11 export")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if optimization == "all"
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        if profile:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile)
        self.session = ort.InferenceSession(
            str(self.model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        size = self.metadata["image_size"]
        if (
            len(inputs) != 1
            or inputs[0].shape != [1, 3, size, size]
            or inputs[0].type != "tensor(float)"
        ):
            raise ValueError("Expected static batch-1 float32 NCHW input")
        self.input_name = inputs[0].name
        self.image_size = size

    def predict(self, image, confidence=0.25, nms_iou=0.7, multi_label=False):
        started = time.perf_counter()
        pixels, geometry = letterbox(image, self.image_size)
        preprocessed = time.perf_counter()
        raw = self.session.run(None, {self.input_name: pixels})[0]
        inferred = time.perf_counter()
        result = decode(raw, geometry, confidence, nms_iou, multi_label=multi_label)
        completed = time.perf_counter()
        result["timing_ms"] = {
            "preprocess": (preprocessed - started) * 1000,
            "inference": (inferred - preprocessed) * 1000,
            "postprocess": (completed - inferred) * 1000,
            "pipeline": (completed - started) * 1000,
        }
        return result
