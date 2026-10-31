"""一级 YOLO 靶标检测：使用 OpenCV DNN 运行 Ultralytics 导出的 ONNX 权重。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


class TargetDetector:
    def __init__(self, model: Path, confidence=0.20, nms_iou=0.45, input_size=640):
        if not model.is_file():
            raise FileNotFoundError(model)
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.confidence, self.nms_iou, self.input_size = confidence, nms_iou, input_size

    def infer(self, image):
        height, width = image.shape[:2]
        ratio = min(self.input_size / width, self.input_size / height)
        resized_width, resized_height = int(round(width * ratio)), int(round(height * ratio))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_width, pad_height = self.input_size - resized_width, self.input_size - resized_height
        left, top = int(round(pad_width / 2 - 0.1)), int(round(pad_height / 2 - 0.1))
        padded = cv2.copyMakeBorder(resized, top, pad_height - top, left, pad_width - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self.net.setInput(cv2.dnn.blobFromImage(padded, 1 / 255.0, (self.input_size, self.input_size), swapRB=True))
        prediction = np.squeeze(self.net.forward())
        if prediction.ndim != 2:
            return []
        if prediction.shape[0] <= 256 and prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T
        if prediction.shape[1] < 5:
            return []
        scores = prediction[:, 4:].max(axis=1)
        detections = []
        for row, score in zip(prediction[scores >= self.confidence], scores[scores >= self.confidence]):
            cx, cy, box_width, box_height = map(float, row[:4])
            box = (
                min(max((cx - box_width / 2 - left) / ratio, 0.0), width - 1.0),
                min(max((cy - box_height / 2 - top) / ratio, 0.0), height - 1.0),
                min(max((cx + box_width / 2 - left) / ratio, 0.0), width - 1.0),
                min(max((cy + box_height / 2 - top) / ratio, 0.0), height - 1.0),
            )
            if box[2] > box[0] and box[3] > box[1]:
                detections.append(Detection(box, float(score)))
        remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept = []
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [item for item in remaining if box_iou(best.box, item.box) < self.nms_iou]
        return kept
