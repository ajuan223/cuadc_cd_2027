"""一级 YOLO 靶标检测：仅依赖 OpenCV DNN 和 NumPy 运行 ONNX 权重。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float


def letterbox(image: np.ndarray, size: int):
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    new_width, new_height = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_width, pad_height = size - new_width, size - new_height
    left, top = int(round(pad_width / 2 - 0.1)), int(round(pad_height / 2 - 0.1))
    right, bottom = pad_width - left, pad_height - top
    output = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return output, ratio, (float(left), float(top))


def iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


class TargetDetector:
    def __init__(
        self, model: Path, confidence: float = 0.25, nms_iou: float = 0.45,
        input_size: int = 640,
    ) -> None:
        if not model.is_file():
            raise FileNotFoundError(model)
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.input_size = input_size

    def infer(self, image: np.ndarray) -> list[Detection]:
        padded, ratio, (pad_x, pad_y) = letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 255.0, (self.input_size, self.input_size), swapRB=True
        )
        self.net.setInput(blob)
        prediction = np.squeeze(self.net.forward())
        if prediction.ndim != 2:
            return []
        if prediction.shape[0] <= 256 and prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T
        if prediction.shape[1] < 5:
            return []
        class_scores = prediction[:, 4:]
        scores = class_scores.max(axis=1)
        frame_height, frame_width = image.shape[:2]
        detections = []
        for row, score in zip(prediction[scores >= self.confidence], scores[scores >= self.confidence]):
            cx, cy, width, height = map(float, row[:4])
            box = (
                min(max((cx - width / 2 - pad_x) / ratio, 0.0), frame_width - 1.0),
                min(max((cy - height / 2 - pad_y) / ratio, 0.0), frame_height - 1.0),
                min(max((cx + width / 2 - pad_x) / ratio, 0.0), frame_width - 1.0),
                min(max((cy + height / 2 - pad_y) / ratio, 0.0), frame_height - 1.0),
            )
            if box[2] > box[0] and box[3] > box[1]:
                detections.append(Detection(box, float(score)))
        remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept = []
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [item for item in remaining if iou(best.box, item.box) < self.nms_iou]
        return kept


def padded_crop(frame: np.ndarray, box, padding: float = 0.15) -> np.ndarray:
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(frame.shape[1], int(x2 + pad_x))
    bottom = min(frame.shape[0], int(y2 + pad_y))
    return frame[top:bottom, left:right].copy()
