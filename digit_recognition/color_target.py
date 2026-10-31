"""数字任务的 HSV 红蓝候选、五边形几何评分和背景误检过滤。"""

from __future__ import annotations

import cv2
import numpy as np

from detector import Detection, iou


def mask_convex_hull(mask):
    """Use all component boundaries; interior pixels cannot change the hull."""
    if cv2.countNonZero(mask) < 20:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    points = np.concatenate(contours).reshape(-1, 2)
    # 保持旧版行优先点序以稳定多边形起点选择。
    points = points[np.lexsort((points[:, 0], points[:, 1]))]
    return cv2.convexHull(points)


def pentagon_shape_score(image, box):
    """检查候选区域是否满足比赛靶标的等边直角五边形几何特征。"""
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(image_w, int(np.ceil(x2)))
    bottom = min(image_h, int(np.ceil(y2)))
    crop = image[top:bottom, left:right]
    if min(crop.shape[:2]) < 12:
        return False, 0.0, {"reason": "small"}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array((0, 70, 40)), np.array((12, 255, 255)))
    red |= cv2.inRange(hsv, np.array((165, 70, 40)), np.array((179, 255, 255)))
    blue = cv2.inRange(hsv, np.array((90, 70, 40)), np.array((135, 255, 255)))
    colour = cv2.bitwise_or(red, blue)
    colour = cv2.morphologyEx(
        colour, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    hull = mask_convex_hull(colour)
    if hull is None:
        return False, 0.0, {"reason": "no_colour"}
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 1.0:
        return False, 0.0, {"reason": "short_perimeter"}
    best = None
    for epsilon in (0.018, 0.020, 0.022, 0.025, 0.028, 0.030, 0.035, 0.040):
        polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(polygon) != 5:
            continue
        edges = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1)
        if float(edges.max()) <= 1.0:
            continue
        side_ratio = float(edges.min() / edges.max())
        angles = []
        for index in range(5):
            first = polygon[index - 1] - polygon[index]
            second = polygon[(index + 1) % 5] - polygon[index]
            cosine = np.dot(first, second) / (
                np.linalg.norm(first) * np.linalg.norm(second) + 1e-9
            )
            angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        ordered = sorted(angles)
        angle_ok = (
            35 <= ordered[0] <= 105
            and 55 <= ordered[1] <= 125
            and 65 <= ordered[2] <= 135
            and 110 <= ordered[3] <= 175
            and 115 <= ordered[4] <= 178
        )
        # 将边长一致性作为质量分并保留五顶点和角度拓扑硬约束。
        if side_ratio < 0.16 or not angle_ok:
            continue
        side_quality = min(1.0, side_ratio / 0.75)
        right_error = min(abs(angle - 90.0) for angle in angles)
        right_quality = max(0.0, 1.0 - right_error / 35.0)
        quality = 0.75 * side_quality + 0.25 * right_quality
        candidate = (quality, side_ratio, angles, epsilon)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return False, 0.0, {"reason": "not_equal_sided_pentagon"}
    quality, side_ratio, angles, epsilon = best
    return True, float(quality), {
        "reason": "ok", "vertices": 5, "side_ratio": float(side_ratio),
        "angles_deg": [round(value, 1) for value in angles],
        "epsilon": float(epsilon),
    }


def _touches(a, b, margin: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + margin < bx1 or bx2 + margin < ax1
        or ay2 + margin < by1 or by2 + margin < ay1
    )


def _merge_components(components, margin: int):
    groups = []
    for box, area in components:
        matching = [
            index for index, (other, _) in enumerate(groups)
            if _touches(box, other, margin)
        ]
        if not matching:
            groups.append((box, area))
            continue
        x1, y1, x2, y2 = box
        total_area = area
        for index in reversed(matching):
            other, other_area = groups.pop(index)
            x1, y1 = min(x1, other[0]), min(y1, other[1])
            x2, y2 = max(x2, other[2]), max(y2, other[3])
            total_area += other_area
        groups.append(((x1, y1, x2, y2), total_area))
    return groups


class ColorTargetDetector:
    """Find compact red/blue target bodies and reject long colour edges."""

    def __init__(
        self, min_box_area_ratio=0.0002, max_box_area_ratio=0.45,
        min_aspect=0.35, max_aspect=2.8, expansion=0.04,
    ):
        self.min_box_area_ratio = min_box_area_ratio
        self.max_box_area_ratio = max_box_area_ratio
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.expansion = expansion

    @staticmethod
    def masks(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array((0, 70, 40)), np.array((12, 255, 255)))
        red |= cv2.inRange(
            hsv, np.array((165, 70, 40)), np.array((179, 255, 255))
        )
        blue = cv2.inRange(
            hsv, np.array((90, 70, 40)), np.array((135, 255, 255))
        )
        return red, blue

    def infer(self, image):
        height, width = image.shape[:2]
        image_area = float(width * height)
        minimum_component_area = max(60.0, image_area * 0.00004)
        merge_margin = max(5, int(round(min(width, height) * 0.008)))
        candidates = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for mask in self.masks(image):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            components = []
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < minimum_component_area:
                    continue
                x, y, box_width, box_height = cv2.boundingRect(contour)
                components.append(
                    ((x, y, x + box_width, y + box_height), area)
                )

            for box, colour_area in _merge_components(components, merge_margin):
                x1, y1, x2, y2 = box
                box_width, box_height = x2 - x1, y2 - y1
                if min(box_width, box_height) < 15:
                    continue
                aspect = box_width / max(box_height, 1)
                box_area_ratio = box_width * box_height / image_area
                fill = colour_area / max(float(box_width * box_height), 1.0)
                if not self.min_aspect <= aspect <= self.max_aspect:
                    continue
                if not self.min_box_area_ratio <= box_area_ratio <= self.max_box_area_ratio:
                    continue
                if not 0.06 <= fill <= 0.92:
                    continue

                pad_x = box_width * self.expansion
                pad_y = box_height * self.expansion
                expanded = (
                    max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
                    min(float(width - 1), x2 + pad_x),
                    min(float(height - 1), y2 + pad_y),
                )
                shape_ok, shape_quality, _ = pentagon_shape_score(image, expanded)
                if not shape_ok:
                    continue
                colour_quality = min(1.0, fill / 0.35)
                confidence = 0.55 + 0.35 * (
                    0.45 * colour_quality + 0.55 * shape_quality
                )
                candidates.append(Detection(expanded, min(0.90, confidence)))

        remaining = sorted(
            candidates, key=lambda item: item.confidence, reverse=True
        )
        kept = []
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [
                item for item in remaining if iou(best.box, item.box) < 0.5
            ]
        return kept


class ColorFirstDetector:
    """Use HSV proposals first and optionally fall back to YOLO."""

    def __init__(self, color_detector, yolo_detector=None):
        self.color_detector = color_detector
        self.yolo_detector = yolo_detector

    def infer(self, image):
        detections = self.color_detector.infer(image)
        if detections or self.yolo_detector is None:
            return detections
        return self.yolo_detector.infer(image)
