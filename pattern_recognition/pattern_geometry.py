"""Colour-driven central pattern extraction without orientation rectification."""

from __future__ import annotations

"""图案靶标的颜色筛选、五边形校验和中央白色图案面板提取。"""

import cv2
import numpy as np


# 用HSV提出靶标候选并交给五边形和中央结构继续校验。
def target_colour_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array((0, 70, 40)), np.array((25, 255, 255)))
    red |= cv2.inRange(hsv, np.array((155, 70, 40)), np.array((180, 255, 255)))
    blue = cv2.inRange(hsv, np.array((80, 70, 40)), np.array((145, 255, 255)))
    colour = cv2.bitwise_or(red, blue)
    colour = cv2.morphologyEx(
        colour, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    return cv2.morphologyEx(colour, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _angles(points: np.ndarray) -> list[float]:
    output = []
    for index in range(len(points)):
        first = points[index - 1] - points[index]
        second = points[(index + 1) % len(points)] - points[index]
        cosine = np.dot(first, second) / (
            np.linalg.norm(first) * np.linalg.norm(second) + 1e-9
        )
        output.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return output


# 用凸包和多边形近似排除普通红蓝物体。
def target_pentagon(image: np.ndarray):
    mask = target_colour_mask(image)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 1.0:
        return None
    candidates = []
    for epsilon in (0.018, 0.020, 0.022, 0.025, 0.028, 0.030, 0.035, 0.040):
        polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(polygon) != 5:
            continue
        polygon = polygon.astype(np.float32)
        angles = _angles(polygon)
        ordered = sorted(angles)
        edges = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1)
        if not (
            35 <= ordered[0] <= 105 and 55 <= ordered[1] <= 125
            and 65 <= ordered[2] <= 135 and 110 <= ordered[3] <= 175
            and 115 <= ordered[4] <= 178
            and float(edges.min() / max(float(edges.max()), 1.0)) >= 0.16
        ):
            continue
        candidates.append((float(edges.min() / edges.max()), polygon, angles))
    if not candidates:
        return None
    _, polygon, angles = max(candidates, key=lambda item: item[0])
    center = polygon.mean(axis=0)
    order = np.argsort(np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0]))
    polygon = polygon[order]
    angles = [angles[index] for index in order]
    return np.roll(polygon, -int(np.argmin(angles)), axis=0)


# 按白色区域、矩形度和黑色图案占比提取中央面板并拒绝背景。
def extract_white_pattern(
    image: np.ndarray, target_polygon: np.ndarray, output_size: int = 320
):
    """Crop the observed white panel while preserving its image orientation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 从中央白色面板轮廓中排除褪色红蓝材料。
    white = cv2.inRange(hsv, np.array((0, 0, 105)), np.array((180, 85, 255)))
    interior = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(interior, np.round(target_polygon).astype(np.int32), 255)
    kernel_size = max(3, int(round(min(image.shape[:2]) * 0.015)) | 1)
    interior = cv2.erode(
        interior, np.ones((kernel_size, kernel_size), np.uint8), iterations=1
    )
    white = cv2.bitwise_and(white, interior)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_area = max(float(cv2.contourArea(target_polygon)), 1.0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 165)
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        (_, _), (rotated_width, rotated_height), _ = cv2.minAreaRect(contour)
        rectangle_area = float(rotated_width * rotated_height)
        if area < target_area * 0.012 or rectangle_area <= 1.0:
            continue
        aspect = rotated_width / max(float(rotated_height), 1.0)
        rectangularity = area / rectangle_area
        panel_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(panel_mask, [contour], -1, 255, -1)
        dark_ratio = cv2.countNonZero(cv2.bitwise_and(dark, panel_mask)) / max(
            cv2.countNonZero(panel_mask), 1
        )
        center_x = (x + width / 2.0) / image.shape[1]
        center_y = (y + height / 2.0) / image.shape[0]
        if not (
            0.48 <= aspect <= 2.1 and rectangularity >= 0.32
            and 0.008 <= dark_ratio <= 0.84
            and 0.12 <= center_x <= 0.88 and 0.18 <= center_y <= 0.94
        ):
            continue
        centrality = 1.0 - min(1.0, abs(center_x - 0.5) / 0.5)
        score = area * (0.6 + rectangularity) * (0.8 + 0.2 * centrality)
        if best is None or score > best[0]:
            best = (score, contour)
    if best is None:
        return None

    contour = best[1]
    x, y, width, height = cv2.boundingRect(contour)
    padding = max(1, int(round(max(width, height) * 0.025)))
    left, top = max(0, x - padding), max(0, y - padding)
    right = min(image.shape[1], x + width + padding)
    bottom = min(image.shape[0], y + height + padding)
    panel_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(panel_mask, [contour], -1, 255, -1)
    roi = image[top:bottom, left:right].copy()
    roi_mask = panel_mask[top:bottom, left:right]
    roi[roi_mask == 0] = 255

    roi_height, roi_width = roi.shape[:2]
    side = max(roi_height, roi_width)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    offset_x = (side - roi_width) // 2
    offset_y = (side - roi_height) // 2
    square[offset_y:offset_y + roi_height, offset_x:offset_x + roi_width] = roi
    interpolation = cv2.INTER_AREA if side > output_size else cv2.INTER_CUBIC
    return cv2.resize(square, (output_size, output_size), interpolation=interpolation)


def extract_pattern_crop(image: np.ndarray, output_size: int = 320):
    """输出等比例缩放并补边的分类器输入；保留图案原方向，不做旋正。"""
    polygon = target_pentagon(image)
    return None if polygon is None else extract_white_pattern(image, polygon, output_size)
