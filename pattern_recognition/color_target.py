"""图案任务的 HSV 红蓝候选、五边形几何评分和背景误检过滤。"""

from __future__ import annotations

import cv2
import numpy as np

from detector import Detection, box_iou


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
    """检查候选区域是否满足比赛靶标的等边直角五边形几何特征。

    透视和低分辨率会扰动测得的顶点，因此会搜索一小段多边形近似容差，
    而不是强制原始轮廓必须恰好得到五个角点。
    """
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
        "reason": "ok",
        "vertices": 5,
        "side_ratio": float(side_ratio),
        "angles_deg": [round(value, 1) for value in angles],
        "epsilon": float(epsilon),
    }


def target_structure_score(image, box):
    """Validate that a colour proposal contains a central white panel and glyph.

    The HSV body alone can also describe flags, clothes, signs and painted
    objects.  A competition target additionally has a sizeable white panel
    inside the convex hull of its red/blue body and a dark symbol in that
    panel.  Return ``(accepted, quality, details)`` for diagnostics.
    """
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
    ys, xs = np.nonzero(colour)
    if len(xs) < 20:
        return False, 0.0, {"reason": "no_colour"}

    hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
    hull_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)
    erosion = max(1, int(round(min(crop.shape[:2]) * 0.01)))
    hull_mask = cv2.erode(
        hull_mask, np.ones((erosion, erosion), np.uint8), iterations=1
    )

    # 排除饱和度达到靶标颜色范围的浅色材料。
    white = cv2.inRange(hsv, np.array((0, 0, 105)), np.array((180, 85, 255)))
    white = cv2.bitwise_and(white, hull_mask)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = float(crop.shape[0] * crop.shape[1])
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        rectangle_area = float(width * height)
        if rectangle_area <= 1.0:
            continue
        center_x = (x + width / 2) / crop.shape[1]
        center_y = (y + height / 2) / crop.shape[0]
        rectangularity = area / rectangle_area
        score = area * rectangularity
        if 0.15 <= center_x <= 0.85 and 0.15 <= center_y <= 0.85:
            if best is None or score > best[0]:
                best = (score, contour, area, rectangularity, center_x, center_y)
    if best is None:
        return False, 0.0, {"reason": "no_central_white_panel"}

    _, panel, panel_area, rectangularity, center_x, center_y = best
    panel_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(panel_mask, [panel], -1, 255, -1)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 150)
    dark_ratio = cv2.countNonZero(cv2.bitwise_and(dark, panel_mask)) / max(
        cv2.countNonZero(panel_mask), 1
    )
    panel_ratio = panel_area / crop_area
    accepted = (
        0.055 <= panel_ratio <= 0.82
        and rectangularity >= 0.35
        and 0.015 <= dark_ratio <= 0.82
    )
    quality = min(1.0, panel_ratio / 0.20) * min(1.0, rectangularity / 0.65)
    quality *= min(1.0, dark_ratio / 0.12)
    return accepted, float(quality), {
        "reason": "ok" if accepted else "panel_geometry",
        "panel_ratio": float(panel_ratio),
        "rectangularity": float(rectangularity),
        "dark_ratio": float(dark_ratio),
        "center_x": float(center_x),
        "center_y": float(center_y),
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
        matching = [index for index, (other, _) in enumerate(groups) if _touches(box, other, margin)]
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


def _merge_aligned_target_parts(groups):
    """Join target-colour pieces separated by the central white panel."""
    merged = list(groups)
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            first_box, first_area = merged[first_index]
            for second_index in range(first_index + 1, len(merged)):
                second_box, second_area = merged[second_index]
                ax1, ay1, ax2, ay2 = first_box
                bx1, by1, bx2, by2 = second_box
                overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                minimum_width = max(1, min(ax2 - ax1, bx2 - bx1))
                if overlap / minimum_width < 0.55:
                    continue
                vertical_gap = max(0, max(ay1, by1) - min(ay2, by2))
                reference_width = max(ax2 - ax1, bx2 - bx1)
                if vertical_gap > 0.50 * reference_width:
                    continue
                merged[first_index] = ((
                    min(ax1, bx1), min(ay1, by1),
                    max(ax2, bx2), max(ay2, by2),
                ), first_area + second_area)
                merged.pop(second_index)
                changed = True
                break
            if changed:
                break
    return merged


class ColorTargetDetector:
    """Find compact red or blue target bodies and reject long color edges."""

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
    def _masks(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array((0, 70, 40)), np.array((12, 255, 255)))
        red |= cv2.inRange(hsv, np.array((165, 70, 40)), np.array((179, 255, 255)))
        blue = cv2.inRange(hsv, np.array((90, 70, 40)), np.array((135, 255, 255)))
        return red, blue

    def infer(self, image):
        height, width = image.shape[:2]
        image_area = float(width * height)
        minimum_component_area = max(60.0, image_area * 0.00004)
        merge_margin = max(5, int(round(min(width, height) * 0.008)))
        candidates = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for mask in self._masks(image):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            components = []
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < minimum_component_area:
                    continue
                x, y, box_width, box_height = cv2.boundingRect(contour)
                components.append(((x, y, x + box_width, y + box_height), area))

            groups = _merge_components(components, merge_margin)
            groups = _merge_aligned_target_parts(groups)
            for box, color_area in groups:
                x1, y1, x2, y2 = box
                box_width, box_height = x2 - x1, y2 - y1
                # 拒绝在1080p下无法保留足够图案细节的小区域。
                if min(box_width, box_height) < 32:
                    continue
                aspect = box_width / max(box_height, 1)
                box_area_ratio = box_width * box_height / image_area
                fill = color_area / max(float(box_width * box_height), 1.0)
                if not self.min_aspect <= aspect <= self.max_aspect:
                    continue
                if not self.min_box_area_ratio <= box_area_ratio <= self.max_box_area_ratio:
                    continue
                if not 0.06 <= fill <= 0.92:
                    continue

                pad_x, pad_y = box_width * self.expansion, box_height * self.expansion
                expanded = (
                    max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
                    min(float(width - 1), x2 + pad_x),
                    min(float(height - 1), y2 + pad_y),
                )
                shape_ok, shape_quality, _ = pentagon_shape_score(image, expanded)
                if not shape_ok:
                    continue
                accepted, structure_quality, _ = target_structure_score(
                    image, expanded
                )
                if not accepted:
                    continue
                colour_quality = min(1.0, fill / 0.35)
                confidence = 0.55 + 0.35 * (
                    0.25 * colour_quality
                    + 0.35 * shape_quality
                    + 0.40 * structure_quality
                )
                candidates.append(Detection(expanded, min(0.90, confidence)))

        remaining = sorted(candidates, key=lambda item: item.confidence, reverse=True)
        kept = []
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [item for item in remaining if box_iou(best.box, item.box) < 0.5]
        return kept


class ColorFirstDetector:
    def __init__(self, color_detector, yolo_detector=None):
        self.color_detector = color_detector
        self.yolo_detector = yolo_detector

    def infer(self, image):
        detections = self.color_detector.infer(image)
        if detections or self.yolo_detector is None:
            return detections
        return self.yolo_detector.infer(image)
