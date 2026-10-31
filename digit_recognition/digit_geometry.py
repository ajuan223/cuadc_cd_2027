"""Red/blue target rectification and two-digit strip extraction."""

import math

"""数字靶标的颜色筛选、五边形校验、方向旋正和白色数字面板提取。"""

import cv2
import numpy as np

from color_target import mask_convex_hull

# 定义官方方形加等边三角箭头靶标的标准透视坐标。
RECTIFIED_WIDTH = 360
RECTIFIED_SHOULDER_Y = int(round(RECTIFIED_WIDTH * math.sqrt(3.0) / 2.0))
RECTIFIED_HEIGHT = RECTIFIED_SHOULDER_Y + RECTIFIED_WIDTH
RECTIFIED_PENTAGON = np.array(
    [
        (RECTIFIED_WIDTH / 2.0, 0.0),
        (RECTIFIED_WIDTH - 1.0, RECTIFIED_SHOULDER_Y),
        (RECTIFIED_WIDTH - 1.0, RECTIFIED_HEIGHT - 1.0),
        (0.0, RECTIFIED_HEIGHT - 1.0),
        (0.0, RECTIFIED_SHOULDER_Y),
    ],
    dtype=np.float32,
)


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Return four unique corners clockwise from the visual top-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    first_edge = ordered[1] - ordered[0]
    second_edge = ordered[2] - ordered[1]
    cross = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
    if cross < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


# 用HSV提出红蓝候选并交给几何和内部结构继续校验。
def target_colour_mask(image: np.ndarray):
    """Return a cleaned mask containing both allowed target colours."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array((0, 70, 40)), np.array((25, 255, 255))),
        cv2.inRange(hsv, np.array((155, 70, 40)), np.array((180, 255, 255))),
    )
    blue = cv2.inRange(hsv, np.array((80, 70, 40)), np.array((145, 255, 255)))
    colour = cv2.bitwise_or(red, blue)
    colour = cv2.morphologyEx(
        colour, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    return cv2.morphologyEx(colour, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def target_colour_contour(image: np.ndarray):
    colour = target_colour_mask(image)
    contours, _ = cv2.findContours(colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * 0.015:
        return None
    return contour


def _polygon_angles(polygon: np.ndarray):
    angles = []
    for index in range(len(polygon)):
        first = polygon[index - 1] - polygon[index]
        second = polygon[(index + 1) % len(polygon)] - polygon[index]
        cosine = np.dot(first, second) / (
            np.linalg.norm(first) * np.linalg.norm(second) + 1e-9
        )
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return angles


# 用凸包和多边形近似校验官方五边形几何。
def target_pentagon(image: np.ndarray):
    """Recover and clockwise-order the five physical target corners."""
    colour = target_colour_mask(image)
    hull = mask_convex_hull(colour)
    if hull is None:
        return None
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 1.0:
        return None

    candidates = []
    for epsilon in (0.018, 0.020, 0.022, 0.025, 0.028, 0.030, 0.035, 0.040):
        polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(polygon) != 5:
            continue
        angles = _polygon_angles(polygon.astype(np.float32))
        ordered_angles = sorted(angles)
        if not (
            35 <= ordered_angles[0] <= 105
            and 55 <= ordered_angles[1] <= 125
            and 65 <= ordered_angles[2] <= 135
            and 110 <= ordered_angles[3] <= 175
            and 115 <= ordered_angles[4] <= 178
        ):
            continue
        edges = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1)
        side_ratio = float(edges.min() / max(float(edges.max()), 1.0))
        if side_ratio < 0.16:
            continue
        candidates.append((side_ratio, polygon.astype(np.float32), angles))
    if not candidates:
        return None

    _, polygon, angles = max(candidates, key=lambda item: item[0])
    center = polygon.mean(axis=0)
    order = np.argsort(np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0]))
    polygon = polygon[order]
    angles = [angles[index] for index in order]

    # 结合尖角与白色面板中心确定箭头尖端。
    hull_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, np.round(polygon).astype(np.int32), 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array((0, 0, 125)), np.array((180, 115, 255)))
    white = cv2.bitwise_and(white, hull_mask)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    large = [item for item in contours if cv2.contourArea(item) >= image_area * 0.002]
    if large:
        panel = max(large, key=cv2.contourArea)
        moments = cv2.moments(panel)
        if abs(moments["m00"]) > 1e-8:
            panel_center = np.array(
                (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                dtype=np.float32,
            )
            apex_index = int(np.argmax(np.linalg.norm(polygon - panel_center, axis=1)))
        else:
            apex_index = int(np.argmin(angles))
    else:
        apex_index = int(np.argmin(angles))
    return np.roll(polygon, -apex_index, axis=0)


# 根据五个角点计算单应矩阵并将靶标旋正到箭头朝上。
def rectify_target_perspective(image: np.ndarray):
    """Map a detected physical pentagon to one canonical, arrow-up view."""
    polygon = target_pentagon(image)
    if polygon is None:
        return None
    # 用五个角点拟合投影变换以降低单个角点误差。
    matrix, _ = cv2.findHomography(
        polygon.astype(np.float32), RECTIFIED_PENTAGON, method=0
    )
    if matrix is None:
        return None
    return cv2.warpPerspective(
        image,
        matrix,
        (RECTIFIED_WIDTH, RECTIFIED_HEIGHT),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )


def target_colour_hull(image: np.ndarray):
    contour = target_colour_contour(image)
    if contour is None:
        return None
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(contour)], -1, 255, -1)
    return mask


# 在已确认靶标内按颜色定位中央白色数字面板。
def white_panel_contour(image: np.ndarray, hull_mask: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array((0, 0, 125)), np.array((180, 115, 255)))
    kernel_size = max(3, int(round(min(image.shape[:2]) * 0.025)))
    kernel_size += 1 - kernel_size % 2
    interior = cv2.erode(
        hull_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1
    )
    white = cv2.bitwise_and(white, interior)
    join_width = max(3, int(round(image.shape[1] * 0.055)))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, join_width), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    best, best_score = None, -math.inf
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.002:
            continue
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        rectangle_area = width * height
        if rectangle_area <= 1.0:
            continue
        aspect = max(width, height) / max(min(width, height), 1.0)
        rectangularity = area / rectangle_area
        if aspect > 3.4 or rectangularity < 0.42:
            continue
        score = area * (0.5 + rectangularity)
        if score > best_score:
            best, best_score = contour, score
    return best


def glyph_candidates(image: np.ndarray, hull_mask: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, np.array((0, 0, 0)), np.array((180, 165, 185)))
    dark = cv2.bitwise_and(dark, hull_mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    output = []
    for contour in contours:
        area = cv2.contourArea(contour)
        _, _, box_width, box_height = cv2.boundingRect(contour)
        if area < height * width * 0.0006:
            continue
        if box_width < max(5, width * 0.012) or box_height < max(9, height * 0.075):
            continue
        if box_width <= width * 0.35 and box_height <= height * 0.42:
            output.append(contour)
    return output


def choose_glyph_pair(contours):
    best_pair, best_score = None, -math.inf
    for index, first in enumerate(contours):
        x1, y1, width1, height1 = cv2.boundingRect(first)
        center1 = np.array((x1 + width1 / 2, y1 + height1 / 2))
        for second in contours[index + 1:]:
            x2, y2, width2, height2 = cv2.boundingRect(second)
            center2 = np.array((x2 + width2 / 2, y2 + height2 / 2))
            distance = float(np.linalg.norm(center1 - center2))
            mean_height = (height1 + height2) / 2
            similarity = min(height1, height2) / max(height1, height2)
            if not mean_height * 0.45 <= distance <= mean_height * 2.8 or similarity < 0.55:
                continue
            alignment = min(abs(center1[0] - center2[0]), abs(center1[1] - center2[1])) / max(mean_height, 1)
            score = math.log1p(cv2.contourArea(first) + cv2.contourArea(second)) + 2.5 * similarity - 2 * alignment
            if score > best_score:
                best_pair, best_score = (first, second), score
    return best_pair


def arrow_tip_and_center(contour: np.ndarray):
    x, y, width, height = cv2.boundingRect(contour)
    local = contour.copy()
    local[:, 0, 0] -= x
    local[:, 0, 1] -= y
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [local], -1, 255, -1)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    points = np.column_stack((xs + x, ys + y)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    center, axis = mean[0], eigenvectors[0]
    projections = (points - center) @ axis
    low, high = float(projections.min()), float(projections.max())
    span = high - low
    if span <= 1e-6:
        return None
    low_count = np.count_nonzero(projections <= low + span * 0.22)
    high_count = np.count_nonzero(projections >= high - span * 0.22)
    sign = -1.0 if low_count < high_count else 1.0
    return center + axis * sign * span, center


def rotate_bound(image: np.ndarray, angle: float):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    output_width = int(math.ceil(height * sine + width * cosine))
    output_height = int(math.ceil(height * cosine + width * sine))
    matrix[0, 2] += output_width / 2 - center[0]
    matrix[1, 2] += output_height / 2 - center[1]
    return cv2.warpAffine(
        image, matrix, (output_width, output_height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114),
    )


# 用白色面板中心与最远顶点确定方向并在几何不完整时回退。
def normalize_arrow_up(image: np.ndarray, prefer_panel: bool = True):
    contour = target_colour_contour(image)
    if contour is None:
        return None
    hull_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(hull_mask, [cv2.convexHull(contour)], -1, 255, -1)
    panel = white_panel_contour(image, hull_mask) if prefer_panel else None
    geometry = None
    if panel is not None:
        moments = cv2.moments(panel)
        if abs(moments["m00"]) > 1e-8:
            center = np.array(
                (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                dtype=np.float32,
            )
            hull_points = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
            geometry = hull_points[int(np.argmax(np.linalg.norm(hull_points - center, axis=1)))], center
    if geometry is None:
        pair = choose_glyph_pair(glyph_candidates(image, hull_mask))
        if pair is not None:
            centers = []
            for glyph in pair:
                x, y, width, height = cv2.boundingRect(glyph)
                centers.append((x + width / 2, y + height / 2))
            center = np.mean(np.asarray(centers, dtype=np.float32), axis=0)
            hull_points = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
            geometry = hull_points[int(np.argmax(np.linalg.norm(hull_points - center, axis=1)))], center
    if geometry is None:
        geometry = arrow_tip_and_center(contour)
    if geometry is None:
        return None
    tip, center = geometry
    vector = tip - center
    return rotate_bound(image, math.degrees(math.atan2(float(vector[1]), float(vector[0]))) + 90)


def rectify_white_panel(image: np.ndarray, contour: np.ndarray):
    x, y, width, height = cv2.boundingRect(contour)
    if width < 4 or height < 4:
        return None
    panel = image[y:y + height, x:x + width].copy()
    local = contour.copy()
    local[:, 0, 0] -= x
    local[:, 0, 1] -= y
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [local], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    panel[mask == 0] = 255
    pad_y, pad_x = max(2, int(height * 0.10)), max(2, int(width * 0.05))
    return cv2.copyMakeBorder(panel, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255))


# 按实测白板四边形二次透视校正并输出横向OCR数字条。
def rectified_colour_panel(image: np.ndarray):
    """Extract the white digit area by colour from a canonical target view.

    Expected physical dimensions influence candidate scoring only.  The crop
    boundaries themselves always come from observed white pixels, so printing,
    placement and perspective errors do not turn into fixed-pixel crop errors.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array((0, 0, 125)), np.array((180, 110, 255)))
    polygon_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(
        polygon_mask, np.round(RECTIFIED_PENTAGON).astype(np.int32), 255
    )
    polygon_mask = cv2.erode(
        polygon_mask, np.ones((7, 7), np.uint8), iterations=1
    )
    white = cv2.bitwise_and(white, polygon_mask)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 连接两块官方数字板之间的小间隙。
    joined = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        np.ones((5, max(9, int(round(image.shape[1] * 0.075)))), np.uint8),
        iterations=1,
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(image.shape[0] * image.shape[1])
    dark = cv2.inRange(hsv, np.array((0, 0, 0)), np.array((180, 175, 190)))
    best = None
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        box_area = float(width * height)
        if box_area < image_area * 0.008 or width < 18 or height < 18:
            continue
        aspect = width / max(float(height), 1.0)
        if not 0.42 <= aspect <= 2.8:
            continue
        white_ratio = cv2.countNonZero(white[y:y + height, x:x + width]) / box_area
        dark_ratio = cv2.countNonZero(dark[y:y + height, x:x + width]) / box_area
        if white_ratio < 0.32 or not 0.008 <= dark_ratio <= 0.48:
            continue

        center_x = (x + width / 2.0) / image.shape[1]
        center_y = (y + height / 2.0) / image.shape[0]
        # 用位置和近方形尺寸评分并强制检查白色区域与深色字符。
        centrality = max(0.0, 1.0 - abs(center_x - 0.5) / 0.5)
        body_position = max(0.0, 1.0 - abs(center_y - 0.70) / 0.55)
        score = box_area * (0.55 + white_ratio) * (
            0.65 + 0.20 * centrality + 0.15 * body_position
        )
        if best is None or score > best[0]:
            best = (score, contour, x, y, width, height)
    if best is None:
        return None

    _, contour, x, y, width, height = best
    # 从白色边界估计面板四边并映射为正方形。
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    corners = None
    for epsilon in (0.015, 0.020, 0.025, 0.030, 0.040, 0.050):
        candidate = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(candidate) == 4 and cv2.isContourConvex(candidate.astype(np.int32)):
            corners = candidate.astype(np.float32)
            break
    if corners is None:
        corners = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    ordered = _order_quad(corners)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    square_size = int(round(np.mean((top_width, bottom_width, left_height, right_height))))
    if square_size >= 24 and len(np.unique(np.round(ordered).astype(np.int32), axis=0)) == 4:
        destination = np.array(
            [(0, 0), (square_size - 1, 0),
             (square_size - 1, square_size - 1), (0, square_size - 1)],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, destination)
        strip = cv2.warpPerspective(
            image, matrix, (square_size, square_size), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
        pad = max(2, int(round(square_size * 0.045)))
        return cv2.copyMakeBorder(
            strip, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )

    # 四角不可用时按颜色框和相对边距回退裁剪。
    margin_x = max(2, int(round(width * 0.035)))
    margin_y = max(2, int(round(height * 0.055)))
    left, top = max(0, x - margin_x), max(0, y - margin_y)
    right = min(image.shape[1], x + width + margin_x)
    bottom = min(image.shape[0], y + height + margin_y)
    strip = image[top:bottom, left:right].copy()
    if not strip.size:
        return None
    pad_y, pad_x = max(2, int(strip.shape[0] * 0.06)), max(2, int(strip.shape[1] * 0.03))
    return cv2.copyMakeBorder(
        strip, pad_y, pad_y, pad_x, pad_x,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


def rectify_pair(image: np.ndarray, pair):
    x, y, width, height = cv2.boundingRect(np.vstack(pair))
    margin_x, margin_y = int(width * 0.28), int(height * 0.38)
    return image[
        max(0, y - margin_y):min(image.shape[0], y + height + margin_y),
        max(0, x - margin_x):min(image.shape[1], x + width + margin_x),
    ].copy()


def digit_candidate_check(image: np.ndarray):
    """Cheap rotation-independent gate before warping or OCR.

    Check observed white regions inside the colour hull for darker contents.
    Do not require exactly two connected glyphs: touching digits and split
    printed panels are valid. This is a structural check, not digit recognition.
    """
    if image.size == 0 or min(image.shape[:2]) < 4:
        return False, "empty_or_small_crop"
    scale = min(1.0, 384.0 / max(image.shape[:2]))
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    hull = mask_convex_hull(target_colour_mask(preview))
    if hull is None:
        return False, "no_colour_body"
    interior = np.zeros(preview.shape[:2], np.uint8)
    cv2.drawContours(interior, [hull], -1, 255, -1)
    white = cv2.inRange(hsv, (0, 0, 125), (180, 115, 255))
    white = cv2.bitwise_and(white, interior)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(8.0, preview.shape[0] * preview.shape[1] * 0.002)
    found_panel = False
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        found_panel = True
        x, y, width, height = cv2.boundingRect(contour)
        panel_mask = np.zeros((height, width), np.uint8)
        cv2.drawContours(panel_mask, [contour - (x, y)], -1, 255, -1)
        panel_hsv = hsv[y:y + height, x:x + width]
        white_pixels = (white[y:y + height, x:x + width] != 0) & (panel_mask != 0)
        if not np.any(white_pixels):
            continue
        # 用相对对比度避免把均匀灰白面板误判成字符。
        dark_limit = min(185, max(0, int(np.median(panel_hsv[:, :, 2][white_pixels])) - 35))
        dark = cv2.inRange(panel_hsv, (0, 0, 0), (180, 165, dark_limit))
        ink_pixels = cv2.countNonZero(cv2.bitwise_and(dark, panel_mask))
        if ink_pixels >= max(3, cv2.countNonZero(panel_mask) * 0.003):
            return True, "structure_pass"
    return False, "no_dark_content" if found_panel else "no_white_panel"


def extract_digit_strips(image: np.ndarray, diagnostics=None, *, early_reject=True) -> list[np.ndarray]:
    """汇总主路径和备用路径，返回可直接送入 OCR 的候选数字条。"""
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(early_rejected=False, rejection_reason=None, geometry_path=None)
    if image.size == 0:
        diagnostics.update(early_rejected=True, rejection_reason="empty_crop")
        return []
    if early_reject:
        allowed, reason = digit_candidate_check(image)
        if not allowed:
            diagnostics.update(early_rejected=True, rejection_reason=reason)
            return []
    panel_image = image
    short_side = min(image.shape[:2])
    if short_side < 160:
        scale = min(3.0, 160.0 / max(short_side, 1))
        panel_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 先按几何校正透视和方向再按白色边界确定OCR裁剪。
    rectified = rectify_target_perspective(panel_image)
    if rectified is not None:
        strip = rectified_colour_panel(rectified)
        if strip is not None:
            diagnostics["geometry_path"] = "pentagon"
            return [strip]

    # 五角轮廓残缺或模糊时走兼容回退。
    strips = []
    normalized = normalize_arrow_up(panel_image, prefer_panel=True)
    if normalized is not None:
        hull = target_colour_hull(normalized)
        panel = white_panel_contour(normalized, hull) if hull is not None else None
        strip = rectify_white_panel(normalized, panel) if panel is not None else None
        if strip is not None:
            strips.append(strip)
    legacy = normalize_arrow_up(image, prefer_panel=False)
    if legacy is not None:
        hull = target_colour_hull(legacy)
        pair = choose_glyph_pair(glyph_candidates(legacy, hull)) if hull is not None else None
        if pair is not None:
            strips.append(rectify_pair(legacy, pair))
    strips = [strip for strip in strips if strip.size]
    diagnostics["geometry_path"] = "fallback"
    if not strips:
        diagnostics["rejection_reason"] = "no_digit_strip"
    return strips


def split_strip(strip: np.ndarray):
    """把两位数字条左右分开，供单字 OCR 与整条 OCR 交叉验证。"""
    height, width = strip.shape[:2]
    output = []
    for half in (strip[:, :width // 2], strip[:, width // 2:]):
        cropped = half[int(height * 0.03):max(1, int(height * 0.97)), int(half.shape[1] * 0.02):max(1, int(half.shape[1] * 0.98))]
        output.append(cv2.resize(cropped, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    return output
