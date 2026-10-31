"""轻量多靶标框关联：为画面中的每个靶标分别维护识别投票历史。"""

from __future__ import annotations

import math


def limit_detections(detections, maximum: int):
    items = list(detections)
    return items if maximum <= 0 else items[:maximum]


def _geometry(box):
    x1, y1, x2, y2 = map(float, box)
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return (x1 + x2) / 2, (y1 + y2) / 2, width, height, width * height


def _iou(left, right):
    ax1, ay1, ax2, ay2 = map(float, left)
    bx1, by1, bx2, by2 = map(float, right)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


class BoxTrackManager:
    """按框的位置、尺度和重叠关系关联连续帧，无需额外跟踪神经网络。"""

    def __init__(self, max_age: int = 5):
        self.max_age = max(1, int(max_age))
        self.tick = 0
        self.next_id = 0
        self.tracks = {}

    @property
    def active_ids(self):
        return set(self.tracks)

    def assign(self, detections):
        detections = list(detections)
        self.tick += 1
        self.tracks = {
            track_id: state for track_id, state in self.tracks.items()
            if self.tick - state["last_seen"] <= self.max_age
        }
        candidates = []
        for track_id, state in self.tracks.items():
            old_box = state["box"]
            old_cx, old_cy, old_w, old_h, old_area = _geometry(old_box)
            old_scale = math.hypot(old_w, old_h)
            for detection_index, detection in enumerate(detections):
                new_box = detection.box
                new_cx, new_cy, new_w, new_h, new_area = _geometry(new_box)
                new_scale = math.hypot(new_w, new_h)
                overlap = _iou(old_box, new_box)
                distance = math.hypot(new_cx - old_cx, new_cy - old_cy) / max((old_scale + new_scale) / 2, 1.0)
                area_ratio = new_area / max(old_area, 1.0)
                if overlap < 0.02 and not (distance <= 2.0 and 0.35 <= area_ratio <= 2.85):
                    continue
                cost = distance - 2.0 * overlap + 0.25 * abs(math.log(max(area_ratio, 1e-6)))
                candidates.append((cost, track_id, detection_index))

        assignments = [None] * len(detections)
        used_tracks, used_detections = set(), set()
        for _, track_id, detection_index in sorted(candidates):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            track_id = assignments[detection_index]
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                assignments[detection_index] = track_id
            self.tracks[track_id] = {"box": tuple(map(float, detection.box)), "last_seen": self.tick}
        return assignments
