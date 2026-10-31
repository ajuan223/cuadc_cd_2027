"""图案任务纯视觉入口：靶标检测、中央面板提取、图案分类与跨帧投票。"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from detector import TargetDetector
from color_target import ColorFirstDetector, ColorTargetDetector
from pattern_geometry import extract_pattern_crop
from target_tracking import BoxTrackManager, limit_detections
from camera_calibration import CameraCalibration, CameraUndistorter, DEFAULT_CALIBRATION_PATH
from camera_runtime import add_camera_arguments, build_live_camera


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = sorted([
    "jiqiangbing", "huojianbing", "duoxuanyi", "gudingyi", "kache",
    "fangkongpao", "tanke", "zhishengji", "zhandouji", "yunshuji",
    "zhenchaji", "hongzhaji", "invalid",
])
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# 生成靶标候选区域。
def source_value(value: str):
    return int(value) if value.isdigit() else value


def build_target_detector(args):
    """选择 HSV 几何检测、HSV 优先并用 YOLO 兜底，或仅使用 YOLO。"""
    color = ColorTargetDetector()
    if args.target_proposal == "color":
        return ColorFirstDetector(color)
    yolo = TargetDetector(
        args.target_model, args.target_conf, args.nms_iou, args.target_size
    )
    if args.target_proposal == "color-first":
        return ColorFirstDetector(color, yolo)
    return yolo


def crop_target(image, box, expansion=1.15):
    """Crop target context without changing its aspect ratio."""
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    pad_x = (x2 - x1) * max(0.0, expansion - 1.0) / 2.0
    pad_y = (y2 - y1) * max(0.0, expansion - 1.0) / 2.0
    left, top = max(0, int(np.floor(x1 - pad_x))), max(0, int(np.floor(y1 - pad_y)))
    right = min(image_width, int(np.ceil(x2 + pad_x)))
    bottom = min(image_height, int(np.ceil(y2 + pad_y)))
    if right - left < 12 or bottom - top < 12:
        raise ValueError("target crop is too small")
    return image[top:bottom, left:right].copy()


def prepare_pattern_crop(image, box, expansion=1.15, output_size=320):
    """按实际白色区域裁取中央图案面板；图案分类不进行方向旋正。"""
    target = crop_target(image, box, expansion)
    crop = extract_pattern_crop(target, output_size)
    if crop is None:
        raise ValueError("central white pattern panel could not be extracted")
    return crop


class PatternClassifier:
    """通过 OpenCV DNN 在 CPU 上运行 ONNX 图案分类模型。"""
    def __init__(self, model: Path, input_size=320):
        if not model.is_file():
            raise FileNotFoundError(model)
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.input_size = input_size

    def infer(self, crop):
        blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (self.input_size, self.input_size), swapRB=True)
        self.net.setInput(blob)
        output = np.asarray(self.net.forward(), dtype=np.float32).reshape(-1)
        if len(output) != len(CLASS_NAMES):
            raise RuntimeError(f"classifier returned {len(output)} outputs; expected {len(CLASS_NAMES)}")
        if np.any(output < 0) or not 0.98 <= float(output.sum()) <= 1.02:
            output = np.exp(output - output.max())
            output /= output.sum()
        index = int(np.argmax(output))
        return CLASS_NAMES[index], float(output[index])


class LabelVote:
    """对同一 track_id 的分类结果进行置信度加权和最少命中次数约束。"""
    def __init__(self, window):
        self.items = deque(maxlen=window)

    def add(self, label, confidence):
        self.items.append((label, confidence))
        totals = defaultdict(float)
        for item, score in self.items:
            totals[item] += score
        return max(totals, key=totals.get)

    def winner(self, minimum_hits=1):
        if len(self.items) < minimum_hits:
            return None
        totals = defaultdict(float)
        counts = defaultdict(int)
        for item, score in self.items:
            totals[item] += score
            counts[item] += 1
        winner = max(totals, key=totals.get)
        return winner if counts[winner] >= minimum_hits else None


def open_capture(source, width, height, fps):
    """打开摄像头或视频；USB 摄像头把内部缓冲限制为 1，减少画面延迟。"""
    capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return capture


def process_frame(frame, detector, classifier, votes, tracker, args, frame_index):
    """单帧视觉链路：检测靶标、裁取白色面板、分类、拒识和跨帧投票。"""
    # 检测多个靶标并为跨帧目标分配稳定编号。
    annotated = frame.copy()
    detections = limit_detections(detector.infer(frame), args.max_targets)
    track_ids = tracker.assign(detections)
    for stale_id in set(votes) - tracker.active_ids:
        votes.pop(stale_id, None)
    for track_id, detection in zip(track_ids, detections):
        # 提取每个靶标的中央白色面板并送入含invalid类的分类器。
        preprocess_error = None
        try:
            crop = prepare_pattern_crop(
                frame, detection.box, args.crop_expansion, args.classifier_size
            )
            label, confidence = classifier.infer(crop)
        except ValueError as exc:
            label, confidence = "PREPROCESS_REJECT", 0.0
            preprocess_error = str(exc)
        x1, y1, x2, y2 = map(int, detection.box)
        confirmed = preprocess_error is None and label != "invalid" and confidence >= args.class_conf
        # 拒绝invalid、预处理失败和低置信度结果。
        if confirmed:
            history = votes.setdefault(track_id, LabelVote(args.vote_window))
            history.add(label, confidence)
            minimum_hits = 1 if args._single_image else args.min_class_hits
            voted = history.winner(minimum_hits)
        else:
            voted = None
        accepted = confirmed and voted is not None
        color = (0, 210, 0) if accepted else (0, 180, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        state = voted if accepted else "REJECT"
        text = f"T{track_id} {label} p={confidence:.2f} vote={state}"
        cv2.putText(annotated, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        print(f"frame={frame_index} track={track_id} target={detection.confidence:.3f} class={label} score={confidence:.3f} vote={voted} accepted={accepted} preprocess_error={preprocess_error}")
    return annotated


def run(args):
    """初始化模型并根据输入类型运行静态图片、摄像头或视频检测。"""
    detector = build_target_detector(args)
    classifier = PatternClassifier(args.classifier_model, args.classifier_size)
    votes = {}
    tracker = BoxTrackManager(args.track_max_age)
    source = source_value(args.source)
    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES
    args._single_image = is_image
    calibration = CameraCalibration.load(args.camera_calibration)
    undistorter = CameraUndistorter(calibration) if args.undistort else None
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {source}")
        if undistorter:
            frame = undistorter.apply(frame)
        annotated = process_frame(frame, detector, classifier, votes, tracker, args, 0)
        output = args.output or Path("pattern_result.jpg")
        cv2.imwrite(str(output), annotated)
        print(f"saved={output.resolve()}")
        return 0

    if isinstance(source, int):
        camera = build_live_camera(args, undistorter)
        writer, last_seen, last_inferred = None, -1, -args.infer_every
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Camera backend={camera.backend} {camera.width}x{camera.height} "
            f"@ {camera.fps:.1f} FPS settings={camera.settings}"
        )
        try:
            while True:
                item = camera.read_latest(last_seen)
                if item is None:
                    continue
                frame_index, _, frame, _ = item
                last_seen = frame_index
                if frame_index - last_inferred < args.infer_every:
                    continue
                last_inferred = frame_index
                started = time.perf_counter()
                shown = process_frame(
                    frame, detector, classifier, votes, tracker, args, frame_index
                )
                cv2.putText(
                    shown, f"inference {(time.perf_counter() - started) * 1000:.0f} ms",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (30, 230, 230), 2,
                )
                if args.output:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                            max(1.0, camera.fps / args.infer_every),
                            (shown.shape[1], shown.shape[0]),
                        )
                    writer.write(shown)
                if not args.headless:
                    cv2.imshow("pattern recognition - q to quit", shown)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            camera.close()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        return 0

    capture = open_capture(source, args.width, args.height, args.fps)
    writer, frame_index, last_annotated = None, 0, None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if undistorter:
                frame = undistorter.apply(frame)
            if frame_index % args.infer_every == 0:
                started = time.perf_counter()
                last_annotated = process_frame(frame, detector, classifier, votes, tracker, args, frame_index)
                elapsed = time.perf_counter() - started
                cv2.putText(last_annotated, f"inference {elapsed * 1000:.0f} ms", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 230, 230), 2)
            shown = last_annotated if last_annotated is not None else frame
            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (shown.shape[1], shown.shape[0]))
                writer.write(shown)
            if not args.headless:
                cv2.imshow("pattern recognition - q to quit", shown)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", default="0", help="camera index, image, or video")
    result.add_argument("--target-model", type=Path, default=ROOT / "models/target_detector.onnx")
    result.add_argument("--classifier-model", type=Path, default=ROOT / "models/pattern_classifier.onnx")
    result.add_argument("--target-conf", type=float, default=0.20)
    result.add_argument("--class-conf", type=float, default=0.50)
    result.add_argument("--nms-iou", type=float, default=0.45)
    result.add_argument("--target-size", type=int, default=640)
    result.add_argument(
        "--target-proposal", choices=("color", "color-first", "yolo"),
        default="color",
        help="red/blue color proposals (default), with optional YOLO fallback",
    )
    result.add_argument("--classifier-size", type=int, default=224)
    result.add_argument("--crop-expansion", type=float, default=1.15)
    result.add_argument("--max-targets", type=int, default=0, help="maximum targets per frame; 0 processes all")
    result.add_argument("--track-max-age", type=int, default=5, help="inference passes to retain a missing target track")
    result.add_argument("--vote-window", type=int, default=5)
    result.add_argument(
        "--min-class-hits", type=int, default=3,
        help="matching confident video observations required before acceptance",
    )
    result.add_argument("--infer-every", type=int, default=2)
    result.add_argument("--width", type=int, default=2304)
    result.add_argument("--height", type=int, default=1296)
    result.add_argument("--fps", type=int, default=30)
    result.set_defaults(undistort=False)
    result.add_argument("--undistort", dest="undistort", action="store_true", help="opt in to calibrated full-frame correction")
    result.add_argument("--no-undistort", dest="undistort", action="store_false", help="keep raw frames (default)")
    result.add_argument("--camera-calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    add_camera_arguments(result)
    result.add_argument("--headless", action="store_true")
    result.add_argument("--output", type=Path)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
