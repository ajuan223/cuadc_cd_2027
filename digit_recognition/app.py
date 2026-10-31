"""数字任务纯视觉入口：靶标检测、方向旋正、数字 OCR 与跨帧投票。"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from detector import TargetDetector, padded_crop
from color_target import ColorFirstDetector, ColorTargetDetector
from digit_geometry import extract_digit_strips, split_strip
from target_tracking import BoxTrackManager, limit_detections
from camera_calibration import CameraCalibration, CameraUndistorter, DEFAULT_CALIBRATION_PATH
from camera_runtime import add_camera_arguments, build_live_camera


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def clean_digits(text: str, length: int):
    digits = "".join(re.findall(r"\d", text))
    return digits if len(digits) == length else None


def text_score(result):
    return str(result["rec_text"]), float(result["rec_score"])


def choose(whole: dict, split: dict) -> dict:
    """在整条两位数 OCR 与左右两个单字 OCR 之间选择更可靠的结果。"""
    if whole["number"] is None:
        return split if split["number"] is not None else max((whole, split), key=lambda item: item["score"])
    if split["number"] is None:
        return whole
    if whole["number"] == split["number"]:
        return max((whole, split), key=lambda item: item["score"])
    if whole["number"][0] != split["number"][0]:
        return whole
    fixes_one = whole["number"][1] == "0" and split["number"][1] == "1"
    return split if split["score"] > whole["score"] or (fixes_one and split["score"] >= 0.75) else whole


class DigitOCR:
    """使用 ONNX Runtime 在 CPU 上执行数字识别模型。"""
    def __init__(self, model_path: Path, *, mode="full", fast_conf=0.90, threads=2) -> None:
        if mode not in ("full", "adaptive") or not 0 < fast_conf <= 1 or threads < 1:
            raise ValueError("invalid OCR mode, confidence or thread count")
        self.mode, self.fast_conf = mode, fast_conf
        self.last_stats = {}
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Follow README.md or run with --no-ocr."
            ) from exc
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def from_args(cls, args):
        return cls(args.ocr_model, mode=args.ocr_mode,
                   fast_conf=max(args.ocr_fast_conf, args.ocr_conf), threads=args.ocr_threads)

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        resized_width = min(3200, max(320, int(np.ceil(48 * width / max(height, 1)))))
        content_width = min(resized_width, int(np.ceil(48 * width / max(height, 1))))
        resized = cv2.resize(image, (max(1, content_width), 48)).astype(np.float32)
        resized = (resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        output = np.zeros((3, 48, resized_width), dtype=np.float32)
        output[:, :, :content_width] = resized
        return output

    @staticmethod
    def _decode(prediction: np.ndarray) -> dict:
        indices = prediction.argmax(axis=-1)
        probabilities = prediction.max(axis=-1)
        selected = [
            index for index, value in enumerate(indices)
            if value != 0 and (index == 0 or value != indices[index - 1])
        ]
        digit_positions = [index for index in selected if 1 <= indices[index] <= 10]
        text = "".join(str(int(indices[index]) - 1) for index in digit_positions)
        score = float(np.mean([probabilities[index] for index in digit_positions])) if digit_positions else 0.0
        return {"rec_text": text, "rec_score": score}

    def _predict(self, inputs):
        prepared = [self._prepare(image) for image in inputs]
        max_width = max(item.shape[2] for item in prepared)
        batch = np.zeros((len(prepared), 3, 48, max_width), dtype=np.float32)
        for index, item in enumerate(prepared):
            batch[index, :, :, :item.shape[2]] = item
        predictions = self.session.run(None, {self.input_name: batch})[0]
        self.last_stats["model_calls"] += 1
        self.last_stats["input_images"] += len(inputs)
        return [self._decode(prediction) for prediction in predictions]

    def recognize(self, strips: list[np.ndarray]):
        if not strips:
            raise ValueError("OCR requires at least one strip")
        self.last_stats = {"model_calls": 0, "input_images": 0, "path": "full"}
        if self.mode == "adaptive" and len(strips) == 1:
            whole_result = self._predict([strips[0]])[0]
            whole_text, whole_score = text_score(whole_result)
            number = clean_digits(whole_text, 2)
            # 尾数为0时始终用拆分识别器复核0和1混淆。
            if number is not None and whole_score >= self.fast_conf and not number.endswith("0"):
                self.last_stats["path"] = "whole_fast"
                return {"number": number, "score": whole_score, "raw": whole_text,
                        "kind": "whole", "ocr_stats": dict(self.last_stats)}
            self.last_stats["path"] = "split_fallback"
            results = [whole_result] + self._predict(list(split_strip(strips[0])))
            groups = [(0, 1, 2)]
        else:
            # 多个候选数字条继续使用完整交叉识别。
            inputs, groups = [], []
            for strip in strips:
                left, right = split_strip(strip)
                groups.append((len(inputs), len(inputs) + 1, len(inputs) + 2))
                inputs.extend((strip, left, right))
            results = self._predict(inputs)
        choices = []
        for whole_index, left_index, right_index in groups:
            whole_text, whole_score = text_score(results[whole_index])
            left_text, left_score = text_score(results[left_index])
            right_text, right_score = text_score(results[right_index])
            whole = {
                "number": clean_digits(whole_text, 2), "score": whole_score,
                "raw": whole_text, "kind": "whole",
            }
            left_digit, right_digit = clean_digits(left_text, 1), clean_digits(right_text, 1)
            split = {
                "number": left_digit + right_digit if left_digit and right_digit else None,
                "score": (left_score + right_score) / 2, "raw": f"{left_text}|{right_text}",
                "kind": "split",
            }
            choices.append(choose(whole, split))
        valid = [choice for choice in choices if choice["number"] is not None]
        choice = max(valid if valid else choices, key=lambda item: item["score"])
        return {**choice, "ocr_stats": dict(self.last_stats)}


def add_ocr_arguments(result):
    result.add_argument("--ocr-mode", choices=("full", "adaptive"), default="full",
                        help="full cross-check (default), or experimental whole-first OCR")
    result.add_argument("--ocr-fast-conf", type=float, default=0.90,
                        help="adaptive whole-only confidence; trailing zero still cross-checks")
    result.add_argument("--ocr-threads", type=int, choices=range(1, 5), default=2)
    result.add_argument("--no-early-reject", action="store_true",
                        help="disable the pre-warp structure gate for A/B validation")


class WeightedVote:
    """按 OCR 置信度进行滑动窗口投票，抑制单帧跳变。"""
    def __init__(self, window: int) -> None:
        self.items = deque(maxlen=window)

    def add(self, number: str | None, score: float):
        if number:
            self.items.append((number, score))
        totals = defaultdict(float)
        for value, weight in self.items:
            totals[value] += weight
        return max(totals, key=totals.get) if totals else None


def open_capture(source, width: int, height: int, fps: int):
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


def process_frame(frame, detector, ocr, votes, tracker, args, strip_directory, frame_index):
    """单帧视觉链路：检测靶标、旋正并裁数字、OCR、置信度门控和跨帧投票。"""
    # 检测多个靶标并为跨帧目标分配稳定编号。
    annotated = frame.copy()
    detections = limit_detections(detector.infer(frame), args.max_targets)
    track_ids = tracker.assign(detections)
    for stale_id in set(votes) - tracker.active_ids:
        votes.pop(stale_id, None)
    for track_id, detection in zip(track_ids, detections):
        # 用颜色和五边形几何提取并旋正中央数字面板。
        crop = padded_crop(frame, detection.box, args.padding)
        geometry_info = {}
        strips = extract_digit_strips(crop, geometry_info,
                                      early_reject=not getattr(args, "no_early_reject", False))
        number, score, raw = None, 0.0, "no strip"
        ocr_stats = {}
        if strips:
            if strip_directory is not None:
                cv2.imwrite(str(strip_directory / f"frame_{frame_index:06d}_track_{track_id}.jpg"), strips[0])
            if ocr is not None:
                result = ocr.recognize(strips)
                number, score, raw = result["number"], result["score"], result["raw"]
                ocr_stats = result.get("ocr_stats", {})
            else:
                raw = "rectified"
        accepted = number is not None and score >= args.ocr_conf
        # 只让超过OCR阈值的数字参加加权投票。
        voted = votes.setdefault(track_id, WeightedVote(args.vote_window)).add(
            number if accepted else None, score
        )
        x1, y1, x2, y2 = map(int, detection.box)
        color = (0, 210, 0) if accepted else (0, 180, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        labels = [
            f"T{track_id} num={number or '--'} vote={voted or '--'}",
            f"det={detection.confidence:.2f} OCR={score:.2f} "
            + ("OK" if accepted else "CHECK" if ocr is not None else "OFF"),
        ]
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.58
        label_width = max(cv2.getTextSize(text, font, scale, 1)[0][0] for text in labels) + 12
        label_x = max(0, min(x1, annotated.shape[1] - label_width))
        label_y = max(0, min(y1 - 52, annotated.shape[0] - 52))
        cv2.rectangle(annotated, (label_x, label_y), (label_x + label_width, label_y + 52), (25, 25, 25), -1)
        for row, text in enumerate(labels):
            cv2.putText(annotated, text, (label_x + 6, label_y + 20 + row * 24), font, scale, color, 1, cv2.LINE_AA)
        print(f"frame={frame_index} track={track_id} target={detection.confidence:.3f} number={number} score={score:.3f} accepted={accepted} vote={voted} raw={raw} rejection={geometry_info.get('rejection_reason')} ocr_path={ocr_stats.get('path')} ocr_inputs={ocr_stats.get('input_images', 0)}")
    return annotated


def run(args) -> int:
    """初始化模型并根据输入类型运行静态图片、摄像头或视频检测。"""
    detector = build_target_detector(args)
    ocr = None if args.no_ocr else DigitOCR.from_args(args)
    votes = {}
    tracker = BoxTrackManager(args.track_max_age)
    strip_directory = args.save_strips
    if strip_directory:
        strip_directory.mkdir(parents=True, exist_ok=True)
    source = source_value(args.source)
    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES
    calibration = CameraCalibration.load(args.camera_calibration)
    undistorter = CameraUndistorter(calibration) if args.undistort else None
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {source}")
        if undistorter is not None:
            frame = undistorter.apply(frame)
        annotated = process_frame(frame, detector, ocr, votes, tracker, args, strip_directory, 0)
        output = args.output or Path("digit_result.jpg")
        cv2.imwrite(str(output), annotated)
        print(f"saved={output.resolve()}")
        return 0

    if isinstance(source, int):
        # IMX708共用只保留最新帧的低延迟采集接口。
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
                    frame, detector, ocr, votes, tracker, args,
                    strip_directory, frame_index,
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
                    cv2.imshow("digit recognition - q to quit", shown)
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
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = float(args.fps)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    window = "digit recognition - Space: pause | Q/Esc: quit"
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window, 1280, 720)
        while True:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            # 在视觉推理前校正刚采集的原始帧。
            if undistorter is not None:
                frame = undistorter.apply(frame)
            if frame_index % args.infer_every == 0:
                started = time.perf_counter()
                last_annotated = process_frame(frame, detector, ocr, votes, tracker, args, strip_directory, frame_index)
                elapsed = time.perf_counter() - started
                cv2.putText(last_annotated, f"inference {elapsed * 1000:.0f} ms", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 230, 230), 2)
            shown = last_annotated.copy() if last_annotated is not None else frame.copy()
            progress = f"Frame {frame_index + 1}/{total_frames or '?'}  {frame_index / source_fps:.1f}s  source {source_fps:.1f} FPS"
            cv2.putText(shown, progress, (12, shown.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 230, 230), 2, cv2.LINE_AA)
            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (shown.shape[1], shown.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {args.output}")
                writer.write(shown)
            if not args.headless:
                cv2.imshow(window, shown)
                delay = max(1, round((1 / source_fps - (time.perf_counter() - frame_started)) * 1000))
                key = cv2.waitKey(delay) & 0xFF
                if key == ord(" "):
                    while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
                        key = cv2.waitKey(50) & 0xFF
                        if key in (ord(" "), ord("q"), 27):
                            break
                if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
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
    result.add_argument("--ocr-model", type=Path, default=ROOT / "models/ocr_rec.onnx")
    result.add_argument("--ocr-conf", type=float, default=0.65, help="minimum OCR confidence accepted for voting and output")
    add_ocr_arguments(result)
    result.add_argument("--target-conf", type=float, default=0.4)
    result.add_argument("--nms-iou", type=float, default=0.45)
    result.add_argument("--target-size", type=int, default=640)
    result.add_argument(
        "--target-proposal", choices=("color", "color-first", "yolo"),
        default="color",
        help="HSV only (default), HSV with YOLO fallback, or YOLO only",
    )
    result.add_argument("--padding", type=float, default=0.15)
    result.add_argument("--max-targets", type=int, default=0, help="maximum targets per frame; 0 processes all")
    result.add_argument("--track-max-age", type=int, default=5, help="inference passes to retain a missing target track")
    result.add_argument("--vote-window", type=int, default=5)
    result.add_argument("--infer-every", type=int, default=2, help="infer every N frames; use 1 for frame-accurate video boxes")
    result.add_argument("--width", type=int, default=2304)
    result.add_argument("--height", type=int, default=1296)
    result.add_argument("--fps", type=int, default=30)
    result.add_argument("--headless", action="store_true")
    result.set_defaults(undistort=False)
    result.add_argument("--undistort", dest="undistort", action="store_true", help="opt in to calibrated full-frame correction")
    result.add_argument("--no-undistort", dest="undistort", action="store_false", help="keep raw frames (default)")
    result.add_argument("--camera-calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    add_camera_arguments(result)
    result.add_argument("--no-ocr", action="store_true", help="test detection and strip extraction only")
    result.add_argument("--save-strips", type=Path)
    result.add_argument("--output", type=Path)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
