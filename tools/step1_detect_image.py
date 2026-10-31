"""V1 step 1: draw target detector results on one image.

This script intentionally does one thing only:
    image -> target_detector -> boxes drawn on image + console output

It does not load the digit model or the pattern model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_DIR = ROOT / "digit_recognition"
sys.path.insert(0, str(DETECTOR_DIR))

from detector import TargetDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "test.jpg",
        help="input image (default: test.jpg)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DETECTOR_DIR / "models" / "target_detector.onnx",
        help="target detector model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "step1_detected.jpg",
        help="output image",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="minimum target confidence, from 0 to 1 (demo default: 0.6)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="square model input size",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
        help="IoU threshold used to remove duplicate boxes",
    )
    return parser.parse_args()


def draw_detections(image, detections):
    annotated = image.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"target {index}: {detection.confidence:.2f}"
        text_x = max(0, x1)
        text_y = max(24, y1 - 8)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        cv2.rectangle(
            annotated,
            (text_x, text_y - text_height - baseline - 4),
            (text_x + text_width + 8, text_y + baseline),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (text_x + 4, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    args = parse_args()
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be greater than 0 and at most 1")
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"could not read image: {args.image}")

    detector = TargetDetector(
        args.model,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        input_size=args.input_size,
    )
    detections = detector.infer(image)
    annotated = draw_detections(image, detections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"could not write output image: {args.output}")

    print(f"image={args.image.resolve()}")
    print(f"image_size={image.shape[1]}x{image.shape[0]}")
    print(f"targets={len(detections)}")
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box
        print(
            f"target_{index}: "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) "
            f"confidence={detection.confidence:.4f}"
        )
    print(f"output={args.output.resolve()}")
    return 0 if detections else 1


if __name__ == "__main__":
    raise SystemExit(main())
