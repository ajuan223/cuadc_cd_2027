"""V1 step 2: detect targets and crop each bounding box.

Pipeline:
    image -> target_detector -> boxes -> OpenCV/NumPy crop -> saved images

No digit model or pattern model is loaded in this step.
"""

from __future__ import annotations

import argparse
import math
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
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "step2_crops",
        help="directory for cropped target images",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="minimum target confidence, from 0 to 1",
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


def clip_box(box, image_width: int, image_height: int):
    """Convert a floating-point box to safe integer image boundaries."""
    x1, y1, x2, y2 = box
    left = max(0, min(image_width - 1, math.floor(x1)))
    top = max(0, min(image_height - 1, math.floor(y1)))
    right = max(0, min(image_width, math.ceil(x2)))
    bottom = max(0, min(image_height, math.ceil(y2)))
    return left, top, right, bottom


def crop_detections(image, detections):
    height, width = image.shape[:2]
    crops = []
    for detection in detections:
        left, top, right, bottom = clip_box(detection.box, width, height)
        if right - left < 2 or bottom - top < 2:
            continue
        crop = image[top:bottom, left:right].copy()
        crops.append((detection, (left, top, right, bottom), crop))
    return crops


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
    crops = crop_detections(image, detections)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"image={args.image.resolve()}")
    print(f"image_size={image.shape[1]}x{image.shape[0]}")
    print(f"targets={len(crops)}")

    for index, (detection, box, crop) in enumerate(crops, start=1):
        output = args.output_dir / f"target_{index:03d}.jpg"
        if not cv2.imwrite(str(output), crop):
            raise RuntimeError(f"could not write crop: {output}")
        left, top, right, bottom = box
        print(
            f"target_{index}: "
            f"box=({left}, {top}, {right}, {bottom}) "
            f"confidence={detection.confidence:.4f} "
            f"crop_size={crop.shape[1]}x{crop.shape[0]} "
            f"output={output.resolve()}"
        )

    return 0 if crops else 1


if __name__ == "__main__":
    raise SystemExit(main())