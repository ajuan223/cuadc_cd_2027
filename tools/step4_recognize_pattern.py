"""V1 step 4: detect a pattern target and classify its central panel.

Pipeline:
    image -> target_detector -> padded target crop
          -> pattern geometry -> central panel
          -> pattern_model -> class + confidence

The digit model is not loaded in this step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
PATTERN_DIR = ROOT / "pattern_recognition"
sys.path.insert(0, str(PATTERN_DIR))

from app import PatternClassifier, prepare_pattern_crop  # noqa: E402
from detector import TargetDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "testdata" / "pattern_yunshuji_frame.jpg",
        help="input image",
    )
    parser.add_argument(
        "--target-model",
        type=Path,
        default=PATTERN_DIR / "models" / "target_detector.onnx",
        help="shared target detector model",
    )
    parser.add_argument(
        "--pattern-model",
        type=Path,
        default=PATTERN_DIR / "models" / "pattern_classifier.onnx",
        help="independent pattern classification model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "step4_pattern",
        help="directory for extracted pattern panels",
    )
    parser.add_argument("--target-confidence", type=float, default=0.4)
    parser.add_argument("--class-confidence", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--classifier-size", type=int, default=224)
    parser.add_argument("--max-targets", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"could not read image: {args.image}")

    detector = TargetDetector(
        args.target_model,
        confidence=args.target_confidence,
        input_size=args.input_size,
    )
    classifier = PatternClassifier(args.pattern_model, input_size=args.classifier_size)
    detections = detector.infer(image)[: max(0, args.max_targets)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"image={args.image.resolve()}")
    print(f"image_size={image.shape[1]}x{image.shape[0]}")
    print(f"targets={len(detections)}")

    accepted = []
    for index, detection in enumerate(detections, start=1):
        try:
            panel = prepare_pattern_crop(
                image,
                detection.box,
                expansion=1.0 + args.padding,
                output_size=args.classifier_size,
            )
        except ValueError as exc:
            print(
                f"target_{index}: detection={detection.confidence:.4f} "
                f"preprocess_reject={exc}"
            )
            continue

        label, score = classifier.infer(panel)
        is_accepted = label != "invalid" and score >= args.class_confidence
        panel_path = args.output_dir / f"pattern_panel_{index:03d}.jpg"
        cv2.imwrite(str(panel_path), panel)
        if is_accepted:
            accepted.append((label, score, panel_path))

        print(
            f"target_{index}: detection={detection.confidence:.4f} "
            f"class={label} score={score:.4f} accepted={is_accepted} "
            f"panel={panel_path.resolve()}"
        )

    if accepted:
        label, score, panel_path = max(accepted, key=lambda item: item[1])
        print(f"result: class={label} score={score:.4f} panel={panel_path}")
        return 0

    print("result: no pattern target accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())