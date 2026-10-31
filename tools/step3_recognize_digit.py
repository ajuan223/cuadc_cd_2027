"""V1 step 3: detect a target and recognize its two-digit number.

Pipeline:
    image -> target_detector -> padded target crop
          -> digit geometry -> digit strip -> digit_model -> number

The pattern model is not loaded in this step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DIGIT_DIR = ROOT / "digit_recognition"
sys.path.insert(0, str(DIGIT_DIR))

from app import DigitOCR  # noqa: E402
from detector import TargetDetector, padded_crop  # noqa: E402
from digit_geometry import extract_digit_strips  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "testdata" / "digit_17_frame.jpg",
        help="input image",
    )
    parser.add_argument(
        "--target-model",
        type=Path,
        default=DIGIT_DIR / "models" / "target_detector.onnx",
        help="shared target detector model",
    )
    parser.add_argument(
        "--digit-model",
        type=Path,
        default=DIGIT_DIR / "models" / "ocr_rec.onnx",
        help="digit recognition model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "step3_digit",
        help="directory for intermediate and result images",
    )
    parser.add_argument("--target-confidence", type=float, default=0.5)
    parser.add_argument("--ocr-confidence", type=float, default=0.65)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--input-size", type=int, default=640)
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
    detections = detector.infer(image)[: max(0, args.max_targets)]
    if not detections:
        print(f"image={args.image.resolve()}")
        print("targets=0")
        return 1

    ocr = DigitOCR(args.digit_model, mode="full", threads=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"image={args.image.resolve()}")
    print(f"image_size={image.shape[1]}x{image.shape[0]}")
    print(f"targets={len(detections)}")

    for index, detection in enumerate(detections, start=1):
        target_crop = padded_crop(image, detection.box, args.padding)
        target_path = args.output_dir / f"target_{index:03d}.jpg"
        cv2.imwrite(str(target_path), target_crop)

        diagnostics = {}
        strips = extract_digit_strips(
            target_crop,
            diagnostics,
            early_reject=True,
        )
        print(
            f"target_{index}: detection={detection.confidence:.4f} "
            f"crop={target_crop.shape[1]}x{target_crop.shape[0]} "
            f"geometry={diagnostics.get('geometry_path')} "
            f"strips={len(strips)} "
            f"rejection={diagnostics.get('rejection_reason')}"
        )

        if not strips:
            print("result: no digit strip extracted")
            continue

        strip_path = args.output_dir / f"digit_strip_{index:03d}.jpg"
        cv2.imwrite(str(strip_path), strips[0])

        result = ocr.recognize(strips)
        number = result["number"]
        score = float(result["score"])
        accepted = number is not None and score >= args.ocr_confidence
        print(
            f"result: number={number} score={score:.4f} "
            f"accepted={accepted} raw={result['raw']} "
            f"strip={strip_path.resolve()}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
