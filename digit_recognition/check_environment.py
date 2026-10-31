"""Offline deployment self-check for the digit-recognition task."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify_model_hashes() -> None:
    manifest = ROOT / "models.sha256"
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localized", action="store_true", help="also require pymavlink")
    parser.add_argument("--camera", type=int, help="optionally read one frame from this camera index")
    parser.add_argument("--camera-backend", choices=("picamera2", "opencv"), default="picamera2")
    parser.add_argument("--width", type=int, default=2304)
    parser.add_argument("--height", type=int, default=1296)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        print("FAIL Python 3.10 or newer is required", file=sys.stderr)
        return 1

    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:
        print(f"FAIL core imports: {exc}", file=sys.stderr)
        return 1

    print(
        f"Machine={platform.machine()} Python={sys.version.split()[0]} OpenCV={cv2.__version__} "
        f"NumPy={np.__version__} ONNXRuntime={ort.__version__}"
    )
    target_model = ROOT / "models" / "target_detector.onnx"
    ocr_model = ROOT / "models" / "ocr_rec.onnx"
    try:
        verify_model_hashes()
        if not target_model.is_file():
            raise FileNotFoundError(target_model)
        if not ocr_model.is_file():
            raise FileNotFoundError(ocr_model)
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CPUExecutionProvider is unavailable")
        from app import DigitOCR
        from detector import TargetDetector
        dummy_frame = np.full((640, 640, 3), 114, dtype=np.uint8)
        TargetDetector(target_model).infer(dummy_frame)
        DigitOCR(ocr_model).recognize([
            np.full((96, 192, 3), 255, dtype=np.uint8)
        ])
        print(f"model_forward_ok={target_model.name} bytes={target_model.stat().st_size}")
        print(f"model_forward_ok={ocr_model.name} bytes={ocr_model.stat().st_size}")
        print("model_hashes_ok=models.sha256")
        print(f"onnx_providers={ort.get_available_providers()}")
    except Exception as exc:
        print(f"FAIL model loading: {exc}", file=sys.stderr)
        return 1

    if args.localized:
        try:
            from pymavlink import mavutil
            print(f"pymavlink_ok={Path(mavutil.__file__).resolve()}")
        except Exception as exc:
            print(f"FAIL pymavlink: {exc}", file=sys.stderr)
            return 1

    if args.camera is not None:
        if args.camera_backend == "picamera2":
            try:
                from camera_runtime import LatestFramePicamera2
                camera = LatestFramePicamera2(
                    args.camera, args.width, args.height, args.fps,
                    ae=True, exposure_value=-0.75, awb=True,
                    lens_position=0.0, contrast=1.3,
                )
                item = camera.read_latest(timeout=5.0)
                metadata = {} if item is None else item[3]
                camera.close()
                if item is None:
                    raise RuntimeError("no Picamera2 frame within 5 seconds")
                frame = item[2]
                if frame.shape[1::-1] != (args.width, args.height):
                    raise RuntimeError(f"unexpected frame size {frame.shape[1]}x{frame.shape[0]}")
                if metadata.get("SensorTimestamp") is None:
                    raise RuntimeError("SensorTimestamp missing from Picamera2 metadata")
                print(
                    f"camera_ok=picamera2:{args.camera} size={frame.shape[1]}x{frame.shape[0]} "
                    f"lens={metadata.get('LensPosition')} crop={metadata.get('ScalerCrop')} "
                    f"timestamp={metadata.get('CaptureTimestampBasis')}"
                )
            except Exception as exc:
                print(f"FAIL Picamera2 camera {args.camera}: {exc}", file=sys.stderr)
                return 1
        else:
            capture = cv2.VideoCapture(args.camera)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            capture.set(cv2.CAP_PROP_FPS, args.fps)
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            actual_fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            if not ok or frame is None:
                print(f"FAIL camera {args.camera}: cannot read a frame", file=sys.stderr)
                return 1
            print(f"camera_ok=opencv:{args.camera} size={frame.shape[1]}x{frame.shape[0]} reported_fps={actual_fps:.1f}")

    print("OFFLINE_ENVIRONMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
