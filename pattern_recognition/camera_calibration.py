"""Load and scale the camera model shared by recognition and localization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_PATH = ROOT / "camera_calibration.json"


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics at one reference resolution.

    The deployed IMX708 profile deliberately uses zero distortion because the
    available near-field board did not constrain the high-order coefficients
    safely at the image corners.
    """

    camera: str
    model_name: str
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray
    source_path: Path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "CameraCalibration":
        source = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
        payload = json.loads(source.read_text(encoding="utf-8"))
        width = int(payload["image_width"])
        height = int(payload["image_height"])
        matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(payload.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
        if width <= 0 or height <= 0 or matrix.shape != (3, 3):
            raise ValueError(f"invalid camera calibration dimensions in {source}")
        if distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError(f"invalid distortion vector in {source}")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
            raise ValueError(f"non-finite camera calibration in {source}")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"camera focal lengths must be positive in {source}")
        return cls(
            str(payload.get("camera", "unknown")),
            str(payload.get("model_name", source.stem)),
            width,
            height,
            matrix,
            distortion,
            source.resolve(),
        )

    def matrix_for_size(self, width: int, height: int) -> np.ndarray:
        """Scale fx/fy/cx/cy for an ISP resize that preserves the full FoV."""
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        matrix = self.matrix.copy()
        matrix[0, :] *= float(width) / self.width
        matrix[1, :] *= float(height) / self.height
        matrix[2, :] = (0.0, 0.0, 1.0)
        return matrix

    @property
    def uses_distortion(self) -> bool:
        return bool(np.any(np.abs(self.distortion) > 1e-12))

    def as_dict(self) -> dict:
        return {
            "camera": self.camera,
            "model_name": self.model_name,
            "reference_size": [self.width, self.height],
            "camera_matrix": self.matrix.tolist(),
            "distortion": self.distortion.tolist(),
            "uses_distortion": self.uses_distortion,
            "source": str(self.source_path),
        }


class CameraUndistorter:
    """Build and cache a full-frame remap for an explicitly enabled model."""

    def __init__(self, calibration: CameraCalibration | None = None) -> None:
        self.calibration = calibration or CameraCalibration.load()
        self.maps = None
        self.size = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if not self.calibration.uses_distortion:
            return frame
        if self.size != (width, height):
            matrix = self.calibration.matrix_for_size(width, height)
            self.maps = cv2.initUndistortRectifyMap(
                matrix, self.calibration.distortion, None, matrix,
                (width, height), cv2.CV_16SC2,
            )
            self.size = (width, height)
        return cv2.remap(
            frame, self.maps[0], self.maps[1], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

