"""Image loading boundary for the vision pipeline."""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a three-channel BGR NumPy array.

    Args:
        path: Image file path.

    Returns:
        OpenCV image array with shape ``(height, width, 3)`` and uint8 dtype.

    Raises:
        FileNotFoundError: The path does not exist.
        IsADirectoryError: The path exists but is a directory.
        ValueError: OpenCV cannot decode the file as a color image.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    if not image_path.is_file():
        raise IsADirectoryError(f"image path is not a file: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"expected a three-channel BGR image, got shape {image.shape}: {image_path}"
        )
    return image