from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.image_io import load_image


def test_load_image_returns_bgr_array(tmp_path: Path) -> None:
    expected = np.zeros((12, 18, 3), dtype=np.uint8)
    expected[2:7, 4:11] = (10, 120, 240)
    image_path = tmp_path / "sample.png"
    assert cv2.imwrite(str(image_path), expected)

    loaded = load_image(image_path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, expected)


def test_load_image_accepts_string_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    assert cv2.imwrite(str(image_path), np.zeros((5, 7, 3), dtype=np.uint8))

    loaded = load_image(str(image_path))

    assert loaded.shape == (5, 7, 3)


def test_load_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_image(tmp_path / "missing.jpg")


def test_load_image_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="not a file"):
        load_image(tmp_path)


def test_load_image_rejects_undecodable_file(tmp_path: Path) -> None:
    image_path = tmp_path / "not-an-image.jpg"
    image_path.write_text("this is not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="could not decode"):
        load_image(image_path)