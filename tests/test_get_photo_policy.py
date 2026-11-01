from pathlib import Path


def test_get_photo_module_has_no_image_persistence_calls() -> None:
    source = (Path(__file__).resolve().parents[1] / "get_photo.py").read_text(encoding="utf-8")

    forbidden = (
        "cv2.imwrite",
        "cv2.VideoWriter",
        "Image.save",
        ".tofile(",
    )
    assert all(item not in source for item in forbidden)