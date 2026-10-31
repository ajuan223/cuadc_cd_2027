"""Check the isolated training and inference environment for the two checkpoints."""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def load_pattern_classifier(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes, checkpoint["size"]


def main() -> int:
    detector = YOLO(ROOT / "part.pt")
    detector_result = detector.predict(
        source=torch.zeros(640, 640, 3, dtype=torch.uint8).numpy(),
        imgsz=640,
        device="cpu",
        verbose=False,
    )[0]
    classifier, classes, image_size = load_pattern_classifier(ROOT / "best.pt")
    with torch.inference_mode():
        classifier_result = classifier(torch.zeros(1, 3, image_size, image_size))

    print(f"python={__import__('sys').executable}")
    print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"part.pt task={detector.task} names={detector.names} boxes={len(detector_result.boxes)}")
    print(f"best.pt architecture=mobilenet_v3_small classes={len(classes)} output={tuple(classifier_result.shape)}")
    print("TRAINING_ENVIRONMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
