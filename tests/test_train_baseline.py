from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.training.train_baseline import (  # noqa: E402
    CLASS_NAMES,
    build_dataloaders,
    build_model,
    build_transforms,
    class_weights,
)


def create_image_folder_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (40, 30), color=(80, 120, 160))
            image.save(class_dir / f"{class_name}.jpg")


def test_build_transforms_returns_three_channel_tensor() -> None:
    image = Image.new("RGB", (40, 20), color=(255, 0, 0))
    transform = build_transforms(image_size=32)["val"]

    tensor = transform(image)

    assert tuple(tensor.shape) == (3, 32, 32)


def test_build_model_resnet18_has_binary_classifier() -> None:
    model = build_model(
        model_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
        num_classes=2,
    )

    assert model.fc.out_features == 2


def test_build_model_can_freeze_resnet_backbone() -> None:
    model = build_model(
        model_name="resnet18",
        pretrained=False,
        freeze_backbone=True,
        num_classes=2,
    )

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    assert trainable_names == ["fc.weight", "fc.bias"]


def test_class_weights_are_inverse_to_class_counts() -> None:
    class DummyDataset:
        targets = [0, 0, 0, 1]

    weights = class_weights(DummyDataset(), torch.device("cpu"))

    assert weights.tolist() == pytest.approx([2 / 3, 2.0])


def test_build_dataloaders_uses_expected_class_order(tmp_path: Path) -> None:
    create_image_folder_dataset(tmp_path)

    dataloaders, datasets_by_split = build_dataloaders(
        data_dir=tmp_path,
        image_size=32,
        batch_size=2,
        num_workers=0,
    )

    assert datasets_by_split["train"].classes == CLASS_NAMES
    assert len(dataloaders["train"].dataset) == 2
