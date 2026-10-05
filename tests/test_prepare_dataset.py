import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.prepare_dataset import (
    CLASS_TO_IDX,
    ImageRecord,
    build_record,
    extract_species,
    prepare_output_dir,
    preprocess_image,
    read_image_metadata,
    write_metadata,
)


def make_image(path: Path, size: tuple[int, int] = (80, 40), mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, size, color=(120, 80, 40))
    image.save(path)
    return path


def test_extract_species_from_dataset_filename() -> None:
    assert extract_species(Path("Agaricus_bisporus_000003_1.jpg")) == "Agaricus_bisporus"
    assert extract_species(Path("Tuber_melanosporum_000020_0.png")) == "Tuber_melanosporum"
    assert extract_species(Path("unknown_name.jpg")) == "unknown_name"


def test_read_image_metadata_reads_valid_image(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "sample.png", size=(64, 32))

    width, height, mode, image_format = read_image_metadata(image_path)

    assert width == 64
    assert height == 32
    assert mode == "RGB"
    assert image_format == "PNG"


def test_preprocess_image_outputs_rgb_jpeg_square(tmp_path: Path) -> None:
    source_path = make_image(tmp_path / "raw" / "mushroom.png", size=(100, 50))
    output_path = tmp_path / "processed" / "mushroom.jpg"

    preprocess_image(
        source_path=source_path,
        output_path=output_path,
        image_size=32,
        resize_mode="pad",
        jpeg_quality=90,
    )

    with Image.open(output_path) as image:
        assert image.size == (32, 32)
        assert image.mode == "RGB"
        assert image.format == "JPEG"


def test_prepare_output_dir_refuses_existing_directory_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        prepare_output_dir(output_dir, overwrite=False, dry_run=False)


def test_prepare_output_dir_creates_split_class_directories(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"

    prepare_output_dir(output_dir, overwrite=False, dry_run=False)

    for split in ("train", "val", "test"):
        for label in CLASS_TO_IDX:
            assert (output_dir / split / label).is_dir()


def test_build_record_maps_poisonous_to_non_edible(tmp_path: Path) -> None:
    source_path = make_image(tmp_path / "Amanita_muscaria_000001_0.jpg")
    output_dir = tmp_path / "processed"

    record = build_record(
        split="train",
        source_label="poisonous",
        source_path=source_path,
        output_dir=output_dir,
        image_index=7,
    )

    assert record.label == "non_edible"
    assert record.label_id == CLASS_TO_IDX["non_edible"]
    assert record.species == "Amanita_muscaria"
    assert record.processed_path.endswith("train\\non_edible\\non_edible_000007.jpg") or (
        record.processed_path.endswith("train/non_edible/non_edible_000007.jpg")
    )


def test_write_metadata_counts_classes_and_cross_split_duplicates(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    output_dir.mkdir()

    records = [
        ImageRecord(
            "train",
            "edible",
            "edible",
            0,
            "A",
            "train_a.jpg",
            "out_1.jpg",
            10,
            10,
            "RGB",
            "JPEG",
            "hash_a",
        ),
        ImageRecord(
            "val",
            "edible",
            "edible",
            0,
            "A",
            "val_a.jpg",
            "out_2.jpg",
            10,
            10,
            "RGB",
            "JPEG",
            "hash_a",
        ),
        ImageRecord(
            "test",
            "poisonous",
            "non_edible",
            1,
            "B",
            "test_b.jpg",
            "out_3.jpg",
            10,
            10,
            "RGB",
            "JPEG",
            "hash_b",
        ),
    ]

    write_metadata(
        output_dir=output_dir,
        raw_dir=raw_dir,
        records=records,
        image_size=224,
        resize_mode="pad",
        jpeg_quality=95,
    )

    metadata = json.loads((output_dir / "dataset_metadata.json").read_text(encoding="utf-8"))

    assert metadata["total_images"] == 3
    assert metadata["class_counts"] == {"edible": 2, "non_edible": 1}
    assert metadata["duplicate_hash_count"] == 1
    assert metadata["cross_split_duplicate_hash_count"] == 1
