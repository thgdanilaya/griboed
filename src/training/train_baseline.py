from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "baseline"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
CLASS_NAMES = ["edible", "non_edible"]
POSITIVE_CLASS = 1


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline CNN for mushroom edibility classification."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--model-name", choices=("resnet18", "mobilenet_v3_small"), default="resnet18"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
        help="Stop training after this many epochs without validation F1 improvement.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="Reduce learning rate after this many epochs without validation F1 improvement.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained ImageNet weights if they are available locally.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the final classification layer.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, DataLoader], dict[str, datasets.ImageFolder]]:
    transform_by_split = build_transforms(image_size)
    datasets_by_split = {
        split: datasets.ImageFolder(data_dir / split, transform=transform_by_split[split])
        for split in ("train", "val", "test")
    }

    if datasets_by_split["train"].classes != CLASS_NAMES:
        raise ValueError(
            f"Unexpected class order: {datasets_by_split['train'].classes}. Expected: {CLASS_NAMES}"
        )

    dataloaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return dataloaders, datasets_by_split


def build_model(
    model_name: str,
    pretrained: bool,
    freeze_backbone: bool,
    num_classes: int = 2,
) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    labels = np.array(dataset.targets)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return running_loss / len(loader.dataset), accuracy_score(targets, predictions)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    running_loss = 0.0
    probabilities: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, POSITIVE_CLASS]

        running_loss += loss.item() * inputs.size(0)
        probabilities.extend(probs.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    try:
        roc_auc = roc_auc_score(targets, probabilities)
    except ValueError:
        roc_auc = None

    return {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(
            targets, predictions, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "recall": recall_score(targets, predictions, pos_label=POSITIVE_CLASS, zero_division=0),
        "f1": f1_score(targets, predictions, pos_label=POSITIVE_CLASS, zero_division=0),
        "roc_auc": roc_auc,
        "targets": targets,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def save_history(report_dir: Path, history: list[EpochMetrics]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    history_path = report_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(asdict(row))


def plot_history(report_dir: Path, history: list[EpochMetrics]) -> None:
    epochs = [item.epoch for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item.train_loss for item in history], label="train")
    axes[0].plot(epochs, [item.val_loss for item in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [item.train_accuracy for item in history], label="train")
    axes[1].plot(epochs, [item.val_accuracy for item in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(report_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrix(report_dir: Path, targets: list[int], predictions: list[int]) -> None:
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_metrics(
    report_dir: Path,
    model_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    dataset_sizes: dict[str, int],
    history: list[EpochMetrics],
    val_metrics: dict[str, object],
    test_metrics: dict[str, object],
    elapsed_seconds: float,
) -> None:
    serializable_val = {
        key: value
        for key, value in val_metrics.items()
        if key not in {"targets", "predictions", "probabilities"}
    }
    serializable_test = {
        key: value
        for key, value in test_metrics.items()
        if key not in {"targets", "predictions", "probabilities"}
    }
    payload = {
        "model_name": args.model_name,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "data_dir": str(args.data_dir.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "classes": CLASS_NAMES,
        "positive_class": "non_edible",
        "dataset_sizes": dataset_sizes,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "early_stopping_patience": args.early_stopping_patience,
            "lr_patience": args.lr_patience,
            "seed": args.seed,
        },
        "best_val_f1": max(item.val_f1 for item in history),
        "best_epoch": max(history, key=lambda item: item.val_f1).epoch,
        "last_val_metrics": serializable_val,
        "test_metrics": serializable_test,
        "elapsed_seconds": elapsed_seconds,
    }

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    report_text = classification_report(
        test_metrics["targets"],
        test_metrics["predictions"],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    with (report_dir / "classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(report_text)


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.report_dir = args.report_dir.resolve()
    args.model_dir = args.model_dir.resolve()

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {args.data_dir}. Run src/data/prepare_dataset.py first."
        )

    set_seed(args.seed)
    device = get_device()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    dataloaders, datasets_by_split = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dataset_sizes = {split: len(dataset) for split, dataset in datasets_by_split.items()}

    model = build_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(datasets_by_split["train"], device))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-6,
    )

    start_time = perf_counter()
    history: list[EpochMetrics] = []
    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Dataset sizes: {dataset_sizes}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=float(val_metrics["loss"]),
            val_accuracy=float(val_metrics["accuracy"]),
            val_precision=float(val_metrics["precision"]),
            val_recall=float(val_metrics["recall"]),
            val_f1=float(val_metrics["f1"]),
        )
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={epoch_metrics.val_loss:.4f} "
            f"val_acc={epoch_metrics.val_accuracy:.4f} "
            f"val_recall_non_edible={epoch_metrics.val_recall:.4f} "
            f"val_f1={epoch_metrics.val_f1:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if epoch_metrics.val_f1 > best_val_f1:
            best_val_f1 = epoch_metrics.val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        scheduler.step(epoch_metrics.val_f1)

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Early stopping: "
                f"validation F1 did not improve for {args.early_stopping_patience} epochs."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, dataloaders["val"], criterion, device)
    test_metrics = evaluate(model, dataloaders["test"], criterion, device)
    elapsed_seconds = perf_counter() - start_time

    model_path = args.model_dir / f"baseline_{args.model_name}.pt"
    torch.save(
        {
            "model_name": args.model_name,
            "class_names": CLASS_NAMES,
            "image_size": args.image_size,
            "state_dict": model.state_dict(),
        },
        model_path,
    )

    save_history(args.report_dir, history)
    plot_history(args.report_dir, history)
    plot_confusion_matrix(
        args.report_dir,
        targets=test_metrics["targets"],
        predictions=test_metrics["predictions"],
    )
    save_metrics(
        report_dir=args.report_dir,
        model_path=model_path,
        args=args,
        device=device,
        dataset_sizes=dataset_sizes,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        elapsed_seconds=elapsed_seconds,
    )

    print("Training complete.")
    print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall_non_edible={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']}"
    )
    print(f"Model saved to: {model_path}")
    print(f"Reports saved to: {args.report_dir}")


if __name__ == "__main__":
    main()
