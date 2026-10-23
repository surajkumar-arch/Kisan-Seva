"""Train an EfficientNet-B0 soil image classifier.

The bundled soil dataset is loaded from `data/soil/Soil dataset/` by default:
    python train_soil.py --verify
    python train_soil.py --epochs 15 --batch-size 32

The dataset contains Alluvial, Black, Clay, and Red Soil and saves to a separate output
directory so it does not overwrite the backend's six-class model.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from src.edaphnet.ml.calibration import TemperatureScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainSoil")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "soil" / "Soil dataset"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "soil_health" / "soil_dataset_experiment"
SOIL_CLASSES = [
    "Alluvial Soil",
    "Black Soil",
    "Clay Soil",
    "Red Soil",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".gif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an EfficientNet-B0 soil classifier")
    parser.add_argument(
        "--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Soil dataset root"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for model artifacts",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of each training class used for validation",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Square model input size")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the stratified train/validation split",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the EfficientNet backbone with ImageNet weights",
    )
    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the best model to ONNX",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validate dataset structure without training",
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Fit confidence calibration for the existing checkpoint without retraining",
    )
    return parser.parse_args()


def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def _split_directory(
    data_dir: Path, preferred_name: str, fallback_name: Optional[str] = None
) -> Path:
    preferred = data_dir / preferred_name
    if preferred.is_dir():
        return preferred
    if fallback_name:
        fallback = data_dir / fallback_name
        if fallback.is_dir():
            return fallback
    raise FileNotFoundError(
        f"Dataset split '{preferred_name}' was not found under '{data_dir}'."
    )


def _discover_classes(train_dir: Path) -> List[str]:
    folders = [
        path for path in train_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    if not folders:
        raise ValueError(f"No soil class folders found in '{train_dir}'.")

    folder_names = {folder.name.casefold(): folder.name for folder in folders}
    unknown_classes = sorted(
        name for name in folder_names.values()
        if name.casefold() not in {soil_class.casefold() for soil_class in SOIL_CLASSES}
    )
    if unknown_classes:
        raise ValueError(f"Unknown soil class folders in '{train_dir}': {unknown_classes}")

    missing_classes = [
        name for name in SOIL_CLASSES if name.casefold() not in folder_names
    ]
    if missing_classes:
        raise ValueError(
            f"Expected all four soil classes in '{train_dir}'; "
            f"missing={missing_classes}."
        )
    return SOIL_CLASSES.copy()


class SoilImageDataset(Dataset):
    """Image dataset that maps case-insensitive folder names to canonical soil labels."""

    def __init__(
        self,
        split_dir: Path,
        class_names: List[str],
        transform=None,
        samples: Optional[List[Tuple[Path, int]]] = None,
    ):
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_counts = {name: 0 for name in class_names}
        class_to_idx = {name.casefold(): index for index, name in enumerate(class_names)}

        if samples is not None:
            self.samples = list(samples)
            for _, label in self.samples:
                self.class_counts[class_names[label]] += 1
            return

        folders = [
            path for path in split_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        found_classes = {folder.name.casefold() for folder in folders}
        unexpected = sorted(found_classes - set(class_to_idx))
        missing = sorted(set(class_to_idx) - found_classes)
        if unexpected or missing:
            raise ValueError(
                f"Soil classes in '{split_dir}' do not match training classes; "
                f"unexpected={unexpected}, missing={missing}."
            )

        for folder in folders:
            class_name = class_names[class_to_idx[folder.name.casefold()]]
            label = class_to_idx[folder.name.casefold()]
            image_paths = sorted(
                path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not image_paths:
                raise ValueError(f"No supported soil images found in '{folder}'.")
            self.samples.extend((path, label) for path in image_paths)
            self.class_counts[class_name] += len(image_paths)

        if not self.samples:
            raise ValueError(f"No supported soil images found in '{split_dir}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_datasets(data_dir: Path, val_split: float, img_size: int, batch_size: int, seed: int):
    if not 0 < val_split < 1:
        raise ValueError(f"Validation split must be between 0 and 1, got {val_split}.")

    train_dir = _split_directory(data_dir, "train_data", "train")
    test_dir = _split_directory(data_dir, "test")
    class_names = _discover_classes(train_dir)
    train_transform, eval_transform = get_transforms(img_size)

    source_dataset = SoilImageDataset(train_dir, class_names)
    test_dataset = SoilImageDataset(test_dir, class_names, eval_transform)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for class_id, class_name in enumerate(class_names):
        class_indices = [
            index for index, (_, label) in enumerate(source_dataset.samples) if label == class_id
        ]
        if len(class_indices) < 2:
            raise ValueError(
                f"Class '{class_name}' needs at least two training images "
                "for a train/validation split."
            )
        rng.shuffle(class_indices)
        val_count = min(len(class_indices) - 1, max(1, round(len(class_indices) * val_split)))
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    train_dataset = SoilImageDataset(
        train_dir,
        class_names,
        train_transform,
        [source_dataset.samples[index] for index in train_indices],
    )
    val_dataset = SoilImageDataset(
        train_dir,
        class_names,
        eval_transform,
        [source_dataset.samples[index] for index in val_indices],
    )

    num_workers = 2 if sys.platform != "win32" else 0
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    try:
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        logger.exception(
            "Could not load pretrained EfficientNet-B0 weights; "
            "using random initialization."
        )
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )
    return model


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += torch.sum(outputs.argmax(dim=1) == labels).item()
            total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total), correct, total


def fit_validation_temperature(model, data_loader, device) -> float:
    """Fit a single temperature on validation logits for calibrated probabilities."""
    model.eval()
    logits_batches = []
    label_batches = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            logits_batches.append(model(images).cpu())
            label_batches.append(labels.cpu())
    if not logits_batches:
        raise ValueError("Cannot calibrate soil confidence: validation split is empty")

    calibrator = TemperatureScaler(initial_temperature=1.0)
    return calibrator.fit(torch.cat(logits_batches), torch.cat(label_batches))


def write_class_metadata(
    output_dir: Path,
    class_names: List[str],
    temperature: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    metadata_path = output_dir / "class_labels.json"
    metadata_path.write_text(
        json.dumps(
            {
                "classes": class_names,
                "num_classes": len(class_names),
                "backbone": "efficientnet_b0",
                "dataset": str(data_dir.resolve()),
                "temperature": temperature,
                "calibration_method": "temperature_scaling_validation_nll",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path


def train_and_evaluate(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not data_dir.is_dir():
        logger.error("Dataset directory '%s' does not exist.", data_dir)
        return False
    if args.epochs <= 0 or args.batch_size <= 0 or args.img_size <= 0 or args.lr <= 0:
        logger.error("Epochs, batch size, image size, and learning rate must all be positive.")
        return False

    train_loader, val_loader, test_loader, class_names = load_datasets(
        data_dir, args.val_split, args.img_size, args.batch_size, args.seed
    )
    logger.info("Soil classes: %s", class_names)
    logger.info(
        "Dataset samples: train=%d, validation=%d, test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    if args.verify:
        logger.info("Dataset verification complete.")
        return True

    best_weights_path = output_dir / "soil_efficientnet_b0.pt"
    if args.calibrate_only:
        if not best_weights_path.is_file():
            logger.error("Cannot calibrate: checkpoint not found at %s", best_weights_path)
            return False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(len(class_names), pretrained=False).to(device)
        model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
        temperature = fit_validation_temperature(model, val_loader, device)
        _, test_accuracy, correct, total = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device
        )
        metadata_path = write_class_metadata(output_dir, class_names, temperature, data_dir)
        logger.info(
            "Existing checkpoint held-out accuracy: %.2f%% (%d/%d).",
            test_accuracy * 100,
            correct,
            total,
        )
        logger.info(
            "Saved validation-fitted temperature %.4f to %s; no model weights were changed.",
            temperature,
            metadata_path,
        )
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Compute device: %s", device)
    model = build_model(len(class_names), args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_acc = -1.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += torch.sum(outputs.argmax(dim=1) == labels).item()
            train_total += labels.size(0)

        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(
            "Epoch [%02d/%02d] Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%%",
            epoch,
            args.epochs,
            train_loss / max(1, train_total),
            train_correct / max(1, train_total) * 100,
            val_loss,
            val_accuracy * 100,
        )
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_weights_path)
            logger.info("Saved best validation checkpoint: %s", best_weights_path)

    elapsed_minutes = (time.time() - start_time) / 60
    logger.info(
        "Training completed in %.2f minutes. Best validation accuracy: %.2f%%",
        elapsed_minutes,
        best_val_acc * 100,
    )
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    temperature = fit_validation_temperature(model, val_loader, device)
    logger.info("Validation-fitted confidence temperature: %.4f", temperature)
    test_loss, test_accuracy, correct, total = evaluate(model, test_loader, criterion, device)
    logger.info(
        "Held-out test Loss: %.4f Acc: %.2f%% (%d/%d)",
        test_loss,
        test_accuracy * 100,
        correct,
        total,
    )

    metadata_path = write_class_metadata(output_dir, class_names, temperature, data_dir)
    logger.info("Saved class metadata: %s", metadata_path)

    if args.export_onnx:
        onnx_path = output_dir / "soil_efficientnet_b0.onnx"
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
                opset_version=14,
                dynamo=False,
            )
            logger.info("Exported ONNX model: %s", onnx_path)
        except Exception:
            logger.exception("ONNX export failed.")
            return False
    return True


if __name__ == "__main__":
    cli_args = parse_args()
    if not train_and_evaluate(cli_args):
        sys.exit(1)
