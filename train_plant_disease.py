
"""Plant Disease Model Training & Fine-Tuning Pipeline (MobileNetV3 / ResNet).

Prepares, trains, and exports vision models for foliage disease diagnosis.
The bundled Multi-Crop Disease Dataset is read from `data/plant/` by default:
    python train_plant_disease.py --verify
    python train_plant_disease.py --epochs 15 --batch-size 32
"""

import argparse
import ast
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainPlantDisease")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "plant" / "Multi-Crop Disease Dataset"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "plant_health" / "multi_crop_30_class"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Plant Foliage Disease Classifier")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to plant images dataset")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for weights")
    parser.add_argument("--backbone", type=str, choices=["mobilenet_v3", "resnet50"], default="mobilenet_v3", help="Backbone model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split for ImageFolder datasets")
    parser.add_argument("--img-size", type=int, default=224, help="Image input dimension")
    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export model to ONNX format",
    )
    parser.add_argument("--verify", action="store_true", help="Verify dataset structure without training")
    return parser.parse_args()


def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


class YOLOCropDataset(Dataset):
    """Classification samples made from the object boxes in a YOLO detection split."""

    def __init__(self, images_dir: Path, labels_dir: Path, class_names: List[str], transform):
        self.transform = transform
        self.samples: List[Tuple[Path, int, Tuple[float, float, float, float]]] = []
        self.skipped_images = 0
        class_counts: Counter = Counter()

        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected YOLO split directories '{images_dir}' and '{labels_dir}'."
            )

        image_paths = sorted(
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(f"No supported images found in '{images_dir}'.")

        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                self.skipped_images += 1
                continue

            with label_path.open("r", encoding="utf-8") as label_file:
                for line_number, line in enumerate(label_file, start=1):
                    fields = line.split()
                    if not fields:
                        continue
                    if len(fields) != 5:
                        raise ValueError(
                            f"Expected 5 YOLO fields in '{label_path}' line {line_number}, "
                            f"found {len(fields)}."
                        )

                    try:
                        class_id = int(fields[0])
                        x_center, y_center, width, height = map(float, fields[1:])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid YOLO annotation in '{label_path}' line {line_number}."
                        ) from exc

                    if not 0 <= class_id < len(class_names):
                        raise ValueError(
                            f"Class id {class_id} in '{label_path}' line {line_number} "
                            f"is outside the configured class range."
                        )
                    box_values = (x_center, y_center, width, height)
                    if (
                        not all(math.isfinite(value) for value in box_values)
                        or not 0 <= x_center <= 1
                        or not 0 <= y_center <= 1
                        or not 0 < width <= 1
                        or not 0 < height <= 1
                    ):
                        raise ValueError(
                            f"Invalid normalized bounding box in '{label_path}' "
                            f"line {line_number}: {line.strip()}"
                        )

                    self.samples.append((image_path, class_id, box_values))
                    class_counts[class_id] += 1

        if not self.samples:
            raise ValueError(f"No labeled bounding boxes found in '{labels_dir}'.")
        self.class_counts = class_counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_id, box = self.samples[index]
        x_center, y_center, width, height = box

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image_width, image_height = image.size
        left = max(0, math.floor((x_center - width / 2) * image_width))
        top = max(0, math.floor((y_center - height / 2) * image_height))
        right = min(image_width, math.ceil((x_center + width / 2) * image_width))
        bottom = min(image_height, math.ceil((y_center + height / 2) * image_height))
        if right <= left or bottom <= top:
            raise ValueError(f"YOLO bounding box in '{image_path}' has no visible area.")

        sample = image.crop((left, top, right, bottom))
        if self.transform:
            sample = self.transform(sample)
        return sample, class_id


def load_class_names(data_yaml: Path) -> List[str]:
    """Read the dataset's inline `names` list without requiring a YAML dependency."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: '{data_yaml}'.")

    content = data_yaml.read_text(encoding="utf-8")
    names_match = re.search(r"^\s*names\s*:\s*(.+?)\s*$", content, re.MULTILINE)
    if not names_match:
        raise ValueError(f"No inline 'names' list was found in '{data_yaml}'.")

    try:
        parsed_names = ast.literal_eval(names_match.group(1))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse class names in '{data_yaml}'.") from exc

    if isinstance(parsed_names, dict):
        try:
            sorted_names = sorted(parsed_names.items(), key=lambda item: int(item[0]))
            class_names = [str(name) for _, name in sorted_names]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Class-name mapping in '{data_yaml}' must use integer keys.") from exc
    elif isinstance(parsed_names, (list, tuple)):
        class_names = [str(name).strip() for name in parsed_names]
    else:
        raise ValueError(f"The 'names' entry in '{data_yaml}' must be a list or mapping.")

    if not class_names or any(not name for name in class_names):
        raise ValueError(f"The 'names' entry in '{data_yaml}' contains no usable class names.")

    count_match = re.search(r"^\s*nc\s*:\s*(\d+)\s*$", content, re.MULTILINE)
    if count_match and int(count_match.group(1)) != len(class_names):
        raise ValueError(
            f"Configured class count ({count_match.group(1)}) does not match "
            f"the {len(class_names)} class names in '{data_yaml}'."
        )
    return class_names


def _make_loaders(train_dataset, val_dataset, test_dataset, class_names, batch_size):
    num_workers = 2 if sys.platform != "win32" else 0
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = (
        DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        if test_dataset is not None else None
    )
    return train_loader, val_loader, test_loader, class_names


def load_datasets(data_dir: Path, val_split: float, img_size: int, batch_size: int):
    train_transform, val_transform = get_transforms(img_size)

    train_images = data_dir / "train" / "images"
    train_labels = data_dir / "train" / "labels"
    val_images = data_dir / "valid" / "images"
    val_labels = data_dir / "valid" / "labels"
    is_yolo_dataset = any(
        path.exists() for path in (train_images, train_labels, val_images, val_labels)
    )
    if is_yolo_dataset:
        if not train_images.is_dir() or not val_images.is_dir():
            raise FileNotFoundError(
                f"YOLO datasets require both '{train_images}' and '{val_images}'."
            )
        class_names = load_class_names(data_dir / "data.yaml")
        logger.info("Loading YOLO-annotated crops from train/valid/test splits.")
        train_dataset = YOLOCropDataset(
            train_images, train_labels, class_names, train_transform
        )
        val_dataset = YOLOCropDataset(val_images, val_labels, class_names, val_transform)
        test_images = data_dir / "test" / "images"
        test_labels = data_dir / "test" / "labels"
        if test_images.exists() != test_labels.exists():
            raise FileNotFoundError(
                f"YOLO test split must provide both '{test_images}' and '{test_labels}'."
            )
        test_dataset = YOLOCropDataset(
            test_images, test_labels, class_names, val_transform
        ) if test_images.is_dir() else None
        logger.info(
            "YOLO samples: train=%d, valid=%d, test=%s",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset) if test_dataset is not None else "unavailable",
        )
        for split_name, split_dataset in (
            ("train", train_dataset),
            ("valid", val_dataset),
            ("test", test_dataset),
        ):
            if split_dataset is None:
                continue
            missing_classes = [
                class_names[class_id]
                for class_id in range(len(class_names))
                if split_dataset.class_counts[class_id] == 0
            ]
            if missing_classes:
                logger.warning("%s split has no annotations for: %s", split_name, missing_classes)
        if train_dataset.skipped_images or val_dataset.skipped_images:
            logger.warning(
                "Skipped images without annotation files: train=%d, valid=%d",
                train_dataset.skipped_images,
                val_dataset.skipped_images,
            )
        return _make_loaders(
            train_dataset, val_dataset, test_dataset, class_names, batch_size
        )

    train_path = data_dir / "train"
    val_path = data_dir / "val"
    if train_path.exists() and val_path.exists():
        logger.info("Found distinct 'train' and 'val' subdirectories.")
        train_dataset = datasets.ImageFolder(str(train_path), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(val_path), transform=val_transform)
        class_names = train_dataset.classes
        if val_dataset.classes != class_names:
            raise ValueError(
                "The ImageFolder training and validation splits have different class mappings."
            )
        test_dataset = None
    else:
        logger.info(f"Loading dataset directly from: {data_dir}")
        full_dataset = datasets.ImageFolder(str(data_dir))
        class_names = full_dataset.classes

        total_samples = len(full_dataset)
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size

        if train_size == 0 or val_size == 0:
            raise ValueError(
                f"ImageFolder dataset needs at least 2 samples and a valid validation split; "
                f"got {total_samples} samples and val_split={val_split}."
            )
        logger.info(f"Splitting {total_samples} samples: {train_size} training, {val_size} validation.")
        train_subset, val_subset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        class TransformedDataset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                x, y = self.subset[idx]
                return self.transform(x), y

        train_dataset = TransformedDataset(train_subset, train_transform)
        val_dataset = TransformedDataset(val_subset, val_transform)
        test_dataset = None

    return _make_loaders(train_dataset, val_dataset, test_dataset, class_names, batch_size)


def build_model(backbone_name: str, num_classes: int) -> nn.Module:
    if backbone_name == "mobilenet_v3":
        logger.info("Initializing MobileNetV3-Large with pretrained ImageNet weights...")
        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights=weights)
        except Exception:
            model = models.mobilenet_v3_large(weights=None)

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )
    elif backbone_name == "resnet50":
        logger.info("Initializing ResNet50 with pretrained ImageNet weights...")
        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        except Exception:
            model = models.resnet50(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return model


def train_and_evaluate(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Dataset directory '{data_dir}' does not exist.")
        logger.info(
            "Provide the YOLO dataset root containing data.yaml and train/valid/test "
            "image/label folders, or an ImageFolder dataset."
        )
        return False

    if args.epochs <= 0 or args.batch_size <= 0 or args.img_size <= 0 or args.lr <= 0:
        logger.error("Epochs, batch size, image size, and learning rate must all be positive.")
        return False
    if not 0 < args.val_split < 1:
        logger.error("The ImageFolder validation split must be between 0 and 1.")
        return False

    train_loader, val_loader, test_loader, class_names = load_datasets(
        data_dir, args.val_split, args.img_size, args.batch_size
    )
    num_classes = len(class_names)
    logger.info(f"Target classes ({num_classes}): {class_names}")
    if args.verify:
        logger.info(
            "Verification complete: %d training samples and %d validation samples are ready.",
            len(train_loader.dataset),
            len(val_loader.dataset),
        )
        if test_loader is not None:
            logger.info("Test split contains %d samples.", len(test_loader.dataset))
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    model = build_model(args.backbone, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_weights_path = output_dir / f"plant_{args.backbone}.pt"

    logger.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [Train]",
            leave=False
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{args.epochs} [Val]",
                leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)
        scheduler.step()

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), str(best_weights_path))
            logger.info(f"  --> Saved new best checkpoint to: {best_weights_path} (Val Acc: {best_val_acc * 100:.2f}%)")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.2f} minutes. Best Val Accuracy: {best_val_acc * 100:.2f}%")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))

    # Save Class Metadata
    class_meta_path = output_dir / "class_labels.json"
    with open(class_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": class_names,
                "num_classes": num_classes,
                "backbone": args.backbone,
                "dataset": str(data_dir.resolve()),
            },
            f,
            indent=2,
        )
    logger.info(f"Saved class metadata mapping to: {class_meta_path}")

    if test_loader is not None:
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct_test += torch.sum(predictions == labels).item()
                total_test += labels.size(0)
        test_accuracy = correct_test / max(1, total_test)
        logger.info(
            "Held-out test accuracy: %.2f%% (%d/%d)",
            test_accuracy * 100,
            correct_test,
            total_test,
        )

    # Export to ONNX for mobile / edge inference
    if args.export_onnx:
        try:
            onnx_path = output_dir / "plant_disease_model.onnx"
            model.eval()
            dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
                opset_version=14,
            )
            logger.info(f"Successfully exported ONNX model to: {onnx_path}")
        except Exception as exc:
            logger.warning(f"ONNX export skipped: {exc}")

    return True


if __name__ == "__main__":
    cli_args = parse_args()
    success = train_and_evaluate(cli_args)
    if not success:
        sys.exit(1)

