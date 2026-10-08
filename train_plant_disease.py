"""Plant Disease Model Training & Fine-Tuning Pipeline (MobileNetV3 / ResNet).

Prepares, trains, and exports vision models for foliage disease diagnosis.
Once the user provides their leaf dataset in `data/plant/`, run:
    python train_plant_disease.py --data-dir data/plant --epochs 15 --batch-size 32
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainPlantDisease")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "plant"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "plant_health"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Plant Foliage Disease Classifier")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to plant images dataset")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for weights")
    parser.add_argument("--backbone", type=str, choices=["mobilenet_v3", "resnet50"], default="mobilenet_v3", help="Backbone model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction (if no separate val dir)")
    parser.add_argument("--img-size", type=int, default=224, help="Image input dimension")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Export model to ONNX format")
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


def load_datasets(data_dir: Path, val_split: float, img_size: int, batch_size: int):
    train_transform, val_transform = get_transforms(img_size)

    train_path = data_dir / "train"
    val_path = data_dir / "val"

    if train_path.exists() and val_path.exists():
        logger.info("Found distinct 'train' and 'val' subdirectories.")
        train_dataset = datasets.ImageFolder(str(train_path), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(val_path), transform=val_transform)
        class_names = train_dataset.classes
    else:
        logger.info(f"Loading dataset directly from: {data_dir}")
        full_dataset = datasets.ImageFolder(str(data_dir))
        class_names = full_dataset.classes

        total_samples = len(full_dataset)
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size

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

    num_workers = 2 if sys.platform != "win32" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names


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
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error(f"Dataset directory '{data_dir}' does not exist.")
        logger.info("Please place your leaf disease image folders inside 'data/plant/' e.g.:")
        logger.info("  data/plant/Healthy_Foliage/*.jpg")
        logger.info("  data/plant/Early_Blight/*.jpg")
        logger.info("  data/plant/Late_Blight/*.jpg")
        return False

    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        logger.error(f"No class subfolders found inside '{data_dir}'.")
        logger.info("Expected subdirectories for each disease class.")
        return False

    logger.info(f"Found {len(subdirs)} classes: {[d.name for d in subdirs]}")
    if args.verify:
        logger.info("Verification complete: Dataset directory structure is valid.")
        return True

    train_loader, val_loader, class_names = load_datasets(
        data_dir, args.val_split, args.img_size, args.batch_size
    )
    num_classes = len(class_names)
    logger.info(f"Target classes ({num_classes}): {class_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    model = build_model(args.backbone, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_weights_path = output_dir / "plant_mobilenet_v3.pt"

    logger.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
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
            for images, labels in val_loader:
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

    # Save Class Metadata
    class_meta_path = output_dir / "class_labels.json"
    with open(class_meta_path, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "num_classes": num_classes, "backbone": args.backbone}, f, indent=2)
    logger.info(f"Saved class metadata mapping to: {class_meta_path}")

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
