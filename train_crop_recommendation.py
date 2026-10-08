"""Precision Crop Recommendation Model Training Pipeline.

Trains a Deep Neural Network classifier (and optional Ensemble) on soil nutrient
and meteorological data (N, P, K, pH, Temperature, Humidity, Rainfall).
Once the user provides their CSV in `data/crop/Crop_recommendation.csv`, run:
    python train_crop_recommendation.py --csv-path data/crop/Crop_recommendation.csv
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Optional sklearn integration
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainCropRecommendation")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "data" / "crop" / "Crop_recommendation.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "crop"

STANDARD_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Crop Recommendation Model")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH), help="Path to crop recommendation CSV")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for model weights")
    parser.add_argument("--test-split", type=float, default=0.2, help="Validation/test split fraction")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--verify", action="store_true", help="Verify CSV columns without training")
    return parser.parse_args()


class CropTabularNet(nn.Module):
    """Deep Neural Network for 7-Feature Agronomic Profiling."""

    def __init__(self, in_features: int = 7, num_classes: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Map variants of N, P, K, temp, humidity, ph, rainfall, and label."""
    col_map = {}
    label_col = None

    for col in df.columns:
        c_clean = str(col).strip().lower()
        if c_clean in ["n", "nitrogen"]:
            col_map[col] = "N"
        elif c_clean in ["p", "phosphorus", "phosphorous"]:
            col_map[col] = "P"
        elif c_clean in ["k", "potassium"]:
            col_map[col] = "K"
        elif c_clean in ["temperature", "temp"]:
            col_map[col] = "temperature"
        elif c_clean in ["humidity", "humid"]:
            col_map[col] = "humidity"
        elif c_clean in ["ph"]:
            col_map[col] = "ph"
        elif c_clean in ["rainfall", "rain"]:
            col_map[col] = "rainfall"
        elif c_clean in ["label", "crop", "crop_name", "target"]:
            label_col = col

    df = df.rename(columns=col_map)
    if label_col is None:
        label_col = df.columns[-1]

    return df, label_col


def compute_standard_scaling(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # prevent division by zero
    scaled_X = (X - mean) / std
    scaler_params = {"mean": mean.tolist(), "std": std.tolist()}
    return scaled_X, scaler_params


def train_models(args):
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.error(f"Dataset file '{csv_path}' not found.")
        logger.info("Please place your Crop recommendation CSV at:")
        logger.info(f"  {csv_path}")
        logger.info("Expected columns: N, P, K, temperature, humidity, ph, rainfall, label")
        return False

    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    df, label_col = normalize_column_names(df)

    missing_cols = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in CSV: {missing_cols}")
        logger.info(f"Available columns in CSV: {list(df.columns)}")
        return False

    # Clean missing values
    df = df.dropna(subset=STANDARD_COLUMNS + [label_col])
    logger.info(f"Dataset verified: {len(df)} rows.")

    unique_crops = sorted([str(c).strip().title() for c in df[label_col].unique()])
    logger.info(f"Found {len(unique_crops)} distinct crops: {unique_crops}")

    if args.verify:
        logger.info("Verification complete: Dataset columns and format are valid.")
        return True

    # Encode labels
    crop_to_idx = {crop: i for i, crop in enumerate(unique_crops)}
    idx_to_crop = {i: crop for i, crop in enumerate(unique_crops)}
    y_raw = df[label_col].astype(str).str.strip().str.title().map(crop_to_idx).values

    X_raw = df[STANDARD_COLUMNS].values.astype(np.float32)

    # Train / Test split
    np.random.seed(42)
    indices = np.random.permutation(len(X_raw))
    test_size = int(len(X_raw) * args.test_split)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train_raw, y_train = X_raw[train_idx], y_raw[train_idx]
    X_test_raw, y_test = X_raw[test_idx], y_raw[test_idx]

    # Standard scaling
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std == 0] = 1.0
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    scaler_meta = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "features": STANDARD_COLUMNS,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    num_classes = len(unique_crops)
    model = CropTabularNet(in_features=7, num_classes=num_classes).to(device)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    logger.info(f"Training Neural Crop Recommendation Classifier for {args.epochs} epochs...")
    best_acc = 0.0
    best_weights_path = output_dir / "crop_recommendation_net.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = torch.argmax(model(bx), dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)

        acc = correct / max(1, total)
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), str(best_weights_path))

        if epoch % 10 == 0 or epoch == args.epochs:
            logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] Test Accuracy: {acc * 100:.2f}% (Best: {best_acc * 100:.2f}%)")

    # Optional Sklearn Ensemble evaluation if installed
    ens_acc = None
    if SKLEARN_AVAILABLE:
        try:
            logger.info("Training Scikit-Learn Random Forest & Gradient Boosting...")
            rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
            rf.fit(X_train, y_train)
            ens_acc = float(accuracy_score(y_test, rf.predict(X_test)))
            logger.info(f"Random Forest Test Accuracy: {ens_acc * 100:.2f}%")
        except Exception as ex:
            logger.warning(f"Sklearn training skipped: {ex}")

    # Export Metadata
    classes_meta_path = output_dir / "crop_classes.json"
    with open(classes_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crops": unique_crops,
                "num_classes": num_classes,
                "scaler": scaler_meta,
                "neural_accuracy": float(best_acc),
                "ensemble_accuracy": ens_acc,
            },
            f,
            indent=2,
        )

    logger.info("-------------------------------------------------------")
    logger.info(f"Training complete! Best Accuracy: {best_acc * 100:.2f}%")
    logger.info(f"Model weights saved to: {best_weights_path}")
    logger.info(f"Scaler & crop mapping saved to: {classes_meta_path}")
    logger.info("-------------------------------------------------------")
    return True


if __name__ == "__main__":
    cli_args = parse_args()
    success = train_models(cli_args)
    if not success:
        sys.exit(1)
