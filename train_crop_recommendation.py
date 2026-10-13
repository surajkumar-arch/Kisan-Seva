"""Precision Crop Recommendation Model Training Pipeline.

Trains a Deep Neural Network classifier (and optional Ensemble) on soil nutrient
and meteorological data (N, P, K, pH, Temperature, Humidity, Rainfall).
By default this combines the original benchmark with the added dataset after its
rainfall column has been estimated by `prepare_crop_recommendation_data.py`.
Run: python train_crop_recommendation.py
"""

import argparse
import json
import logging
import sys
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
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
DEFAULT_ADDITIONAL_CSV_PATH = BASE_DIR / "data" / "crop" / "Crop_recommendation_with_estimated_rainfall.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "crop"

STANDARD_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Crop Recommendation Model")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH), help="Path to crop recommendation CSV")
    parser.add_argument(
        "--additional-csv-path",
        type=str,
        default=str(DEFAULT_ADDITIONAL_CSV_PATH),
        help="Second dataset, prepared with rainfall (default: added dataset with estimated rainfall)",
    )
    parser.add_argument(
        "--no-additional-data",
        action="store_true",
        help="Train only from --csv-path instead of combining both datasets",
    )
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
        elif c_clean in ["ph", "soil_ph", "soil ph"]:
            col_map[col] = "ph"
        elif c_clean in ["rainfall", "rain"]:
            col_map[col] = "rainfall"
        elif c_clean in ["label", "crop", "crops", "crop_name", "target"]:
            label_col = col

    df = df.rename(columns=col_map)
    if label_col is None:
        # Find the original label column after feature aliases were renamed.
        candidates = [column for column in df.columns if column not in STANDARD_COLUMNS]
        if not candidates:
            raise ValueError("Could not identify a crop label column")
        label_col = candidates[-1]
    elif label_col in col_map:
        label_col = col_map[label_col]

    return df, label_col


def canonicalize_crop_label(value: object) -> str:
    """Normalize spelling variants shared by the two source datasets."""
    label = str(value).strip().casefold()
    aliases = {
        "kidneybeans": "Kidney Beans",
        "pigeonpeas": "Pigeon Peas",
        "redgram": "Pigeon Peas",
        "mungbean": "Mung Bean",
        "greengram": "Mung Bean",
        "bengalgram": "Chickpea",
        "soyabean": "Soybean",
    }
    return aliases.get(label, label.title())


def load_and_combine_datasets(args) -> Tuple[pd.DataFrame, List[Dict[str, object]], Dict[str, object]]:
    """Load source files, keep only the seven requested features plus the label."""
    csv_paths = [Path(args.csv_path)]
    if not args.no_additional_data:
        csv_paths.append(Path(args.additional_csv_path))

    frames = []
    sources = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        raw = pd.read_csv(path)
        normalized, label_col = normalize_column_names(raw)
        required = STANDARD_COLUMNS + [label_col]
        missing = [column for column in required if column not in normalized.columns]
        if missing:
            raise ValueError(
                f"{path} is missing required columns {missing}; available columns: {list(normalized.columns)}"
            )

        # Discard unrelated columns (including TYPE_OF_CROP, soil, season, etc.).
        selected = normalized[required].copy().rename(columns={label_col: "label"})
        selected = selected.dropna(subset=STANDARD_COLUMNS + ["label"])
        selected["label"] = selected["label"].map(canonicalize_crop_label)
        selected = selected[selected["label"].str.len() > 0]
        frames.append(selected)
        sources.append({"path": str(path.resolve()), "rows_used": int(len(selected))})
        logger.info("Loaded %s: %s usable rows", path, f"{len(selected):,}")

    combined = pd.concat(frames, ignore_index=True)
    # Numeric conversion catches malformed cells before they can poison scaling.
    for column in STANDARD_COLUMNS:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.dropna(subset=STANDARD_COLUMNS + ["label"])
    combined = combined[np.isfinite(combined[STANDARD_COLUMNS]).all(axis=1)]
    if combined.empty:
        raise ValueError("No usable rows remain after cleaning the supplied datasets")

    rainfall_metadata = {}
    additional_path = Path(args.additional_csv_path)
    sidecar = additional_path.with_suffix(".metadata.json")
    if not args.no_additional_data and sidecar.exists():
        try:
            rainfall_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            logger.warning("Could not read rainfall provenance from %s: %s", sidecar, ex)
    return combined, sources, rainfall_metadata


def compute_standard_scaling(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # prevent division by zero
    scaled_X = (X - mean) / std
    scaler_params = {"mean": mean.tolist(), "std": std.tolist()}
    return scaled_X, scaler_params


def train_models(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df, data_sources, rainfall_metadata = load_and_combine_datasets(args)
    except (OSError, ValueError, pd.errors.ParserError) as ex:
        logger.error("Unable to load training datasets: %s", ex)
        logger.info("Expected columns: N, P, K, temperature, humidity, ph, rainfall, label")
        if not args.no_additional_data:
            logger.info("For the added dataset, run prepare_crop_recommendation_data.py first to create its rainfall column.")
        return False

    logger.info("Combined dataset verified: %s rows from %s file(s).", f"{len(df):,}", len(data_sources))
    unique_crops = sorted(df["label"].unique().tolist())
    logger.info(f"Found {len(unique_crops)} distinct crops: {unique_crops}")

    if args.verify:
        logger.info("Source row counts: %s", data_sources)
        logger.info("Features used: %s", STANDARD_COLUMNS)
        logger.info("Verification complete: both datasets are valid and only the requested columns will be used.")
        return True

    # Encode labels
    crop_to_idx = {crop: i for i, crop in enumerate(unique_crops)}
    idx_to_crop = {i: crop for i, crop in enumerate(unique_crops)}
    y_raw = df["label"].map(crop_to_idx).values

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
    best_weights_path = output_dir / "crop_model.pth"

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

    # Restore the selected neural checkpoint before evaluating or ensembling it.
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        neural_probabilities = torch.softmax(
            model(torch.tensor(X_test, dtype=torch.float32, device=device)), dim=1
        ).cpu().numpy()
    neural_test_acc = float(accuracy_score(y_test, neural_probabilities.argmax(axis=1))) if SKLEARN_AVAILABLE else float(best_acc)

    rf_acc = None
    ensemble_acc = None
    rf_model_path = output_dir / "crop_random_forest.joblib"
    ensemble_meta_path = output_dir / "crop_ensemble.json"
    if SKLEARN_AVAILABLE:
        try:
            logger.info("Training Random Forest and probability-averaged NN + RF ensemble...")
            rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
            rf.fit(X_train, y_train)
            joblib.dump(rf, rf_model_path)
            rf_probabilities = rf.predict_proba(X_test)
            rf_acc = float(accuracy_score(y_test, rf_probabilities.argmax(axis=1)))
            ensemble_probabilities = 0.5 * neural_probabilities + 0.5 * rf_probabilities
            ensemble_acc = float(accuracy_score(y_test, ensemble_probabilities.argmax(axis=1)))
            logger.info("Neural Network Test Accuracy: %.2f%%", neural_test_acc * 100)
            logger.info("Random Forest Test Accuracy: %.2f%%", rf_acc * 100)
            logger.info("NN + RF Soft-Voting Ensemble Test Accuracy: %.2f%%", ensemble_acc * 100)
            with ensemble_meta_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "method": "soft_voting_probability_average",
                        "models": {
                            "neural_network": {"artifact": best_weights_path.name, "weight": 0.5},
                            "random_forest": {"artifact": rf_model_path.name, "weight": 0.5},
                        },
                        "feature_order": STANDARD_COLUMNS,
                        "num_classes": num_classes,
                        "test_accuracy": ensemble_acc,
                        "datasets": data_sources,
                        "rainfall_provenance": rainfall_metadata,
                    },
                    f,
                    indent=2,
                )
        except Exception as ex:
            rf_acc = None
            ensemble_acc = None
            logger.warning(f"Random Forest / ensemble training skipped: {ex}")

    # Export the model metadata as separate files for inference pipelines.
    label_mapping_path = output_dir / "label_mapping.json"
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crop_to_idx": crop_to_idx,
                "idx_to_crop": {str(index): crop for index, crop in idx_to_crop.items()},
                "crops": unique_crops,
                "num_classes": num_classes,
            },
            f,
            indent=2,
        )

    scaler_path = output_dir / "scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_meta, f, indent=2)

    # Keep the combined metadata file for existing project tooling.
    classes_meta_path = output_dir / "crop_classes.json"
    with open(classes_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crops": unique_crops,
                "num_classes": num_classes,
                "scaler": scaler_meta,
                "neural_accuracy": neural_test_acc,
                "random_forest_accuracy": rf_acc,
                "ensemble_accuracy": ensemble_acc,
                "datasets": data_sources,
                "rainfall_provenance": rainfall_metadata,
            },
            f,
            indent=2,
        )

    logger.info("-------------------------------------------------------")
    logger.info(f"Training complete! Best Neural Accuracy: {best_acc * 100:.2f}%")
    logger.info(f"Model weights saved to: {best_weights_path}")
    if rf_acc is not None and ensemble_acc is not None:
        logger.info(f"Random Forest model saved to: {rf_model_path}")
        logger.info(f"Ensemble configuration saved to: {ensemble_meta_path}")
    logger.info(f"Label mapping saved to: {label_mapping_path}")
    logger.info(f"Scaler saved to: {scaler_path}")
    logger.info(f"Combined crop metadata saved to: {classes_meta_path}")
    logger.info("-------------------------------------------------------")
    return True


if __name__ == "__main__":
    cli_args = parse_args()
    success = train_models(cli_args)
    if not success:
        sys.exit(1)
