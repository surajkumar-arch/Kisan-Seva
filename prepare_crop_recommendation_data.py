"""Prepare the added Tamil Nadu crop dataset for the 7-feature trainer.

The added dataset has no rainfall measurements. This utility estimates rainfall
from N, P, K, temperature, humidity, and pH using the existing crop CSV, then
writes a derived CSV with the trainer's expected columns. The original data is
left unchanged; a sidecar JSON records that rainfall is estimated.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "data" / "crop" / "Crop recommendation dataset.csv"
DEFAULT_REFERENCE = BASE_DIR / "data" / "crop" / "Crop_recommendation.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "crop" / "Crop_recommendation_with_estimated_rainfall.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Add an estimated rainfall column for crop training")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    new_df = pd.read_csv(args.input)
    ref_df = pd.read_csv(args.reference)

    target_column_map = {
        "TEMP": "temperature",
        "RELATIVE_HUMIDITY": "humidity",
        "SOIL_PH": "ph",
        "CROPS": "label",
    }
    new_df = new_df.rename(columns=target_column_map)

    required_reference = FEATURES + ["rainfall"]
    required_target = FEATURES + ["label"]
    missing_reference = [column for column in required_reference if column not in ref_df.columns]
    missing_target = [column for column in required_target if column not in new_df.columns]
    if missing_reference:
        raise ValueError(f"Reference CSV is missing columns: {missing_reference}")
    if missing_target:
        raise ValueError(f"Input CSV is missing columns: {missing_target}")

    ref_df = ref_df.dropna(subset=required_reference)
    new_df = new_df.dropna(subset=required_target).copy()

    x_train, x_test, y_train, y_test = train_test_split(
        ref_df[FEATURES], ref_df["rainfall"], test_size=0.2, random_state=42
    )
    validation_model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    validation_model.fit(x_train, y_train)
    validation_predictions = validation_model.predict(x_test)
    validation_mae = float(mean_absolute_error(y_test, validation_predictions))
    validation_r2 = float(r2_score(y_test, validation_predictions))

    rainfall_model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rainfall_model.fit(ref_df[FEATURES], ref_df["rainfall"])
    new_df["rainfall"] = rainfall_model.predict(new_df[FEATURES]).round(2)

    ordered_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_df[ordered_columns].to_csv(args.output, index=False)

    out_of_range = {}
    for feature in FEATURES:
        min_value = float(ref_df[feature].min())
        max_value = float(ref_df[feature].max())
        outside = (new_df[feature] < min_value) | (new_df[feature] > max_value)
        out_of_range[feature] = {
            "reference_min": min_value,
            "reference_max": max_value,
            "target_rows_outside_range": int(outside.sum()),
        }

    provenance = {
        "rainfall_status": "estimated, not measured",
        "method": "ExtraTreesRegressor (300 trees, min_samples_leaf=2)",
        "estimator_features": FEATURES,
        "crop_label_used_to_estimate_rainfall": False,
        "reference_csv": str(args.reference.resolve()),
        "reference_rows": int(len(ref_df)),
        "held_out_reference_mae_mm": validation_mae,
        "held_out_reference_r2": validation_r2,
        "target_rows": int(len(new_df)),
        "target_crop_classes": int(new_df["label"].nunique()),
        "target_features_outside_reference_ranges": out_of_range,
        "warning": (
            "The reference data is a small crop-recommendation benchmark and may not "
            "represent Tamil Nadu conditions. Estimated rainfall is not observed weather; "
            "model accuracy trained on this derived column is not evidence of real-world accuracy."
        ),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Rows: {len(new_df):,}; crops: {new_df['label'].nunique()}")
    print(f"Estimated rainfall validation: MAE={validation_mae:.2f} mm, R2={validation_r2:.3f}")
    print(f"Provenance: {metadata_path}")


if __name__ == "__main__":
    main()
