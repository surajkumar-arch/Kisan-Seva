"""Model evaluation suite (Phase 8) -- honest edition.

What this module does and does NOT claim
----------------------------------------
* **Held-out evaluation (real accuracy)** only happens when you pass image
  folders (``--soil-dir`` / ``--plant-dir``) laid out as ``<root>/<class_name>/*.jpg``
  using data that was *not* used for training. Accuracy, macro-F1, per-class
  scores and the confusion matrix are then computed from the deployed
  models' real predictions.
* **Smoke tests** (the default, no data needed) only verify that each model
  loads, produces the right number of finite probabilities that sum to 1, and
  report latency. They deliberately report ``accuracy = None``.
* **Crop engine** is checked for *catalog self-consistency*: feeding a crop's
  own optimal conditions should rank that crop highly. This is a rule-engine
  sanity check, not a measure of real-world recommendation accuracy.

Nothing here injects ground truth into predictions or uses random labels.

Usage::

    python -m backend.app.evaluation.evaluate_models \
        --soil-dir path/to/soil_test --plant-dir path/to/plant_test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.models.crop_net import CROP_CATALOG
from backend.app.models.fusion_net import get_fusion_model
from backend.app.models.plant_net import get_plant_class_names, get_plant_model, is_plant_model_trained
from backend.app.models.soil_net import get_soil_class_names, get_soil_model, is_soil_model_trained
from backend.app.utils.image_utils import preprocess_image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_REPORTS_DIR = BASE_DIR / "data" / "evaluations"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PredictFn = Callable[[Image.Image], int]


# --------------------------------------------------------------------------
# Metrics core (pure functions, unit-testable)
# --------------------------------------------------------------------------
def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> Dict[str, Any]:
    """Accuracy, macro-F1, per-class P/R/F1/support and confusion matrix.

    Classes with no support are excluded from macro-F1 (they have no ground
    truth to be right or wrong about) instead of being scored as 1.0.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    f1_with_support: List[float] = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1_score": round(f1, 4), "support": support,
        }
        if support:
            f1_with_support.append(f1)

    return {
        "overall_accuracy": round(float((yt == yp).mean()), 4) if len(yt) else None,
        "macro_f1_score": round(float(np.mean(f1_with_support)), 4) if f1_with_support else None,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": int(len(yt)),
    }


def evaluate_image_folder(
    predict_fn: PredictFn,
    root: Path,
    class_names: Sequence[str],
    max_per_class: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ``predict_fn`` over ``root/<class>/*.img`` and score against folder names."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation folder not found: {root}")
    index = {name: i for i, name in enumerate(class_names)}
    y_true: List[int] = []
    y_pred: List[int] = []
    latencies: List[float] = []
    unmatched: List[str] = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name not in index:
            unmatched.append(folder.name)
            continue
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            img = Image.open(f).convert("RGB")
            t0 = time.perf_counter()
            pred = int(predict_fn(img))
            latencies.append((time.perf_counter() - t0) * 1000)
            y_true.append(index[folder.name])
            y_pred.append(pred)

    if not y_true:
        raise ValueError(
            f"No images matched class folders in {root}. Folders must be named exactly like the "
            f"model's classes. Unmatched folders: {unmatched[:10]}"
        )
    out = classification_metrics(y_true, y_pred, class_names)
    out.update(
        {
            "evaluation_type": "held_out_images",
            "data_dir": str(root),
            "unmatched_folders": unmatched,
            "avg_inference_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        }
    )
    return out


def _make_predict_fn(model: torch.nn.Module) -> PredictFn:
    model.eval()

    def predict(img: Image.Image) -> int:
        arr = preprocess_image(img, target_size=(224, 224), normalize_mode="imagenet")
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            return int(torch.argmax(model(tensor), dim=1).item())

    return predict


def _smoke_test(model: torch.nn.Module, class_names: Sequence[str], trained: bool, trials: int = 8) -> Dict[str, Any]:
    """Forward-pass sanity check on noise images. Reports NO accuracy."""
    model.eval()
    rng = np.random.default_rng(0)
    latencies: List[float] = []
    ok = True
    for _ in range(trials):
        noise = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        arr = preprocess_image(noise, target_size=(224, 224), normalize_mode="imagenet")
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).squeeze(0).numpy()
        latencies.append((time.perf_counter() - t0) * 1000)
        ok &= bool(
            probs.shape == (len(class_names),) and np.isfinite(probs).all() and abs(float(probs.sum()) - 1.0) < 1e-3
        )
    return {
        "evaluation_type": "smoke_test_no_accuracy",
        "note": "Accuracy not measured. Pass --soil-dir/--plant-dir with held-out images to measure it.",
        "forward_pass_ok": bool(ok),
        "trained_weights_loaded": bool(trained),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "overall_accuracy": None,
        "macro_f1_score": None,
        "per_class_metrics": {},
        "confusion_matrix": None,
        "total_test_samples": 0,
        "avg_inference_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
    }


# --------------------------------------------------------------------------
# Per-model evaluations
# --------------------------------------------------------------------------
def evaluate_soil_model(data_dir: Optional[str] = None, max_per_class: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate the deployed soil classifier (real data if ``data_dir`` is given)."""
    model = get_soil_model()
    names = get_soil_class_names()
    base = {"model_architecture": "EfficientNet-B0 (Transfer Learning)", "num_classes": len(names), "class_names": names}
    if data_dir:
        res = evaluate_image_folder(_make_predict_fn(model), Path(data_dir), names, max_per_class)
        return {**base, **res, "trained_weights_loaded": bool(is_soil_model_trained())}
    return {**base, **_smoke_test(model, names, is_soil_model_trained())}


def evaluate_plant_model(data_dir: Optional[str] = None, max_per_class: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate the deployed plant-disease classifier (real data if ``data_dir`` is given)."""
    model = get_plant_model()
    names = get_plant_class_names()
    base = {"model_architecture": "MobileNetV3-Large (Edge-Optimized)", "num_classes": len(names), "class_names": names}
    if data_dir:
        res = evaluate_image_folder(_make_predict_fn(model), Path(data_dir), names, max_per_class)
        return {**base, **res, "trained_weights_loaded": bool(is_plant_model_trained())}
    return {**base, **_smoke_test(model, names, is_plant_model_trained())}


def evaluate_fusion_model(num_eval_trials: int = 50) -> Dict[str, Any]:
    """Check latent-space invariants of the gated fusion network on random inputs."""
    model = get_fusion_model()
    model.eval()
    torch.manual_seed(0)

    latencies, latent_norms, gating_entropies, fvi_scores = [], [], [], []
    for _ in range(num_eval_trials):
        soil_vec = torch.rand(1, 10, dtype=torch.float32)
        plant_vec = torch.rand(1, 8, dtype=torch.float32)
        climate_vec = torch.rand(1, 3, dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            latent_vec, fvi_score, _stress, modality_weights = model(soil_vec, plant_vec, climate_vec)
        latencies.append((time.perf_counter() - t0) * 1000)

        latent_np = latent_vec.squeeze(0).numpy()
        assert latent_np.shape == (128,), f"Expected shape (128,), got {latent_np.shape}"
        assert np.all(latent_np >= -1.0) and np.all(latent_np <= 1.0), "Latent vector exceeded tanh [-1, 1] bounds"

        latent_norms.append(float(np.linalg.norm(latent_np)))
        w = modality_weights.squeeze(0).numpy()
        gating_entropies.append(float(-np.sum(w * np.log(w + 1e-9))))
        fvi_scores.append(int(round(fvi_score.item())))

    return {
        "model_architecture": "Gated Multi-Modal Fusion Net",
        "evaluation_type": "invariant_check_on_random_inputs",
        "latent_dimension": 128,
        "total_eval_trials": num_eval_trials,
        "avg_inference_latency_ms": round(float(np.mean(latencies)), 3),
        "mean_latent_l2_norm": round(float(np.mean(latent_norms)), 3),
        "mean_gating_entropy": round(float(np.mean(gating_entropies)), 4),
        "fvi_range": [min(fvi_scores), max(fvi_scores)],
        "latent_boundedness": f"[-1.0, 1.0] (asserted on all {num_eval_trials} trials)",
        "modality_weights_sum": "softmax normalised",
    }


def evaluate_crop_model() -> Dict[str, Any]:
    """Catalog self-consistency of the rule-based crop engine (NOT real-world accuracy)."""
    from backend.app.services.crop_service import recommend_crops_direct

    def mid(rng):
        return (float(rng[0]) + float(rng[1])) / 2.0

    ranks: List[int] = []
    latencies: List[float] = []
    for crop, prof in CROP_CATALOG.items():
        t0 = time.perf_counter()
        res = recommend_crops_direct(
            ph=mid(prof["optimal_ph"]), nitrogen=mid(prof["optimal_n"]),
            phosphorus=mid(prof["optimal_p"]), potassium=mid(prof["optimal_k"]),
            temperature=mid(prof["optimal_temp"]), humidity=65.0,
            rainfall=mid(prof["optimal_rainfall"]), soil_type=prof["preferred_soils"][0],
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        order = [c["crop_name"] for c in res["all_rankings"]]
        ranks.append(order.index(crop) + 1)

    n = len(ranks)
    return {
        "model_architecture": "Rule-based agronomic suitability engine (catalog envelopes)",
        "evaluation_type": "catalog_self_consistency",
        "note": "Feeds each crop its own optimal conditions. Measures rule consistency, not field accuracy.",
        "num_commercial_crops": n,
        "top_1_accuracy": round(sum(r == 1 for r in ranks) / n, 4),
        "top_3_accuracy": round(sum(r <= 3 for r in ranks) / n, 4),
        "mean_rank": round(float(np.mean(ranks)), 2),
        "avg_inference_latency_ms": round(float(np.mean(latencies)), 3),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_full_system_evaluation(
    soil_dir: Optional[str] = None,
    plant_dir: Optional[str] = None,
    max_per_class: Optional[int] = None,
    write_report: bool = True,
) -> Dict[str, Any]:
    """Run the evaluation suite and (optionally) persist ``evaluation_report.json``."""
    start = time.time()
    soil_eval = evaluate_soil_model(soil_dir, max_per_class)
    plant_eval = evaluate_plant_model(plant_dir, max_per_class)
    fusion_eval = evaluate_fusion_model()
    crop_eval = evaluate_crop_model()

    measured = bool(soil_dir or plant_dir)
    report = {
        "system": "KisanSeva Smart Agriculture System",
        "evaluation_phase": "PHASE 8: Testing, Evaluation & Deployment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_evaluation_runtime_seconds": round(time.time() - start, 2),
        "executive_summary": {
            "evaluation_mode": "held_out_images" if measured else "smoke_test_only",
            "soil_classification_accuracy": soil_eval["overall_accuracy"],
            "plant_disease_accuracy": plant_eval["overall_accuracy"],
            "crop_recommendation_top3_accuracy": crop_eval["top_3_accuracy"],
            "crop_metric_meaning": "catalog self-consistency, not field accuracy",
            "end_to_end_ml_latency_ms": round(
                soil_eval["avg_inference_latency_ms"] + plant_eval["avg_inference_latency_ms"]
                + fusion_eval["avg_inference_latency_ms"] + crop_eval["avg_inference_latency_ms"], 2),
            "production_readiness": "NOT_ASSESSED",
            "readiness_note": (
                "Readiness requires accuracy measured on held-out real images; "
                "run with --soil-dir/--plant-dir."
            ),
        },
        "phase2_soil_evaluation": soil_eval,
        "phase3_plant_evaluation": plant_eval,
        "phase4_fusion_evaluation": fusion_eval,
        "phase5_crop_evaluation": crop_eval,
    }
    if write_report:
        os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)
        with open(EVAL_REPORTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def _fmt(x: Optional[float]) -> str:
    return "n/a (not measured)" if x is None else f"{x * 100:.1f}%"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="KisanSeva model evaluation")
    ap.add_argument("--soil-dir", help="held-out soil images: <dir>/<class>/*.jpg")
    ap.add_argument("--plant-dir", help="held-out plant images: <dir>/<class>/*.jpg")
    ap.add_argument("--max-per-class", type=int, default=None)
    a = ap.parse_args()
    rep = run_full_system_evaluation(a.soil_dir, a.plant_dir, a.max_per_class)
    s = rep["executive_summary"]
    print("=" * 60)
    print("KISANSEVA MODEL EVALUATION")
    print("=" * 60)
    print(f"Mode:              {s['evaluation_mode']}")
    print(f"Soil Accuracy:     {_fmt(s['soil_classification_accuracy'])}")
    print(f"Plant Accuracy:    {_fmt(s['plant_disease_accuracy'])}")
    print(f"Crop Top-3 (self-consistency): {_fmt(s['crop_recommendation_top3_accuracy'])}")
    print(f"Total ML Latency:  {s['end_to_end_ml_latency_ms']} ms")
    print(f"Readiness:         {s['production_readiness']}")
    print("=" * 60)
