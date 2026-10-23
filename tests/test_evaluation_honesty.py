"""Evaluation must be honest: metrics come only from real predictions."""

import numpy as np
import pytest
from PIL import Image

from backend.app.evaluation import evaluate_models as ev


def test_metrics_perfect_and_imperfect():
    names = ["a", "b", "c"]
    perfect = ev.classification_metrics([0, 1, 2, 2], [0, 1, 2, 2], names)
    assert perfect["overall_accuracy"] == 1.0 and perfect["macro_f1_score"] == 1.0

    mixed = ev.classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert mixed["overall_accuracy"] == 0.75
    assert mixed["confusion_matrix"] == [[1, 1], [0, 2]]
    assert mixed["per_class_metrics"]["a"]["recall"] == 0.5
    assert mixed["per_class_metrics"]["b"]["precision"] == round(2 / 3, 4)


def test_macro_f1_ignores_classes_without_support():
    res = ev.classification_metrics([0, 0], [0, 0], ["a", "b", "never_seen"])
    assert res["macro_f1_score"] == 1.0  # unsupported class is not scored as 1.0 or 0.0
    assert res["per_class_metrics"]["never_seen"]["support"] == 0


def _make_folder(root, classes, per_class=3):
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (32, 32), (10 * i, 20, 30)).save(d / f"{i}.png")


def test_folder_eval_uses_predictor_not_labels(tmp_path):
    classes = ["Alluvial Soil", "Black Soil"]
    _make_folder(tmp_path, classes)
    # A predictor that is always wrong must produce 0% accuracy: nothing may leak labels.
    always_class_zero = lambda img: 0
    res = ev.evaluate_image_folder(always_class_zero, tmp_path, classes)
    assert res["overall_accuracy"] == 0.5  # right only for the class-0 folder
    assert res["total_test_samples"] == 6
    assert res["evaluation_type"] == "held_out_images"


def test_folder_eval_reports_unmatched_and_empty(tmp_path):
    (tmp_path / "Mystery").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "Mystery" / "x.png")
    with pytest.raises(ValueError):
        ev.evaluate_image_folder(lambda i: 0, tmp_path, ["Alluvial Soil"])
    with pytest.raises(FileNotFoundError):
        ev.evaluate_image_folder(lambda i: 0, tmp_path / "nope", ["Alluvial Soil"])


def test_smoke_mode_never_reports_accuracy():
    res = ev.evaluate_soil_model()
    assert res["overall_accuracy"] is None
    assert res["evaluation_type"] == "smoke_test_no_accuracy"
    assert res["forward_pass_ok"] is True


def test_crop_self_consistency_is_deterministic():
    a, b = ev.evaluate_crop_model(), ev.evaluate_crop_model()
    assert a["top_1_accuracy"] == b["top_1_accuracy"]
    assert a["evaluation_type"] == "catalog_self_consistency"
