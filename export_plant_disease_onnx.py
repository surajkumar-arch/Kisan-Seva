import json
from pathlib import Path

import torch
import onnx
from train_plant_disease import build_model

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "plant_health" / "multi_crop_69_class"
CHECKPOINT = MODEL_DIR / "plant_mobilenet_v3.pt"
METADATA = MODEL_DIR / "class_labels.json"
OUTPUT = MODEL_DIR / "plant_disease_model.onnx"
IMAGE_SIZE = 160  # Use the image size used during training.

classes = json.loads(METADATA.read_text(encoding="utf-8"))["classes"]

model = build_model(
    "mobilenet_v3",
    num_classes=len(classes),
    pretrained_backbone=False,
)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
model.eval()

torch.onnx.export(
    model,
    torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
    str(OUTPUT),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=18,
    dynamo=False,
)

onnx.checker.check_model(str(OUTPUT))
print(f"Exported and validated: {OUTPUT}")