"""Optuna hyper-parameter search for the plant (MobileNetV3) and soil (EfficientNet-B0) classifiers.

Usage (dataset in ImageFolder layout: <data-dir>/<class_name>/*.jpg):
    python tune_hyperparameters.py --task plant --data-dir "data/plant/Multi-Crop Disease Dataset" --trials 20
    python tune_hyperparameters.py --task soil  --data-dir "data/soil/Soil dataset" --trials 15

Searches learning rate, weight decay, optimizer, batch size, dropout, label smoothing,
augmentation strength and frozen-backbone epochs. Uses a TPE sampler with median pruning.
Writes the best configuration to models/tuning/<task>_best_params.json, then prints the
exact `train_*.py` command to retrain with it. Study history is kept in a SQLite file so
the search can be resumed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger("Tune")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "models" / "tuning"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(task: str, num_classes: int, dropout: float) -> nn.Module:
    if task == "plant":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[3] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(m.classifier[3].in_features, num_classes))
        head = m.classifier
    else:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(m.classifier[1].in_features, num_classes))
        head = m.classifier
    return m, head


def make_loaders(data_dir: str, img_size: int, batch_size: int, aug: float, val_split: float, seed: int, limit: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(int(30 * aug)),
        transforms.ColorJitter(brightness=0.4 * aug, contrast=0.4 * aug, saturation=0.4 * aug),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    full_train = datasets.ImageFolder(data_dir, transform=train_tf)
    full_val = datasets.ImageFolder(data_dir, transform=val_tf)
    n = len(full_train)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    if limit and limit < n:
        idx = idx[:limit]
    n_val = max(1, int(len(idx) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    sub = torch.utils.data.Subset
    return (
        DataLoader(sub(full_train, train_idx), batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(sub(full_val, val_idx), batch_size=batch_size, shuffle=False, num_workers=0),
        len(full_train.classes),
    )


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def objective_factory(args, device):
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        opt_name = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
        batch = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        aug = trial.suggest_float("augmentation_strength", 0.3, 1.5)
        freeze_epochs = trial.suggest_int("frozen_backbone_epochs", 0, 2)

        train_loader, val_loader, num_classes = make_loaders(
            args.data_dir, args.img_size, batch, aug, args.val_split, args.seed, args.max_samples
        )
        model, head = build_model(args.task, num_classes, dropout)
        model.to(device)
        head_ids = {id(p) for p in head.parameters()}

        def make_opt():
            params = [p for p in model.parameters() if p.requires_grad]
            if opt_name == "adamw":
                return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
            if opt_name == "adam":
                return torch.optim.Adam(params, lr=lr, weight_decay=wd)
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)

        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        acc = 0.0
        for epoch in range(args.epochs):
            if epoch < freeze_epochs:
                for p in model.parameters():
                    p.requires_grad = id(p) in head_ids
                optimizer = make_opt()
            elif epoch == freeze_epochs:
                for p in model.parameters():
                    p.requires_grad = True
                optimizer = make_opt()
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            acc = evaluate(model, val_loader, device)
            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return acc
    return objective


def main():
    ap = argparse.ArgumentParser(description="Optuna hyper-parameter tuning")
    ap.add_argument("--task", choices=["plant", "soil"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=4, help="Epochs per trial (keep small)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--max-samples", type=int, default=0, help="Subsample for faster trials (0 = all)")
    ap.add_argument("--timeout", type=int, default=None, help="Stop after N seconds")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"kisanseva_{args.task}",
        storage=f"sqlite:///{(OUT_DIR / 'optuna_studies.db').as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(objective_factory(args, device), n_trials=args.trials, timeout=args.timeout)

    best = {"task": args.task, "best_val_accuracy": study.best_value, "params": study.best_params}
    out = OUT_DIR / f"{args.task}_best_params.json"
    out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    log.info("Best val accuracy %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)
    log.info("Saved %s", out)
    p = study.best_params
    script = "train_plant_disease.py" if args.task == "plant" else "train_soil.py"
    print(f"\nRetrain with: python {script} --lr {p['lr']:.2e} --batch-size {p['batch_size']} --epochs 25")


if __name__ == "__main__":
    main()
