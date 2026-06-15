import json
import os
import sys
import time
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch

from device import device
from inception_time import InceptionNetwork
from resnet1d import SUBSET_FRACTION, ResNet18
from training import make_mmap_datasets, train, validate

# ============================================================
# KONFIGURACJA
# ============================================================
DIRPATH = "processed-data"
ECG_PATH = os.path.join(DIRPATH, "ecg_merged_100hz.npy")
LABELS_PATH = os.path.join(DIRPATH, "labels_merged.npy")
METADATA_PATH = os.path.join(DIRPATH, "metadata_merged.npy")
RESULTS_DIR = "results"

# Parametry treningu
BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-3
SEED = 42
SUBSET_FRACTION = 1.0
EPOCHS_NO_AUG = 25
PATIENCE_NO_AUG = 5
EPOCHS_AUG = 45
PATIENCE_AUG = 10
EXPERIMENTS = [
    {"name": "resnet18", "model": "resnet18", "use_meta": False, "augmentation": False},
    {"name": "resnet18_meta", "model": "resnet18", "use_meta": True, "augmentation": False},
    {"name": "resnet18_meta_aug", "model": "resnet18", "use_meta": True, "augmentation": True},
    {"name": "inception", "model": "inception", "use_meta": False, "augmentation": False},
    {"name": "inception_meta", "model": "inception", "use_meta": True, "augmentation": False},
    {"name": "inception_meta_aug", "model": "inception", "use_meta": True, "augmentation": True},
]


def build_model(name, use_meta):
    if name == "resnet18":
        return ResNet18(use_meta=use_meta)
    if name == "inception":
        return InceptionNetwork(in_channels=12, use_meta=use_meta)


def make_loaders(train_ds, val_ds, test_ds):
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    return train_loader, val_loader, test_loader


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================
def run_one_experiment(cfg, run_dir):
    """Trenuje jeden wariant, zapisuje wyniki do run_dir/. Zwraca dict z metrykami."""
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    num_epochs = EPOCHS_AUG if cfg["augmentation"] else EPOCHS_NO_AUG
    patience = PATIENCE_AUG if cfg["augmentation"] else PATIENCE_NO_AUG
    metadata_path = METADATA_PATH if cfg["use_meta"] else None

    # Datasety
    train_ds, val_ds, test_ds, y_train, y_val, y_test = make_mmap_datasets(
        ECG_PATH,
        LABELS_PATH,
        metadata_path=metadata_path,
        subset_fraction=SUBSET_FRACTION,
        seed=SEED,
        augmentation=cfg["augmentation"],
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Pozytywnych w train: {int(y_train.sum())} / {len(y_train)}")

    # Loaders
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    # Model
    model = build_model(cfg["model"], cfg["use_meta"])
    assert model is not None
    model = model.to(device)
    compiled_model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())

    # Optymalizator + loss
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Trening
    checkpoint_path = os.path.join(run_dir, "best.pt")
    t0 = time.time()
    best_auprc, best_epoch, history = train(
        optimizer,
        criterion,
        compiled_model,
        num_epochs,
        train_loader,
        val_loader,
        early_stopping=patience,
        checkpoint_path=checkpoint_path,
    )
    duration_s = time.time() - t0

    # Final test (model ma juz przywrocone best wagi z train())
    test_auroc, test_auprc, test_acc = validate(test_loader, compiled_model)
    print(f"\n=== TEST ({cfg['name']}) ===")
    print(f"AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, ACC: {test_acc:.4f}")
    print(f"Czas treningu: {duration_s/60:.1f} min")

    # Zapis metryk
    metrics = {
        "variant": cfg["name"],
        "config": {
            "model": cfg["model"],
            "use_meta": cfg["use_meta"],
            "augmentation": cfg["augmentation"],
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "num_epochs_max": num_epochs,
            "early_stopping_patience": patience,
            "seed": SEED,
            "subset_fraction": SUBSET_FRACTION,
            "n_params": n_params,
        },
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_auprc),
        "test": {
            "auroc": float(test_auroc),
            "auprc": float(test_auprc),
            "acc": float(test_acc),
        },
        "duration_s": float(duration_s),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Cleanup
    del model, compiled_model, optimizer, train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds
    torch.cuda.empty_cache()

    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = []
    total_t0 = time.time()

    for i, cfg in enumerate(EXPERIMENTS):
        print("\n" + "#" * 60)
        print(f"# EKSPERYMENT {i+1}/{len(EXPERIMENTS)}: {cfg['name']}")
        print("#" * 60)
        run_dir = os.path.join(RESULTS_DIR, cfg["name"])
        try:
            metrics = run_one_experiment(cfg, run_dir)
            summary.append(
                {
                    "name": cfg["name"],
                    "best_val_auprc": metrics["best_val_auprc"],
                    "test_auroc": metrics["test"]["auroc"],
                    "test_auprc": metrics["test"]["auprc"],
                    "test_acc": metrics["test"]["acc"],
                    "duration_s": metrics["duration_s"],
                    "best_epoch": metrics["best_epoch"],
                    "n_params": metrics["config"]["n_params"],
                }
            )
            # Zapisuj summary po kazdym eksperymencie (na wypadek crashu kolejnego)
            with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"BLAD w wariancie {cfg['name']}: {e}")
            summary.append({"name": cfg["name"], "error": str(e)})
            with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

    total_duration_s = time.time() - total_t0
    print("\n" + "=" * 60)
    print(f"WSZYSTKIE EKSPERYMENTY ZAKONCZONE ({total_duration_s/3600:.2f} h)")
    print("=" * 60)
    print(
        f"{'wariant':<25} {'val AUPRC':>10} {'test AUPRC':>10} {'test AUROC':>10} {'czas [min]':>10}"
    )
    for r in summary:
        if "error" in r:
            print(f"{r['name']:<25} BLAD: {r['error']}")
        else:
            print(
                f"{r['name']:<25} {r['best_val_auprc']:>10.4f} {r['test_auprc']:>10.4f} {r['test_auroc']:>10.4f} {r['duration_s']/60:>10.1f}"
            )


if __name__ == "__main__":
    main()
