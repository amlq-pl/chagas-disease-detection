import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from device import device
from inception_time import InceptionNetwork
from training import make_mmap_datasets, train, validate

DIRPATH = "processed-data"
ECG_PATH = os.path.join(DIRPATH, "ecg_merged_100hz.npy")
LABELS_PATH = os.path.join(DIRPATH, "labels_merged.npy")
RESULTS_DIR = "results/ensemble"

BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 25
PATIENCE = 5
SUBSET_FRACTION = 1.0

SPLIT_SEED = 42
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]


def make_loaders(train_ds, val_ds, test_ds, shuffle_seed):
    # Wlasny generator dla DataLoader, zeby shuffle zalezal od seeda modelu
    g = torch.Generator()
    g.manual_seed(shuffle_seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        generator=g,
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


def train_one_seed(seed, train_ds, val_ds, test_ds, y_train, run_dir):
    """Trenuje jeden model z danym seedem. Zwraca dict z metrykami + sciezka do checkpointu."""
    os.makedirs(run_dir, exist_ok=True)

    # Seed wplywa na init wag i shuffle, ale NIE na split
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, shuffle_seed=seed
    )

    model = InceptionNetwork(in_channels=12, use_meta=False).to(device)
    # model = ResNet18(use_meta=False).to(device)
    compiled_model = torch.compile(model)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    checkpoint_path = os.path.join(run_dir, "best.pt")
    t0 = time.time()
    best_auprc, best_epoch, history = train(
        optimizer,
        criterion,
        compiled_model,
        NUM_EPOCHS,
        train_loader,
        val_loader,
        early_stopping=PATIENCE,
        checkpoint_path=checkpoint_path,
    )
    duration_s = time.time() - t0

    # Per-model test
    test_auroc, test_auprc, test_acc = validate(test_loader, compiled_model)
    print(f"\n=== TEST (seed={seed}) ===")
    print(f"AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, ACC: {test_acc:.4f}")
    print(f"Czas: {duration_s/60:.1f} min")

    metrics = {
        "seed": seed,
        "best_val_auprc": float(best_auprc),
        "best_epoch": int(best_epoch),
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc),
        "test_acc": float(test_acc),
        "duration_s": float(duration_s),
        "history": history,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    del model, compiled_model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    return metrics


@torch.no_grad()
def collect_test_probs(checkpoint_path, test_ds):
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
    )

    model = InceptionNetwork(in_channels=12, use_meta=False).to(device)
    # model = ResNet18(use_meta=False).to(device)
    compiled_model = torch.compile(model)
    state = torch.load(checkpoint_path, map_location=device)
    compiled_model.load_state_dict(state)
    compiled_model.eval()

    all_probs = []
    all_labels = []
    for X, meta, y in test_loader:
        X = X.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        logits = compiled_model(X, meta)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())

    del model, compiled_model, test_loader
    torch.cuda.empty_cache()

    return np.concatenate(all_probs).ravel(), np.concatenate(all_labels).ravel()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Buduje datasety (split seed = 42)...")
    train_ds, val_ds, test_ds, y_train, y_val, y_test = make_mmap_datasets(
        ECG_PATH,
        LABELS_PATH,
        metadata_path=None,
        subset_fraction=SUBSET_FRACTION,
        seed=SPLIT_SEED,
        augmentation=False,
    )
    # train_ds, val_ds, test_ds, y_train, y_val, y_test = make_cached_spectrogram_datasets(
    #     ECG_PATH,
    #     LABELS_PATH,
    #     cache_fraction=SUBSET_FRACTION,
    #     seed=SPLIT_SEED,
    # )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Pozytywnych w train: {int(y_train.sum())} / {len(y_train)}")
    print(f"Pozytywnych w test:  {int(y_test.sum())} / {len(y_test)}\n")

    per_model_metrics = []
    total_t0 = time.time()
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print("\n" + "#" * 60)
        print(f"# MODEL {i+1}/{len(ENSEMBLE_SEEDS)}: seed={seed}")
        print("#" * 60)
        run_dir = os.path.join(RESULTS_DIR, f"seed{seed}")
        m = train_one_seed(seed, train_ds, val_ds, test_ds, y_train, run_dir)
        per_model_metrics.append(m)

        with open(os.path.join(RESULTS_DIR, "_progress.json"), "w") as f:
            json.dump(per_model_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION (sredniaarytmetyczna sigmoid outputow)")
    print("=" * 60)

    all_probs = []
    labels = None
    for seed in ENSEMBLE_SEEDS:
        checkpoint_path = os.path.join(RESULTS_DIR, f"seed{seed}", "best.pt")
        probs, lbls = collect_test_probs(checkpoint_path, test_ds)
        all_probs.append(probs)
        if labels is None:
            labels = lbls
        else:
            assert np.array_equal(labels, lbls), "Labels mismatched between seeds!"
        print(f"  seed={seed} | indywidualne AUPRC: {average_precision_score(lbls, probs):.4f}")

    ensemble_probs = np.mean(all_probs, axis=0)
    ens_auroc = roc_auc_score(labels, ensemble_probs)
    ens_auprc = average_precision_score(labels, ensemble_probs)
    ens_acc = ((ensemble_probs > 0.5) == labels).mean()

    total_duration_s = time.time() - total_t0

    print(f"\n=== ENSEMBLE TEST (mean of 5 sigmoid outputs) ===")
    print(f"AUROC: {ens_auroc:.4f}")
    print(f"AUPRC: {ens_auprc:.4f}")
    print(f"ACC:   {ens_acc:.4f}")
    print(f"\nLaczny czas: {total_duration_s/3600:.2f} h")

    # Tabelka per-model vs ensemble
    print(f"\n{'seed':>6} | {'test AUPRC':>10} | {'test AUROC':>10}")
    print("-" * 36)
    for m in per_model_metrics:
        print(f"{m['seed']:>6} | {m['test_auprc']:>10.4f} | {m['test_auroc']:>10.4f}")
    print("-" * 36)
    print(f"{'ENS':>6} | {ens_auprc:>10.4f} | {ens_auroc:>10.4f}")

    mean_indiv_auprc = np.mean([m["test_auprc"] for m in per_model_metrics])
    print(f"\nSrednia indywidualnych AUPRC: {mean_indiv_auprc:.4f}")
    print(f"Ensemble AUPRC:               {ens_auprc:.4f}")
    print(f"Boost ensemble vs srednia:    {ens_auprc - mean_indiv_auprc:+.4f}")

    summary = {
        "split_seed": SPLIT_SEED,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "per_model": [
            {
                "seed": m["seed"],
                "test_auprc": m["test_auprc"],
                "test_auroc": m["test_auroc"],
                "test_acc": m["test_acc"],
                "best_epoch": m["best_epoch"],
                "duration_s": m["duration_s"],
            }
            for m in per_model_metrics
        ],
        "ensemble": {
            "test_auprc": float(ens_auprc),
            "test_auroc": float(ens_auroc),
            "test_acc": float(ens_acc),
        },
        "mean_individual_test_auprc": float(mean_indiv_auprc),
        "ensemble_boost_vs_mean": float(ens_auprc - mean_indiv_auprc),
        "total_duration_s": float(total_duration_s),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
