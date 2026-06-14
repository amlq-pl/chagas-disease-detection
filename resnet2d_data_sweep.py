#!/usr/bin/env python3
"""
Sweep danych dla ResNet2D: trenuje model na frakcjach danych treningowych
[2%, 5%, 10%, 25%, 50%, 75%, 100%] i rysuje AUPRC / AUROC / ACC vs % danych.

Test set jest zawsze taki sam (pełny, ustalony przed pętlą).
Val set jest subsamplowany proporcjonalnie do frakcji train (ale może być pełny).

Użycie:
    python3 resnet2d_data_sweep.py [--dirpath processed-data] [--epochs 20]
                                   [--batch 64] [--patience 5] [--seed 42]
                                   [--out sweep_results.json]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from device import device
from resnet2d import ResNet18
from resnet2d_data import SpectrogramDataset
from training import train, validate, split_indices

FRACTIONS = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


def subsample_indices(idx, y_full, fraction, seed):
    """Stratyfikowany podzbiór train_idx do `fraction` rozmiaru."""
    if fraction >= 1.0:
        return idx
    n_keep = max(2, int(round(len(idx) * fraction)))
    # Potrzebujemy min 2 próbek na klasę; jeśli nie ma — bierzemy tyle ile jest
    labels = y_full[idx]
    try:
        sub, _ = train_test_split(
            idx,
            train_size=n_keep,
            stratify=labels,
            random_state=seed,
        )
    except ValueError:
        # Za mało próbek na stratyfikację
        rng = np.random.default_rng(seed)
        sub = rng.choice(idx, size=n_keep, replace=False)
    return sub


def build_loaders(ecg_path, y_full, train_idx, val_idx, batch_size, num_workers=4):
    train_ds = SpectrogramDataset(ecg_path, y_full[train_idx], train_idx, cache_fraction=1.0)
    val_ds = SpectrogramDataset(ecg_path, y_full[val_idx], val_idx, cache_fraction=1.0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def run_sweep(args):
    ecg_path = os.path.join(args.dirpath, "ecg_merged_100hz_resampled.npy")
    labels_path = os.path.join(args.dirpath, "labels_merged.npy")

    for p in (ecg_path, labels_path):
        if not os.path.isfile(p):
            sys.exit(f"Brak pliku: {p}")

    y_full = np.load(labels_path)
    train_idx, val_idx, test_idx = split_indices(y_full, seed=args.seed)

    # Test set jest stały dla wszystkich frakcji
    test_ds = SpectrogramDataset(ecg_path, y_full[test_idx], test_idx, cache_fraction=1.0)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        pin_memory=True, num_workers=4,
    )

    results = []

    for frac in FRACTIONS:
        pct = int(frac * 100)
        print(f"\n{'='*60}")
        print(f"  FRAKCJA DANYCH: {pct}%")
        print(f"{'='*60}")

        sub_train_idx = subsample_indices(train_idx, y_full, frac, args.seed)
        # Val proporcjonalnie (opcjonalnie można zostawić pełny)
        sub_val_idx = subsample_indices(val_idx, y_full, frac, args.seed)

        print(f"  Train: {len(sub_train_idx)} | Val: {len(sub_val_idx)} | Test: {len(test_idx)}")

        train_loader, val_loader = build_loaders(
            ecg_path, y_full, sub_train_idx, sub_val_idx, args.batch, num_workers=4
        )

        model = ResNet18().to(device)
        compiled_model = torch.compile(model)

        y_tr = y_full[sub_train_idx]
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4)

        checkpoint = f"sweep_checkpoint_{pct}pct.pt"
        best_auprc, best_epoch, history = train(
            optimizer, criterion, compiled_model,
            args.epochs, train_loader, val_loader,
            early_stopping=args.patience,
            checkpoint_path=checkpoint,
        )

        auroc, auprc, acc = validate(test_loader, compiled_model)
        print(f"\n  >> TEST [{pct}%]: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  ACC={acc:.4f}")

        results.append({
            "fraction": frac,
            "pct": pct,
            "n_train": int(len(sub_train_idx)),
            "best_val_auprc": float(best_auprc),
            "best_epoch": int(best_epoch),
            "test_auroc": float(auroc),
            "test_auprc": float(auprc),
            "test_acc": float(acc),
        })

        # Usuń checkpoint (nie potrzebujemy go po sweepie)
        if os.path.isfile(checkpoint):
            os.remove(checkpoint)

    # Zapis wyników
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWyniki zapisano do: {args.out}")

    return results


def plot_results(results, out_path="resnet2d_sweep_plot.png"):
    pcts = [r["pct"] for r in results]
    aurocs = [r["test_auroc"] for r in results]
    auprcs = [r["test_auprc"] for r in results]
    accs = [r["test_acc"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(pcts, aurocs, "o-", color="#2196F3", linewidth=2, markersize=7, label="AUROC")
    ax.plot(pcts, auprcs, "s-", color="#E91E63", linewidth=2, markersize=7, label="AUPRC")
    ax.plot(pcts, accs,   "^-", color="#4CAF50", linewidth=2, markersize=7, label="Dokładność")

    # Annotacje wartości
    for pct, auroc, auprc, acc in zip(pcts, aurocs, auprcs, accs):
        ax.annotate(f"{auroc:.3f}", (pct, auroc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color="#2196F3")
        ax.annotate(f"{auprc:.3f}", (pct, auprc), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7.5, color="#E91E63")
        ax.annotate(f"{acc:.3f}", (pct, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color="#4CAF50")

    ax.set_xlabel("% danych treningowych", fontsize=12)
    ax.set_ylabel("Wartość metryki", fontsize=12)
    ax.set_title("ResNet2D: AUROC / AUPRC / Dokładność vs. % danych", fontsize=13)
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wykres zapisano do: {out_path}")


def get_parser():
    p = argparse.ArgumentParser(description="Sweep frakcji danych dla ResNet2D.")
    p.add_argument("--dirpath", default="processed-data",
                   help="Folder z przetworzonymi danymi (domyślnie: processed-data).")
    p.add_argument("--epochs", type=int, default=20, help="Maks. epok per run (domyślnie: 20).")
    p.add_argument("--batch", type=int, default=64, help="Batch size (domyślnie: 64).")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience (domyślnie: 5).")
    p.add_argument("--seed", type=int, default=42, help="Seed (domyślnie: 42).")
    p.add_argument("--out", default="sweep_results.json", help="Plik JSON z wynikami.")
    p.add_argument("--plot-only", dest="plot_only", default=None,
                   help="Pomiń trening - wczytaj wyniki z podanego pliku JSON i narysuj wykres.")
    p.add_argument("--plot-out", dest="plot_out", default="resnet2d_sweep_plot.png",
                   help="Ścieżka do wyjściowego wykresu (domyślnie: resnet2d_sweep_plot.png).")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.plot_only:
        # Tryb rysowania z gotowych wyników
        with open(args.plot_only) as f:
            results = json.load(f)
        plot_results(results, out_path=args.plot_out)
    else:
        results = run_sweep(args)
        plot_results(results, out_path=args.plot_out)
