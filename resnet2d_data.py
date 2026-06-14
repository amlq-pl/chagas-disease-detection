#!/usr/bin/env python3
"""Przetwarzanie danych dla resnet2d.

Surowe okno EKG (12, L) -> log-spektrogram (12, F, T) liczony on-the-fly w __getitem__.
Dane wczytywane są z częściowym cache w RAM: pierwsze `cache_fraction` rekordów danego
splitu trzymamy na stałe w RAM (przypięte, bez eksmisji), resztę czytamy leniwie z mmap.

Dzięki temu działa to dla dowolnego rozmiaru zbioru (np. pełne CODE-15 ~45 GB):
- cache_fraction=1.0 -> cały split w RAM (jeśli się mieści),
- cache_fraction=0.5 -> połowa w RAM, reszta z dysku,
- cache_fraction=0.0 -> czysty mmap (zachowanie jak wcześniej).

Spektrogram liczony jest dopiero przy pobraniu próbki, więc w RAM trzymamy lekki surowy
sygnał (12*L*4 B), a nie cięższą reprezentację 2D.
"""

import numpy as np
import torch
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from training import RamCache, split_indices  # wspólny cache + split (jedna implementacja)

# Parametry STFT dobrane pod fs = 100 Hz (okno 7 s -> 700 próbek)
FS = 100
WIN = hann(128, sym=True)
HOP = 32
MFFT = 128
EPSILON = 1e-12


class SpectrogramDataset(torch.utils.data.Dataset):
    """Zwraca (x, meta, label): x to log-spektrogram (12, F, T), meta to pusty wektor
    (model 2D nie używa metadanych), label kształtu (1,). 3-krotka jest zgodna z
    train/validate w training.py (które wołają model(X, meta))."""

    def __init__(self, ecg_path, labels, indices, cache_fraction=1.0, fs=FS):
        self.cache = RamCache(ecg_path, indices, cache_fraction)
        self.y = np.asarray(labels)
        self.stft = ShortTimeFFT(
            win=WIN, hop=HOP, fs=fs, mfft=MFFT, fft_mode="onesided", scale_to="magnitude"
        )

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        signals = self.cache.get(i)  # surowy sygnał (12, L)
        spec = self.stft.spectrogram(signals)  # (12, F, T), |STFT|^2
        spec = np.log(spec + EPSILON).astype(np.float32)
        x = torch.from_numpy(spec)
        meta = torch.zeros(0, dtype=torch.float32)  # placeholder - model 2D ignoruje
        label = torch.tensor([self.y[i]], dtype=torch.float32)
        return x, meta, label


def make_cached_spectrogram_datasets(ecg_path, labels_path, cache_fraction=1.0, seed=42):
    """Buduje train/val/test SpectrogramDataset z częściowym cache w RAM.

    Zwraca (train_ds, val_ds, test_ds, y_train, y_val, y_test)."""
    y_full = np.load(labels_path)
    train_idx, val_idx, test_idx = split_indices(y_full, seed=seed)

    train_ds = SpectrogramDataset(ecg_path, y_full[train_idx], train_idx, cache_fraction)
    val_ds = SpectrogramDataset(ecg_path, y_full[val_idx], val_idx, cache_fraction)
    test_ds = SpectrogramDataset(ecg_path, y_full[test_idx], test_idx, cache_fraction)

    return train_ds, val_ds, test_ds, y_full[train_idx], y_full[val_idx], y_full[test_idx]
