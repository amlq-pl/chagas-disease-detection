import numpy as np
import torch
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from training import split_indices

FS = 100
WIN = hann(128, sym=True)
HOP = 32
MFFT = 128
EPSILON = 1e-12


class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, ecg_path, labels, indices, cache_fraction=1.0, fs=FS):
        self.ecg_path = ecg_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(labels, dtype=np.float32)
        self._mmap = None
        self.stft = ShortTimeFFT(
            win=WIN, hop=HOP, fs=fs, mfft=MFFT, fft_mode="onesided", scale_to="magnitude"
        )

    def __len__(self):
        return len(self.indices)

    def _get_signal(self, i):
        if self._mmap is None:
            self._mmap = np.load(self.ecg_path, mmap_mode="r")
        return self._mmap[int(self.indices[i])].astype(np.float32, copy=True)

    def __getitem__(self, i):
        signals = self._get_signal(i)
        spec = self.stft.spectrogram(signals)
        spec = np.log(spec + EPSILON).astype(np.float32)
        x = torch.from_numpy(spec)
        meta = torch.zeros(0, dtype=torch.float32)  # placeholder
        label = torch.tensor([self.y[i]], dtype=torch.float32)
        return x, meta, label


def make_cached_spectrogram_datasets(ecg_path, labels_path, cache_fraction=1.0, seed=42):
    y_full = np.load(labels_path)
    train_idx, val_idx, test_idx = split_indices(y_full, seed=seed)

    train_ds = SpectrogramDataset(ecg_path, y_full[train_idx], train_idx, cache_fraction)
    val_ds = SpectrogramDataset(ecg_path, y_full[val_idx], val_idx, cache_fraction)
    test_ds = SpectrogramDataset(ecg_path, y_full[test_idx], test_idx, cache_fraction)

    return train_ds, val_ds, test_ds, y_full[train_idx], y_full[val_idx], y_full[test_idx]
