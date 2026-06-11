#!/usr/bin/env python3
"""
Łączy i przetwarza macierze z prepare_samitrop_data.py i prepare_code15_data.py

Skrypty prepare_* zapisują surowe okna (bez filtra i normalizacji).
Tutaj robimy wspólne przetwarzanie obu zbiorów naraz:
1. filtr pasmowy low..high Hz - górny próg sterowany argumentem --high,
2. standaryzacja z-score
3. sklejenie SaMi-Trop + CODE-15% w jedną macierz na każdą częstotliwość

Obie bazy są natywnie 400 Hz, a 100/250 Hz mamy z resamplingu po obu stronach.
Łączymy (obie strony mają zgodne kształty, 7 s okna):
- 100 Hz -> ecg_merged_100hz_resampled.npy : ecg_samitrop_100hz_resampled + ecg_code15_100hz_resampled
- 250 Hz -> ecg_merged_250hz.npy           : ecg_samitrop_250hz_resampled + ecg_code15_250hz_resampled
- 400 Hz -> ecg_merged_400hz.npy           : ecg_samitrop_400hz           + ecg_code15_400hz

Użycie:
    python3 merge_and_process_data.py [-i semi-processed-data] [-o processed-data] [--low 0.5] [--high 40]
    -i / --input_folder  : folder z plikami .npy z prepare_* (domyślnie semi-processed-data)
    -o / --output_folder : folder na pliki ecg_merged_* (domyślnie processed-data)
    --low                : dolny próg pasma w Hz (domyślnie 0.5, usuwa dryf bazy)
    --high               : górny próg pasma w Hz (domyślnie 40, powyżej usuwamy)
"""

import argparse
import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt

# Częstotliwość -> (fs, plik SaMi-Trop, plik CODE-15%, plik wyjściowy). Macierze mają układ (N, 12, L)
SPECS = [
    (100, 'ecg_samitrop_100hz_resampled.npy', 'ecg_code15_100hz_resampled.npy', 'ecg_merged_100hz_resampled.npy'),
    (250, 'ecg_samitrop_250hz_resampled.npy', 'ecg_code15_250hz_resampled.npy', 'ecg_merged_250hz_resampled.npy'),
    (400, 'ecg_samitrop_400hz.npy',           'ecg_code15_400hz.npy',           'ecg_merged_400hz.npy'),
]


def get_parser():
    parser = argparse.ArgumentParser(
        description='Łączy i przetwarza (filtr + standaryzacja) macierze SaMi-Trop i CODE-15%.')
    parser.add_argument('-i', '--input_folder', type=str, default='semi-processed-data',
                        help='Folder z plikami .npy z prepare_* (domyślnie semi-processed-data).')
    parser.add_argument('-o', '--output_folder', type=str, default='processed-data',
                        help='Folder na pliki wyjściowe (domyślnie processed-data).')
    parser.add_argument('--low', type=float, default=0.5,
                        help='Dolny próg pasma w Hz (domyślnie 0.5).')
    parser.add_argument('--high', type=float, default=40.0,
                        help='Górny próg pasma w Hz - powyżej usuwamy (domyślnie 40).')
    return parser


# Filtr pasmowy
def bandpass(X, fs, low, high, order=4):
    nyq = fs / 2.0
    hi = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, X, axis=2).astype(np.float32)


# Standaryzacja z-score
def standardize(X):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    return ((X - mean) / (std + 1e-8)).astype(np.float32)


# Wczytanie pliku .npy z folderu wejściowego
def load(folder, name):
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        sys.exit(f'Nie znaleziono pliku: {path}.')
    return np.load(path)


def run(args):
    inp, out = args.input_folder, args.output_folder
    os.makedirs(out, exist_ok=True)

    # Etykiety, metadane, id - łączone raz
    labels_s = load(inp, 'labels_samitrop.npy')
    labels_c = load(inp, 'labels_code15.npy')
    n_s, n_c = len(labels_s), len(labels_c)

    labels = np.concatenate([labels_s, labels_c])
    metadata = np.concatenate([load(inp, 'metadata_samitrop.npy'), load(inp, 'metadata_code15.npy')])
    # Globalnie unikalne id z prefiksem zbioru ('samitrop_<id>' / 'code15_<id>'), bo numery z obu baz mogą się powtarzać
    ids_s = load(inp, 'exam_ids_samitrop.npy')
    ids_c = load(inp, 'exam_ids_code15.npy')
    ids = np.array([f'samitrop_{i}' for i in ids_s] + [f'code15_{i}' for i in ids_c])
    source = np.concatenate([np.zeros(n_s, dtype=np.int64), np.ones(n_c, dtype=np.int64)])  # 0=SaMi-Trop, 1=CODE-15%

    np.save(os.path.join(out, 'labels_merged.npy'), labels)
    np.save(os.path.join(out, 'metadata_merged.npy'), metadata)
    np.save(os.path.join(out, 'ids_merged.npy'), ids)
    np.save(os.path.join(out, 'source_merged.npy'), source)

    n_pos = int((labels == 1).sum())
    print(f'SaMi-Trop: {n_s} recordów | CODE-15%: {n_c} recordów | razem: {n_s + n_c}')
    print(f'Pozytywnych w sumie: {n_pos} | negatywnych: {n_s + n_c - n_pos}')
    print(f'Filtr pasmowy: {args.low}-{args.high} Hz\n')

    # Dla każdego wpisu: wczytaj obie macierze, przefiltruj, standaryzuj, sklej i zapisz
    for fs, samitrop_name, code15_name, out_name in SPECS:
        Xs = load(inp, samitrop_name)
        Xc = load(inp, code15_name)
        if Xs.shape[0] != n_s or Xc.shape[0] != n_c:
            sys.exit(f'Niezgodna liczba recordów dla {out_name} (EKG vs etykiety).')

        Xs = standardize(bandpass(Xs, fs, args.low, args.high))
        Xc = standardize(bandpass(Xc, fs, args.low, args.high))
        X = np.concatenate([Xs, Xc], axis=0)

        np.save(os.path.join(out, out_name), X)
        print(f'  {out_name:32s}: {X.shape}')

    print(f'\nlabels_merged.npy   : {labels.shape}')
    print(f'metadata_merged.npy : {metadata.shape}')
    print(f'ids_merged.npy      : {ids.shape}')
    print(f'source_merged.npy   : {source.shape}')


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
