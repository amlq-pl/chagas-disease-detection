#!/usr/bin/env python3
"""
Łączy i przetwarza macierze z prepare_samitrop_data.py i prepare_code15_data.py

Skrypty prepare_* zapisują surowe okna (bez filtra i normalizacji).
Tutaj robimy wspólne przetwarzanie obu zbiorów naraz:
1. filtr pasmowy low..high Hz
2. standaryzacja z-score
3. sklejenie SaMi-Trop + CODE-15% w jedną macierz na każdą częstotliwość

Użycie:
    python3 merge_and_process_data.py [-i semi-processed-data] [-o processed-data]
    -i / --input_folder  : folder z plikami .npy z prepare_* (domyślnie semi-processed-data)
    -o / --output_folder : folder na pliki ecg_merged_* (domyślnie processed-data)
"""

import argparse
import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt


FS = 100
SAMITROP_NAME = 'ecg_samitrop_100hz.npy'
CODE15_NAME =  'ecg_code15_100hz.npy'
OUT_NAME =  'ecg_merged_100hz.npy'
LOW = 0.5
HIGH = 40


def get_parser():
    parser = argparse.ArgumentParser(
        description='Łączy i przetwarza (filtr + standaryzacja) macierze SaMi-Trop i CODE-15%.')
    parser.add_argument('-i', '--input_folder', type=str, default='semi-processed-data',
                        help='Folder z plikami .npy z prepare_* (domyślnie semi-processed-data).')
    parser.add_argument('-o', '--output_folder', type=str, default='processed-data',
                        help='Folder na pliki wyjściowe (domyślnie processed-data).')
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


    np.save(os.path.join(out, 'labels_merged.npy'), labels)
    np.save(os.path.join(out, 'metadata_merged.npy'), metadata)

    n_pos = int((labels == 1).sum())
    print(f'SaMi-Trop: {n_s} recordów | CODE-15%: {n_c} recordów | razem: {n_s + n_c}')
    print(f'Pozytywnych w sumie: {n_pos} | negatywnych: {n_s + n_c - n_pos}')
    print(f'Filtr pasmowy: {LOW}-{HIGH} Hz\n')

    # Dla każdego wpisu: wczytaj obie macierze, przefiltruj, standaryzuj, sklej i zapisz
    Xs = load(inp, SAMITROP_NAME)
    Xc = load(inp, CODE15_NAME)
    if Xs.shape[0] != n_s or Xc.shape[0] != n_c:
        sys.exit(f'Niezgodna liczba recordów dla {OUT_NAME} (EKG vs etykiety).')

    Xs = standardize(bandpass(Xs, FS, LOW, HIGH))
    Xc = standardize(bandpass(Xc, FS, LOW, HIGH))
    X = np.concatenate([Xs, Xc], axis=0)

    np.save(os.path.join(out, OUT_NAME), X)
    print(f'  {OUT_NAME:32s}: {X.shape}')

    print(f'\nlabels_merged.npy   : {labels.shape}')
    print(f'metadata_merged.npy : {metadata.shape}')


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
