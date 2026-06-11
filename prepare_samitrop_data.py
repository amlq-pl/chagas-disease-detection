#!/usr/bin/env python3
"""
Skrypt przygotowuje dane badania Sami-Trop - tu wszystkie EKG są chore, więc etykiety to same jedynki.
W pliku exams.hdf5 znajduje się gotowa macierz sygnałów: 400 Hz, 12 leadów,
sygnał już w miliwoltach. Niektóre recordy mają 10s, a niektóre 7s - dopełnione zerami do 10s,
więc najpierw obcinamy tyle zer, ile możemy, a potem działamy na realnym sygnale.
Jeśli jakiś sygnał jest krótszy niż 7s, pomijamy go.

Z każdego recordu wycinamy 7-sekundowe okno. Jeśli początkowa długość to 10 sekund - losujemy.

Zapisujemy 3 macierze EKG w częstotliwościach:
- 100 po resamplingu
- 250 po resamplingu
- 400, oryginalne
Dodatkowo: macierz etykiet (same jedynki) i macierz metadanych [płeć, wiek, normal_ecg].

Układ macierzy EKG: (N, 12, L)
N - liczba recordów
12 - leady w EKG
L - liczba próbek w czasie

Użycie:
    python3 prepare_samitrop_data.py [-i samitrop-data] [-o semi-processed-data] [--limit N]
    -i / --data_folder   : folder z danymi SaMi-Trop (exams.hdf5 + exams.csv), domyślnie samitrop-data
    -o / --output_folder : folder na pliki .npy (domyślnie semi-processed-data)
    --limit N            : tylko pierwsze N recordów
"""

import argparse
import math
import os
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.signal import resample_poly

SIGNAL_HDF5 = 'exams.hdf5'        # gotowa macierz sygnałów
DEMOGRAPHICS_CSV = 'exams.csv'    # metadane (exam_id, age, is_male)
NATIVE_FS = 400                   # częstotliwość natywna SaMi-Trop
WINDOW_S = 7                      # długość okna w sekundach
SEED = 67
_rng = np.random.default_rng(SEED)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Buduje macierze EKG (100/250 resampled, 400 oryginal), etykiet i metadanych z Sami-Trop.')
    parser.add_argument('-i', '--data_folder', type=str, default='samitrop-data',
                        help='Folder z danymi Sami-Trop (domyślnie: samitrop-data).')
    parser.add_argument('-o', '--output_folder', type=str, default='semi-processed-data',
                        help='Folder na pliki wyjściowe .npy (domyślnie: semi-processed-data).')
    parser.add_argument('--limit', type=int, default=0,
                        help='Opcjonalnie: weź tylko pierwsze N recordów (0 = wszystkie).')
    return parser


# Resampling sygnału wzdłuż osi czasu z fs_in do fs_out
def resample_signal(sig, fs_in, fs_out):
    if fs_in == fs_out:
        return sig.astype(np.float32)
    g = math.gcd(int(fs_in), int(fs_out))
    up, down = fs_out // g, fs_in // g
    return resample_poly(sig, up, down, axis=0).astype(np.float32)


# Obcięcie zer z początku i końca sygnału. Jeśli cały sygnał to zera, zwraca None
def strip_zero_padding(sig):
    nonzero = np.where(~np.all(sig == 0, axis=1))[0]
    if nonzero.size == 0:
        return None
    return sig[nonzero[0]:nonzero[-1] + 1]


# Wycięcie losowego 7-sekundowego okna z sygnału 400 Hz (2800 próbek)
# Zwraca None, gdy sygnał jest krótszy niż 7 s
def random_window_400(sig):
    win = WINDOW_S * NATIVE_FS          # 2800 próbek
    if sig.shape[0] < win:
        return None
    max_start = sig.shape[0] - win
    start = int(_rng.integers(0, max_start + 1))
    return sig[start:start + win]


# Kontrola jakości: odrzuca rekordy z NaN/Inf albo całkowicie płaskie (martwy zapis).
def is_valid(sig):
    return bool(np.all(np.isfinite(sig))) and float(sig.std()) >= 1e-6


# Zbudowanie wiersza metadanych [płeć, wiek, normal_ecg] dla recordu
def build_metadata_row(is_male, age, normal_ecg):
    return [float(is_male), float(age), float(normal_ecg)]


# Główna procedura: czyta metadane i sygnały z HDF5, dla każdego rekordu obcina zera,
# losuje 7 s okno, robi resampling do 250 i 100 Hz, a na końcu zapisuje macierze
def run(args):
    hdf5_path = os.path.join(args.data_folder, SIGNAL_HDF5)
    csv_path = os.path.join(args.data_folder, DEMOGRAPHICS_CSV)
    if not os.path.isfile(hdf5_path):
        sys.exit(f'Nie znaleziono pliku sygnałów: {hdf5_path}')
    if not os.path.isfile(csv_path):
        sys.exit(f'Nie znaleziono pliku metadanych: {csv_path}')

    # Metadane w kolejności z CSV. Parujemy po pozycji: i-ty wiersz CSV odpowiada tracings[i]
    df = pd.read_csv(csv_path)
    csv_rows = [(int(r['exam_id']), r['age'], r['is_male'], r['normal_ecg'])
                for _, r in df.iterrows()]

    # Długości okna w próbkach dla każdej częstotliwości
    len_100 = WINDOW_S * 100
    len_250 = WINDOW_S * 250
    len_400 = WINDOW_S * 400

    ecg_100, ecg_250, ecg_400 = [], [], []
    metadata, exam_ids = [], []
    n_skipped = 0

    with h5py.File(hdf5_path, 'r') as f:
        tracings = f['tracings']                       # (N, próbki, 12)
        n_total = min(len(csv_rows), tracings.shape[0])
        if args.limit and args.limit > 0:
            n_total = min(n_total, args.limit)

        for i in range(n_total):
            # i-ty wiersz CSV <-> tracings[i].
            exam_id, age, is_male, normal_ecg = csv_rows[i]

            sig = np.asarray(tracings[i], dtype=np.float32)   # (próbki, 12), 400 Hz, mV
            sig = strip_zero_padding(sig)                     # usuń padding zerowy
            if sig is None:
                n_skipped += 1
                continue
            # Kontrola jakości - odrzucamy NaN/Inf i martwe zapisy
            if not is_valid(sig):
                n_skipped += 1
                continue

            sig400 = random_window_400(sig)                   # losowe 7 s okno @ 400 Hz
            if sig400 is None:                                # sygnał < 7 s -> pomijamy
                n_skipped += 1
                continue

            # 250 i 100 Hz z resamplingu tego samego 7 s okna (filtr i standaryzacja - w merge_and_process_data.py)
            sig250 = resample_signal(sig400, NATIVE_FS, 250)
            sig100 = resample_signal(sig400, NATIVE_FS, 100)

            # Transpozycja do (12, L) i dodanie do list (surowe mV, bez filtra/normalizacji)
            ecg_400.append(sig400[:len_400].T)
            ecg_250.append(sig250[:len_250].T)
            ecg_100.append(sig100[:len_100].T)
            metadata.append(build_metadata_row(is_male, age, normal_ecg))
            exam_ids.append(exam_id)

            if len(exam_ids) % 250 == 0:
                print(f'  przetworzono {len(exam_ids)} recordów')

    n = len(exam_ids)

    # Złożenie macierzy (N, 12, L)
    ecg_100 = np.stack(ecg_100).astype(np.float32)
    ecg_250 = np.stack(ecg_250).astype(np.float32)
    ecg_400 = np.stack(ecg_400).astype(np.float32)
    labels = np.ones((n,), dtype=np.int64)             # wszyscy chorzy -> 1
    metadata = np.asarray(metadata, dtype=np.float32)  # [płeć, wiek, normal_ecg]
    exam_ids = np.asarray(exam_ids, dtype=np.int64)

    # Zapis wyników
    os.makedirs(args.output_folder, exist_ok=True)
    out = args.output_folder
    np.save(os.path.join(out, 'ecg_samitrop_100hz_resampled.npy'), ecg_100)
    np.save(os.path.join(out, 'ecg_samitrop_250hz_resampled.npy'), ecg_250)
    np.save(os.path.join(out, 'ecg_samitrop_400hz.npy'), ecg_400)
    np.save(os.path.join(out, 'labels_samitrop.npy'), labels)
    np.save(os.path.join(out, 'metadata_samitrop.npy'), metadata)
    np.save(os.path.join(out, 'exam_ids_samitrop.npy'), exam_ids)

    # Podsumowanie
    print(f'\nPrzetworzono {n} rekordów (pominięto {n_skipped}).')
    print('Zapisano:')
    print(f'  ecg_samitrop_100hz_resampled.npy : {ecg_100.shape}')
    print(f'  ecg_samitrop_250hz_resampled.npy : {ecg_250.shape}')
    print(f'  ecg_samitrop_400hz.npy           : {ecg_400.shape}')
    print(f'  labels_samitrop.npy              : {labels.shape}')
    print(f'  metadata_samitrop.npy            : {metadata.shape}')
    print(f'  exam_ids_samitrop.npy            : {exam_ids.shape}')

    # Podgląd pierwszego recordu dla metadanych i sygnału
    # print('metadata[0] :', metadata[0])
    # print('ecg_400 record 0, próbka 0, wszystkie 12 leadów:\n', ecg_400[0, :, 0])


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
