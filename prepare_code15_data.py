#!/usr/bin/env python3
"""
Skrypt przygotowuje dane badania CODE-15% - mamy zarówno zdrowe, jak i chore EKG.
Różnica w stosunku do pozostałych badań jest taka, że labele nie są pewne.
Osoby oznaczone jako chore, są faktycznie chore z prawdopodobieństwem graniczącym z pewnością.
Osoby oznaczone jako zdrowe natomiast, mogą być zdrowe, ale równie dobrze mogą być chore
bez objawów.
W plikach exams_part*.hdf5 (pliki są podzielone ze względu na wielkość całej bazy danych)
znajduje się gotowa macierz sygnałów: 400 Hz, 12 leadów, po 10 lub 7 sekund.
Przygotowujemy go analogicznie do Sami-Trop, aby otrzymać 7-sekundowe okna.

Etykiety bierzemy z osobnego pliku code15_chagas_labels.csv, a metadane z exams.csv. 
W odróżnieniu od Sami-Trop nie parujemy po pozycji - exams.csv jest globalny dla całej
bazy, a pliki .hdf5 to jej fragmenty. Dlatego parujemy po exam_id. Jeśli dla danego exam_id
nie ma etykiety albo metadanych, pomijamy record.

Zapisujemy macierz EKG w częstotliwości 100 Hz.
Dodatkowo - macierz etykiet i macierz metadanych [płeć, wiek, normal_ecg].

Układ macierzy EKG: (N, 12, L)
N - liczba recordów
12 - leady w EKG
L - liczba próbek w czasie

Użycie:
    python3 prepare_code15_data.py [-i code15-data] [-o semi-processed-data] [--limit N]
    -i / --data_folder   : folder z danymi CODE-15% (exams_part*.hdf5 + exams.csv
                           + code15_chagas_labels.csv), domyślnie code15-data
    -o / --output_folder : folder na pliki .npy (domyślnie semi-processed-data)
    --limit N            : tylko pierwsze N plików .hdf5 (0 = wszystkie)

Uwaga:  pliki numerowane są od zera (exams_part0.hdf5, exams_part1.hdf5, ...).
        Zakładamy, że jeśli na dysku jest niepełna liczba plików, to są one pobrane w kolejności
        rosnącej indeksów. Pętla po plikach kończy się, gdy nie ma kolejnego pliku albo gdy
        przekroczymy limit plików z --limit; brak pierwszego pliku (exams_part0.hdf5) to błąd.
"""

import argparse
import math
import os
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.signal import resample_poly

SIGNAL_HDF5_PREFIX = 'exams_part'                   # prefix nazwy plików z macierzą sygnałów
SIGNAL_HDF5_SUFFIX = '.hdf5'                        # sufix nazwy plików z macierzą sygnałów
DEMOGRAPHICS_CSV = 'exams.csv'                      # metadane
CHAGAS_LABELS_CSV = 'code15_chagas_labels.csv'      # etykiety
NATIVE_FS = 400                                     # częstotliwość natywna
OUT_FS = 100                                        # częstotliwość docelowa
WINDOW_S = 7                                        # długość okna w sekundach
SEED = 67
_rng = np.random.default_rng(SEED)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Buduje macierze EKG (w 100 Hz), etykiet i metadanych z CODE-15%.')
    parser.add_argument('-i', '--data_folder', type=str, default='code15-data',
                        help='Folder z danymi CODE-15%% (domyślnie: code15-data).')
    parser.add_argument('-o', '--output_folder', type=str, default='semi-processed-data',
                        help='Folder na pliki wyjściowe .npy (domyślnie: semi-processed-data).')
    parser.add_argument('--limit', type=int, default=0,
                        help='Opcjonalnie: weź tylko pierwsze N plików .hdf5 (0 = wszystkie).')
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


# Wczytanie metadanych i etykiet do słowników po exam_id
# (meta_by_id -> wiersz [płeć, wiek, normal_ecg], label_by_id -> 0/1 chagas)
def load_lookup_tables(data_folder):
    csv_path = os.path.join(data_folder, DEMOGRAPHICS_CSV)
    labels_path = os.path.join(data_folder, CHAGAS_LABELS_CSV)
    if not os.path.isfile(csv_path):
        sys.exit(f'Nie znaleziono pliku metadanych: {csv_path}')
    if not os.path.isfile(labels_path):
        sys.exit(f'Nie znaleziono pliku etykiet: {labels_path}')

    df_meta = pd.read_csv(
        csv_path, usecols=['exam_id', 'is_male', 'age', 'normal_ecg'])
    meta_by_id = {
        int(r.exam_id): build_metadata_row(r.is_male, r.age, r.normal_ecg)
        for r in df_meta.itertuples(index=False)
    }

    df_lab = pd.read_csv(labels_path, usecols=['exam_id', 'chagas'])
    label_by_id = {int(r.exam_id): int(bool(r.chagas)) for r in df_lab.itertuples(index=False)}

    return meta_by_id, label_by_id


# Iteruje po plikach exams_part{i}.hdf5 i zwraca kolejno (sig, meta_row, label) dla recordów,
# które przeszły kontrolę.
def iter_valid_records(args, meta_by_id, label_by_id):
    win400 = WINDOW_S * NATIVE_FS
    n_files = 0
    file_idx = 0
    while True:
        if args.limit and args.limit > 0 and n_files >= args.limit:
            break

        hdf5_path = os.path.join(
            args.data_folder, f'{SIGNAL_HDF5_PREFIX}{file_idx}{SIGNAL_HDF5_SUFFIX}')
        if not os.path.isfile(hdf5_path):
            if file_idx == 0:
                sys.exit(f'Nie znaleziono pierwszego pliku sygnałów: {hdf5_path}')
            break                                          # skończyły się kolejne części

        print(f'Plik {os.path.basename(hdf5_path)} ...')
        with h5py.File(hdf5_path, 'r') as f:
            tracings = f['tracings']
            file_exam_ids = np.asarray(f['exam_id'])
            n_in_file = min(tracings.shape[0], file_exam_ids.shape[0])

            for j in range(n_in_file):
                exam_id = int(file_exam_ids[j])

                # Parujemy po exam_id - bez etykiety albo metadanych nie ma sensu brać recordu
                if exam_id not in label_by_id or exam_id not in meta_by_id:
                    continue

                sig = np.asarray(tracings[j], dtype=np.float32)   # (4096, 12), 400 Hz, mV
                sig = strip_zero_padding(sig)                     # usuń padding zerowy
                if sig is None:
                    continue
                # Kontrola jakości - odrzucamy NaN/Inf i martwe zapisy
                if not is_valid(sig):
                    continue
                # Sygnał krótszy niż 7 s -> pomijamy (okna nie da się wyciąć)
                if sig.shape[0] < win400:
                    continue

                yield sig, meta_by_id[exam_id], label_by_id[exam_id]

        n_files += 1
        file_idx += 1


# Główna procedura:
# Najpierw liczymy recordy, które później zapiszemy.
# Następnie zapisujemy sygnały do pliku .npy przez memmap.
def run(args):
    meta_by_id, label_by_id = load_lookup_tables(args.data_folder)

    # Długość okna dla częstotliwości docelowej
    window_len = WINDOW_S * OUT_FS

    # Liczymy recordy
    print('Liczenie recordów ...')
    n = sum(1 for _ in iter_valid_records(args, meta_by_id, label_by_id))
    if n == 0:
        sys.exit('Brak recordów spełniających kryteria.')
    print(f'Recordów do zapisania: {n}')

    os.makedirs(args.output_folder, exist_ok=True)
    out = args.output_folder
    ecg_path = os.path.join(out, 'ecg_code15_100hz.npy')

    # Zapisujemy sygnały przez memmap (N, 12, L)
    print('Zapis sygnałów ...')
    ecg_100 = np.lib.format.open_memmap(
        ecg_path, mode='w+', dtype=np.float32, shape=(n, 12, window_len))
    labels = np.empty(n, dtype=np.int64)        # 0 -> zdrowy, 1 -> chory
    metadata = np.empty((n, 3), dtype=np.float32)  # [płeć, wiek, normal_ecg]

    i = 0
    for sig, meta_row, label in iter_valid_records(args, meta_by_id, label_by_id):
        sig400 = random_window_400(sig)                   # losowe 7 s okno @ 400 Hz
        # 100 Hz z resamplingu 7 s okna
        sig100 = resample_signal(sig400, NATIVE_FS, 100)
        ecg_100[i] = sig100[:window_len].T                # surowe mV, bez filtra/normalizacji
        metadata[i] = meta_row
        labels[i] = label
        i += 1

        if i % 250 == 0:
            print(f'  zapisano {i} recordów')

    ecg_100.flush()
    np.save(os.path.join(out, 'labels_code15.npy'), labels)
    np.save(os.path.join(out, 'metadata_code15.npy'), metadata)

    # Podsumowanie
    n_pos = int((labels == 1).sum())
    print(f'\nPrzetworzono {n} rekordów.')
    print(f'{n_pos} chorych, {n - n_pos} zdrowych.')
    print('Zapisano:')
    print(f'  {os.path.basename(ecg_path)} : {ecg_100.shape}')
    print(f'  labels_code15.npy        : {labels.shape}')
    print(f'  metadata_code15.npy      : {metadata.shape}')


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
