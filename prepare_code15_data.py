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

Zapisujemy 3 macierze EKG w częstotliwościach:
- 100 po resamplingu
- 250 po resamplingu
- 400, oryginalne
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
WINDOW_S = 7                                        # długość okna w sekundach
SEED = 67
_rng = np.random.default_rng(SEED)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Buduje macierze EKG (100/250 resampled, 400 oryginal), etykiet i metadanych z CODE-15%.')
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


# Główna procedura: czyta metadane i etykiety, iteruje po plikach exams_part{i}.hdf5, dla
# każdego rekordu obcina zera, losuje 7 s okno, resampling do 250/100 Hz i zapis macierzy
def run(args):
    meta_by_id, label_by_id = load_lookup_tables(args.data_folder)

    # Długości okna w próbkach dla każdej częstotliwości
    len_100 = WINDOW_S * 100
    len_250 = WINDOW_S * 250
    len_400 = WINDOW_S * 400

    ecg_100, ecg_250, ecg_400 = [], [], []
    metadata, labels, exam_ids = [], [], []
    n_skipped = 0
    n_files = 0

    # Pętla po plikach: budujemy nazwę z prefixu + numeru + sufixu
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
                    n_skipped += 1
                    continue
                meta_row = meta_by_id[exam_id]
                label = label_by_id[exam_id]

                sig = np.asarray(tracings[j], dtype=np.float32)   # (4096, 12), 400 Hz, mV
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
                metadata.append(meta_row)
                labels.append(label)
                exam_ids.append(exam_id)

                if len(exam_ids) % 250 == 0:
                    print(f'  przetworzono {len(exam_ids)} recordów')

        n_files += 1
        file_idx += 1

    n = len(exam_ids)

    # Złożenie macierzy (N, 12, L)
    ecg_100 = np.stack(ecg_100).astype(np.float32)
    ecg_250 = np.stack(ecg_250).astype(np.float32)
    ecg_400 = np.stack(ecg_400).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)        # 0 -> zdrowy, 1 -> chory
    metadata = np.asarray(metadata, dtype=np.float32)  # [płeć, wiek, normal_ecg]
    exam_ids = np.asarray(exam_ids, dtype=np.int64)

    # Zapis wyników
    os.makedirs(args.output_folder, exist_ok=True)
    out = args.output_folder
    np.save(os.path.join(out, 'ecg_code15_100hz_resampled.npy'), ecg_100)
    np.save(os.path.join(out, 'ecg_code15_250hz_resampled.npy'), ecg_250)
    np.save(os.path.join(out, 'ecg_code15_400hz.npy'), ecg_400)
    np.save(os.path.join(out, 'labels_code15.npy'), labels)
    np.save(os.path.join(out, 'metadata_code15.npy'), metadata)
    np.save(os.path.join(out, 'exam_ids_code15.npy'), exam_ids)

    # Podsumowanie
    n_pos = int((labels == 1).sum())
    print(f'\nPrzetworzono {n} rekordów (pominięto {n_skipped}).')
    print(f'{n_pos} chorych, {n - n_pos} zdrowych.')
    print('Zapisano:')
    print(f'  ecg_code15_100hz_resampled.npy : {ecg_100.shape}')
    print(f'  ecg_code15_250hz_resampled.npy : {ecg_250.shape}')
    print(f'  ecg_code15_400hz.npy           : {ecg_400.shape}')
    print(f'  labels_code15.npy              : {labels.shape}')
    print(f'  metadata_code15.npy            : {metadata.shape}')
    print(f'  exam_ids_code15.npy            : {exam_ids.shape}')

    # Podgląd pierwszego recordu dla metadanych i sygnału
    # print('metadata[0] :', metadata[0])
    # print('ecg_400 record 0, próbka 0, wszystkie 12 leadów:\n', ecg_400[0, :, 0])


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
