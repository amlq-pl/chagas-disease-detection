#!/usr/bin/env python3
"""
Skrypt przygotowuje dane badania PTB-XL - w tym przypadku mamy tylko zdrowe EKG.
Ogólnie w tym badaniu mamy 10-sekundowe próbki po 500hz, ale mamy też wersję 100 hz w danych.

Zapisujemy na dysku 3 macierze EKG w różnych częstotliwościach próbkowania:
- 100, które czytamy bezpośrednio z records100 - pewnie jest to dokładniejsze niż nasz resampling
- 100, ale z naszym resamplingiem (niżej dlaczego)
- 250
- 400

100 z naszym resamplingiem, bo mimo, że jest raczej gorszy niż oryginalny, to pozostałe badania nie mają próbek 100 hz
i też będziemy je resamplować, więc będziemy porównywać to samo.

Z każdego 10-sekundowego zapisu wycinamy losowe 7-sekundowe okno (ten sam fragment czasu dla wszystkich
częstotliwości danego rekordu). 7 s, bo część próbek z Sami-Trop ma tylko 7 s.

Dodatkowo zapisujemy:
- macierz etykiet (same zera)
- macierz metadanych: [płeć, wiek, >90]
W danych osoby >90 lat mają wiek ustawiony na 300, więc mamy flagę, która jest ustawiona na 1,
jeśli osoba ma pow. 90 lat - pewnie można byłoby po prostu ustawić na 90,
ale tak jest bardziej clear.

Jeśli chcemy wrzucić mniej danych do macierzy, to możemy użyć argumentu --limit N, żeby wziąć tylko pierwsze N rekordów.

Metadane użyjemy docelowe do middle fusion - wiek i płeć wpływają na EKG, więc na razie są dodane, ale w MVP nie używamy.

Układ macierzy EKG: (N, 12, L)
N - liczba recordów
12 - leady w EKG
L - liczba próbek w czasie

Użycie:
    python3 prepare_ptbxl_data.py [-i ptbxl-data] [-o semi-processed-data] [--limit N]
    -i / --data_folder   : folder z danymi PTB-XL
    -o / --output_folder : folder na pliki .npy (domyślnie semi-processed-data)
    --limit N            : tylko pierwsze N recordów
"""

import argparse
import math
import os
import sys
import numpy as np
import pandas as pd

PTBXL_SUBPATH = os.path.join('files', 'ptb-xl', '1.0.3')  # stała podścieżka wewnątrz folderu danych
DATABASE_CSV = 'ptbxl_database.csv'                       # plik z metadanymi
RECORDS100_DIR = 'records100'                             # sygnały 100 Hz (*_lr)
RECORDS500_DIR = 'records500'                             # sygnały 500 Hz (*_hr)
DURATION_S = 10                                           # długość każdego zapisu PTB-XL w sekundach
WINDOW_S = 7                                              # długość okna - 7 sekund, bo część próbek z Sami-Trop ma tylko 7 s
SEED = 67
_rng = np.random.default_rng(SEED)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Buduje macierze EKG (100/250/400 Hz), etykiet i metadanych z bazy PTB-XL.')
    parser.add_argument('-i', '--data_folder', type=str, default='ptbxl-data',
                        help='Folder z danymi PTB-XL (domyślnie: ptbxl-data).')
    parser.add_argument('-o', '--output_folder', type=str, default='semi-processed-data',
                        help='Folder na pliki wyjściowe .npy (domyślnie: semi-processed-data).')
    parser.add_argument('--limit', type=int, default=0,
                        help='Opcjonalnie: weź tylko pierwsze N rekordów (0 = wszystkie dostępne).')
    return parser


# Wczytanie sygnału z pliku .dat i przeliczenie na miliwolty
# W całym PTB-XL parametry są stałe: 12 leadów, gain 1000, baseline 0
# fs podajemy z zewnątrz (100 dla *_lr, 500 dla *_hr).
# Zwraca tablicę o kształcie (liczba_próbek, 12) oraz fs
def read_signal(record_basepath, fs):
    raw = np.fromfile(record_basepath + '.dat', dtype='<i2')  # 16-bit ze znakiem, little-endian
    raw = raw.reshape(-1, 12)
    return raw.astype(np.float32) / 1000.0, fs


# Resampling sygnału wzdłuż osi czasu z fs_in do fs_out
def resample_signal(sig, fs_in, fs_out):
    if fs_in == fs_out:
        return sig.astype(np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(int(fs_in), int(fs_out))
    up, down = fs_out // g, fs_in // g
    return resample_poly(sig, up, down, axis=0).astype(np.float32)


# Znalezienie recordów: zwraca listę krotek (ecg_id, ścieżka_lr, ścieżka_hr)
# tylko dla tych ecg_id, które mają komplet plików w 100 Hz i 500 Hz.
# Resztę pomijamy
def list_available_records(df, base_dir):
    available = []
    for ecg_id, row in df.iterrows():
        lr = os.path.join(base_dir, row['filename_lr'])
        hr = os.path.join(base_dir, row['filename_hr'])
        if all(os.path.isfile(p) for p in
               (lr + '.dat', lr + '.hea', hr + '.dat', hr + '.hea')):
            available.append((int(ecg_id), lr, hr))
    available.sort(key=lambda t: t[0])
    return available


# Zbudowanie wiersza metadanych [płeć, wiek, >90] dla recordu
# W PTB-XL osoby >90 lat mają ustawiony wiek na 300 — wtedy ustawiamy
# wiek na 89 i flagę na 1. Dla pozostałych flaga to 0
def build_metadata_row(row):
    sex = float(row['sex'])          # 0 = mężczyzna, 1 = kobieta
    age = float(row['age'])
    if age >= 90:
        age = 89.0
        over_90 = 1.0
    else:
        over_90 = 0.0
    return [sex, age, over_90]


# Wycięcie okna WINDOWS_S (w naszym przypadku 7) sekund z sygnału
# start_frac w [0, 1) mapuje to samo położenie okna w czasie na każdą częstotliwość, więc
# wszystkie reprezentacje (100/250/400 Hz) dostają ten sam fragment czasowy recordu
def crop_window(sig, fs, start_frac):
    win = WINDOW_S * fs                          # liczba próbek w oknie
    max_start = sig.shape[0] - win               # ostatni dopuszczalny indeks startu (10s - 7s = 3s zapasu)
    start = int(round(start_frac * max_start)) if max_start > 0 else 0
    return sig[start:start + win]


# Odrzuca rekordy z NaN/Inf albo całkowicie płaskie
def is_valid(sig):
    return bool(np.all(np.isfinite(sig))) and float(sig.std()) >= 1e-6


# Główna procedura: czyta metadane, ustala dostępne rekordy, alokuje i wypełnia
# cztery macierze EKG (100 natywne, 100 z resamplingu, 250, 400), macierz etykiet
# i macierz metadanych, na końcu zapis na dysk
def run(args):
    base_dir = os.path.join(args.data_folder, PTBXL_SUBPATH)
    csv_path = os.path.join(base_dir, DATABASE_CSV)
    if not os.path.isfile(csv_path):
        sys.exit(f'Nie znaleziono pliku metadanych: {csv_path}')

    # Wczytanie metadanych (index = ecg_id)
    df = pd.read_csv(csv_path, index_col='ecg_id')

    # Lista recordów
    records = list_available_records(df, base_dir)
    if args.limit and args.limit > 0:
        records = records[:args.limit]
    n = len(records)
    if n == 0:
        sys.exit('Nie znaleziono żadnych rekordów z kompletem plików (100 Hz i 500 Hz).')
    print(f'Znaleziono {n} rekordów z kompletem sygnałów.')

    # Długości okna dla każdej częstotliwości
    len_100 = WINDOW_S * 100
    len_250 = WINDOW_S * 250
    len_400 = WINDOW_S * 400

    # Alokacja macierzy
    ecg_100 = np.zeros((n, 12, len_100), dtype=np.float32)   # 100 Hz natywne (z records100)
    ecg_100r = np.zeros((n, 12, len_100), dtype=np.float32)  # 100 Hz z resamplingu
    ecg_250 = np.zeros((n, 12, len_250), dtype=np.float32)
    ecg_400 = np.zeros((n, 12, len_400), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)            # wszyscy zdrowi -> 0
    metadata = np.zeros((n, 3), dtype=np.float32)      # [płeć, wiek, >90]
    ecg_ids = np.zeros((n,), dtype=np.int64)           # do identyfikacji wierszy

    # Wypełnianie macierzy. k = pozycja zapisu (rośnie tylko dla dobrych rekordów), n_skipped = odrzucone QC.
    k = 0
    n_skipped = 0
    for i, (ecg_id, lr_path, hr_path) in enumerate(records):
        # 100 Hz natywne (*_lr) i 500 Hz (*_hr)
        sig_lr, _ = read_signal(lr_path, 100)
        sig_hr, _ = read_signal(hr_path, 500)

        # Kontrola jakości na surowym sygnale - odrzucamy NaN/Inf i martwe zapisy
        if not (is_valid(sig_lr) and is_valid(sig_hr)):
            n_skipped += 1
            continue

        # Resampling 500 Hz do 100, 250 i 400 Hz (filtr i standaryzacja - w merge_and_process_data.py)
        sig_100r = resample_signal(sig_hr, 500, 100)
        sig_250 = resample_signal(sig_hr, 500, 250)
        sig_400 = resample_signal(sig_hr, 500, 400)

        # Losowe 7-sekundowe okno (ten sam fragment czasu dla wszystkich częstotliwości)
        start_frac = _rng.random()
        sig_lr = crop_window(sig_lr, 100, start_frac)
        sig_100r = crop_window(sig_100r, 100, start_frac)
        sig_250 = crop_window(sig_250, 250, start_frac)
        sig_400 = crop_window(sig_400, 400, start_frac)

        # Transpozycja do (12, L) i wpisanie do macierzy (surowe mV, bez filtra/normalizacji)
        ecg_100[k] = sig_lr.T
        ecg_100r[k] = sig_100r.T
        ecg_250[k] = sig_250.T
        ecg_400[k] = sig_400.T

        metadata[k] = build_metadata_row(df.loc[ecg_id])
        ecg_ids[k] = ecg_id
        k += 1

        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f'  przetworzono {i + 1}/{n}')

    # Przycięcie macierzy do liczby faktycznie zapisanych recordów
    ecg_100, ecg_100r = ecg_100[:k], ecg_100r[:k]
    ecg_250, ecg_400 = ecg_250[:k], ecg_400[:k]
    labels, metadata, ecg_ids = labels[:k], metadata[:k], ecg_ids[:k]
    if n_skipped:
        print(f'Odrzucono {n_skipped} rekordów (kontrola jakości).')

    # Zapis wyników
    os.makedirs(args.output_folder, exist_ok=True)
    out = args.output_folder
    np.save(os.path.join(out, 'ecg_ptbxl_100hz.npy'), ecg_100)
    np.save(os.path.join(out, 'ecg_ptbxl_100hz_resampled.npy'), ecg_100r)
    np.save(os.path.join(out, 'ecg_ptbxl_250hz.npy'), ecg_250)
    np.save(os.path.join(out, 'ecg_ptbxl_400hz.npy'), ecg_400)
    np.save(os.path.join(out, 'labels_ptbxl.npy'), labels)
    np.save(os.path.join(out, 'metadata_ptbxl.npy'), metadata)
    np.save(os.path.join(out, 'ecg_ids_ptbxl.npy'), ecg_ids)

    # Podsumowanie
    print('\nZapisano:')
    print(f'  ecg_ptbxl_100hz.npy           : {ecg_100.shape}')
    print(f'  ecg_ptbxl_100hz_resampled.npy : {ecg_100r.shape}')
    print(f'  ecg_ptbxl_250hz.npy           : {ecg_250.shape}')
    print(f'  ecg_ptbxl_400hz.npy           : {ecg_400.shape}')
    print(f'  labels_ptbxl.npy              : {labels.shape}')
    print(f'  metadata_ptbxl.npy            : {metadata.shape}')
    print(f'  ecg_ids_ptbxl.npy             : {ecg_ids.shape}')

    # Podgląd pierwszego recordu dla metadanych i sygnału
    # print('metadata[0] :', metadata[0])
    # print('ecg_400 record 0, próbka 0, wszystkie 12 leadów:\n', ecg_400[0, :, 0])


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
