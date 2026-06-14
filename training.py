from matplotlib import pyplot as plt
import copy
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from device import device


class MmapECGDataset(torch.utils.data.Dataset):
    def __init__(self, ecg_path, labels, indices, metadata, augmentation=False):
        self.ecg_path = ecg_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)
        self.meta = torch.from_numpy(metadata.astype(np.float32))
        self._mmap = None
        self.augmentation = augmentation

    def _ensure_mmap(self):
        if self._mmap is None:
            self._mmap = np.load(self.ecg_path, mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def _augment(self, x):
        # dodanie szumu w około połowie danych
        if np.random.rand() < 0.5:
            noise = np.random.randn(*x.shape).astype(np.float32) * 0.05
            x = x + torch.from_numpy(noise)
        # wyzerowanie losowego kanału w 20% danych
        if np.random.rand() < 0.2:
            channel_id = np.random.randint(0, 12)
            x[channel_id, :] = 0

    def __getitem__(self, i):
        self._ensure_mmap()
        assert self._mmap is not None
        idx = int(self.indices[i])
        x = torch.from_numpy(self._mmap[idx].astype(np.float32, copy=True))
        if self.augmentation is True:
            self._augment(x)

        return x, self.meta[i], self.labels[i]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmap"] = None
        return state


def train(
    optimizer,
    criterion,
    model,
    num_epochs,
    data_loader,
    val_loader,
    early_stopping=None,
    checkpoint_path=None,
):
    best_auprc = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []  # lista dictow per epoch - do zapisu metryk i pozniejszego rysowania

    for i in range(num_epochs):
        model.train()
        loss = train_one_epoch(optimizer, criterion, model, data_loader, epoch=i)
        auroc, auprc, acc = validate(val_loader, model)
        print(f"Epoch: {i} --- loss = {loss:.4f}")
        print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, ACC: {acc:.4f}")

        history.append(
            {
                "epoch": i,
                "train_loss": float(loss),
                "val_auroc": float(auroc),
                "val_auprc": float(auprc),
                "val_acc": float(acc),
            }
        )

        improved = auprc > best_auprc
        if improved:
            best_auprc = auprc
            best_epoch = i
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
                print(f"  -> nowy best AUPRC ({auprc:.4f}), zapisano {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"  bez poprawy ({epochs_no_improve}/{early_stopping or '-'})")

        if early_stopping is not None and epochs_no_improve >= early_stopping:
            print(f"Early stopping po {i + 1} epokach (best AUPRC = {best_auprc:.4f})")
            break

    # Wracamy do najlepszych wag przed zwróceniem
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auprc, best_epoch, history


def train_one_epoch(optimizer, criterion, model, data_loader: torch.utils.data.DataLoader, epoch=0):
    epoch_loss = 0.0
    n_samples = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False, unit="batch")
    for X_batch, meta_batch, y_batch in pbar:
        X_batch = X_batch.to(device, non_blocking=True)
        meta_batch = meta_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        y_pred = model(X_batch, meta_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)

        pbar.set_postfix(loss=f"{epoch_loss / n_samples:.4f}")

    return epoch_loss / n_samples


@torch.no_grad()
def validate(data_loader: torch.utils.data.DataLoader, model):
    model.eval()

    all_labels = []
    all_labels_pred = []
    for X, meta, y in data_loader:
        X = X.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_pred = model(X, meta)
        labels_pred = torch.sigmoid(y_pred)
        all_labels.append(y.cpu())
        all_labels_pred.append(labels_pred.cpu())

    # Zamieniamy sobie z tensora torchowego na numpy ndarray, a potem flattujemy
    all_labels = torch.cat(all_labels).numpy().ravel()
    all_labels_pred = torch.cat(all_labels_pred).numpy().ravel()

    auroc = roc_auc_score(all_labels, all_labels_pred)
    auprc = average_precision_score(all_labels, all_labels_pred)
    acc = ((all_labels_pred > 0.5) == all_labels).mean()

    return auroc, auprc, acc


def make_tensor_datasets(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train_t = torch.from_numpy(X_train)
    X_val_t = torch.from_numpy(X_val)
    X_test_t = torch.from_numpy(X_test)

    y_train_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1)

    train_dataset_t = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    eval_dataset_t = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    test_dataset_t = torch.utils.data.TensorDataset(X_test_t, y_test_t)

    return train_dataset_t, eval_dataset_t, test_dataset_t


# Stratyfikowany split na train/val/test po indeksach - nie kopiuje samych danych
def split_indices(y, test_size=0.15, val_size=0.15, seed=42):
    n = len(y)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size / (1 - test_size), stratify=y[train_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def make_mmap_datasets(
    ecg_path, labels_path, metadata_path=None, subset_fraction=1.0, seed=42, augmentation=False
):
    y_full = np.load(labels_path)
    n_full = len(y_full)

    keep = np.arange(n_full)
    y_kept = y_full[keep]

    tr_loc, va_loc, te_loc = split_indices(y_kept, seed=seed)
    train_idx = keep[tr_loc]
    val_idx = keep[va_loc]
    test_idx = keep[te_loc]

    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    if metadata_path is not None:
        meta_full = np.load(metadata_path).astype(np.float32)
        meta_train = meta_full[train_idx].copy()
        meta_val = meta_full[val_idx].copy()
        meta_test = meta_full[test_idx].copy()

        age_mean = float(meta_train[:, 1].mean())
        age_std = float(meta_train[:, 1].std()) or 1.0
        meta_train[:, 1] = (meta_train[:, 1] - age_mean) / age_std
        meta_val[:, 1] = (meta_val[:, 1] - age_mean) / age_std
        meta_test[:, 1] = (meta_test[:, 1] - age_mean) / age_std
    else:
        # Pusta meta - zerowy wymiar; model wtedy musi miec meta_dim=0
        meta_train = np.zeros((len(train_idx), 0), dtype=np.float32)
        meta_val = np.zeros((len(val_idx), 0), dtype=np.float32)
        meta_test = np.zeros((len(test_idx), 0), dtype=np.float32)

    train_ds = MmapECGDataset(ecg_path, y_train, train_idx, meta_train, augmentation=augmentation)
    val_ds = MmapECGDataset(ecg_path, y_val, val_idx, meta_val, augmentation=False)
    test_ds = MmapECGDataset(ecg_path, y_test, test_idx, meta_test, augmentation=False)
    return train_ds, val_ds, test_ds, y_train, y_val, y_test


def load_data(DATA_FILEPATH, LEABELS_FILE_PATH):
    ecg250 = np.load(DATA_FILEPATH)
    y = np.load(LEABELS_FILE_PATH)

    return ecg250, y


def split_data(X, y):
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15 / 0.85,
        stratify=y_train,
        random_state=seed,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_sample_ecg(example):
    offset = float(np.ptp(example, axis=1).max()) + 1.0
    positions = np.arange(12) * offset

    plt.figure(figsize=(14, 10))
    for i in range(12):
        plt.plot(example[i] + positions[i], linewidth=0.8)
    plt.title("Example ECG Record (12 leads)")
    plt.xlabel("Time (samples at 100 Hz)")
    plt.ylabel("Lead")
    plt.yticks(positions, [f"Lead {i+1}" for i in range(12)])
    plt.grid(alpha=0.3)
    plt.savefig("example_ecg.png", dpi=120, bbox_inches="tight")
    plt.close()
