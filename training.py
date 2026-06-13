from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from device import device


def train(optimizer, criterion, model, num_epochs, data_loader, val_loader):
    for i in range(num_epochs):
        model.train()
        loss = train_one_epoch(optimizer, criterion, model, data_loader, epoch=i)
        auroc, auprc, acc = validate(val_loader, model)
        print(f"Epoch: {i} --- loss = {loss:.4f}")
        print(f"AUROC: {auroc}, AUPRC: {auprc}, ACC: {acc}")


def train_one_epoch(optimizer, criterion, model, data_loader: torch.utils.data.DataLoader, epoch=0):
    epoch_loss = 0.0
    n_samples = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False, unit="batch")
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        y_pred = model(X_batch)
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
    for X, y in data_loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_pred = model(X)
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
