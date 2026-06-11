from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=None, stride=1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsampling = downsampling
        self.stride = stride

        self.conv1 = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            stride=self.stride,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm1d(self.out_channels)
        self.conv2 = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm1d(self.out_channels)

        self.relu = torch.nn.ReLU()

    def forward(self, input: torch.Tensor):
        feed_forward = input
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsampling != None:
            feed_forward = self.downsampling(feed_forward)

        output = output + feed_forward
        output = self.relu(output)
        return output


class ResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_channels = 12
        self.input_channels = 64

        self.conv_pre = torch.nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bne_pre = torch.nn.BatchNorm1d(self.input_channels)

        self.layer1 = self._layer(in_channels=64, out_channels=64, stride=1, num_blocks=2)
        self.layer2 = self._layer(in_channels=64, out_channels=128, stride=2, num_blocks=2)
        self.layer3 = self._layer(in_channels=128, out_channels=256, stride=2, num_blocks=2)
        self.layer4 = self._layer(in_channels=256, out_channels=512, stride=2, num_blocks=2)

        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, input: torch.Tensor):
        out = self.conv_pre(input)
        out = self.bne_pre(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _layer(self, in_channels, out_channels, stride, num_blocks):
        downsampling = None
        if stride != 1 or in_channels != out_channels:
            downsampling = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                torch.nn.BatchNorm1d(out_channels),
            )

        blocks = []
        block_with_downsampling = SmallResidualBlock(
            in_channels, out_channels, downsampling, stride=stride
        )
        blocks.append(block_with_downsampling)

        for _ in range(num_blocks - 1):
            blocks.append(SmallResidualBlock(out_channels, out_channels))

        return torch.nn.Sequential(*blocks)


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


# Add AUROC, AUPRC, acc
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


def load_data():
    DIRPATH = "processed-data"
    FILEPATH_250HZ = DIRPATH + "/" + "ecg_merged_250hz.npy"
    LABELS = DIRPATH + "/" + "labels_merged.npy"

    ecg250 = np.load(FILEPATH_250HZ)
    y = np.load(LABELS)

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


if __name__ == "__main__":
    NUM_EPOCHS = 20
    BATCH_SIZE = 64

    X, y = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    train_dataset_t, eval_dataset_t, test_dataset_t = make_tensor_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    example_np = np.array(X_train[0])
    example = torch.from_numpy(example_np).to(device)
    print(f"Example device: {example.device}")
    plot_sample_ecg(example_np)

    model = ResNet18().to(device)
    compiled_model = torch.compile(model)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset_t,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset_t,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    # Parametry do treningu
    lr = 0.0001
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=lr)

    train(optimizer, criterion, compiled_model, NUM_EPOCHS, train_data_loader, eval_data_loader)
