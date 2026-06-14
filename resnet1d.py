import torch
import numpy as np
import matplotlib.pyplot as plt
from device import device

from inception_time import InceptionNetwork
from training import make_mmap_datasets, plot_sample_ecg, train, validate


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
    def __init__(self, use_meta=False, meta_dim=3):
        super().__init__()
        self.ecg_channels = 12
        self.input_channels = 64
        self.use_meta = use_meta
        self.meta_dim = meta_dim if use_meta else 0

        self.conv_pre = torch.nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bne_pre = torch.nn.BatchNorm1d(self.input_channels)

        self.layer1 = self._layer(in_channels=64, out_channels=64, stride=1, num_blocks=2)
        self.layer2 = self._layer(in_channels=64, out_channels=128, stride=2, num_blocks=2)
        self.layer3 = self._layer(in_channels=128, out_channels=256, stride=2, num_blocks=2)
        self.layer4 = self._layer(in_channels=256, out_channels=512, stride=2, num_blocks=2)

        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(512 + self.meta_dim, 1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, input: torch.Tensor, meta: torch.Tensor):
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
        if self.use_meta:
            out = torch.cat([out, meta], dim=1)
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


if __name__ == "__main__":
    NUM_EPOCHS = 30
    BATCH_SIZE = 64  # 2.5x dłuższe sygnały niż przy 100 Hz, więc mniejszy batch
    NUM_WORKERS = 4
    SUBSET_FRACTION = 1.0  # 1.0 = pełny zbiór; ustaw mniej dla szybkiego sanity check
    EARLY_STOPPING_PATIENCE = 5
    DIRPATH = "processed-data"
    FILEPATH = DIRPATH + "/" + "ecg_merged_100hz_resampled.npy"
    LABELS = DIRPATH + "/" + "labels_merged.npy"
    METADATA = DIRPATH + "/" + "metadata_merged.npy"
    CHECKPOINT = "best_resnet1d.pt"

    # Datasety przez mmap - nie ładujemy całego pliku do RAM
    train_ds, val_ds, test_ds, y_train, y_val, y_test = make_mmap_datasets(
        FILEPATH, LABELS, metadata_path=METADATA, subset_fraction=SUBSET_FRACTION, seed=42
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Pozytywnych w train: {int(y_train.sum())} / {len(y_train)}")

    # Podgląd pierwszego przykładu (też przez mmap)
    example_x, _, _ = train_ds[0]
    plot_sample_ecg(example_x.numpy())

    model = InceptionNetwork(in_channels=12).to(device)
    compiled_model = torch.compile(model)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    train_data_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    eval_data_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    # Parametry do treningu
    lr = 0.0001
    weight_decay = (
        1e-3  # silniejsza regularyzacja L2 niż domyślny AdamW=0.01 - opcja walki z overfittem
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=lr, weight_decay=weight_decay)

    train(
        optimizer,
        criterion,
        compiled_model,
        NUM_EPOCHS,
        train_data_loader,
        eval_data_loader,
        early_stopping=EARLY_STOPPING_PATIENCE,
        checkpoint_path=CHECKPOINT,
    )

    test_auroc, test_auprc, test_acc = validate(test_loader, compiled_model)
    print(f"\n=== TEST ===\nAUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, ACC: {test_acc:.4f}")
