import torch
import numpy as np
import matplotlib.pyplot as plt
from device import device

from training import load_data, make_tensor_datasets, plot_sample_ecg, split_data, train


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


if __name__ == "__main__":
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    DIRPATH = "processed-data"
    FILEPATH_250HZ = DIRPATH + "/" + "ecg_merged_250hz.npy"
    LABELS = DIRPATH + "/" + "labels_merged.npy"

    X, y = load_data(FILEPATH_250HZ, LABELS)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    train_dataset_t, eval_dataset_t, test_dataset_t = make_tensor_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    example_np = np.array(X_train[0])
    example = torch.from_numpy(example_np).to(device)
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
