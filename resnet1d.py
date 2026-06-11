from sklearn.model_selection import train_test_split
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

        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1, stride=self.stride, bias = False)
        self.bn1 = torch.nn.BatchNorm1d(self.out_channels)
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias = False)
        self.bn2 = torch.nn.BatchNorm1d(self.out_channels)

        self.relu = torch.nn.ReLU()

    def forward(self, input: torch.Tensor):
        feed_forward = input
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if (self.downsampling != None):
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
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm1d(out_channels)
            )

        blocks = []
        block_with_downsampling = SmallResidualBlock(in_channels, out_channels, downsampling, stride=stride)
        blocks.append(block_with_downsampling)

        for _ in range(num_blocks - 1):
            blocks.append(SmallResidualBlock(out_channels, out_channels))

        return torch.nn.Sequential(*blocks)

def train(optimizer, criterion, model: torch.nn.Module):
    model.train()
    optimizer.zero_grad()

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
    plt.savefig('example_ecg.png', dpi=120, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    DIRPATH = 'processed-data'
    FILEPATH_250HZ = DIRPATH + '/' + 'ecg_merged_250hz.npy'
    LABELS = DIRPATH + '/' + 'labels_merged.npy'

    ecg250 = np.load(FILEPATH_250HZ, mmap_mode='r')
    print(f"Shape of ECG data: {ecg250.shape}")
    print(f"Data type: {ecg250.dtype}")

    y = np.load(LABELS)
    indices = np.arange(len(y))

    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=y, random_state=42,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.15 / 0.85,
        stratify=y[trainval_idx],
        random_state=42,
    )

    example_np = np.array(ecg250[0])
    example = torch.from_numpy(example_np).to(device)
    print(f"Example device: {example.device}")

    model = ResNet18()

    n_pos = int(y[train_idx].sum())
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

