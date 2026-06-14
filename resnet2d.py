import torch
from torch.utils.data import DataLoader

from device import device
from resnet2d_data import make_cached_spectrogram_datasets
from training import train, validate


class SmallResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=None, stride=1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsampling = downsampling
        self.stride = stride
        
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size=3,
            padding=1, 
            stride=stride,
            bias=False)
        
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.relu = torch.nn.ReLU()

    def forward(self, input):
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
    def __init__(self, in_channels=12):
        super().__init__()
        self.ecg_channels = 12
        self.input_channels = 64
        
        self.conv_pre = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bne_pre = torch.nn.BatchNorm2d(64)
        
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._layer(in_channels=64, out_channels=64, stride=1, num_blocks=2)
        self.layer2 = self._layer(in_channels=64, out_channels=128, stride=2, num_blocks=2)
        self.layer3 = self._layer(in_channels=128, out_channels=256, stride=2, num_blocks=2)
        self.layer4 = self._layer(in_channels=256, out_channels=512, stride=2, num_blocks=2)

        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, 1)

    def forward(self, input, meta=None):
        out = self.relu(self.bne_pre(self.conv_pre(input)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

    def _layer(self, in_channels, out_channels, stride, num_blocks):
        downsampling = None
        if stride != 1 or in_channels != out_channels:
            downsampling = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )
        blocks = [SmallResidualBlock(in_channels, out_channels, downsampling, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(SmallResidualBlock(out_channels, out_channels))
        return torch.nn.Sequential(*blocks)


if __name__ == "__main__":
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    CACHE_FRACTION = 1.0  # ułamek każdego splitu trzymany w RAM; reszta z mmap (dysk)
    EARLY_STOPPING_PATIENCE = 5
    CHECKPOINT = "best_resnet2d.pt"
    DIRPATH = "processed-data"
    ECG_PATH = DIRPATH + "/" + "ecg_merged_100hz_resampled.npy"
    LABELS = DIRPATH + "/" + "labels_merged.npy"

    train_dataset_t, eval_dataset_t, test_dataset_t, y_train, y_val, y_test = (
        make_cached_spectrogram_datasets(ECG_PATH, LABELS, cache_fraction=CACHE_FRACTION, seed=42)
    )

    model = ResNet18().to(device)
    compiled_model = torch.compile(model)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    # num_workers>0, bo spektrogram liczony jest on-the-fly (CPU-heavy).
    # Uwaga: na macOS (spawn) cache RAM jest duplikowany per-worker; przy dużym
    # cache_fraction rozważ mniej workerów albo niższy cache_fraction.
    train_data_loader = DataLoader(
        train_dataset_t,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    eval_data_loader = DataLoader(
        eval_dataset_t,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    test_data_loader = DataLoader(
        test_dataset_t,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # Parametry do treningu
    lr = 0.0001
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=lr)

    best_auprc, best_epoch, history = train(
        optimizer,
        criterion,
        compiled_model,
        NUM_EPOCHS,
        train_data_loader,
        eval_data_loader,
        early_stopping=EARLY_STOPPING_PATIENCE,
        checkpoint_path=CHECKPOINT,
    )

    # train() przywraca najlepsze wagi (po AUPRC na walidacji) przed zwróceniem
    test_auroc, test_auprc, test_acc = validate(test_data_loader, compiled_model)
    print(f"\n=== TEST ===\nAUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, ACC: {test_acc:.4f}")