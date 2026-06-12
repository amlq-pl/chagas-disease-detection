from black.output import out
import torch
import numpy as np


class InceptionModule(torch.nn.Module):
    def __init__(self, in_channels, num_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels

        self.input_channels = min(self.in_channels, self.num_channels)

        # To jest zarówno dla bottleneck jak i kolejna warstwa dla MaxPool1D
        self.bottleneck = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=num_channels, kernel_size=1, stride=1, bias=False
        )
        self.conv0 = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=num_channels, kernel_size=1, stride=1, bias=False
        )

        self.max_pool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # Kernel size zmniejszony o 1, bo jak jest parzysty to są problemy z paddingiem
        self.conv1 = torch.nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=num_channels,
            kernel_size=9,
            padding="same",
            bias=False,
        )

        self.conv2 = torch.nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=num_channels,
            kernel_size=19,
            padding="same",
            bias=False,
        )

        self.conv3 = torch.nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=num_channels,
            kernel_size=39,
            padding="same",
            bias=False,
        )

        self.bn = torch.nn.BatchNorm1d(num_features=num_channels * 4)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        max_pool_out = self.max_pool(input)
        conv_in = input

        if self.in_channels > self.num_channels:
            conv_in = self.bottleneck(input)

        conv1_out = self.conv1(conv_in)
        conv2_out = self.conv2(conv_in)
        conv3_out = self.conv3(conv_in)

        parallel_out = self.conv0(max_pool_out)

        # konkatenacja po dim=1, czyli po liczbie kanalow 32+32+32+32 = 128
        concat_out = torch.cat((conv1_out, conv2_out, conv3_out, parallel_out), dim=1)
        norm_out = self.bn(concat_out)
        return self.relu(norm_out)


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, num_channels=32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.inter_channels = 4 * num_channels

        self.module1 = InceptionModule(in_channels=in_channels, num_channels=num_channels)
        self.module2 = InceptionModule(in_channels=self.inter_channels, num_channels=num_channels)
        self.module3 = InceptionModule(in_channels=self.inter_channels, num_channels=num_channels)

        self.conv = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=self.inter_channels, kernel_size=1, bias=False
        )

        self.bn = torch.nn.BatchNorm1d(self.inter_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        res_conv_out = self.conv(input)
        output = self.module1(input)
        output = self.module2(output)
        output = self.module3(output)
        res_out = self.bn(res_conv_out)
        output = output + res_out
        return self.relu(output)


class InceptionNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_channels=32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.inter_channels = 4 * self.num_channels

        self.in_block = InceptionBlock(in_channels=self.in_channels, num_channels=self.num_channels)
        self.out_block = InceptionBlock(
            in_channels=self.inter_channels, num_channels=self.num_channels
        )

        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(self.inter_channels, 1)

    def forward(self, input):
        output = self.in_block(input)
        output = self.out_block(output)
        output = self.pool(output)
        output = torch.flatten(output, 1)
        return self.fc(output)
