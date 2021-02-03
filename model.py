import torch
import torch.nn as nn

# Torch setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model():
    return CNN().to(device)


class Bottleneck(nn.Module):
    def __init__(self, input_filters, output_filters, downsampling=False, first_layer=False):
        super(Bottleneck, self).__init__()

        self.downsampling = downsampling
        self.first_layer = first_layer
        conv_stride = (2, 2) if self.downsampling else (1, 1)

        self.sequential = nn.Sequential(
            nn.Conv2d(input_filters, output_filters//4, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_filters//4),
            nn.Conv2d(output_filters//4, output_filters//4, kernel_size=(3, 3), padding=1, stride=conv_stride, bias=False),
            nn.BatchNorm2d(output_filters//4),
            nn.Conv2d(output_filters//4, output_filters, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_filters),
            nn.Dropout2d(0.1),
            nn.ReLU(),
        )

        if self.downsampling or self.first_layer:
            ds_stride = (1, 1) if self.first_layer else (2, 2)
            self.downsample = nn.Sequential(
                nn.Conv2d(input_filters, output_filters, kernel_size=(1, 1), stride=ds_stride, bias=False),
                nn.BatchNorm2d(output_filters)
            )

    def forward(self, x):
        initial = x
        x = self.sequential(x)

        if self.downsampling or self.first_layer:
            x += self.downsample(initial)
        else:
            x += initial
        return x


class ResNetLayer(nn.Module):
    def __init__(self, input_filters, output_filters, num_blocks, downsampling=True, first_layer=False):
        super(ResNetLayer, self).__init__()

        self.sequence = nn.Sequential()
        for i in range(num_blocks):
            if i == 0:
                self.sequence.add_module(f'{i}', Bottleneck(input_filters, output_filters, downsampling, first_layer))
            else:
                self.sequence.add_module(f'{i}', Bottleneck(output_filters, output_filters))

    def forward(self, x):
        x = self.sequence(x)
        return x


class CNN(nn.Module):
    # Ctor
    def __init__(self):
        super(CNN, self).__init__()

        # 1x32x32
        self.in_sequence = nn.Sequential(
            nn.Conv2d(1, 16, (1, 1), padding=0, stride=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv_sequence = nn.Sequential(
            # 16x32x32
            ResNetLayer(16, 64, 2, downsampling=False, first_layer=True),
            # 64x32x32
            ResNetLayer(64, 128, 2),
            # 128x16x16
            ResNetLayer(128, 256, 2),
            # 256x8x8
            ResNetLayer(256, 512, 2),
        )

        self.out_sequence = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

        self.softmax = nn.Softmax(dim=1)

    # Forward
    def forward(self, x):
        x = self.in_sequence(x)
        x = self.conv_sequence(x)
        x = self.out_sequence(x)
        return x

    # Prediction
    def predict(self, x):
        _, cls = torch.max(self.softmax(self(x).cpu().detach()), 1)
        return cls

    # Prediction with probability
    def predict_proba(self, x):
        rate, cls = torch.max(self.softmax(self(x).cpu().detach()), 1)
        return rate, cls
