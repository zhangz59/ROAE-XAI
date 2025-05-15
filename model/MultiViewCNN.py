import torch
import torch.nn as nn


class SixLayerCNN(nn.Module):
    def __init__(self, input_channels):
        super(SixLayerCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class MultiViewCNN(nn.Module):
    def __init__(self, mlp_dims):
        super(MultiViewCNN, self).__init__()

        # Modify CNNs for each direction
        self.cnn1 = SixLayerCNN(91)
        self.cnn2 = SixLayerCNN(109)
        self.cnn3 = SixLayerCNN(91)

        # Define an MLP to combine outputs
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x1 = x.permute(0, 3, 1, 2)  # Assuming the input shape is [batch_size, height, width, channels]
        x2 = x.permute(0, 2, 1, 3)
        x3 = x

        out1 = self.cnn1(x1)
        out2 = self.cnn2(x2)
        out3 = self.cnn3(x3)

        combined_out = torch.cat([out1, out2, out3], dim=1)
        return self.mlp(combined_out)