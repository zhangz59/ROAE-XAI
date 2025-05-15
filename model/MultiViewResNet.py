import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet(nn.Module):
    def __init__(self, base_model, input_channels):
        super(ModifiedResNet, self).__init__()

        # Modify the first convolution layer to accept `input_channels`
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(base_model.children())[1:-1]  # Excluding the original first conv and last fc layers
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten the features

class MultiViewResNet(nn.Module):
    def __init__(self, mlp_dims):
        super(MultiViewResNet, self).__init__()

        base_model = models.resnet18(pretrained=True)

        # Modify ResNets for each direction
        self.resnet1 = ModifiedResNet(base_model, 91)
        self.resnet2 = ModifiedResNet(base_model, 109)
        self.resnet3 = ModifiedResNet(base_model, 91)

        # Define an MLP to combine outputs
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.5))  # Added dropout for regularization
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x1 = x.permute(0, 3, 1, 2)
        x2 = x.permute(0, 2, 1, 3)
        x3 = x
        out1 = self.resnet1(x1)
        out2 = self.resnet2(x2)
        out3 = self.resnet3(x3)

        combined_out = torch.cat([out1, out2, out3], dim=1)
        return self.mlp(combined_out)