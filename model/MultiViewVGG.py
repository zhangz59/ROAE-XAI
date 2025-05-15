import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedVGG(nn.Module):
    def __init__(self, base_model, input_channels):
        super(ModifiedVGG, self).__init__()

        # Modify the first convolution layer to accept `input_channels`
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            *list(base_model.features)[1:]
        )
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MultiViewVGG(nn.Module):
    def __init__(self, mlp_dims):
        super(MultiViewVGG, self).__init__()

        base_model = models.vgg19(pretrained=True)
        # Remove the last fully connected layer to obtain features instead of predictions
        base_model.classifier = nn.Sequential(*list(base_model.classifier)[:-1])

        # Modify VGGs for each direction
        self.vgg1 = ModifiedVGG(base_model, 91)
        self.vgg2 = ModifiedVGG(base_model, 109)
        self.vgg3 = ModifiedVGG(base_model, 91)

        # Define an MLP to combine outputs
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x1 = x.permute(0, 3, 1, 2)
        x2 = x.permute(0, 2, 1, 3)
        x3 = x

        out1 = self.vgg1(x1)
        out2 = self.vgg2(x2)
        out3 = self.vgg3(x3)

        combined_out = torch.cat([out1, out2, out3], dim=1)
        return self.mlp(combined_out)