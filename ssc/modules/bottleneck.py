try:
    from ssc.modules.conv2d import SphericallyPaddedConv2d
except ImportError:
    from conv2d import SphericallyPaddedConv2d

import torch

class Bottleneck(torch.nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        strided: bool,
    ):
        super(Bottleneck, self).__init__()
        self.W1 = SphericallyPaddedConv2d(
            in_channels=in_features,
            out_channels=bottleneck_features,
            stride=2 if strided else 1,
            kernel_size=1,
        )
        self.A1 = torch.nn.ModuleDict({
            'bn': torch.nn.BatchNorm2d(bottleneck_features),
            'activation': torch.nn.ReLU(inplace=True)
        })
        self.W2 = SphericallyPaddedConv2d(
            in_channels=bottleneck_features,
            out_channels=bottleneck_features,
            kernel_size=3,
            padding=1,
        )
        self.A2 = torch.nn.ModuleDict({
            'bn': torch.nn.BatchNorm2d(bottleneck_features),
            'activation': torch.nn.ReLU(inplace=True)
        })
        self.W3 = SphericallyPaddedConv2d(
            in_channels=bottleneck_features,
            out_channels=out_features,
            kernel_size=1,
        )
        self.A3 = torch.nn.ModuleDict({
            'bn': torch.nn.BatchNorm2d(out_features),
            'activation': torch.nn.ReLU(inplace=True)
        })
        self.S = torch.nn.Identity() if in_features == out_features\
            else SphericallyPaddedConv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
            ) if not strided else SphericallyPaddedConv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                stride=2,
                padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W3(self.A2.bn(self.A2.activation(
            self.W2(self.A1.bn(self.A1.activation(
                self.W1(x)
            )))
        )))
        return self.A3.bn(self.A3.activation(self.S(x) + y))

class PreActivatedBottleneck(torch.nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        strided: bool,
    ):
        super(PreActivatedBottleneck, self).__init__()
        self.A1 = torch.nn.ModuleDict({
            'bn': torch.nn.BatchNorm2d(in_features),
            'activation': torch.nn.ReLU(inplace=True)
        })
        self.W1 = SphericallyPaddedConv2d(
            in_channels=in_features,
            out_channels=bottleneck_features,
            kernel_size=1,
            stride=2 if strided else 1,
        )
        self.A2 = torch.nn.ModuleDict({
            'bn': torch.nn.BatchNorm2d(bottleneck_features),
            'activation': torch.nn.ReLU(inplace=True)
        })
        self.W2 = SphericallyPaddedConv2d(
            in_channels=bottleneck_features,
            out_channels=bottleneck_features,
            kernel_size=3,
            padding=1,
        )
        self.A3 = torch.nn.ModuleDict({
            'bn': torch.nn.BatchNorm2d(bottleneck_features),
            'activation': torch.nn.ReLU(inplace=True)
        })
        self.W3 = SphericallyPaddedConv2d(
            in_channels=bottleneck_features,
            out_channels=out_features,
            kernel_size=1,
        )
        self.S = torch.nn.Identity() if in_features == out_features\
            else SphericallyPaddedConv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
            ) if not strided else SphericallyPaddedConv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1,
                stride=2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W3(self.A3.bn(self.A3.activation(
            self.W2(self.A2.bn(self.A2.activation(
                self.W1(self.A1.bn(self.A1.activation(x)))
            )))
        )))
        return self.S(x) + y    