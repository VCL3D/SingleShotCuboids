import torch
import functools

def __pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """
    if isinstance(dim, int):
        dim = [dim]
    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")
        idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)
        idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass
    return x

horizontal_circular_pad2d = functools.partial(__pad_circular_nd, dim=[3])

class SphericalPad2d(torch.nn.Module):
    def __init__(self,
        padding:        int = 1,
    ):
        super(SphericalPad2d, self).__init__()
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # assumes [B, C, H, W] tensor inputs
        return torch.nn.functional.pad(
            horizontal_circular_pad2d(
                x, pad=self.padding
            ),
            pad=[0, 0, self.padding, self.padding], mode='replicate'
        )

class SphericallyPaddedConv2d(SphericalPad2d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,        
        stride: int=1,
        padding: int=0,
    ):
        super(SphericallyPaddedConv2d, self).__init__(padding=padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,            
            stride=stride,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = super(SphericallyPaddedConv2d, self).forward(x)
        return self.conv2d(padded)

class ConvActiv2d(torch.nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        kernel_size: int=3,        
        stride: int=1,
        padding: int=0,
        batch_norm: bool=True,
    ):
        super(ConvActiv2d, self).__init__()
        self.conv = SphericallyPaddedConv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.activation = torch.nn.ModuleDict({
            'bn': torch.nn.BatchNorm2d(out_features)\
                if batch_norm else torch.nn.Identity(),
            'activation': torch.nn.ReLU(inplace=True)
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation.bn(self.activation.activation(self.conv(x)))