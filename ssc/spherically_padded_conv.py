import torch
import functools

__all__ = [
    "SphericalConv2d",
]

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
        padding:        int = 1
    ):
        super(SphericalPad2d, self).__init__()
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(
            horizontal_circular_pad2d(
                x, pad=self.padding
            ),
            pad=[0, 0, self.padding, self.padding], mode='replicate'
        )

class SphericalConv2d(SphericalPad2d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        dilation: int=1,
        stride: int=1,
        padding: int=0,
        groups: int=1,
        bias: bool=True
    ):
        super(SphericalConv2d, self).__init__(padding=padding if kernel_size > 1 else 0)
        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            groups=groups,
            padding=0,
            padding_mode='zeros'
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = super(SphericalConv2d, self).forward(x)
        return self.conv2d(padded)