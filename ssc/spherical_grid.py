import torch
import typing
import numpy as np

__all__ = ["SphericalGrid"]

def _create_exclusive(size: typing.List[int]) -> torch.Tensor:    
    return torch.stack(
        torch.meshgrid(
            *[(torch.arange(dim) + 0.5) / dim for dim in size],
        )
    ).unsqueeze(0)

class Grid(torch.nn.Module):

    def __init__(self,
        width:              int=512,
        height:             int=256,
    ):
        super(Grid, self).__init__()
        size = [height, width]
        unit_grid = _create_exclusive(size)
        grid = torch.addcmul(
            torch.scalar_tensor(-1.0),
            unit_grid,
            torch.scalar_tensor(2.0)
        ) # from [0, 1] to [-1, 1]
        self.register_buffer("grid", torch.flip(grid, dims=[1]))
        
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        b = tensor.shape[0]
        return self.grid.expand(b, *self.grid.shape[1:])

class SphericalGrid(Grid):
    def __init__(self,
        width:              int=512,
    ):
        super(SphericalGrid, self).__init__(width, width // 2)
        scale = torch.Tensor([[np.pi, 0.5 * np.pi]])
        self.grid = self.grid * scale.unsqueeze(-1).unsqueeze(-1)