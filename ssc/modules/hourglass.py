#NOTE: adapted from https://github.com/princeton-vl/pytorch_stacked_hourglass

try:
    from ssc.modules.bottleneck import PreActivatedBottleneck
    from ssc.modules.maxpool2d_aa import AntialiasedMaxPool2d
    from ssc.modules.upsample2d import Upsample2d
except ImportError:
    from bottleneck import PreActivatedBottleneck
    from maxpool2d_aa import AntialiasedMaxPool2d
    from upsample2d import Upsample2d

import torch

class Hourglass(torch.nn.Module):
    def __init__(self, 
        depth:              int=4,
        features:           int=128,
    ):
        super(Hourglass, self).__init__()
        new_features = features
        self.up1 = PreActivatedBottleneck(
            in_features=features,
            out_features=features,
            bottleneck_features=features,
            strided=False,
        )   
        self.pool1 = AntialiasedMaxPool2d(
            features=features,
            kernel_size=3,
        )
        self.low1 = PreActivatedBottleneck(
            in_features=features,
            out_features=new_features,
            bottleneck_features=new_features,
            strided=False,
        )    
        self.low2 = Hourglass(
                depth=depth-1, 
                features=new_features,     
            ) if depth > 1 \
            else PreActivatedBottleneck(
                in_features=new_features,
                out_features=new_features,
                bottleneck_features=new_features,
                strided=False,
            )
        self.low3 = PreActivatedBottleneck(
                in_features=new_features,
                out_features=features,
                bottleneck_features=features,
                strided=False,
            )
        self.up2 = Upsample2d(
            mode="bilinear"
        )

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2