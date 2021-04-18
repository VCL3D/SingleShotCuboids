#NOTE: adapted from https://github.com/princeton-vl/pytorch_stacked_hourglass

try:
    from ssc.modules import ConvActiv2d, SphericallyPaddedConv2d
    from ssc.modules import Bottleneck
    from ssc.modules import AntialiasedMaxPool2d
    from ssc.modules import Hourglass
except ImportError:
    from conv2d import ConvActiv2d, SphericallyPaddedConv2d
    from bottleneck import Bottleneck
    from maxpool2d_aa import AntialiasedMaxPool2d
    from hourglass import Hourglass

import torch
import typing

class StackedHourglass(torch.nn.Module):
    def __init__(self,
        stacks:             int=3,
        depth:              int=3,
        in_features:        int=3,
        out_features:       int=8,
        hourglass_features: int=128,
    ):
        super(StackedHourglass, self).__init__()
        self.pre = torch.nn.Sequential(
            ConvActiv2d(
                in_features=in_features,
                out_features=hourglass_features // 4,
                kernel_size=7,
                stride=2,
                padding=3,
            ),            
            Bottleneck(
                in_features=hourglass_features // 4,
                out_features=hourglass_features // 2,
                bottleneck_features=hourglass_features // 2,
                strided=False,
            ),
            AntialiasedMaxPool2d(
                features=hourglass_features // 2,
                kernel_size=3,
            ),
            Bottleneck(
                in_features=hourglass_features // 2,
                out_features=hourglass_features // 2,
                bottleneck_features=hourglass_features // 2,
                strided=False,
            ),
            Bottleneck(
                in_features=hourglass_features // 2,
                out_features=hourglass_features,
                bottleneck_features=hourglass_features,
                strided=False,
            ),
        )
        
        self.hgs = torch.nn.ModuleList([
                torch.nn.Sequential(
                    Hourglass(
                        depth=depth,
                        features=hourglass_features
                    )
                ) for i in range(stacks)
            ] 
        )
        self.features = torch.nn.ModuleList([
                torch.nn.Sequential(
                    Bottleneck(
                        in_features=hourglass_features,
                        out_features=hourglass_features,
                        bottleneck_features=hourglass_features,
                        strided=False,
                    ),
                    ConvActiv2d(
                        in_features=hourglass_features, 
                        out_features=hourglass_features,
                        kernel_size=1,
                        batch_norm=False,
                    )
                ) for i in range(stacks)
            ]
        )
        
        self.outs = torch.nn.ModuleList([
            ConvActiv2d(
                in_features=hourglass_features, 
                out_features=out_features,
                kernel_size=1,
                batch_norm=False,
            ) for i in range(stacks)         
        ])
        self.merge_features = torch.nn.ModuleList([
            torch.nn.Sequential(
                SphericallyPaddedConv2d(
                    in_channels=hourglass_features,
                    out_channels=hourglass_features,
                    kernel_size=1,
                ),
            ) for i in range(stacks-1)
        ])
        self.merge_preds = torch.nn.ModuleList([
            torch.nn.Sequential(
                SphericallyPaddedConv2d(
                    in_channels=out_features,
                    out_channels=hourglass_features,
                    kernel_size=1,
                ),
            ) for i in range(stacks-1)
        ])
        self.stacks = stacks

    def forward(self, 
        image: torch.Tensor,
    ) -> typing.List[torch.Tensor]:
        x = self.pre(image)
        combined_hm_preds = []
        for i in range(self.stacks):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.stacks - 1:                
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return combined_hm_preds

if __name__ == '__main__':
    import toolz

    sh = StackedHourglass()
    CKPT_PATH = './ckpts/ssc.pth'
    state_dict = torch.load(CKPT_PATH)
    sh.load_state_dict(state_dict, strict=True)
    print(toolz.last(sh(torch.rand(5, 3, 256, 512))).shape)
    