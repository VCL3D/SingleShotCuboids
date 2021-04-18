#NOTE: Code adapted from https://github.com/sunset1995/HorizonNet  

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools

class HorizonNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])
      
    class Resnet(nn.Module):
        def __init__(self, backbone='resnet50', pretrained=False):
            super(HorizonNet.Resnet, self).__init__()
            self.encoder = getattr(models, backbone)(pretrained=pretrained)
            del self.encoder.fc, self.encoder.avgpool

        def forward(self, x):
            features = []
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)

            x = self.encoder.layer1(x);  features.append(x)  # 1/4
            x = self.encoder.layer2(x);  features.append(x)  # 1/8
            x = self.encoder.layer3(x);  features.append(x)  # 1/16
            x = self.encoder.layer4(x);  features.append(x)  # 1/32
            return features

        def list_blocks(self):
            lst = [m for m in self.encoder.children()]
            block0 = lst[:4]
            block1 = lst[4:5]
            block2 = lst[5:6]
            block3 = lst[6:7]
            block4 = lst[7:8]
            return block0, block1, block2, block3, block4

    class LR_PAD(nn.Module):
        ''' Pad left/right-most to each other instead of zero padding '''
        def __init__(self, padding=1):
            super(HorizonNet.LR_PAD, self).__init__()
            self.padding = padding

        def forward(self, x):
            return torch.cat([x[..., -self.padding:], x, x[..., :self.padding]], dim=3)

    @classmethod
    def wrap_lr_pad(cls, net):
        for name, m in net.named_modules():
            if not isinstance(m, nn.Conv2d):
                continue
            if m.padding[1] == 0:
                continue
            w_pad = int(m.padding[1])
            m.padding = (m.padding[0], 0)
            names = name.split('.')
            root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
            setattr(
                root, names[-1],
                nn.Sequential(HorizonNet.LR_PAD(w_pad), m)
            )

    class ConvCompressH(nn.Module):
        ''' Reduce feature height by factor of two '''
        def __init__(self, in_c, out_c, ks=3):
            super(HorizonNet.ConvCompressH, self).__init__()
            assert ks % 2 == 1
            self.layers = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.layers(x)

    class GlobalHeightConv(nn.Module):
        def __init__(self, in_c, out_c):
            super(HorizonNet.GlobalHeightConv, self).__init__()
            self.layer = nn.Sequential(
                HorizonNet.ConvCompressH(in_c, in_c//2),
                HorizonNet.ConvCompressH(in_c//2, in_c//2),
                HorizonNet.ConvCompressH(in_c//2, in_c//4),
                HorizonNet.ConvCompressH(in_c//4, out_c),
            )

        def forward(self, x, out_w):
            x = self.layer(x)

            assert out_w % x.shape[3] == 0
            factor = out_w // x.shape[3]
            x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
            x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
            x = x[..., factor:-factor]
            return x

    class GlobalHeightStage(nn.Module):
        def __init__(self, c1, c2, c3, c4, out_scale=8):
            ''' Process 4 blocks from encoder to single multiscale features '''
            super(HorizonNet.GlobalHeightStage, self).__init__()
            self.cs = c1, c2, c3, c4
            self.out_scale = out_scale
            self.ghc_lst = nn.ModuleList([
                HorizonNet.GlobalHeightConv(c1, c1//out_scale),
                HorizonNet.GlobalHeightConv(c2, c2//out_scale),
                HorizonNet.GlobalHeightConv(c3, c3//out_scale),
                HorizonNet.GlobalHeightConv(c4, c4//out_scale),
            ])

        def forward(self, conv_list, out_w):
            assert len(conv_list) == 4
            bs = conv_list[0].shape[0]
            feature = torch.cat([
                f(x, out_w).reshape(bs, -1, out_w)
                for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
            ], dim=1)
            return feature

    def __init__(self):
        super(HorizonNet, self).__init__()
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512
        self.feature_extractor = HorizonNet.Resnet()        
        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]
            c_last = (c1*8 + c2*4 + c3*2 + c4*1) // self.out_scale
        # Convert features from 4 blocks of the encoder into B x C x 1 x W'
        self.reduce_height_module = HorizonNet.GlobalHeightStage(c1, c2, c3, c4, self.out_scale)

        self.bi_rnn = nn.LSTM(input_size=c_last,
                                hidden_size=self.rnn_hidden_size,
                                num_layers=2,
                                dropout=0.5,
                                batch_first=False,
                                bidirectional=True)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                out_features=3 * self.step_cols)
        self.linear.bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
        self.linear.bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
        self.linear.bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        HorizonNet.wrap_lr_pad(self)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x):
        if x.shape[2] != 512 or x.shape[3] != 1024:
            raise NotImplementedError()
        x = self._prepare_x(x)
        conv_list = self.feature_extractor(x)
        feature = self.reduce_height_module(conv_list, x.shape[3]//self.step_cols)
        feature = feature.permute(2, 0, 1)  # [w, b, c*h]
        output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
        output = self.drop_out(output)
        output = self.linear(output)  # [seq_len, b, 3 * step_cols]
        output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
        output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
        output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W
        return bon, cor