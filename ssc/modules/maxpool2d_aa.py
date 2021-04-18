#NOTE: adapted from https://github.com/adobe/antialiased-cnns 

try:
    from ssc.modules.conv2d import SphericalPad2d
except ImportError:
    from conv2d import SphericalPad2d

import torch
import numpy as np

class AntialiasedMaxPool2d(torch.nn.MaxPool2d):
    def __init__(self,
        features: int,
        kernel_size: int=2,
        stride: int=2,
        padding: int=0,
        pad_type: str='reflect',
    ):
        super(AntialiasedMaxPool2d, self).__init__(
            kernel_size, stride=1, padding=int(np.ceil(1.0 * (kernel_size - 1.0) / 2.0))
        )
        self.filt_size = kernel_size
        self.pad_off = 0
        self.pad_sizes = [
            int(1.0 * (kernel_size - 1) / 2.0), int(np.ceil(1.0 * (kernel_size - 1.0) / 2.0)),
            int(1.0 * (kernel_size - 1) / 2.0), int(np.ceil(1.0 * (kernel_size - 1) / 2.0))
        ]
        self.pad_sizes = [pad_size + self.pad_off for pad_size in self.pad_sizes]
        self.aa_stride = stride if stride else kernel_size #NOTE: different name due to subclassing
        self.off = int((self.aa_stride - 1) / 2.0)
        self.channels = features

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None] * a[None,:])
        filt = filt / torch.sum(filt)
        self.register_buffer('filter', filt[None,None,:,:].repeat((self.channels, 1, 1, 1)))
        self.pad = self._get_pad_layer(pad_type)(self.pad_sizes)

    def _get_pad_layer(self, pad_type: str):
        if(pad_type in ['refl', 'reflect']):
            PadLayer = torch.nn.ReflectionPad2d
        elif(pad_type in ['repl', 'replicate']):
            PadLayer = torch.nn.ReplicationPad2d
        elif(pad_type=='zero'):
            PadLayer = torch.nn.ZeroPad2d
        elif(pad_type=="spherical"):
            PadLayer = SphericalPad2d
        return PadLayer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pooled = super(AntialiasedMaxPool2d, self).forward(input)
        if(self.kernel_size==1):
            if(self.pad_off==0):
                return pooled[:,:,::self.aa_stride,::self.aa_stride]    
            else:
                return self.pad(pooled)[:,:,::self.aa_stride,::self.aa_stride]
        else:
            return torch.nn.functional.conv2d(self.pad(pooled), 
                self.filter, stride=self.aa_stride, groups=pooled.shape[1]
            )