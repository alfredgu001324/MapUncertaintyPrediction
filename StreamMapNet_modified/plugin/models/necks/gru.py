import torch
import torch.nn as nn
from IPython import embed
from mmdet.models import NECKS
from mmcv.cnn.utils import kaiming_init, constant_init


@NECKS.register_module()
class ConvGRU(nn.Module):
    def __init__(self, out_channels):
        super(ConvGRU, self).__init__()
        kernel_size = 1
        padding = kernel_size // 2
        self.convz = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.convr = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.convq = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.ln = nn.LayerNorm(out_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, h, x):
        if len(h.shape) == 3:
            h = h.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        hx = torch.cat([h, x], dim=1) # [1, 2c, h, w]
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        new_x = torch.cat([r * h, x], dim=1) # [1, 2c, h, w]
        q = self.convq(new_x)

        out = ((1 - z) * h + z * q).squeeze(0) # (1, C, H, W)
        out = self.ln(out.permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        return out
