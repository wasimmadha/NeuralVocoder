import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from layers import DilatedConvolution1D, constant_pad_1d
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from utils import mu_law_expansion
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation) -> None:
        super().__init__()
        self.dilation = dilation 
        self.dilated_conv = DilatedConvolution1D(residual_channels, out_channels=residual_channels, dilation=dilation, kernel_size=2, padding=1)
        self.filter_conv = nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=2,
                                                   bias=False)
        self.gate_conv = nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=2,
                                                   bias=False)
        
        self.residual_conv = nn.Conv1d(in_channels=residual_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1, bias=False
                                                   )
        
        self.conv_2 = nn.Conv1d(residual_channels, skip_channels, kernel_size=1, bias=False)

    def forward(self, inputs):
        x, skip = inputs

        x = self.dilated_conv(x)

        filter_x = self.filter_conv(x)
        filter_x = torch.tanh(filter_x)

        gate_x = self.gate_conv(x)
        gate_x = torch.sigmoid(gate_x)

        x = filter_x * gate_x

        skip_x = self.conv_2(x)
        
        try:
            skip = skip[:, :, -x.size(2):]
        except:
            skip = 0

        skip = skip_x + skip

        res_x = self.residual_conv(x)
        x = x + res_x

        return x, skip

class WaveNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, in_channels, residual_channels, skip_channels, out_channels, residual_blocks) -> None:
        super().__init__()
        self.start_conv = nn.Conv1d(in_channels=in_channels,
                            out_channels=residual_channels,
                            kernel_size=1,
                            bias=False)
        
        self.layers = nn.ModuleList([
            ResBlock(residual_channels, skip_channels, 2**i) for i in range(residual_blocks)
        ])

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=residual_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=residual_channels,
                                    out_channels=out_channels,
                                    kernel_size=1, bias=True)
                                   
    def forward(self, inputs):
        print("Inputs Shape: ", inputs.shape)
        x = self.start_conv(inputs)
        print("Shape after Start Convolution: ", x.shape)
        skip = 0
        for i, layer in enumerate(self.layers):
            x, skip = layer((x, skip))
            print(f"After Layer {i}: ", x.shape, skip.shape)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x
     
if __name__ == "__main__":
    dummy_input = torch.randn(1, 1, 16000)
    
    model = WaveNet(in_channels=1, residual_channels=32, skip_channels=80, out_channels=1, residual_blocks=8)
    
    output = model(dummy_input)
    
    print("Output shape:", output.shape)
