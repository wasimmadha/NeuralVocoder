import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from layers import DilatedConvolution1D, constant_pad_1d
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class Upsample1D(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample1D, self).__init__()
        self.scale_factor = 275

    def forward(self, x):
        # Upsample the input tensor
        return F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=False)

class ResBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation) -> None:
        super().__init__()

        self.dilation = dilation
        self.kernel_size = 3
        
        # Calculate padding
        self.padding = (self.kernel_size - 1) * dilation // 2

        self.dilation = dilation 
        self.dilated_conv = DilatedConvolution1D(residual_channels, out_channels=residual_channels, dilation=dilation, kernel_size=self.kernel_size, padding=self.padding)
        self.filter_conv = nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=1,
                                                   bias=True)
        self.gate_conv = nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=1,
                                                   bias=True)
        
        self.residual_conv = nn.Conv1d(in_channels=residual_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1, bias=False
                                                   )
        
        self.conv_2 = nn.Conv1d(residual_channels, skip_channels, kernel_size=1, bias=False)

    def forward(self, inputs):
        x, skip = inputs

        print("Input shape: ", x.shape)
        x = self.dilated_conv(x)
        print("After Dilated Conv shape: ", x.shape)
        
        filter_x = self.filter_conv(x)
        print("After Filter Conv shape: ", filter_x.shape)
        filter_x = torch.tanh(filter_x)

        gate_x = self.gate_conv(x)
        print("After Gate Conv shape: ", gate_x.shape)
        gate_x = torch.sigmoid(gate_x)

        x = filter_x * gate_x
        print("After Gating shape: ", x.shape)

        skip_x = self.conv_2(x)
        print("After Conv_2 (Skip connection) shape: ", skip_x.shape)
        
        try:
            skip = skip[:, :, -x.size(2):]
        except:
            skip = 0

        skip = skip_x + skip
        print("After Skip Connection shape: ", skip.shape)

        res_x = self.residual_conv(x)
        print("After Residual Conv shape: ", res_x.shape)
        x = x + res_x
        print("After Residual Addition shape: ", x.shape)

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
                                  bias=False)

        self.end_conv_2 = nn.Conv1d(in_channels=residual_channels,
                                    out_channels=256,
                                    kernel_size=1, bias=False)

        self.upsample = Upsample1D(scale_factor=[[11, 25]])
        
    def forward(self, inputs):
        print("Inputs Shape: ", inputs.shape)
        x = self.start_conv(inputs)
        print("Shape after Start Convolution: ", x.shape)
        skip = 0
        
        # Process through residual blocks
        for i, layer in enumerate(self.layers):
            print("Dilation: ", 2**i)
            x, skip = layer((x, skip))
            print(f"After Layer {i}: ", x.shape, skip.shape)

        # Apply final convolutions
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        print("Shape after Final Convolutions: ", x.shape)
        
        # Apply upsampling at the end
        x = self.upsample(x)
        print("Shape after Upsampling: ", x.shape)

        return x
         
if __name__ == "__main__":
    dummy_input = torch.randn(1, 80, 387)
    
    model = WaveNet(in_channels=80, residual_channels=128, skip_channels=128, out_channels=256, residual_blocks=19)
    
    output = model(dummy_input)
    print("Output shape:", output.shape)
