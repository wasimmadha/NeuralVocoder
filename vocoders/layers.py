import torch.nn as nn
from torch.autograd import Function

class DilatedConvolution1D(nn.Module):
    """
    Dilated 1D convolution used in WaveNet 
    """
    def __init__(self, in_channels, kernel_size, dilation, padding, out_channels=None):
        super(DilatedConvolution1D, self).__init__()

        if not out_channels:
            out_channels = in_channels

        self.dilated_conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
    
    def initialize_weights(self):
        nn.init.xavier_normal_(self.dilated_conv.weight)
        if self.dilated_conv.bias is not None:
            nn.init.constant_(self.dilated_conv.bias, 0)

    def forward(self, x):
        return self.dilated_conv(x)

class ConstantPad1d(Function):
    @staticmethod
    def forward(ctx, input, target_size, dimension=0, value=0, pad_start=False):
        ctx.target_size = target_size
        ctx.dimension = dimension
        ctx.value = value
        ctx.pad_start = pad_start

        num_pad = ctx.target_size - input.size(ctx.dimension)
        assert num_pad >= 0, 'target size has to be greater than input size'

        ctx.input_size = input.size()

        size = list(input.size())
        size[ctx.dimension] = ctx.target_size
        output = input.new_zeros(*size).fill_(ctx.value)
        c_output = output

        # crop output
        if ctx.pad_start:
            c_output = c_output.narrow(ctx.dimension, num_pad, c_output.size(ctx.dimension) - num_pad)
        else:
            c_output = c_output.narrow(ctx.dimension, 0, c_output.size(ctx.dimension) - num_pad)

        c_output.copy_(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(*ctx.input_size)
        cg_output = grad_output

        # crop grad_output
        if ctx.pad_start:
            cg_output = cg_output.narrow(ctx.dimension, ctx.target_size - ctx.input_size[ctx.dimension], cg_output.size(ctx.dimension) - (ctx.target_size - ctx.input_size[ctx.dimension]))
        else:
            cg_output = cg_output.narrow(ctx.dimension, 0, cg_output.size(ctx.dimension) - (ctx.target_size - ctx.input_size[ctx.dimension]))

        grad_input.copy_(cg_output)
        return grad_input, None, None, None, None

def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d.apply(input, target_size, dimension, value, pad_start)
