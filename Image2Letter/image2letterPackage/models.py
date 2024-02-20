import torch.nn as nn
import torch


class CustomTransposedConv2d(nn.Module):
    def __init__(
        self,
        weights,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        out_padding=0,
    ):
        super(CustomTransposedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = out_padding
        self.weights = nn.Parameter(weights)

        # Define the hard-coded weights (example weights)
        # self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # TODO normalize weights here
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Perform the transposed convolution operation with hard-coded weights
        output = nn.functional.conv_transpose2d(
            x,
            self.weights,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.out_padding,
        )
        return output
