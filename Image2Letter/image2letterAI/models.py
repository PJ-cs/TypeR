import torch.nn as nn
import torchvision.models as models
from utils import load_transp_conv_weights
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, font_path : str, transposed_kernel_size : int, transposed_stride : int, letters : list[str]):
        super().__init__()
        self.conv1 = models.resnet18(pretrained=True).conv1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=100, kernel_size=5, stride=1, padding=2)
        transposed_convs_weights = load_transp_conv_weights(font_path, transposed_kernel_size, letters)
        self.transp_conv = CustomTransposedConv2d(transposed_convs_weights, 100, 1, transposed_kernel_size, transposed_stride, 0)
        self.transp_conv.requires_grad_(False)

    def forward(self, x):
       feat1 = nn.ReLU(self.conv1(x))
       feat2 = nn.ReLU(self.conv2(feat1))
       # TODO insert cap of max five letters per pixel
       out_img = self.transp_conv(feat2)
       return out_img

class CustomTransposedConv2d(nn.Module):
    def __init__(self, weights, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomTransposedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = weights
        
        # Define the hard-coded weights (example weights)
        #self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        # Perform the transposed convolution operation with hard-coded weights
        output = nn.functional.conv_transpose2d(x, self.weights, bias=self.bias,
                                                       stride=self.stride, padding=self.padding)
        return output