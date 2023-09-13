import torch.nn as nn
import torchvision.models as models
from utils import load_transp_conv_weights
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, font_path : str, transposed_kernel_size : int, transposed_stride : int, max_letter_per_pix: int, letters : list[str], eps=1/255.):
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = models.resnet18(pretrained=True).conv1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=len(letters), kernel_size=5, stride=1, padding=2)
        nn.init.xavier_normal_(self.conv2.weight)
        transposed_convs_weights = load_transp_conv_weights(font_path, transposed_kernel_size, letters)
        self.transp_conv = CustomTransposedConv2d(transposed_convs_weights, len(letters), 1, transposed_kernel_size, transposed_stride, 0)
        self.transp_conv.requires_grad_(False)
        self.max_letter_per_pix = max_letter_per_pix
        self.num_letters = len(letters)
        self.eps = eps

    def forward(self, x):
       feat1 = self.tanh(self.conv1(x))
       feat2 = self.sigmoid(self.conv2(feat1))
       # cap of max five letters per pixel
       
       _, indices = torch.topk(feat2, self.max_letter_per_pix, dim=1)
       mask = torch.zeros_like(feat2)
       mask = mask.scatter(1, indices, 1)
       feat2_masked = feat2 * mask
       out_img = self.transp_conv(feat2_masked)
       out_img = torch.where(out_img < self.eps, torch.tensor(0.0), out_img)
       #out_img = torch.where(out_img > 1.0, torch.tensor(1.0), out_img)
       # TODO 

       return out_img, feat2_masked

class CustomTransposedConv2d(nn.Module):
    def __init__(self, weights, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomTransposedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = nn.Parameter(weights)
        
        # Define the hard-coded weights (example weights)
        #self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        # Perform the transposed convolution operation with hard-coded weights
        output = nn.functional.conv_transpose2d(x, self.weights, bias=self.bias,
                                                       stride=self.stride, padding=self.padding)
        return output