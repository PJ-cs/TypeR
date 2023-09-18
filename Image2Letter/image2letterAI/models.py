import torch.nn as nn
import torchvision.models as models
from typing import Any
from utils import load_transp_conv_weights
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
import torchvision
import torch
import pytorch_lightning as pl
from utils import TypeRLoss
import mlflow
from utils import convert_rgb_tensor_for_plot, convert_gray_tensor_for_plot

class TypeRNet(pl.LightningModule):
    # model https://medium.com/analytics-vidhya/lets-discuss-encoders-and-style-transfer-c0494aca6090
    # https://colab.research.google.com/github/usuyama/pytorch-unet/blob/master/pytorch_unet_resnet18_colab.ipynb
    # https://gist.github.com/samson-wang/a6073c18f2adf16e0ab5fb95b53db3e6

    # code https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_ptl_mini.py
    # https://docs.ray.io/en/latest/tune/examples/includes/mlflow_ptl_example.html
    def __init__(self, config):
        
        super().__init__()
        self.config = config
        
        # extract hyperparams
        self.lr  = config["lr"]
        alpha = config["alpha"]
        beta = config["beta"]
        gamma = config["gamma"]

        font_path = config["font_path"]
        transposed_kernel_size = config["transposed_kernel_size"]
        transposed_stride = config["transposed_stride"]
        transposed_padding = config["transposed_padding"]
        max_letter_per_pix = config["max_letter_per_pix"]
        letters = config["letters"]
        eps_out = config["eps_out"]

        self.loss = TypeRLoss(max_letter_per_pix, alpha, beta, gamma)

        # build model
        self.conv1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).conv1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=len(letters), kernel_size=5, stride=1, padding=2)
        nn.init.xavier_normal_(self.conv2.weight)
        transposed_convs_weights = load_transp_conv_weights(font_path, transposed_kernel_size, letters)
        # transposed padding = 31
        self.transp_conv = CustomTransposedConv2d(transposed_convs_weights, len(letters), 1, transposed_kernel_size, transposed_stride, transposed_padding)
        self.transp_conv.requires_grad_(False)

        # set parameters for last layers
        self.max_letter_per_pix = max_letter_per_pix
        self.num_letters = len(letters)
        self.eps_out = eps_out

        self.val_loss_list = []

    def forward(self, x):
       feat1 = torch.tanh(self.conv1(x))
       feat2 = torch.sigmoid(self.conv2(feat1))
       # cap of max five letters per pixel

       _, indices = torch.topk(feat2, self.max_letter_per_pix, dim=1)
       mask = torch.zeros_like(feat2)
       mask = mask.scatter(1, indices, 1)
       feat2_masked = feat2 * mask
       out_img = self.transp_conv(feat2_masked)
       out_img = torch.where(out_img < self.eps_out, torch.tensor(0.0), out_img)
       #out_img = torch.where(out_img > 1.0, torch.tensor(1.0), out_img)
       # TODO 

       return out_img, feat2_masked
    
    def training_step(self, batch, batch_idx : int) -> STEP_OUTPUT:
        img_in, img_target, label = batch
        out_img, key_strokes = self.forward(img_in)
        loss = self.loss.forward(key_strokes, out_img, img_target)
        self.log("train_loss", float(loss))
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img_in, img_target, label = batch
        # TODO use label to get mse per class
        out_img, key_strokes = self.forward(img_in)
        loss = self.loss.forward(key_strokes, out_img, img_target)
        self.val_loss_list.append(loss)
        self.log("val_loss", float(loss))
        if batch_idx % 6 == 0:      
            grid_in = torchvision.utils.make_grid(convert_rgb_tensor_for_plot(img_in[:4])).unsqueeze(0).cpu().numpy()
            grid_out = torchvision.utils.make_grid(convert_gray_tensor_for_plot(out_img[:4])).unsqueeze(0).cpu().numpy()
            mlflow.log_image(grid_in, f'validation_rgb_{self.current_epoch}_{batch_idx}.png')
            mlflow.log_image(grid_out, f'validation_out_{self.current_epoch}_{batch_idx}.png')

        return {"val_loss" : loss}
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss_list).mean()
        self.log("ptl/val_loss", avg_loss)
        self.val_loss_list.clear()
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img_in, img_target, label = batch
        out_img, key_strokes = self.forward(img_in)
        loss = self.loss.forward(key_strokes, out_img, img_target)
        self.log("test_loss", float(loss))
        if batch_idx % 6 == 0:
            grid_in = torchvision.utils.make_grid(convert_rgb_tensor_for_plot(img_in[:4])).unsqueeze(0).cpu().numpy()
            grid_out = torchvision.utils.make_grid(convert_gray_tensor_for_plot(out_img[:4])).unsqueeze(0).cpu().numpy()
            mlflow.log_image(grid_in, f'test_rgb_{self.current_epoch}_{batch_idx}.png')
            mlflow.log_image(grid_out, f'test_out_{self.current_epoch}_{batch_idx}.png')
        return loss
    
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    


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
        # TODO normalize weights here
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        # Perform the transposed convolution operation with hard-coded weights
        output = nn.functional.conv_transpose2d(x, self.weights, bias=self.bias,
                                                       stride=self.stride, padding=self.padding)
        return output