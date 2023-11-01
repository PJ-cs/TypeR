import torch.nn as nn
import torchvision.models as models
from typing import Any
from utils import load_letter_conv_weights, load_letter_conv_weights_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
import torchvision
import torch
import pytorch_lightning as pl
from torch.optim import lr_scheduler
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
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.img_size = self.config["img_size"]

        self.sched_step_size = config["sched_step_size"]
        self.sched_gamma = config["sched_gamma"]

        font_path = config["font_path"]
        transposed_kernel_size = config["transposed_kernel_size"]
        self.transposed_kernel_size = transposed_kernel_size
        transposed_stride = config["transposed_stride"]
        transposed_padding = config["transposed_padding"]
        # keystrokes_mean = config["keystrokes_mean"]
        # keystrokes_std = config["keystrokes_std"]
        letters = config["letters"]
        eps_out = config["eps_out"]

        self.loss = TypeRLoss()

        # build model
        letter_convs_weights = load_letter_conv_weights(font_path, transposed_kernel_size, letters)
        # 1. layer 
        conv_weights_normed = load_letter_conv_weights_norm(letter_convs_weights)
        num_letters = letter_convs_weights.shape[0]
        self.conv1_letters = CustomConv2d(conv_weights_normed, 1, num_letters, transposed_kernel_size, transposed_stride, transposed_kernel_size//2)
        for param in self.conv1_letters.parameters():
            param.requires_grad = False

        assert(transposed_stride < transposed_kernel_size-1) 
        size_conv2 = transposed_kernel_size // transposed_stride * transposed_stride * 2 + transposed_kernel_size 
        self.conv2 = nn.Sequential(nn.Conv2d(num_letters, 64, size_conv2, padding=size_conv2//2), nn.ReLU(inplace=True), nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, size_conv2, padding=size_conv2//2), nn.ReLU(inplace=True), nn.BatchNorm2d(64))
        self.conv4 = nn.Sequential(nn.Conv2d(64, num_letters, 1, padding=0), nn.ReLU(inplace=True))
        
        ### custom init weights for last conv, are key strokes
        #nn.init.trunc_normal_(self.conv_original_size2, keystrokes_mean, keystrokes_std, 0, 1.)

        ###
        transpose_out_padding = transposed_stride -1 
        self.transp_conv = CustomTransposedConv2d(letter_convs_weights, num_letters, 1, transposed_kernel_size, transposed_stride, transposed_padding, transpose_out_padding)
        
        for param in self.transp_conv.parameters():
            param.requires_grad = False

        # set parameters for last layers
        self.num_letters = len(letters)
        self.eps_out = eps_out

        self.val_loss_list = []

        # freeze backbone
        # for l in self.base_layers:
        #     for param in l.parameters():
        #         param.requires_grad = False

    def forward(self, x):
        x = self.conv1_letters(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = nn.functional.softmax(x, dim=1)

        indices = torch.argmax(x, dim=1, keepdim=True)
        mask = torch.zeros_like(x)
        mask = mask.scatter(1, indices, 1)
        x_masked = x * mask
        out_img = self.transp_conv(x_masked)
    
        return out_img, x_masked 
    
    def training_step(self, batch, batch_idx : int) -> STEP_OUTPUT:
        img_in, img_target, label = batch
        out_img, key_strokes = self.forward(img_in)
        mse_loss = self.loss.forward(key_strokes, out_img, img_target)
        loss = self.alpha * mse_loss # + self.beta * key_stroke_loss
        self.log("train_loss", float(loss))
        self.log("train_mse_loss", float(mse_loss))
        # self.log("train_key_stroke_loss", float(key_stroke_loss))
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img_in, img_target, label = batch
        # TODO use label to get mse per class
        out_img, key_strokes = self.forward(img_in)
        mse_loss = self.loss.forward(key_strokes, out_img, img_target)
        loss = self.alpha * mse_loss # + self.beta * key_stroke_loss
        self.log("val_loss", float(loss))
        self.log("val_mse_loss", float(mse_loss))
        # self.log("val_key_stroke_loss", float(key_stroke_loss))

        self.val_loss_list.append(loss)

        if batch_idx % 10 == 0:      
            grid_in = torchvision.utils.make_grid(img_in[:8]).permute(1,2,0).cpu().numpy()
            grid_out = torchvision.utils.make_grid(torchvision.transforms.functional.invert(out_img), normalize=True).permute(1,2,0).cpu().numpy()
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
        mse_loss = self.loss.forward(key_strokes, out_img, img_target)
        loss = self.alpha * mse_loss #+ self.beta * key_stroke_loss
        self.log("test_loss", float(loss))
        self.log("test_mse_loss", float(mse_loss))
        #self.log("test_key_stroke_loss", float(key_stroke_loss))

        self.val_loss_list.append(loss)

        if batch_idx % 10 == 0:      
            grid_in = torchvision.utils.make_grid(img_in[:8]).permute(1,2,0).cpu().numpy()
            grid_out = torchvision.utils.make_grid(torchvision.transforms.functional.invert(out_img[:8]), normalize=True).permute(1,2,0).cpu().numpy()
            mlflow.log_image(grid_in, f'test_rgb_{self.current_epoch}_{batch_idx}.png')
            mlflow.log_image(grid_out, f'test_out_{self.current_epoch}_{batch_idx}.png')

        return {"test_loss" : loss}
    
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        lr_scheduler_tmp = lr_scheduler.StepLR(optimizer, step_size=self.sched_step_size, gamma=self.sched_gamma)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_tmp}
    

class CustomTransposedConv2d(nn.Module):
    def __init__(self, weights, in_channels, out_channels, kernel_size, stride=1, padding=0, out_padding=0):
        super(CustomTransposedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = out_padding
        self.weights = nn.Parameter(weights)
        
        # Define the hard-coded weights (example weights)
        #self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # TODO normalize weights here
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        # Perform the transposed convolution operation with hard-coded weights
        output = nn.functional.conv_transpose2d(x, self.weights, bias=self.bias,
                                                       stride=self.stride, padding=self.padding, output_padding=self.out_padding)
        return output
    

class CustomConv2d(nn.Module):
    def __init__(self, weights, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
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
        output = nn.functional.conv2d(x, self.weights, bias=self.bias,
                                                       stride=self.stride, padding=self.padding)
        return output
    
    
def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
  )

def convsimgmoid(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.BatchNorm2d(out_channels),
    nn.Sigmoid(),
  )
