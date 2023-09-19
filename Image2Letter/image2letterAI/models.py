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
       
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convsimgmoid(64 + 128, 100, 3, 1)

        ###
        transposed_convs_weights = load_transp_conv_weights(font_path, transposed_kernel_size, letters)
        self.transp_conv = CustomTransposedConv2d(transposed_convs_weights, len(letters), 1, transposed_kernel_size, transposed_stride, transposed_padding)
        self.transp_conv.requires_grad_(False)

        # set parameters for last layers
        self.max_letter_per_pix = max_letter_per_pix
        self.num_letters = len(letters)
        self.eps_out = eps_out

        self.val_loss_list = []

        # freeze backbone
        for l in self.base_layers:
            for param in l.parameters():
                param.requires_grad = False

    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        # feat1 = torch.tanh(self.conv1(x))
        # feat2 = torch.sigmoid(self.conv2(feat1))
        # cap of max five letters per pixel

        _, indices = torch.topk(x, self.max_letter_per_pix, dim=1)
        mask = torch.zeros_like(x)
        mask = mask.scatter(1, indices, 1)
        x_masked = x * mask
        out_img = self.transp_conv(x_masked)
        out_img = torch.where(out_img < self.eps_out, torch.tensor(0.0), out_img)
        #out_img = torch.where(out_img > 1.0, torch.tensor(1.0), out_img)
        # TODO 

        return out_img, x_masked     
    
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
        if batch_idx % 10 == 0:      
            grid_in = torchvision.utils.make_grid(convert_rgb_tensor_for_plot(img_in[:4])).permute(1,2,0).cpu().numpy()
            grid_out = torchvision.utils.make_grid(torchvision.transforms.functional.invert(convert_gray_tensor_for_plot(out_img[:4])), normalize=True).permute(1,2,0).cpu().numpy()
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
        if batch_idx % 10 == 0:
            grid_in = torchvision.utils.make_grid(convert_rgb_tensor_for_plot(img_in[:4])).permute(1,2,0).cpu().numpy()
            grid_out = torchvision.utils.make_grid(torchvision.transforms.functional.invert(convert_gray_tensor_for_plot(out_img[:4])), normalize=True).permute(1,2,0).cpu().numpy()
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
    
def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )

def convsimgmoid(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.Sigmoid(),
  )
