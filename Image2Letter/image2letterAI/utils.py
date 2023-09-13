import numpy as np
import random
import torch
import PIL.ImageFont as ImageFont
from PIL import Image, ImageDraw
from config import config
import torchvision.transforms as transforms
import torch.nn as nn


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.seed(seed)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def convert_rgb_tensor_for_plot(tensor_img: torch.Tensor):
    tmp = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_img)
    tmp = tmp.permute(1,2,0)
    return tmp

def load_transp_conv_weights(font_path: str, kernel_size: int, letters: list[str]):
    """loads the font from the file path specified in the config
    and creates the transposed convolutions from it"""

    font: ImageFont.FreeTypeFont = ImageFont.truetype(font=font_path, size=int(kernel_size*.88))

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    all_letters_text = "".join(letters)
    left, top, right, bottom = font.getbbox(all_letters_text)
    y = (kernel_size // 2) -  ((bottom-top)//2)-top

    convolutions: list[torch.Tensor] = []
    for letter in letters:
        im = Image.new("L", (kernel_size, kernel_size), 0)
        draw = ImageDraw.Draw(im)
        left, top, right, bottom = font.getbbox(letter)
        x = (kernel_size // 2) -  ((right-left)//2)
        
        draw.multiline_text((x,y), letter, 255, font=font)

        letter_tensor = transform(im).float().squeeze(0) / 255.
        convolutions.append(letter_tensor)
    
    return torch.stack(convolutions).unsqueeze(1)



# TODO test this code
class TypeRLoss(nn.Module):
    def __init__(self, max_letter_per_pix : int, alpha : float, beta: float, gamma:float) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.max_letter_per_pix = max_letter_per_pix
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, key_strokes: torch.Tensor, out_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        b, key_dims, h, w = key_strokes.shape
        mse_loss =  self.mse(out_img, target_img)
        n_key_strokes_loss =  key_strokes.sum() / (key_strokes.numel() / key_dims * self.max_letter_per_pix)
        key_variety_goal = torch.zeros((b, key_dims))
        key_variety_goal[:] = 1./key_dims / self.max_letter_per_pix * (h*w)
        key_variety_loss =  self.mse(key_strokes.sum(dim=(2,3))-key_variety_goal) 

        return self.alpha * mse_loss + self.beta * n_key_strokes_loss + self.gamma * key_variety_loss
        