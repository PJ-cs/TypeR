import numpy as np
import random
import torch
import PIL.ImageFont as ImageFont
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torch.nn as nn
import os

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for img in tensor:
            for t, m, s in zip(img, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return tensor

def convert_rgb_tensor_for_plot(tensor_img: torch.Tensor) -> torch.Tensor:
    tmp = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_img)
    #tmp = tmp.permute(0,2,3,1)
    return tmp

def convert_gray_tensor_for_plot(tensor_img: torch.Tensor) -> torch.Tensor:
    tmp = UnNormalize(mean=[0.445], std=[0.269])(tensor_img)
    #tmp = tmp.permute(0,2,3,1)
    return tmp

def load_letter_conv_weights(font_path: str, kernel_size: int, letters: list[str]):
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
        letter_tensor = letter_tensor / letter_tensor.sum()
        convolutions.append(letter_tensor)
    
    return torch.stack(convolutions).unsqueeze(1)


def get_rel_area_letters(font_path: str, kernel_size : int, letters: list[str]) -> list[float]:
    font: ImageFont.FreeTypeFont = ImageFont.truetype(font=font_path, size=int(kernel_size*.88))

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    all_letters_text = "".join(letters)
    left, top, right, bottom = font.getbbox(all_letters_text)
    y = (kernel_size // 2) -  ((bottom-top)//2)-top

    letter_areas: list[float] = []

    for letter in letters:
        im = Image.new("L", (kernel_size, kernel_size), 0)
        draw = ImageDraw.Draw(im)
        left, top, right, bottom = font.getbbox(letter)
        x = (kernel_size // 2) -  ((right-left)//2)
        
        draw.multiline_text((x,y), letter, 255, font=font)

        letter_tensor = transform(im).float().squeeze(0) / 255.

        letter_areas.append(float(letter_tensor.sum()))

    # norm areas to maximal area to range [0,1]
    letter_areas_tensor = torch.FloatTensor(letter_areas)
    max_area : float = letter_areas_tensor.max()

    return letter_areas_tensor / max_area
        

def calc_receptive_field(layer_params : list[tuple[int, int, int]]):
    """
    layer_params : (k_l kernel size, s_l stride, p_l padding)
    """
    # https://www.baeldung.com/cs/cnn-receptive-field-size
    r = 1
    S = 1

    for l in range(0, len(layer_params)):
        for i in range(0, l):
            S *= layer_params[i][1]
        r += (layer_params[l][0] - 1) * S
        S=1
    return r


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

