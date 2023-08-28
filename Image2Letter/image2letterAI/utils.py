import numpy as np
import random
import torch
import PIL.ImageFont as ImageFont
from PIL import Image, ImageDraw
from config import config
import torchvision.transforms as transforms


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.seed(seed)


def load_transposed_convolutions(font_path: str, conv_size: int, letters: list[str]):
    """loads the font from the file path specified in the config
    and creates the transposed convolutions from it"""

    font: ImageFont.FreeTypeFont = ImageFont.truetype(font=font_path, size=int(conv_size*.88))

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    all_letters_text = "".join(letters)
    left, top, right, bottom = font.getbbox(all_letters_text)
    y = (conv_size // 2) -  ((bottom-top)//2)-top

    convolutions: list[torch.Tensor] = []
    for letter in letters:
        im = Image.new("L", (conv_size, conv_size), 0)
        draw = ImageDraw.Draw(im)
        left, top, right, bottom = font.getbbox(letter)
        x = (conv_size // 2) -  ((right-left)//2)
        
        draw.multiline_text((x,y), letter, 255, font=font)
        im.show()

        letter_tensor = transform(im).float().squeeze(0) / 255.
        convolutions.append(letter_tensor)
    
    return torch.stack(convolutions)

        