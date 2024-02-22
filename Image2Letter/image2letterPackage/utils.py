import numpy as np
import torch
import PIL.ImageFont as ImageFont
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import cv2


def load_letter_conv_weights(font_path: str, kernel_size: int, letters: list[str]):
    """loads the font from the file path specified in the config
    and creates the transposed convolutions from it"""

    font: ImageFont.FreeTypeFont = ImageFont.truetype(
        font=font_path, size=int(kernel_size * 0.88)
    )

    transform = transforms.Compose([transforms.PILToTensor()])

    all_letters_text = "".join(letters)
    left, top, right, bottom = font.getbbox(all_letters_text)
    y = (kernel_size // 2) - ((bottom - top) // 2) - top

    convolutions: list[torch.Tensor] = []
    for letter in letters:
        im = Image.new("L", (kernel_size, kernel_size), 0)
        draw = ImageDraw.Draw(im)
        left, top, right, bottom = font.getbbox(letter)
        x = (kernel_size // 2) - ((right - left) // 2)

        draw.multiline_text((x, y), letter, 255, font=font)

        letter_tensor = transform(im).float().squeeze(0)
        letter_tensor = letter_tensor / letter_tensor.sum()
        convolutions.append(letter_tensor)

    return torch.stack(convolutions).unsqueeze(1)


def get_rel_area_letters(
    font_path: str, kernel_size: int, letters: list[str]
) -> list[float]:
    font: ImageFont.FreeTypeFont = ImageFont.truetype(
        font=font_path, size=int(kernel_size * 0.88)
    )

    transform = transforms.Compose([transforms.PILToTensor()])

    all_letters_text = "".join(letters)
    left, top, right, bottom = font.getbbox(all_letters_text)
    y = (kernel_size // 2) - ((bottom - top) // 2) - top

    letter_areas: list[float] = []

    for letter in letters:
        im = Image.new("L", (kernel_size, kernel_size), 0)
        draw = ImageDraw.Draw(im)
        left, top, right, bottom = font.getbbox(letter)
        x = (kernel_size // 2) - ((right - left) // 2)

        draw.multiline_text((x, y), letter, 255, font=font)

        letter_tensor = transform(im).float().squeeze(0) / 255.0

        letter_areas.append(float(letter_tensor.sum()))

    # norm areas to maximal area to range [0,1]
    letter_areas_tensor = torch.FloatTensor(letter_areas)
    max_area: float = letter_areas_tensor.max()

    return letter_areas_tensor / max_area


def calc_receptive_field(layer_params: list[tuple[int, int, int]]):
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
        S = 1
    return r


def get_sobel_kernel(k=3):
    # get range
    range_k = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range_k, range_k)
    sobel_2D_numerator = x
    sobel_2D_denominator = x**2 + y**2
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x**2 + y**2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-((distance - mu) ** 2) / (2 * sigma**2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma**2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


# TODO use tiff image format instead of np files, more universal to transfer, and use integer based strength instead of float
def nn_hits_2_np_images(
    letter_hits: torch.Tensor,
    stride: int,
    tw_letters: list[str],
    nn_letters: list[str],
    letters_per_pixel: int,
) -> tuple[np.ndarray, np.array]:
    """
    letter_hits: np array with values in range [0., 1.], #nn_letters channels
    stride: int, convolution stride used in creating letter_hits
    tw_letters: list of letters available for typewriter
    nn_letters: list of letters used by neural net, subset of tw
    letters_per_pixel: amount of overlayed letters for one output pixel
    """

    letter_channels, h_nn, w_nn = letter_hits.shape

    letter_hits_upscaled = np.zeros((letter_channels, h_nn * stride, w_nn * stride))
    letter_hits_upscaled[:, ::stride, ::stride] = letter_hits.detach().cpu().numpy()

    # convert letter hits to two matrices one for letter index, one for strength of that letter
    output_np_strength = np.zeros(
        (letters_per_pixel, h_nn * stride, w_nn * stride), dtype=np.uint8
    )
    output_np_letter = np.zeros_like(output_np_strength, dtype=np.uint8)

    indices_map = {}
    for nn_index, nn_letter in enumerate(nn_letters):
        indices_map[nn_index] = tw_letters.index(nn_letter)

    for letter_index_nn in range(letter_channels):
        channel_mat_nn = letter_hits_upscaled[letter_index_nn]

        for out_channel_num in range(letters_per_pixel):
            if np.sum(channel_mat_nn) == 0:
                break
            output_channel = output_np_strength[out_channel_num]
            valid_pixel_mask = output_channel <= 0 and channel_mat_nn > 0
            output_np_strength[out_channel_num][valid_pixel_mask] = (
                channel_mat_nn[valid_pixel_mask] * 255
            )
            output_np_letter[out_channel_num][valid_pixel_mask] = indices_map[
                letter_index_nn
            ]
            channel_mat_nn[valid_pixel_mask] = 0

    return output_np_letter, output_np_strength


def np_2_bytes(np_arr: np.ndarray) -> bytes:
    """
    returns letter bytes and strength bytes to be easily transfered
    """
    success, np_encode = cv2.imencode(".tiff", np_arr)
    assert success
    return np_encode.tobytes()
