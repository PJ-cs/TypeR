import os
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import v2
import torch
import numpy as np
import pytorch_lightning as pl
from utils import set_seeds
from pathlib import Path
import requests


class BigImagesDataset(Dataset):
    def __init__(self, imgs_dir: str, transform: v2.Compose):
        self.transform = transform
        self.img_paths = []

        for folder in os.scandir(imgs_dir):
            for file in os.scandir(folder.path):
                self.img_paths.append(file.path)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_paths[idx], ImageReadMode.GRAY)
        image: torch.Tensor = self.transform(image)
        target = image.clone()

        return image, target


def get_data_loader(
    imgs_dir: str, max_img_dims: tuple[int, int], batch_size: int, num_workers: int
):
    transforms = get_img_transforms(max_img_dims=max_img_dims)
    dataset = BigImagesDataset(imgs_dir=imgs_dir, transform=transforms)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)


def get_img_transforms(max_img_dims: tuple[int, int]) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(max_img_dims),
            lambda x: v2.functional.invert(x),
            v2.Grayscale(num_output_channels=1),
        ]
    )


def elt_data(images_url_dir: str, training_img_save_dir: str):
    """Extract, load and transform our data assets."""
    # Extract + Load
    url_file_paths: list[str] = [file.path for file in os.scandir(images_url_dir)]
    # courtesy to https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
    for url_file in url_file_paths:
        links = open(url_file).read().strip().split("\n")
        output_path = Path(training_img_save_dir, os.path.basename(url_file)[:-4])
        os.makedirs(output_path, exist_ok=True)
        total = 0
        for url in links:
            try:
                p = os.path.sep.join(
                    [str(output_path), "{}.jpg".format(str(total).zfill(8))]
                )
                if os.path.exists(p):
                    continue
                # try to download the image
                r = requests.get(url, timeout=60)
                # save the image to disk
                f = open(p, "wb")
                f.write(r.content)
                f.close()
                # update the counter
                print("[INFO] downloaded: {}".format(p))
                total += 1
            # handle if any exceptions are thrown during the download process
            except:
                print("[INFO] error downloading {}...skipping".format(p))


def remove_bad_images(training_imgs_save_dir: str):
    for folder in os.scandir(training_imgs_save_dir):
        files: list[str] = os.listdir(folder.path)
        for file_path in files:
            try:
                _ = read_image(os.path.join(folder.path, file_path), ImageReadMode.GRAY)
            except:
                os.remove(os.path.join(folder.path, file_path))
                print("errounous image, deleted ", file_path)
