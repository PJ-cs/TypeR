import os
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import v2
import torch
import numpy as np
import random
import pytorch_lightning as pl
from utils import set_seeds
from pathlib import Path
import image2letterAI.config as configFile
import requests


class BigImagesDataset(Dataset):
    def __init__(self, imgs_dir, transform: v2.Compose, target_transform: v2.Compose):
        self.img_labels = []
        self.img_paths = []
        self.label_dict = {}
        label_cnt = 0
        for folder in os.scandir(imgs_dir):
            for file in os.scandir(folder.path):
                self.img_paths.append(file.path)
                self.img_labels.append(label_cnt)
            self.label_dict[label_cnt] = folder.name
            label_cnt += 1

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_paths[idx], ImageReadMode.GRAY)
        target = image.clone()
        label = self.img_labels[idx]
        seed = np.random.randint(2147483647)

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        image = self.transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        target = self.target_transform(target)
        return image, target, torch.tensor(label)


def get_img_transforms_train(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop((img_size, img_size)),
            lambda x: v2.functional.invert(x),
            v2.Grayscale(num_output_channels=1),
        ]
    )


def get_img_transforms_train_target(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop((img_size, img_size)),
            v2.Grayscale(num_output_channels=1),
        ]
    )


def get_img_transforms_test(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop((img_size, img_size)),
            lambda x: v2.functional.invert(x),
            v2.Grayscale(num_output_channels=1),
        ]
    )


def get_img_transforms_test_target(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop((img_size, img_size)),
            v2.Grayscale(num_output_channels=1),
            # v2.Normalize(mean=[0.445], std=[0.269]),
        ]
    )


class BigImagesDataModule(pl.LightningDataModule):
    # https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_ptl_mini.py
    def __init__(
        self,
        imgs_dir: str,
        img_size: int,
        img_size_test: int,
        batch_size: int,
        num_workers: int,
        val_ratio=0.2,
        test_ratio=0.2,
    ) -> None:
        super().__init__()
        self.imgs_dir = imgs_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.transform_img_train = get_img_transforms_train(img_size)
        self.transform_target_train = get_img_transforms_train_target(img_size)
        self.transform_img_val = get_img_transforms_test(img_size)
        self.transform_target_val = get_img_transforms_test_target(img_size)
        self.transform_img_test = get_img_transforms_test(img_size_test)
        self.transform_target_test = get_img_transforms_test_target(img_size_test)
        set_seeds()

    def setup(self, stage: str) -> None:
        dataset_train_full = BigImagesDataset(self.imgs_dir, self.transform_img_train, self.transform_target_train)
        datset_val_full = BigImagesDataset(self.imgs_dir, self.transform_img_val, self.transform_target_val)
        datset_test_full = BigImagesDataset(self.imgs_dir, self.transform_img_test, self.transform_target_test)

        train_indices, val_indices, test_indices = train_val_test_split(
            len(dataset_train_full), self.val_ratio, self.test_ratio
        )

        if stage == "fit" or stage is None:
            self.ds_train = Subset(dataset_train_full, train_indices)
            self.ds_val = Subset(datset_val_full, val_indices)
        if stage == "test" or stage is None:
            self.ds_test = Subset(datset_test_full, test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers)

    # TODO for distributed training on multiple nodes to get data
    def prepare_data(self) -> None:
        # elt_data()
        # print("removing bad immages...")
        # remove_bad_images()
        return


def elt_data():
    """Extract, load and transform our data assets."""
    # Extract + Load
    url_file_paths: list[str] = [file.path for file in os.scandir(configFile.IMAGES_URL_DIR)]
    # courtesy to https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
    for url_file in url_file_paths:
        links = open(url_file).read().strip().split("\n")
        output_path = Path(configFile.TRAINING_IMGS_DIR, os.path.basename(url_file)[:-4])
        os.makedirs(output_path, exist_ok=True)
        total = 0
        for url in links:
            try:
                p = os.path.sep.join([str(output_path), "{}.jpg".format(str(total).zfill(8))])
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


def remove_bad_images():
    for folder in os.scandir(configFile.TRAINING_IMGS_DIR):
        files: list[str] = os.listdir(folder.path)
        for file_path in files:
            try:
                _ = read_image(os.path.join(folder.path, file_path), ImageReadMode.GRAY)
            except:
                os.remove(os.path.join(folder.path, file_path))
                print("errounous image, deleted ", file_path)


def train_val_test_split(
    dataset_size: int, val_ratio=0.1, test_ratio=0.1, random_seed=42
) -> tuple[list[int], list[int], list[int]]:
    """
    Split a dataset into train, validation, and test sets by generating lists of indices.

    Args:
        dataset_size (int): The total number of data points in the dataset.
        val_ratio (float): The ratio of validation data (default is 0.1).
        test_ratio (float): The ratio of test data (default is 0.1).
        random_seed (int or None): Seed for random number generation (optional).

    Returns:
        train_indices (list of ints): Indices for the training set.
        val_indices (list of ints): Indices for the validation set.
        test_indices (list of ints): Indices for the test set.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)

    val_indices = indices[:val_size]
    test_indices = indices[val_size : val_size + test_size]
    train_indices = indices[val_size + test_size :]

    return train_indices, val_indices, test_indices
