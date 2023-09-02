import os
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch


class BigImagesDataset(Dataset):
    def __init__(self, imgs_dir, transform=None, target_transform=None):
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
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def get_img_transforms_train(img_size):
    return v2.Compose([v2.ToImageTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.RandomCrop(img_size),
                    v2.RandomHorizontalFlip(p=0.2),
                    v2.RandomInvert(0.5),
                    v2.ColorJitter(0.5, 0.5, 0.5),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])                                                   

def get_img_transforms_test(img_size):
    return v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype(torch.float32), v2.CenterCrop(img_size), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_label_transforms():
    return v2.Compose([lambda x: torch.tensor(x)])