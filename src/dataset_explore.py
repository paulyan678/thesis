import numpy as np
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import os
import matplotlib.pyplot as plt
import nibabel as nib
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import cv2


class BraTSDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, index):
        folder = os.listdir(self.path)[index]
        folder_path = self.path + '/' + folder + '/' + folder
        image = nib.load(folder_path + '_flair.nii').get_fdata()[:, :, 77]
        mask = nib.load(folder_path + '_seg.nii').get_fdata()[:, :, 77]

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.astype(np.float32)
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.convert('RGB')

        if self.transform:
            transform_1 = self.transform(image)
            transform_2 = self.transform(image)

            return (transform_1, transform_2), mask

        else:
            return transforms.Compose([transforms.ToTensor()])(image), transforms.Compose([transforms.ToTensor()])(mask)

mean = 0.1276496946811676
std = 0.21015071868896484

transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0))], p=0.3),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.3),
    transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.3),
    transforms.RandomApply([transforms.RandomRotation(degrees=(0, 360))], p=0.3),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


train_dataset = BraTSDataset('MICCAI_BraTS2020_TrainingData', transform)
print(train_dataset[0])