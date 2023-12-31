import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EyesDataset(Dataset):
    def __init__(
        self, img_dir, annotations_file, indices=None, mode="train", task=1
    ):
        """Dataset class.

        Args:
            img_dir (str): Path to the image directory
            annotations_file (str): File with image paths and labels.
            indices (list, optional): List of indices to subset the data.
            Defaults to None.
            mode (str, optional): Choose to return labels or not.
            Defaults to "train".
            task (int, optional): Task to choose the correct annotation file.
            Defaults to 1.
        """
        self.img_dir = img_dir
        self.mode = mode
        self.task = task

        if indices is not None:
            self.img_labels = pd.read_csv(annotations_file).iloc[indices]
        else:
            self.img_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.img_labels["Image"].iloc[idx]
        )
        image = Image.open(img_path).convert("RGB")
        image = transforms.ToTensor()(image)

        if self.mode == "train" or self.mode == "val":
            if self.task == 1:
                label = self.img_labels["Hypertensive"].iloc[idx]
            else:
                label = self.img_labels["Hypertensive Retinopathy"].iloc[idx]

            return image, torch.tensor(label)

        else:
            return image
