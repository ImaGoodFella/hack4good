# Custom dataloader classs for images with corresponding labels such that pytorch can work with the data
import pandas as pd
import numpy as np

import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from sklearn.utils import class_weight


class CustomImageDataset(Dataset):
    
    def __init__(self, img_dir, csv_file, label_name, join_name, class_to_idx, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.label_name = label_name
        self.join_name = join_name # column used to join img_dir and csv_file
        self.transform = transform
        self.target_transform = target_transform

        # Integer class mapping: Label classes -> [0, 1, ..., num(labels) - 1] for pytorch
        self.class_to_idx = class_to_idx
        self.num_classes = len(list(class_to_idx.values()))

    def __len__(self):
        return self.csv_file.shape[0]

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.csv_file.iloc[idx][self.join_name])
        image = Image.open(img_path)

        # Load Label
        label = self.class_to_idx[self.csv_file.iloc[idx][self.label_name]]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

    def calculate_class_weights(self):
        y_vals = self.csv_file[self.label_name].apply(lambda x: self.class_to_idx[x])
        cw = class_weight.compute_class_weight('balanced', classes=y_vals.unique(), y=y_vals).astype(np.float32)
        return torch.tensor(cw)