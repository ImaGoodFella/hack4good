
import pandas as pd
import numpy as np

import torchvision
from torch.utils.data import DataLoader

# Our implementations
from data.dataset import CustomImageDataset

def get_train_val_test_dataloaders(img_size, img_dir, feature_df, label_column, join_column, feature_columns,
                                   split_column=None, split_sizes=[0.6, 0.2, 0.2], train_transforms=None, val_transforms=None, 
                                   random_state=42, batch_size=32, num_workers=64):

    # Pytorch specific stuff (integer class labels, get number of classes)
    class_to_idx = {v:i for i, v in enumerate(feature_df[label_column].unique())} 

    # If transforms are not properly defined, use the default transformations
    if train_transforms is None or val_transforms is None:
        train_transforms, val_transforms = get_default_train_val_transformations(img_size) 

    # Split the provided dataframe on a specific feature. If no specific feature is given, split on join_name. 
    split_column = join_column if split_column is None else split_column
    train_csv, val_csv, test_csv = split_dataframe_on_column(feature_df, split_column, split_sizes, random_state)

    # Get custom dataset
    train_dataset = CustomImageDataset(img_dir=img_dir, feature_df=train_csv, feature_columns=feature_columns, label_column=label_column, join_column=join_column, transform=train_transforms, class_to_idx=class_to_idx)
    val_dataset = CustomImageDataset(img_dir=img_dir, feature_df=val_csv, feature_columns=feature_columns, label_column=label_column, join_column=join_column, transform=val_transforms, class_to_idx=class_to_idx)
    test_dataset = CustomImageDataset(img_dir=img_dir, feature_df=test_csv, feature_columns=feature_columns, label_column=label_column, join_column=join_column, transform=val_transforms, class_to_idx=class_to_idx)

    # Create normal python dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def split_dataframe_on_column(df, column, split_sizes, random_state=42):
    
    if sum(split_sizes) != 1 or len(split_sizes) != 3:
        raise ValueError("split sizes must sum up to 1.0 and have length == 3")

    feature_unique = pd.Series(df[column].unique())
    train_split, val_split, test_split = np.split(feature_unique.sample(frac=1, random_state=random_state), 
                                                  [int(split_sizes[0]*len(feature_unique)), int((split_sizes[0] + split_sizes[1])*len(feature_unique))])

    train_csv = df[df[column].isin(train_split)]
    val_csv = df[df[column].isin(val_split)]
    test_csv = df[df[column].isin(test_split)]

    return train_csv, val_csv, test_csv

def get_default_train_val_transformations(size):

    # Perform image augmentation while training to artificially inflate dataset
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0)),
        torchvision.transforms.ColorJitter(brightness=0.95),#, contrast=None, saturation=None),
        torchvision.transforms.RandomRotation(degrees = (-15,+15)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # Only resize for validation and test images
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms