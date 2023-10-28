import pandas as pd
import time
import torch
import os

from data.dataset import ImageDataset
from data.utils import get_train_val_test_dataloaders
from models.pure_img_model import get_pure_img_model
from callbacks.training import train_pure_step
from callbacks.evaluation import pure_evaluate

# System configs
num_workers = os.cpu_count() // 2
num_gpus = torch.cuda.device_count()
batch_size = 32 * num_gpus
random_state = 42

# Data path configuration
data_path = "/home/rasteiger/datasets/hack4good/"

# Files and Folders of interest
cache_file = data_path + 'time_series_features.csv'
path_weather_data = data_path + 'era5_land_t2m_pev_tp.csv'
img_dir = data_path + "images"
label_path = data_path + "labels.csv"

# Create place to safe model if it does not exist
model_dir = data_path + "models/" 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

timestr = time.strftime("%Y%m%d-%H%M%S")
model_path = model_dir + timestr + 'model.pt'

# Reading the csv file
label_df = pd.read_csv(label_path)
label_df['date'] = pd.to_datetime(label_df['date'], format='mixed')

# Setting the labels and join columns
label_df['is_damage'] = (label_df['extent'] >= 20).astype(int)
label_column = 'is_damage'
join_column = 'filename'
split_column = 'farmer_id'

# Select a subset of the data
label_df = label_df[label_df['crop_name'] == 'maize']

# Define image size for transformations for loading the data
img_size = 224
train_loader, val_loader, test_loader = get_train_val_test_dataloaders(img_size=img_size, img_dir=img_dir, feature_df=label_df, 
                                                                       feature_columns=None, label_column=label_column, join_column=join_column, 
                                                                       split_column=split_column, split_sizes=[0.8, 0.1, 0.1],
                                                                       batch_size=batch_size, num_workers=num_workers, random_state=random_state)

# Define Model
num_classes = train_loader.dataset.num_classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_pure_img_model(num_classes=num_classes, device=device, use_multi_gpu=num_gpus > 1)

# Training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.00)
class_weights = train_loader.dataset.calculate_class_weights().to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Model training

def main():
    best_loss = 1e20
    num_epochs = 20

    print("Starting to train model:")
    for epoch in range(num_epochs):
        
        train_pure_step(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch, device=device)    
        loss = pure_evaluate(model=model, data_loader=val_loader, criterion=criterion, epoch=epoch, device=device, is_test=False)
        
        if (loss < best_loss):
            best_loss = loss
            torch.save(model, model_path)

    # Model evluation on test set
    pure_evaluate(model=model, data_loader=test_loader, criterion=criterion, device=device, is_test=True)

if __name__ == '__main__':
    main()