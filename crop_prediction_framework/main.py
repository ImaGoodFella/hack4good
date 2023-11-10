import pandas as pd
import time
import torch
import os

from data.utils import get_train_val_test_dataloaders
from data.feature_extractor import get_time_series_features_df
from models.basic_models import get_basic_model
from callbacks.training import train_step
from callbacks.evaluation import evaluate

# System configs
if not torch.cuda.is_available():
    raise NotImplementedError('GPU is required for training')

device = torch.device("cuda")
num_workers = os.cpu_count() // 2
num_gpus = torch.cuda.device_count()
batch_size = 16 * num_gpus
random_state = 42

# Data path configuration
data_path = "../data/"

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

# Get time series features dataframe
feature_df, feature_columns = get_time_series_features_df(label_df=label_df, path_weather_data=path_weather_data, 
                                                          join_column=join_column, use_cache=True, cache_file=cache_file)    

# Define image size for transformations for loading the data
img_size = 224
train_loader, val_loader, test_loader = get_train_val_test_dataloaders(img_size=img_size, img_dir=img_dir, feature_df=feature_df, feature_columns=feature_columns,
                                                                       label_column=label_column, join_column=join_column, split_column=split_column,
                                                                       split_sizes=[0.8, 0.1, 0.1], batch_size=batch_size, num_workers=num_workers, random_state=random_state)


# Define Model
num_classes = train_loader.dataset.num_classes
model = get_basic_model(num_classes=num_classes, num_ts_features=len(feature_columns), device=device, use_multi_gpu=num_gpus > 1)

# Training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
class_weights = train_loader.dataset.calculate_class_weights().to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Model training

def main():
    best_loss = 1e20
    num_epochs = 20

    print("Starting to train model:")
    for epoch in range(num_epochs):
        
        train_step(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch, device=device)    
        loss = evaluate(model=model, data_loader=val_loader, criterion=criterion, epoch=epoch, device=device, is_test=False)
        
        if (loss < best_loss):
            best_loss = loss
            torch.save(model, model_path)

    # Model evluation on test set
    evaluate(model=model, data_loader=test_loader, criterion=criterion, device=device, is_test=True)

if __name__ == '__main__':
    main()