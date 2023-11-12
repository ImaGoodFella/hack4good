import pandas as pd
import time
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, EarlyStopping

from data.utils import get_train_val_test_dataloaders
from data.feature_extractor import get_time_series_features_df
from models.basic_models import get_basic_model
from models.pure_img_model import get_pure_img_model
from callbacks.training import train_step
from callbacks.evaluation import evaluate
from train.sequence import ClassificationWrapper

# System configs
if not torch.cuda.is_available():
    raise NotImplementedError('GPU is required for training')

device = torch.device("cuda")
num_workers = os.cpu_count() // 2
num_gpus = torch.cuda.device_count()
batch_size = 32 #* num_gpus
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
time_series_features_path = data_path + "tf_features_full_narm.csv"
relevant_features_path = data_path + "relevant_features.csv"

feature_df = pd.read_csv(time_series_features_path)
relevant = pd.read_csv(relevant_features_path)['x'].values.tolist()

feature_df = feature_df[feature_df.columns.intersection(relevant)]

feature_columns = feature_df.columns

feature_df = pd.concat([label_df, feature_df], axis=1)

#feature_df = feature_df.sample(1000)
# Define image size for transformations for loading the data
img_size = 224
train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
    img_size=img_size, img_dir=img_dir, feature_df=feature_df, feature_columns=feature_columns,
    label_column=label_column, join_column=join_column, split_column=split_column,
    split_sizes=[0.8, 0.1, 0.1], batch_size=batch_size, num_workers=num_workers, random_state=random_state
)


# Define Model
num_classes = train_loader.dataset.num_classes
#model = get_basic_model(num_classes=num_classes, num_ts_features=len(feature_columns), device=None, use_multi_gpu=False)
model = get_pure_img_model(num_classes=num_classes, device=None, use_multi_gpu=False)

# Training parameters
class_weights = train_loader.dataset.calculate_class_weights().to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
callbacks = [EarlyStopping(monitor="val/F1Score", min_delta=0.00, patience=5, verbose=False, mode="max")]
wrapper = ClassificationWrapper(model=model, learning_rate=5e-5, weight_decay=0.01, loss=criterion, num_classes=num_classes)
logger = logger=pl_loggers.TensorBoardLogger(save_dir=data_path, name='lightning_logs', version=None)
trainer = pl.Trainer(accelerator="gpu", strategy='deepspeed', devices=num_gpus, logger=logger, max_epochs=100, callbacks=callbacks)

# Model training
def main():
    
    trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(wrapper, dataloaders=test_loader, ckpt_path='best')

if __name__ == '__main__':
    main()