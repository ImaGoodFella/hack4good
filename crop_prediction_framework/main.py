import pandas as pd
import torch
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data.utils import get_train_val_test_dataloaders
from models.pure_img_model import get_pure_img_model
from models.basic_models import get_basic_model

from train.sequence import ClassificationWrapper
from data.multi_gpu_pred_writer import CustomWriter

import argparse
from pathlib import Path

# System configs
if not torch.cuda.is_available():
    raise NotImplementedError('GPU is required for training')

device = torch.device("cuda")
num_workers = os.cpu_count() // 2
num_gpus = torch.cuda.device_count()
batch_size = 32
random_state = 42

# Data path configuration
data_path = "../data/"

# Files and Folders of interest
img_dir = data_path + "images"
label_path = data_path + "labels.csv"
time_series_features_path = data_path + "tf_features_full_narm.csv"
relevant_features_path = data_path + "relevant_features.csv"

label_column = 'label'
join_column = 'filename'
split_column = 'farmer_id'

# Gets the task
def get_task(task):

    # Reading the csv file
    label_df = pd.read_csv(label_path)

    if task == 'damage':
        # Setting the labels and join columns
        label_df[label_column] = label_df['damage']
    elif task == 'extent':
        label_df[label_column] = (label_df['extent'] >= 20).astype(int)
    else:
        raise NotImplementedError(f'Undefined task: {task}')
    
    return label_df

# Get data loaders
def get_dataloaders(label_df, is_multimodal):

    if is_multimodal:
        feature_df = pd.read_csv(time_series_features_path)
        relevant = pd.read_csv(relevant_features_path)['x'].values.tolist()
        feature_df = feature_df[feature_df.columns.intersection(relevant)]
        feature_columns = feature_df.columns
        feature_df = pd.concat([label_df, feature_df], axis=1)
        num_ts_features = len(feature_columns)
    else:
        feature_df = label_df
        feature_columns = None
        num_ts_features = 0

    img_size = 224
    return get_train_val_test_dataloaders(
        img_size=img_size, 
        img_dir=img_dir, 
        feature_df=feature_df, 
        feature_columns=feature_columns,
        label_column=label_column, 
        join_column=join_column, 
        split_column=split_column,
        split_sizes=[0.8, 0.1, 0.1], 
        batch_size=batch_size, 
        num_workers=num_workers, 
        random_state=random_state
    ), num_ts_features

def get_model(is_multi_modal, num_classes, num_ts_features=0):
    if is_multi_modal:
        model = get_basic_model(num_classes=num_classes, num_ts_features=num_ts_features)
    else:
        model = get_pure_img_model(num_classes=num_classes)

    return model

def get_trainer_args(is_multi_modal, ckpt_path, task):

    model_name = 'mm_convnext_tiny' if is_multi_modal else 'convnext_tiny'
    
    logger = logger=pl_loggers.TensorBoardLogger(save_dir=data_path, name=f'logs/{model_name}/{task}', version=None) if ckpt_path is None else None
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/F1Score",
        mode="max",
        dirpath=data_path + f"checkpoints/{model_name}/{task}",
        filename="model-{epoch:02d}-{val/F1Score:.2f}",
        save_weights_only=True,
    )

    writer = CustomWriter(
        write_interval='epoch',
        output_file=data_path + f'outputs/{task}_{model_name}.pkl'
    )

    early_stop = EarlyStopping(monitor="val/F1Score", min_delta=0.00, patience=10, verbose=False, mode="max")

    if ckpt_path is None:
        return logger, [checkpoint_callback, writer, early_stop]
    else:
        return False, [writer]

# Main function
def main():

    # Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_multi_modal')
    parser.add_argument('--task')
    parser.add_argument('--ckpt_path')

    args = parser.parse_args()
    is_multi_modal = args.is_multi_modal == 'True' # I know no smart way to do this
    task = str(args.task)
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else None

    print(f'Your arguments for (is_multi_modal, task, ckpt_path) are ({is_multi_modal}, {task}, {ckpt_path})')

    label_df = get_task(task)

    # Get the dataloaders
    (train_loader, val_loader, test_loader), num_ts_features = get_dataloaders(label_df, is_multi_modal)
    num_classes = train_loader.dataset.num_classes

    # get the proper model
    model = get_model(is_multi_modal, num_classes, num_ts_features)

    # Define training parameters
    class_weights = train_loader.dataset.calculate_class_weights().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Prepare training, only set logger if you actually train a model
    wrapper = ClassificationWrapper(model=model, learning_rate=5e-5, weight_decay=0.01, loss=criterion, num_classes=num_classes)
    logger, callbacks = get_trainer_args(is_multi_modal, ckpt_path, task)
    
    # Define trainer
    trainer = pl.Trainer(accelerator="gpu", devices=num_gpus, logger=logger, max_epochs=100, callbacks=callbacks)

    # Finally fit model if not given a checkpoint
    if ckpt_path is None:
        trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Evaluate the model on test and train set
    trainer.test(wrapper, dataloaders=test_loader, ckpt_path='best' if not ckpt_path else ckpt_path)
    trainer.predict(wrapper, dataloaders=test_loader, ckpt_path='best' if not ckpt_path else ckpt_path)

if __name__ == '__main__':
    main()