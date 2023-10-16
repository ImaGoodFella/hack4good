import pandas as pd
from data.utils import get_train_val_test_dataloaders

# Data path configuration
data_path = "/home/rasteiger/datasets/hack4good/"
img_dir = data_path + "images"
csv_path = data_path + "labels.csv"

# Reading the csv file
csv_file = pd.read_csv(csv_path)

# Setting the labels and join columns
csv_file['is_damage'] = (csv_file['extent'] >= 20).astype(int)
label_column = 'is_damage'
join_column = 'filename'
split_column = 'farmer_id'

img_size = 224

train_loader, val_loader, test_loader = get_train_val_test_dataloaders(img_size=img_size, img_dir=img_dir, feature_df=csv_file,
                                                                       label_column=label_column, join_column=join_column, split_column=split_column,
                                                                       split_sizes=[0.8, 0.1, 0.1], batch_size=32, num_workers=64, random_state=42)


for x in [train_loader, val_loader, test_loader]:
    print(len(x.dataset))









