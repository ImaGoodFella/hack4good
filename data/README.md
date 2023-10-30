This directory stores the data used for training and evaluating the models of the project.

In order to run the `main.py` model training script, this directory must contain the following files:

- `time_series_features.csv`
- `era5_land_t2m_pev_tp.csv`
- A folder `images/` containing the `*.jpg` files provided by IFPRI
- `labels.csv`

In addition, the subfolder `models/` is used to save the trained model weights. Each saved model is named after the timestamp of training.
