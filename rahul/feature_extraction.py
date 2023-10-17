import numpy as np
import pandas as pd
import xarray as xr


def get_data(path_weather_data, path_labels):
    df = pd.read_csv(path_weather_data, index_col=(0, 1, 2))
    df.index = df.index.set_levels(df.index.levels[2].astype('datetime64[ns]'), level=2)
    weather_data = df.to_xarray()
    labels = pd.read_csv(path_labels)
    labels['date'] = pd.to_datetime(labels['date'], format='mixed')
    return (weather_data, labels)

def get_coords(img_name, labels):
    row = labels[labels["filename"] == img_name]
    return (row.iloc[0]['lat'], row.iloc[0]['lon'])

def extract_features(img_name, weather_data, labels):
    # get relevant data
    (lon, lat) = get_coords(img_name, labels)
    date = labels[labels["filename"] == img_name].iloc[0]['date']
    df = weather_data.sel(latitude= lat, longitude= lon, method='nearest').sel(time = slice(date, date + pd.DateOffset(months=1))).to_dataframe()
    
    # extract features
    temperature = (df["t2m"].values.max(), df["t2m"].values.min(), np.median(df["t2m"].values))
    precipitation = (df["tp"].values.max(), np.median(df["tp"].values), df["tp"].values[:168].sum(), df["tp"].values[:336].sum(), df["tp"].values.sum())
    evaporation = (df["pev"].values.max(), np.median(df["pev"].values), df["pev"].values[:168].sum(), df["pev"].values[:336].sum(), df["pev"].values.sum())
    return np.array(temperature + precipitation + evaporation)
