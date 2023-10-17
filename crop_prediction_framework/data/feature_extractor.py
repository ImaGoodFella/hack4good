import numpy as np
import pandas as pd
import xarray as xr
import os

from tqdm import tqdm
tqdm.pandas()

def get_time_series_features_df(label_df, join_column, path_weather_data, use_cache=True, cache_file=None):
    
    # use cache features if saved
    if use_cache and os.path.isfile(cache_file):
        label_df = pd.read_csv(cache_file)
        ts_columns = [c for c in label_df.columns if c.startswith('ts_columns')]
        return label_df, ts_columns
    
    # open weather data
    df = pd.read_csv(path_weather_data, index_col=(0, 1, 2))
    df.index = df.index.set_levels(df.index.levels[2].astype('datetime64[ns]'), level=2)
    weather_data = df.to_xarray()

    def apply_extract(row):
        return pd.Series(extract_features(row[join_column], weather_data, label_df, join_column))

    ts_df = label_df.progress_apply(apply_extract, axis=1)
    ts_df = ts_df.rename(lambda x: 'ts_columns' + str(x), axis=1)
    ts_columns = ts_df.columns

    label_df = pd.concat([label_df, ts_df], axis=1)
    label_df.to_csv(cache_file, index=False)

    return label_df, ts_columns

def get_coords(img_name, labels, join_column):
    row = labels[labels[join_column] == img_name]
    return (row.iloc[0]['lat'], row.iloc[0]['lon'])

def extract_features(img_name, weather_data, labels, join_column):
    # get relevant data
    (lon, lat) = get_coords(img_name, labels, join_column)
    date = labels[labels[join_column] == img_name].iloc[0]['date']
    df = weather_data.sel(latitude= lat, longitude= lon, method='nearest').sel(time = slice(date, date + pd.DateOffset(months=1))).to_dataframe()
    
    # extract features
    temperature = (df["t2m"].values.max(), df["t2m"].values.min(), np.median(df["t2m"].values))
    precipitation = (df["tp"].values.max(), np.median(df["tp"].values), df["tp"].values[:168].sum(), df["tp"].values[:336].sum(), df["tp"].values.sum())
    evaporation = (df["pev"].values.max(), np.median(df["pev"].values), df["pev"].values[:168].sum(), df["pev"].values[:336].sum(), df["pev"].values.sum())
    return np.array(temperature + precipitation + evaporation)
