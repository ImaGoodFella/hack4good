import numpy as np
import pandas as pd
import xarray as xr
import os

from tqdm import tqdm
tqdm.pandas()

def get_time_series_features_df(label_df, join_column, path_weather_data, use_cache=True, cache_file=None):
    
    # use cache features if saved
    if use_cache and os.path.isfile(cache_file):
        ts_df = pd.read_csv(cache_file)
        ts_columns = ts_df.columns
        label_df = pd.concat([label_df, ts_df], axis=1)
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
    ts_df.to_csv(cache_file, index=False)

    label_df = pd.concat([label_df, ts_df], axis=1)
    
    return label_df, ts_columns

def get_coords(img_name, labels, join_column):
    row = labels[labels[join_column] == img_name]
    return (row.iloc[0]['lat'], row.iloc[0]['lon'])
    
#function that returns number of 'spells' of a parameter
def get_spells(data,param:str,value,higher:bool,spell:int):
    
    #value: threshold to consider
    #higher: whether we want to be higher or lower than the threshold
    #spell: how many days do we consider a spell

    d = data[param].values.reshape((-1,24)) #we consider the daily averages (technically not over a day but 24 hours)
    d = np.mean(d, axis = 1)    
    d = np.where(d>value,1,0)     
    n = len(d)
    if n == 0: 
        return (None)
    else:
        y = d[1:] != d[:-1]               
        i = np.append(np.where(y), n - 1) 
        z = np.diff(np.append(-1, i))
        res = z[d[i]==higher] 
        numspells = sum(k >= spell for k in res)
        return (numspells)

def extract_features(img_name, weather_data, labels, join_column):
    # get relevant data
    (lon, lat) = get_coords(img_name, labels, join_column)
    date = labels[labels[join_column] == img_name].iloc[0]['date']
    df = weather_data.sel(latitude= lat, longitude= lon, method='nearest').sel(time = slice(date - pd.DateOffset(days=30, second=1), date)).to_dataframe()
    
    # extract features
    day = (np.sin(2 * np.pi * date.timetuple().tm_yday/365.0), np.cos(2 * np.pi * date.timetuple().tm_yday/365.0))
    temperature = (df["t2m"].values.max(), df["t2m"].values.min(), np.median(df["t2m"].values))
    precipitation = (df["tp"].values.max(), np.median(df["tp"].values), df["tp"].values[:168].sum(), df["tp"].values[:336].sum(), df["tp"].values.sum())
    evaporation = (df["pev"].values.max(), np.median(df["pev"].values), df["pev"].values[:168].sum(), df["pev"].values[:336].sum(), df["pev"].values.sum())
    return np.array(temperature + precipitation + evaporation + day)