import numpy as np
import pandas as pd
import xarray as xr
import os
import math

from tqdm import tqdm
tqdm.pandas()
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters 
settings = EfficientFCParameters() #extract only time series features which are efficient to compute


data_path = "../data/"

# Files and Folders of interest
cache_file = data_path + 'time_series_features.csv'
path_weather_data = data_path + 'era5_land_t2m_pev_tp.csv' ## we assumme the data is given in the form one would recieve it in from the cds-api
label_path = data_path + "labels.csv"

df = pd.read_csv(path_weather_data, index_col=(0, 1, 2))
df.index = df.index.set_levels(df.index.levels[2].astype('datetime64[ns]'), level=2)
weather_data = df.to_xarray()
labels = pd.read_csv(label_path)
labels['date'] = pd.to_datetime(labels['date'], format='mixed')

def get_coords(img_name, labels, join_column):
    row = labels[labels[join_column] == img_name]
    return (row.iloc[0]['lat'], row.iloc[0]['lon'])

def get_ts_of_coordinates(img_name, weather_data, labels, join_column):
    # get the time series for each coordinate indexed by its position in the labels.csv
    frames=[]
    for v,i in enumerate(img_name):
        (lon, lat) = get_coords(i, labels, join_column)
        date = labels[labels[join_column] == i].iloc[0]['date']
        df = weather_data.sel(latitude= lat, longitude= lon, method='nearest').sel(time = slice(date - pd.DateOffset(days=30, second=1), date)).to_dataframe()
        df.drop('longitude',axis=1,inplace=True)
        df.drop('latitude',axis=1,inplace=True)
        df['id']=v
        frames.append(df)
    final = pd.concat(frames)
    return final


ts_coord = get_ts_of_coordinates(labels['filename'],weather_data,labels,'filename')


def ts_fresh_feature_extraction(time_series):
    frames=[]
    m = math.ceil(time_series.shape[0]/4000)
    for i in range(m):
        df = extract_features(time_series.loc[time_series['id']%m==i], column_id='id', column_sort='time',default_fc_parameters=settings)
        frames.append(df)
    final = pd.concat(frames)
    return final



ts_features = ts_fresh_feature_extraction(ts_coord)
ts_features.dropna(axis=1, inplace = True)

ts_features.to_csv(data_path+"ts_features.csv")