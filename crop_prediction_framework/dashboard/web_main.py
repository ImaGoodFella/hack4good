from flask import Flask, render_template, request, send_from_directory, redirect, json, make_response, Response, jsonify
from dataclasses import dataclass
import datetime
from typing import Optional
from random import randint, choice
from enum import Enum
from pathlib import Path
import csv
import cdsapi #https://pypi.org/project/cdsapi/
import uuid
import xarray as xr
import numpy as np
import netCDF4 #only used to fail early if not installed
import sqlite3
import os
import pickle
import time

# https://pypi.org/project/sqlitedict/
# https://pypi.org/project/result/

DEBUG = True



app = Flask(__name__)

class DamageType(Enum):
    Good = 0 #G
    Drought = 1 #DR
    Disease = 2 #DS
    Flood = 3 #FD
    NutrientDeficient = 4 #ND
    Pest = 5 #PS
    Weed = 6 #WD
    Wind = 7 #WN

@dataclass
class UploadResponse():
    date : datetime.date #not super happy about this being a string
    lon : float
    lat : float
    farmer_id : str
    site_id :str
    damage_extent : int
    damage_type : DamageType

@dataclass
class WeatherSeries():
    """Weather data for a single pixel over a period of time. Might have been smarter to simply use an xarray."""
    end_date : datetime.date
    duration_days : int
    lat : float
    lon : float
    temperature_2m : np.array
    total_precipitation : np.array
    potential_evaporation : np.array




class ClimateData():
    def __init__(self, data_path : Path, url : str = None, key : str = None):
        self.source : str = 'reanalysis-era5-land'
        self.pixels_per_degree : float = 10.0 #era5-land resolution
        self.url : str = url
        self.key : str = key
        self.debug : bool = DEBUG
        self.data_path : Path = data_path
    
    def _round_to_nearest_pixel(self, v : float):
        return round(v*self.pixels_per_degree)/self.pixels_per_degree
    
    def _request_filename(self, date : datetime.date, lat : float, lon : float) -> str:
        lat_rounded = self._round_to_nearest_pixel(lat)
        lon_rounded = self._round_to_nearest_pixel(lon)
        lat_s = f"{lat_rounded:02}".replace('.', '')
        lon_s = f"{lon_rounded:02}".replace('.', '')
        ncf_name = f"era5_land_{date.year}_{date.month}_{date.day}_{lat_s}_{lon_s}.nc"
        return ncf_name

    def _date2datetime(self, date : datetime.date, hour : int) -> datetime.datetime:
            return (datetime.datetime(year = date.year, month = date.month, day = date.day) + datetime.timedelta(hours=hour))

    
    def _get_iso_datetime_from_hour(self, date : datetime.date, hour : int) -> str:
        dt = self._date2datetime(date, hour)
        return dt.isoformat().replace("T", " ")

    def _date_to_unix_timestamp(self, date : datetime.date, hour: int) -> int:
        return int(time.mktime(self._date2datetime(date, hour).timetuple()))

    
    def _get_from_cache(self, lat : float, lon : float, from_date : datetime.date, to_date : datetime.date) -> Optional[WeatherSeries]:
        """Dates are INCLUSIVE that is data will be returned from from_date at 00:00 to do_date at 23:00. Returns None if not in cache"""
        connection = sqlite3.connect(os.path.normpath(self.data_path+"/weather_cache.db"))
        cursor = connection.cursor()
        self._create_cache_table(cursor)
        lat_rounded = self._round_to_nearest_pixel(lat)
        lon_rounded = self._round_to_nearest_pixel(lon)
        from_ts = self._date_to_unix_timestamp(from_date, 0)
        to_ts = self._date_to_unix_timestamp(to_date, 23)
        query = f"SELECT * FROM weather_cache WHERE lat={lat_rounded} AND lon={lon_rounded} AND unixtime<={to_ts} AND unixtime>={from_ts}"
        result = cursor.execute(query)
        result_list = result.fetchall()
        connection.commit()

        print("LENGTH EXPECTED" , (to_date - from_date).days*24, "ACTUAL", len(result_list))

        #TODO could report missing dates instead of just returning None
        return None if len(result_list) == 0 else WeatherSeries(
            end_date = to_date,
            duration_days = (to_date - from_date).days,
            lat = lat_rounded,
            lon = lon_rounded,
            temperature_2m = np.array([x[3] for x in result_list]),
            total_precipitation = np.array([x[4] for x in result_list]),
            potential_evaporation = np.array([x[5] for x in result_list])
        )

    def _insert_whether_data_into_cache(self, cursor, weather_series : WeatherSeries):
        begin_date = weather_series.end_date - datetime.timedelta(days=weather_series.duration_days)
        for i in range(weather_series.duration_days*24):
            query = f"""INSERT INTO weather_cache VALUES (
                {self._round_to_nearest_pixel(weather_series.lat)},
                {self._round_to_nearest_pixel(weather_series.lon)},
                {self._date_to_unix_timestamp(begin_date, i)},
                {weather_series.temperature_2m[i]},
                {weather_series.total_precipitation[i]},
                {weather_series.potential_evaporation[i]})"""
            cursor.execute(query)

    def _create_cache_table(self, cursor):
        cursor.execute("""CREATE TABLE IF NOT EXISTS weather_cache (
                       lat REAL,
                       lon REAL,
                       unixtime INTEGER,
                       temperature_2m REAL,
                       total_precipitation REAL,
                       potential_evaporation REAL
                    )""")

    def _write_to_cache(self, weather_series : WeatherSeries):
        connection = sqlite3.connect(os.path.normpath(self.data_path+"/weather_cache.db"))
        cursor = connection.cursor()
        self._create_cache_table(cursor)
        self._insert_whether_data_into_cache(cursor, weather_series)
        connection.commit()
    
    def _download(self, lat : float, lon : float, date : datetime.date, days : int = 30):
        #it might look expensive to cache everything but it takes only roughtly
        # 60 pixels_lat * 50 pixels_on * 24 hours * 3 variables * 4 bytes per variable = 800kb per day
        # (assuming a a participating farmer in every square of kenya which seemms unlikely)
        lat_rounded = self._round_to_nearest_pixel(lat)
        lon_rounded = self._round_to_nearest_pixel(lon)
        coord_offset = 0.0001#0.5/self.pixels_per_degree
        date_begin = date - datetime.timedelta(days=days)
        ncf_name = self._request_filename(date, lat, lon)
        c = cdsapi.Client(wait_until_complete=True, url=self.url, verify=True, key=self.key, debug=self.debug, quiet = not self.debug)
        request = {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['2m_temperature', 'total_precipitation', 'potential_evaporation'],
                'date': [
                    f'{date_begin.isoformat()}/{date.isoformat()}',
                ],
                'time': ['ALL'],
                'area': [
                    #2, 32, -3,
                    #36,
                    lat_rounded + coord_offset, lon_rounded - coord_offset, lat_rounded - coord_offset, lon_rounded + coord_offset
                ]
            }

        #'reanalysis-era5-single-levels',#Includes sea information which for our purposes is not needed
        c.retrieve(self.source, request, ncf_name)
        #c.retrieve('reanalysis-era5-single-levels', request, ncf_name)
        #df_era5 = xr.open_dataset(ncf_name).to_dataframe()
        #print(df_era5)

    
    def get_time_series(self, lat : float, lon : float, date : datetime.date, days : int = 30) -> WeatherSeries:
        #check cache
        weather_series = self._get_from_cache(lat, lon, date - datetime.timedelta(days=days), date)
        if weather_series is None:
            #if not in cache, download from cdsapi.
            weather_filename = self._request_filename(date, lat, lon)
            if not os.path.exists(weather_filename):
                #if not downloaded previously, download
                self._download(lat, lon, date, days)
            df = xr.open_dataset(weather_filename, engine='netcdf4').to_dataframe()
            weather_series = WeatherSeries(
                end_date = date,
                duration_days = days,
                lat = lat,
                lon = lon,
                temperature_2m = df['t2m'].values,
                total_precipitation = df['tp'].values,
                potential_evaporation = df['pev'].values
            )
            self._write_to_cache(weather_series)
        return weather_series





def get_climate_data(url : str, key : str, lat : float, lon : float, date : datetime.date):
    # https://github.com/ecmwf/cdsapi/blob/master/cdsapi/api.py
    #debug = True
    #c = cdsapi.Client(wait_until_complete=False, url=url, verify=True, key=key, debug=debug, quiet = not debug)
    return ClimateData("./", url, key).get_time_series(lat, lon, date)

def validate_csv(csv_reader : csv.reader):
    #FIXME verify first col parses as date, second as lon, etc.
    return True

def get_matching_csv_row(csv_text : str, img_filename : Path) -> UploadResponse:
    #O(n^2) please keep it to yourselves that I wrote this
    #FIXME should handle decoding errors, might want to use https://pypi.org/project/result/ ??
    csv_row = None
    csv_reader = csv.reader(csv_text.split("\n"), delimiter=',')
    for row in csv_reader:
        if len(list(row)) > 0 and list(row)[0].lower()== str(img_filename).lower():
            csv_row = row
            break

    if csv_row is None or len(csv_row) < 6:
        #TODO create better error message
        return None

    f = "%Y-%m-%d %H:%M:%S"
    #FIXME some dates are of format "%Y-%m-%d" only so this might fail
    date = datetime.datetime.strptime(csv_row[1], f)
    #TODO these hardcoded csv entry positions are a crime against maintainability
    return UploadResponse (
            date = date.date(),
            lon = float(csv_row[5]),
            lat = float(csv_row[4]),
            #damage should be moved away because they are part of analysis response, not upload response
            damage_extent = randint(0, 10)*10,
            damage_type = choice(list(DamageType)).name,
            farmer_id=csv_row[2],
            site_id =csv_row[3]
        )


def analyse_image(img_file, csv_file, url, key):
    csv_text = csv_file.read().decode("utf-8")
    response = get_matching_csv_row(csv_text, Path(img_file.filename))
    weather_data = get_climate_data(url, key, response.lat, response.lon, response.date)
    return response


#TODO should the upload and analysis be separated??
#@app.route("/analysis/<uuid:analysis_id>", methods=["GET"])
#def analysis(analysis_id):
#    return jsonify(analyse_image), 200


@app.route("/upload", methods=["POST"])
def upload():
    #TODO support mutiple files?
    if 'csv_file' not in request.files:
        return "bad request! Missing `csv_file` field in post.", 400
    if 'img_file' not in request.files:
        return "bad request! Missing `img_file` field in post.", 400

    csv_file = request.files['csv_file']
    img_file = request.files['img_file']

    #TODO if one were to use templates one could have a centralised list of acceptable file extension inserted into the form and then checked here
    # not ".csv" == Path(csv_file.filename).suffix.lower()
    if csv_file.filename == '':
        return "bad request! Missing CSV.", 400
    if img_file.filename == '':
        return "bad request! Missing Image.", 400

    #TODO would be nice if the user could choose to upload the `$HOME/.cdsapirc` file
    url = request.form["cdsapi_url"]
    key = request.form["cdsapi_usr_id_key_pair"]

    if not validate_csv(csv_file):
        return "bad request! CSV file is not formatted as required.", 400

    results = analyse_image(img_file, csv_file, url, key)
    if results is None:
        return make_response("CSV file does not contain an entry for the image file name: "+img_file.filename, 400)
    return jsonify(results), 200


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

@app.route("/")
def homepage():
    with open("index.html") as f:
        return f.read()

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=DEBUG)
