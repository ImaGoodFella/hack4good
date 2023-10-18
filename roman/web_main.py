
from flask import Flask, render_template, request, send_from_directory, redirect, json, make_response, Response, jsonify
from dataclasses import dataclass
from random import randint, choice
from enum import Enum
#import pandas as pd
import csv
import uuid
#import pyopenssl #this is just here to mark the dependency

app = Flask(__name__)

#from flask_assets import Bundle, Environment
#assets = Environment(app)
#css = Bundle("src/main.css", output="dist/main.css")
#assets.register("css", css)
#css.build()

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
    date : str #not super happy about this being a string
    lon : float
    lat : float
    farmer_id : str
    site_id :str
    damage_extent : int
    damage_type : DamageType

def validate_csv(csv_reader : csv.reader):
    #FIXME TODO verify first col parses as date, second as lon, etc.
    return True

def analyse_image(img_file, csv_file):
    #O(n^2) please keep it to yourselves that I wrote this
    csv_text = csv_file.read().decode("utf-8")
    csv_row = None
    csv_reader = csv.reader(csv_text.split("\n"), delimiter=',')
    for row in csv_reader: 
        if row[0].lower()== img_file.filename.lower():
            csv_row = row
            break

    print("FOUND CSV ROW: ", csv_row)

    #FIXME these hardcoded csv entry positions are a crime against maintainability
    return UploadResponse \
        (\
            date = csv_row[1],\
            lon = csv_row[5],\
            lat = csv_row[4],\
            damage_extent = str(randint(0, 10)*10),\
            damage_type = choice(list(DamageType)).name,\
            farmer_id=csv_row[2],\
            site_id =csv_row[3]\
        )


#should the upload and analysis be separated??
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

    if csv_file.filename == '':
        return 'bad request! Missing CSV.', 400
    if csv_file.filename == '':
        return 'bad request! Missing Image.', 400

    return jsonify(analyse_image(img_file, csv_file)), 200


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

@app.route("/")
def homepage():
    #return render_template("index.html")
    with open("index.html") as f:
        return f.read()


if __name__ == "__main__":
    #app.run(ssl_context='adhoc', debug=True)
    app.run(debug=True)