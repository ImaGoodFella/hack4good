import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score, classification_report
label_encoder = preprocessing.LabelEncoder() 

random_state = 42

# Data path configuration
data_path = "../data/"

# Files and Folders of interest
time_series_features_path = data_path + "tf_features_full_narm.csv"
label_path = data_path + "labels.csv"
relevant_features_path = data_path + "relevant_features.csv"


data = pd.read_csv(time_series_features_path)
labels = pd.read_csv(label_path)
relevant = pd.read_csv(relevant_features_path)['x'].values.tolist()
relevant.append('damage')

merged = pd.merge(labels, data, left_index=True, right_on='Unnamed: 0')

df = merged[merged.columns.intersection(relevant)]

train, test = train_test_split(df, test_size=0.2,random_state=random_state)
train_x = train.drop(['damage'], axis=1)
train_y = label_encoder.fit_transform(train['damage'])
test_x = test.drop(['damage'], axis=1)
test_y = test['damage']


model = xgb.XGBClassifier()
model.fit(train_x, train_y)

predictions = label_encoder.inverse_transform(model.predict(test_x))
#Calculating accuracy
accuracy = accuracy_score(test_y, predictions)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(test_y, predictions))


