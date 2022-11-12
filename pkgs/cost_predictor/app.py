import sys
sys.executable
import pickle
import datetime
import logging
import math
import os
import pickle
import sys
from datetime import datetime
from flask import Flask, request
import numpy as np
import pandas as pd
# import requests
from shapely.geometry import Point, mapping, shape
from shapely.prepared import prep
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import (OneHotEncoder, PowerTransformer,
                                   RobustScaler, StandardScaler)
from vincenty import vincenty
from xgboost import XGBRegressor, plot_importance
from cost_predictor.inference import FeatenggTransformer, FeatureTransformer, Prediction

app = Flask(__name__)
model_path = '../artifacts_final/stacked_model_final.pkl'
pred_path = "../prediction.csv"
cat_cols = [
        "week_in_month",
        "weekday",
        "is_weekend",
        "month",
        "destn_country",
        "origin_country",
    ]
lat_1,lon_1,lat_2,lon_2 = 'origin_latitude','origin_longitude','destination_latitude','destination_longitude'

@app.route("/predict", methods=['POST'])
def predict():

    instance = request.json
    df = pd.DataFrame(instance)
    print(df)
    df['shipping_date'] = pd.to_datetime(df['shipping_date'])
    fet = FeatenggTransformer()
    X = fet.transform(df)
    num_cols = [col for col in list(X.columns) if col not in cat_cols]
    ft = FeatureTransformer(cat_cols=cat_cols, num_cols=num_cols, path=model_path)
    X_transformed = ft.transform(X)
    p = Prediction(model_path, pred_path)
    pred = p.prediction(X_transformed)

    return {'label': pred}


if __name__ == "__main__":
    app.run(port=5000, debug=True,host='0.0.0.0')