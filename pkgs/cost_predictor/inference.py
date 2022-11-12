import datetime
import logging
import math
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests
# import reverse_geocoder as rg
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

logger = logging.getLogger()
logging.basicConfig(filename="output.log", level=logging.DEBUG)
logger.addHandler(logging.FileHandler("output.log", "a"))

lat_1, lon_1, lat_2, lon_2 = (
    "origin_latitude",
    "origin_longitude",
    "destination_latitude",
    "destination_longitude",
)

## Creating a class to first create new features.
class FeatenggTransformer(BaseEstimator, RegressorMixin, TransformerMixin):
    def transform(self, X):

        """This pipeline transforms the dataframe to the desired
        features and shape"""

        X = self.add_date_feature(X)
        X = self.add_travel_features(X)
        X = self.get_country_feats(X)
        X = self.drop_unrequired_fields(X)
        return X

    def add_travel_features(self, X):

        """This Method creates distance features"""

        if not 0 <= X[lon_1].all() <= 90 or not 0 <= X[lat_1].all() <= 180 or \
            not 0 <= X[lon_2].all() <= 90 or not 0 <= X[lat_2].all() <= 180:
            raise ValueError("Invalid latitude, longitude combination.")
        else:
            X["abs_diff_longitude"] = (X[lon_2] - X[lon_1]).abs()
            X["abs_diff_latitude"] = (X[lat_2] - X[lat_1]).abs()
            X["Vincenty_distance"] = X.apply(
                lambda x: vincenty((x[lat_1], x[lon_1]), (x[lat_2], x[lon_2]), miles=True), axis=1
            )

        # bearing (in degrees converted to radians)
            def bearing_array(lat1, lng1, lat2, lng2):
                AVG_EARTH_RADIUS = 6371  # in km
                lng_delta_rad = np.radians(lng2 - lng1)
                lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
                y = np.sin(lng_delta_rad) * np.cos(lat2)
                x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
                return np.degrees(np.arctan2(y, x))

            X["Bearing"] = bearing_array(X[lat_1], X[lon_1], X[lat_2], X[lon_2])
            X.loc[:, "center_latitude"] = (X[lat_1].values + X[lat_2].values) / 2
            X.loc[:, "center_longitude"] = (X[lon_1].values + X[lon_2].values) / 2
            return X

    def add_date_feature(self, X):

        """This Method creates time series/date features"""

        ref_date = "2017-01-01"
        if not (X['shipping_date'] >= ref_date).all():
            raise ValueError(f"Invalid Date Date should be later than {ref_date}")

        X["weekday"] = X["shipping_date"].dt.day_of_week
        X["is_weekend"] = np.where(X["weekday"] < 5, 0, 1)
        X["week_in_month"] = pd.to_numeric(X["shipping_date"].dt.day / 7).apply(
            lambda x: math.ceil(x)
        )
        X["month"] = X["shipping_date"].dt.month
        X["day_count"] = (
            X["shipping_date"] - datetime.strptime(ref_date, "%Y-%m-%d")
        ) / np.timedelta64(1, "D")
        return X

    def get_country_feats(self, X):
        data = requests.get(
            "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
        ).json()
        countries = {}
        for feature in data["features"]:
            geom = feature["geometry"]
            country = feature["properties"]["ADMIN"]
            countries[country] = prep(shape(geom))

        def get_country(row):
            point = Point(row[0], row[1])
            for country, geom in countries.items():
                if geom.contains(point):
                    return country
            return "unknown"

        X["destn_country"] = X[[lon_2, lat_2]].apply(get_country, axis=1)
        X["origin_country"] = X[[lon_1, lat_1]].apply(get_country, axis=1)
        X["diff_country"] = np.where(X["origin_country"] == X["destn_country"], 0, 1)
        return X

    def drop_unrequired_fields(self, X):

        """Method for dropping unrequired fields"""

        return X.drop(["shipping_date", "is_adr", lat_1, lon_1, lat_2, lon_2], axis=1)


# Feature Transformation
class FeatureTransformer(BaseEstimator, RegressorMixin, TransformerMixin):

    """This Class is for FeatureTransform
    a) categorical feature encoding
    b) feature scaling
    """

    def __init__(self, cat_cols, num_cols, path):
        self.cat_cols, self.num_cols, self.path = cat_cols, num_cols, path
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def transform(self, X):

        """This pipeline transforms the categorical features to One-Hot Encoded and scales the
        features"""

        X = self.encode_categorical(X)
        return X

    def encode_categorical(self, X):

        """This Method creates absolute difference features
        and deserializes(unpickle) the necessary objects for inference"""

        with open(os.path.join(self.path, "ohe.pkl"), "rb") as fout:
            ohe = pickle.load(fout)
        x_cat_df = pd.DataFrame(ohe.transform(X[self.cat_cols]))
        x_cat_df.columns = ohe.get_feature_names_out(self.cat_cols)
        X = pd.concat([X[num_cols], x_cat_df], axis=1)
        return X


# Prediction Class
class Prediction(BaseEstimator, TransformerMixin):

    """Object to make prediction/inference using Model"""

    def __init__(self, model_path=None, data_path=None):
        self.model_path = model_path
        self.pred_path = pred_path

    def prediction(self, X):

        """Method to make prediction loading model and saves data"""

        with open(os.path.join(self.model_path, "stacked_model_final.pkl"), "rb") as fout:
            self.model = pickle.load(fout)
        pred = self.model.predict(X)
        pred = pd.DataFrame({"cost": list(pred)})
        # pred.to_csv(self.pred_path, index=False)
        return pred


if __name__ == "__main__":

    """This main function orchestrates the entire dataprep, scoring process"""

    # Defining all Paths and variables

    path = "../artifacts_final/"
    data_path = "../data/test_data_hold.csv"
    pred_path = "../prediction.csv"
    cat_cols = [
        "week_in_month",
        "weekday",
        "is_weekend",
        "month",
        "destn_country",
        "origin_country",
    ]

    # Implementation

    print(f'process started at {datetime.now().strftime("%d.%b %Y %H:%M:%S")}')
    df = pd.read_csv(data_path, parse_dates=["shipping_date"])
    target = 'cost'
    lat_1,lon_1,lat_2,lon_2 = 'origin_latitude','origin_longitude','destination_latitude','destination_longitude'
    Y = df[target]
    df = df.drop([target],axis=1)
    fet = FeatenggTransformer()
    X = fet.transform(df)
    num_cols = [col for col in list(X.columns) if col not in cat_cols]
    ft = FeatureTransformer(cat_cols=cat_cols, num_cols=num_cols, path=path)
    X_transformed = ft.transform(X)
    p = Prediction(path, pred_path)
    pred = p.prediction(X_transformed)
    print(r2_score(Y,pred))
    print(f"predicted mean:{pred.mean()}")
    print(f'process ended at {datetime.now().strftime("%d.%b %Y %H:%M:%S")}')
