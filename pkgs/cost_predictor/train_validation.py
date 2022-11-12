import datetime
import logging
import math
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
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
from xgboost import XGBRegressor, plot_importance

logger = logging.getLogger()
logging.basicConfig(filename="output.log", level=logging.DEBUG)
logger.addHandler(logging.FileHandler("output.log", "a"))

## Creating a class to first create new features for ML training- Feature Engg.
class FeatenggTransformer(BaseEstimator, RegressorMixin, TransformerMixin):
    def transform(self, X):

        """This pipeline transforms the dataframe to the desired
        features and shape"""

        X = self.add_date_feature(X)
        X = self.add_travel_features(X)
        X["distance"] = np.round(self.add_distance_feature(X), 2)
        X = self.drop_unrequired_fields(X)
        # X = self.one_hot_encode(X)
        return X

    def add_travel_features(self, X):

        """This Method creates absolute difference features"""

        X["abs_diff_longitude"] = (X.destination_longitude - X.origin_longitude).abs()
        X["abs_diff_latitude"] = (X.destination_latitude - X.origin_latitude).abs()
        return X

    def add_date_feature(self, X):

        """This Method creates time series/date features"""

        X["weekday"] = X["shipping_date"].dt.day_of_week
        X["is_weekend"] = np.where(X["weekday"] < 5, 0, 1)
        X["week_in_month"] = pd.to_numeric(X["shipping_date"].dt.day / 7).apply(
            lambda x: math.ceil(x)
        )
        X["month"] = X["shipping_date"].dt.month
        X["year"] = X["shipping_date"].dt.year
        return X

    def add_distance_feature(self, X):

        """Method for calculating distance from start lat/long to end lat/long using haversine-formula
        https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
        """

        radius_of_earth = 6373.0
        lat1 = np.asarray(np.radians(X["origin_latitude"]))
        lon1 = np.asarray(np.radians(X["origin_longitude"]))
        lat2 = np.asarray(np.radians(X["destination_latitude"]))
        lon2 = np.asarray(np.radians(X["destination_longitude"]))

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = radius_of_earth * c

        distance_array = np.asarray(distance) * 0.621
        return pd.Series(distance_array)

    def drop_unrequired_fields(self, X):

        """Method for dropping unrequired fields"""

        return X.drop(
            [
                "origin_latitude",
                "origin_longitude",
                "destination_latitude",
                "destination_longitude",
                "shipping_date",
                "is_adr",
            ],
            axis=1,
        )


## Feature Transformation
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
        X = self.scale_features(X)
        return X

    def encode_categorical(self, X):

        """This Method creates absolute difference features
        and Serializes(pickle) the necessary objects for inference"""

        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        ohe.fit(X[self.cat_cols])
        x_cat_df = pd.DataFrame(ohe.transform(X[self.cat_cols]))
        x_cat_df.columns = ohe.get_feature_names_out(self.cat_cols)
        X = pd.concat([X[num_cols], x_cat_df], axis=1)
        with open(os.path.join(self.path, "ohe.pkl"), "wb") as fout:
            pickle.dump(ohe, fout)
        return X

    def scale_features(self, X):

        """This Method scales features before feeding into ML training
        and Serializes(pickle) the necessary objects for inference"""

        scaling_feats = ["abs_diff_longitude", "abs_diff_latitude"]
        pt = PowerTransformer().fit(X[scaling_feats])
        X[scaling_feats] = pd.DataFrame(pt.transform(X[scaling_feats]))
        with open(os.path.join(self.path, "pt.pkl"), "wb") as fout:
            pickle.dump(pt, fout)
        max_val, min_val = X["distance"].max(), X["distance"].min()
        X["distance"] = (X["distance"] - min_val) / (max_val - min_val)
        return X


# Model Training, validation and model save
class TrainValidateMod(BaseEstimator, TransformerMixin):

    """Splitting data, Model training and Hyperparameter tuning with CrossValidation
    and testing best Hyperparameter tuned model on validation data and save the model
    """

    def __init__(self, model_path=None):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def fit(self, X, Y):

        """Method to Split data, Model training and Hyperparameter tuning with CrossValidation"""
        monotone_feature_names = ["distance"]
        feature_monotones = [0 if f not in monotone_feature_names else 1 for f in X.columns]
        X_train, self.X_valid, y_train, self.y_valid = train_test_split(
            X, Y, test_size=0.2, random_state=101
        )

        # Hyperparameter tuning using GridSearch
        ind_params = {"n_estimators": 300, "seed": 100, "learning_rate": 0.05}
        ind_params["nthread"] = 4
        ind_params["eval_metric"] = "r2"
        ind_params["monotone_constraints"] = (
            "(" + ",".join([str(m) for m in feature_monotones]) + ")"
        )
        xgb_model = XGBRegressor(eval_metric="rmse", objective="reg:squarederror")
        GV = GridSearchCV(
            xgb_model,
            {
                "max_depth": [5, 7, 8],
                "subsample": [0.7, 0.8],
                "colsample_bytree": [0.4, 0.7, 0.8],
                "learning_rate": [0.03, 0.05],
                "gamma": [0.01],
            },
            cv=3,
            scoring="r2",
            refit=True,
            verbose=2,
        )
        GV.fit(X_train, y_train)
        print(f" Best Model Parameter {GV.best_params_}")
        print(f" Best Model CV score {GV.best_score_}")
        print(f"selecting the best Hyperparameter tuned estimator with {GV.best_params_}")
        self.model = GV.best_estimator_

        # best model refit with higher n_estimators and early_stopping_rounds to avoid overfit
        self.model.n_estimators = 1000
        print(self.model)
        self.model.fit(
            X_train,
            y_train,
            verbose=0,
            early_stopping_rounds=20,
            eval_set=[(X_train, y_train), (self.X_valid, self.y_valid)],
        )
        return self.model, self.X_valid, self.y_valid

    def validation(self):

        """Method to test best Hyperparameter tuned model on validation data and save model"""

        pred = self.model.predict(self.X_valid)
        print(f"R2 score: {r2_score(self.y_valid,pred)}")
        print(
            f"Mean absolute percentage error(MAPE): {mean_absolute_percentage_error(self.y_valid,pred)}"
        )
        print(f"Mean absolute error(MAE): {mean_absolute_error(self.y_valid,pred)}")

        with open(os.path.join(self.model_path, "model1.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
        return pred


if __name__ == "__main__":

    """This main function orchestrates the entire dataprep, training and validation process"""

    # Defining all Paths and variables

    target = "cost"
    path = "../artifacts/"
    data_path = "../data/train_data.csv"
    cat_cols = ["week_in_month", "year", "weekday", "is_weekend", "month"]

    # Implementation
    print(f'process started at {datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")}')
    df = pd.read_csv(data_path, sep=";", parse_dates=["shipping_date"])
    X = df.drop([target], axis=1)
    Y = df[target]
    fet = FeatenggTransformer()
    X = fet.transform(X)
    num_cols = [col for col in list(X.columns) if col not in cat_cols]
    ft = FeatureTransformer(cat_cols=cat_cols, num_cols=num_cols, path=path)
    X_transformed = ft.transform(X)
    TV = TrainValidateMod(path)
    model, X_valid, y_valid = TV.fit(X_transformed, Y=Y)
    pred = TV.validation()
    print(f"predicted mean:{pred.mean()}")
    print(f"actual mean:{y_valid.mean()}")
    r_squared = r2_score(TV.y_valid, pred)
    print(f"R2 score validation: {r_squared}")
    print(f'process ended at {datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")}')
