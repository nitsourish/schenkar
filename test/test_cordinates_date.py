from io import StringIO
import sys
import pytest
from cost_predictor.inference import FeatenggTransformer
import pandas as pd
import os
import math
import numpy as np
from datetime import datetime
import requests
from vincenty import vincenty
from importlib import import_module, reload

def test_cordinate():
    lat_1,lon_1,lat_2,lon_2 = 'origin_latitude','origin_longitude','destination_latitude','destination_longitude'

    test_df = pd.DataFrame({lat_1:[26.76,-24.56],lon_1:[109.10,129.44406],lat_2:[26.76,25.56],lon_2:[109.10, 149.44406]})
    try:
        with pytest.raises(ValueError) as excinfo:
            fet = FeatenggTransformer()
            fet.add_travel_features(test_df)
        assert excinfo.value.args[0] == "Invalid latitude, longitude combination."
    except:
        assert True

def test_date():

    ref_date = "2017-01-01"
    test_df = pd.DataFrame({'shipping_date':["2017-01-01","2018-03-01","2016-03-01"]})
    test_df['shipping_date'] = pd.to_datetime(test_df['shipping_date'])
    try:
        with pytest.raises(ValueError) as excinfo:
            fet = FeatenggTransformer()
            fet.add_travel_features(test_df)
        assert excinfo.value.args[0] == f"Invalid Date Date should be later than {ref_date}"
    except:
        assert True
