import pandas as pd
from api import SunCast


def test_api():
    # Define paths to input files
    egauge_file = 'theNetwork/data/data.csv'
    weather_file = 'theNetwork/data/washburn_wi_weather25 - washburn_wi_weather25.csv'
    weather_predict = 'theNetwork/data/br_test/bad_river_weather.csv'
    load_predict = 'load_prediction/load_file.csv'

    # Initialize SunCast object
    sun_cast = SunCast(egauge_file, weather_file, weather_predict, load_predict)

    # Ensure the input DataFrame is correctly generated
    assert isinstance(sun_cast.input, pd.DataFrame)

    # Ensure the solar value is correctly calculated
    assert isinstance(sun_cast.solar_value, float)

    # Ensure the solarModel and loadModel are correctly initialized
    assert hasattr(sun_cast, 'solarModel')
    assert hasattr(sun_cast, 'loadModel')

    # Ensure the solarModel and loadModel are correctly trained
    assert hasattr(sun_cast.solarModel, 'load_state_dict')
    assert hasattr(sun_cast.loadModel, 'load_state_dict')

    print(sun_cast.predict_load(True).head())

    # test prediction
    print(sun_cast.predict_solar().head())

test_api()