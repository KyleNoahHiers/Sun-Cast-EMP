#  SunCast Module Documentation

The `SunCast` Module is designed to facilitate solar production and load demand predictions based on weather and energy usage data. It utilizes trained machine learning models to make predictions and can automatically train these models if they have not been trained already.

## Initialization

To start using the `SunCast` Module, initialize the `SunCast` class with paths to your input data files:

```python 
sun_cast = SunCast(egauge_file, weather_file, weather_predict, load_predict)
```

Parameters:

`egauge_file`: Path to the CSV file containing egauge (energy consumption) data.

`weather_file`: Path to the CSV file containing historical weather data.

`weather_predict`: Path to the CSV file containing weather data for prediction.

`load_predict`: Path to the CSV file containing load data for prediction.

 ## Training Models

The API checks if models are already trained and have saved weights upon initialization. If not, it will train the models using the provided data files.

### `train_load(self, epochs=10)`
Trains the load prediction model.

Parameters:

epochs: The number of training epochs. Default is 10.
### `train_solar(self, epochs=10)`
Trains the solar production prediction model.

#### Parameters:

epochs: The number of training epochs. Default is 10.
Making Predictions

Once the models are trained, you can use them to make predictions.

### `predict_solar(self)`
Predicts solar production based on the weather_predict data provided during initialization.

#### Returns: 
A pandas DataFrame containing the original weather prediction data along with a new column, Solar Prediction, which contains the predicted solar production.

### `predict_load(self, label_present=False)`
Predicts load demand based on the load_predict data provided during initialization.

#### Parameters:

label_present: Indicates whether the label (actual load) is present in the load_predict DataFrame. Default is False.
#### Returns: 
A pandas DataFrame containing the original load prediction data along with a new column, Load Prediction, which contains the predicted load demand.

## Testing Module

You can test the functionality of the Module using the test_api function.

```python
def test_api():
    # Define paths to your input files
    egauge_file = 'path/to/egauge_data.csv'
    weather_file = 'path/to/weather_data.csv'
    weather_predict = 'path/to/weather_predict.csv'
    load_predict = 'path/to/load_predict.csv'

    # Initialize and use SunCast
    sun_cast = SunCast(egauge_file, weather_file, weather_predict, load_predict)

    # Run predictions
    print(sun_cast.predict_solar())
    print(sun_cast.predict_load())
```
