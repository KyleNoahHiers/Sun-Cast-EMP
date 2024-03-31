import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

import load_prediction.prep_load_file as plf
import theNetwork.network_input_creator as nic
import load_prediction.load_predictor as lp
import solar_classification.sun_giggler as sc
import load_prediction.prep_load_file as plf
from load_prediction.load_predictor import Dataset
from solar_classification.sun_giggler import WeatherDataset
import pandas as pd
import numpy as np

class SunCast():
    def __init__(self, egauge_file, weather_file, weather_predict, load_predict):
        self.egauge = nic.prepare_egauge_file(egauge_file)
        self.weather = nic.prepare_weather_file(weather_file)
        self.weather_predict = nic.prepare_weather_file(weather_predict)
        print(pd.read_csv(load_predict).head())
        self.load_predict = plf.prep_load_file(pd.read_csv(load_predict))
        self.solar_value = self.egauge['Production Difference'].mean()
        self.input = nic.combine_and_label(self.solar_value, self.weather,self.egauge)

        # Initialize models
        self.solarModel = sc.SolarNet()
        self.loadModel = lp.LoadModel()

        # Conditionally train or load models
        self.solar_model_path = '../solar_classification/model_weights.pth'
        if not os.path.exists(self.solar_model_path):
            self.train_solar()
        else:
            self.solarModel.load_state_dict(torch.load(self.solar_model_path))

        self.load_model_path = 'load_prediction/model_weights.pth'
        if not os.path.exists(self.load_model_path):
            self.train_load()
        else:
            self.loadModel.load_state_dict(torch.load(self.load_model_path))

    def train_load(self, epochs=10):
        # Prepare dataset and dataloader for load prediction
        load_file = plf.prep_load_file(self.egauge)
        loadDataset = lp.Dataset(load_file)  # Ensure lp.Dataset is defined correctly
        load_loader = torch.utils.data.DataLoader(loadDataset, batch_size=32, shuffle=True)

        # Define optimizer and criterion
        optimizer = optim.Adam(self.loadModel.parameters(), lr=0.001)
        criterion = nn.L1Loss()

        # Train the model
        lp.train_regression(self.loadModel, criterion, optimizer, load_loader, epochs)

        # Save trained model weights
        torch.save(self.loadModel.state_dict(), self.load_model_path)

    def train_solar(self, epochs=10):
        # Prepare dataset and dataloader for solar prediction
        weatherDataset = sc.WeatherDataset(self.input, label_present=True)
        weather_loader = torch.utils.data.DataLoader(weatherDataset, batch_size=32, shuffle=True)

        # Define optimizer and criterion
        optimizer = optim.Adam(self.solarModel.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # Train the model
        sc.train(self.solarModel, criterion, optimizer, weather_loader, epochs)  # Ensure sc.train is defined correctly

        # Save trained model weights
        torch.save(self.solarModel.state_dict(), self.solar_model_path)

    def predict_solar(self):
        # Preprocess the DataFrame
        input_df = self.weather_predict  # Assuming this is already a DataFrame
        # Perform any necessary preprocessing on input_df as per your requirements

        # Wrap preprocessed DataFrame in WeatherDataset
        dataset = WeatherDataset(input_df, label_present=False)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predictions_list = []  # To collect batch predictions

        with torch.no_grad():
            self.solarModel.eval()  # Ensure model is in evaluation mode
            for inputs, _ in data_loader:
                inputs = inputs.float()
                outputs = self.solarModel(inputs).squeeze()
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).long().cpu().numpy()
                predictions_list.append(predictions)

        # Concatenate batch predictions
        all_predictions = np.concatenate(predictions_list, axis=0)

        # Append predictions to the original input DataFrame (ensure it matches the original row count)
        self.weather_predict['Solar Prediction'] = all_predictions[:len(self.weather_predict)]
        return self.weather_predict


    def predict_load(self, label_present=False):
        # Preprocess the DataFrame and ensure it's ready for the model
        preprocessed_df = self.load_predict  # Example preprocessing placeholder

        # Create dataset from preprocessed DataFrame
        dataset = Dataset(preprocessed_df, label_present=label_present)  # Adjust as needed
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loadModel.to(device)
        self.loadModel.eval()  # Ensure model is in evaluation mode

        all_predictions = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = self.loadModel(inputs.float()).squeeze()  # Ensure inputs are float
                all_predictions.extend(outputs.cpu().numpy())

        # Convert list of predictions to a DataFrame and append as a new column
        self.load_predict['Load Prediction'] = np.array(all_predictions)[:len(self.load_predict)]

        # Optionally, you could return the DataFrame
        return self.load_predict



def test_api():
    # Define paths to input files
    egauge_file = 'theNetwork/data/data.csv'
    weather_file = 'src/theNetwork/data/washburn_wi_weather25 - washburn_wi_weather25.csv'
    weather_predict = 'theNetwork/data/br_test/br_egauge_feb.csv'
    load_predict = 'load_prediction/week_of_17.csv'

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

    # test prediction
    print(sun_cast.predict_solar())