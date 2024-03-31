from datetime import datetime
import torch
from torch import nn
import pandas as pd
import numpy as np
import os

class LoadModel(nn.Module):
    def __init__(self):
        super(LoadModel, self).__init__()
        # Define your model architecture here
        self.layer1 = nn.Linear(8, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)
        return x

def load_model(model_path):
    model = LoadModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def prepare_input_for_prediction(month, day):
    year = datetime.now().year
    input_list = []
    for hour in range(6, 19):  # For hours 6 AM to 6 PM
        # Generate features for each hour
        day_of_week = pd.to_datetime(f"{year}-{month}-{day}").dayofweek
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        max_day = pd.Timestamp(year, month, day).days_in_month
        day_sin = np.sin(2 * np.pi * day / max_day)
        day_cos = np.cos(2 * np.pi * day / max_day)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)

        input_list.append([month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, day_of_week_sin, day_of_week_cos])

    return np.array(input_list)

def predict_load(model, month, day):
    inputs = prepare_input_for_prediction(month, day)
    predictions = []

    # Convert inputs to a tensor and make predictions
    inputs_tensor = torch.tensor(inputs, dtype=torch.float)
    with torch.no_grad():
        for single_input in inputs_tensor:
            prediction = model(single_input.unsqueeze(0))  # Model expects a batch dimension
            predictions.append(prediction.item())

    return predictions

# Example usage
if __name__ == "__main__":
    # Path to your local model weights file
    model_path = 'model_weights.pth'
    model = load_model(model_path)
    #initialize an empty dataframe with an hour column and a prediction column
    predictions_df = pd.DataFrame(columns=['hour', 'prediction'])
    for i in range(7):
        month = 3  # March
        day = 17 + i
        predictions = predict_load(model, month, day)

        # Create a temporary DataFrame for the current day's predictions
        day_df = pd.DataFrame({
            'day': [day] * len(range(6, 19)),  # Repeat 'day' for each hour
            'hour': range(6, 19),
            'prediction': predictions
        })

        # Concatenate the current day's predictions with the main DataFrame
        predictions_df = pd.concat([predictions_df, day_df], ignore_index=True)

    # Export the DataFrame to a CSV file
    predictions_df.to_csv('predictions.csv', index=False)



