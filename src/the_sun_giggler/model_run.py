import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sun_giggler import SolarNet, WeatherDataset  # Assuming these are suited for regression
import pandas as pd
# Load the model
model = SolarNet()

# Load the weights and switch to evaluation mode
model_load_path = 'model_weights.pth'
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Assuming 'output.csv' is your test dataset
csv_file_path = '../theNetwork/data/labelled_weather.csv'
df1 = pd.read_csv(csv_file_path)
#exclude all rows with production difference less than 5
df1 = df1[df1['Production Difference'] >20]
test_dataset = WeatherDataset(df1, label_present=True)  # Ensure this is appropriate for regression
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize variables to track predictions and actual values
all_predictions = []
all_actuals = []

# Make predictions
with torch.no_grad():
    for inputs, actuals in test_loader:
        inputs = inputs.to(device)
        actuals = actuals.to(device).float()  # Ensure labels are floats for regression
        outputs = model(inputs).squeeze()  # Outputs are continuous values

        all_predictions.extend(outputs.cpu().numpy())
        all_actuals.extend(actuals.cpu().numpy())

# Evaluate predictions using Mean Absolute Error
mae = mean_absolute_error(all_actuals, all_predictions)
print(f"Mean Absolute Error on the test dataset: {mae}")
import pandas as pd
df = pd = pd.read_csv(csv_file_path)
print(f"The mean of the actuals is {df1['Production Difference'].mean()}")
