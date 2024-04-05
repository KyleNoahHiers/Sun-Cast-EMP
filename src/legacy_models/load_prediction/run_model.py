from load_predictor import LoadModel  # Assuming LoadModel is your regression model class
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from load_predictor import Dataset  # Ensure Dataset is the class handling your data loading

# Load the model
model = LoadModel()

# Load the weights and switch to evaluation mode
model_load_path = 'model_weights.pth'
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Assuming 'load_file_encoded.csv' is your test dataset
csv_file_path = '../../load_prediction/load_file_encoded.csv'
test_dataset = Dataset(csv_file_path, label_present=True)  # Assuming last column is the continuous label
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize lists to track predictions and actual labels
all_predictions = []
all_labels = []

# Make predictions
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs.float()).squeeze()  # Ensure inputs are float and get model outputs
        all_predictions.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Evaluate predictions using regression metrics
mae = mean_absolute_error(all_labels, all_predictions)
mse = mean_squared_error(all_labels, all_predictions)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE) on the test dataset: {mae}")
print(f"Mean Squared Error (MSE) on the test dataset: {mse}")
print(f"Root Mean Squared Error (RMSE) on the test dataset: {rmse}")
