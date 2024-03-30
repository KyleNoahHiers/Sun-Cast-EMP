from sun_giggler import WeatherDataset, SolarNet
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Load the model
model = SolarNet()

# Load the weights and switch to evaluation mode
model_load_path = 'model_weights.pth'
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Assuming 'output.csv' is your test dataset
csv_file_path = 'output.csv'
test_dataset = WeatherDataset(csv_file_path, label_present=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize variables to track predictions and actual labels
all_predictions = []
all_labels = []

# Make predictions
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs).squeeze()  # Assuming binary classification
        predicted_classes = (probabilities > 0.5).long()  # Classify as 0 or 1 based on threshold
        all_predictions.extend(predicted_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Evaluate predictions

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Accuracy on the test dataset: {accuracy}")
