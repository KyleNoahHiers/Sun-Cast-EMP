from src.the_sun_giggler.sun_giggler import SolarNet, WeatherDataset
import torch

# Assuming the first data point for demonstration; replace with actual data as needed
csv_file_path = 'output.csv'  # Update this path to your dataset
dataset = WeatherDataset(csv_file_path, label_present=True)  # Assuming labels are present
test_data_point, _ = dataset[0]  # Getting the first data point (and ignoring its label)

# Ensure the data point is in the correct shape (adding batch dimension) and type
test_data_point = test_data_point.unsqueeze(0)  # Add a batch dimension

# Load the model
model = SolarNet()

# Load the weights
model_load_path = 'model_weights.pth'
model.load_state_dict(torch.load(model_load_path))

# Switch to evaluation mode
model.eval()

# Move the model and data point to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
test_data_point = test_data_point.to(device)

# Make a prediction
with torch.no_grad():  # Ensure gradients are not computed for inference
    prediction = model(test_data_point)

# Assuming a binary classification problem and you're interested in the probability
probability = torch.sigmoid(prediction).item()  # Applying sigmoid to get the probability

print(f"Predicted probability: {probability}")

# Continuation from the previous example where the model makes a prediction

# Threshold to determine class
threshold = 0.5

# Apply threshold to probability to determine class
predicted_class = "good day" if probability > threshold else "bad day"

print(f"Predicted class: {predicted_class}")

