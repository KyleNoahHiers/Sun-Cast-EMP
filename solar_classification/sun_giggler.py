import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class WeatherDataset(Dataset):
    """Weather dataset."""
    def __init__(self, data_frame, label_present=True):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            label_present (bool): Whether the last column contains labels.
        """
        # Load the dataset
        self.data_frame = data_frame
        self.label_present = label_present

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        print(self.data_frame.head())
        if self.label_present:
            # Assuming the last column is the label
            data = self.data_frame.iloc[idx, :-1].values.astype(float)
            label = self.data_frame.iloc[idx, -1]
            return torch.tensor(data, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        else:
            data = self.data_frame.iloc[idx].values.astype(float)
            return torch.tensor(data, dtype=torch.float)

# Usage


# Define the neural network with an input size of 53
class SolarNet(nn.Module):
    def __init__(self):
        super(SolarNet, self).__init__()
        self.layer1 = nn.Linear(52, 128)  # Adjusted input layer to size 53
        self.layer2 = nn.Linear(128, 64)  # Hidden layer
        self.layer3 = nn.Linear(64, 32)   # Hidden layer
        self.output_layer = nn.Linear(32, 1)  # Output layer for binary classification

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)  # Remove sigmoid here for BCEWithLogitsLoss
        return x



def train(model, criterion, optimizer, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def main():
    # Assuming your CSV file's path
    csv_file_path = 'output.csv'

    # Initialize Dataset and DataLoader
    dataset = WeatherDataset(csv_file_path, label_present=True)  # Adjust `label_present` based on your CSV
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = SolarNet()

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, criterion, optimizer, train_loader, epochs=5)

    model_save_path = 'model_weights.pth'
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main();