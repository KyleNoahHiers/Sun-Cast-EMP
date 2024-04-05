from torch import nn
import pandas as pd
import torch as torch
from torch.utils.data import DataLoader
from torch import optim


#basic regression model that predicts load
class LoadModel(nn.Module):
    def __init__(self):
        super(LoadModel, self).__init__()
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

class Dataset():

    def __init__(self, data_frame, label_present=True):
        self.data_frame = pd.read_csv(data_frame)
        self.label_present = label_present

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.label_present:
            #make data but exlude first column

            data = self.data_frame.iloc[idx, 1:].values.astype(float)
            #make label the first column
            label = self.data_frame.iloc[idx, 0]
            #print the columns in data
          




            return torch.tensor(data, dtype=torch.float), torch.tensor(label, dtype=torch.float)
        else:
            data = self.data_frame.iloc[idx].values.astype(float)
            return torch.tensor(data, dtype=torch.float)

# training function
# Initialize the loss function
criterion = nn.L1Loss

def train_regression(model, criterion, optimizer, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))  # Ensure labels are the correct shape
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def main():
    # Assuming your CSV file's path
    csv_file_path = 'load_file_encoded.csv'

    # Initialize Dataset and DataLoader
    dataset = Dataset(csv_file_path)  # No label_present flag needed for regression
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model and move it to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LoadModel().to(device)

    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_regression(model, criterion, optimizer, train_loader, epochs=5)

    # Save the trained model weights
    model_save_path = 'model_weights.pth'
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main()
