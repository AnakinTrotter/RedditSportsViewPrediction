import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Dataset Class
class RedditViewershipDataset(Dataset):
    def __init__(self, reddit_metrics_file, tv_data_file):
        # Load data
        reddit_df = pd.read_csv(reddit_metrics_file)
        tv_data_df = pd.read_csv(tv_data_file)
        
        # Merge datasets on 'Name'
        self.data = pd.merge(reddit_df, tv_data_df, on="Name")
        
        # Feature Engineering: Add derived features
        self.data["Avg Comments Per Post"] = self.data["Total Comments"] / self.data["Total Posts"]
        self.data["Avg Score Per Post"] = self.data["Total Scores"] / self.data["Total Posts"]
        
        # Select features and target
        features = self.data[[
            "Year", "Total Posts", "Total Comments", "Total Scores",
            "Avg Comments Per Post", "Avg Score Per Post"
        ]]
        target = self.data["Average Viewers (Millions)"]

        # Normalize features using Min-Max Scaling
        scaler = MinMaxScaler()
        self.features = torch.tensor(scaler.fit_transform(features), dtype=torch.float32)
        self.target = torch.tensor(target.values.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# Model Definition
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output is a single regression value
        )
    
    def forward(self, x):
        return self.model(x)

# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, target in dataloader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Evaluation Function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for features, target in dataloader:
            preds = model(features)
            loss = criterion(preds, target)
            total_loss += loss.item()
            predictions.append(preds.numpy())
            targets.append(target.numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    # Calculate metrics
    rmse = np.sqrt(total_loss / len(dataloader.dataset))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    print(f"Validation Loss (MSE): {total_loss:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R^2 Score: {r2:.4f}")

# Main Function
def main():
    # Files
    reddit_metrics_file = "reddit_metrics.csv"
    tv_data_file = "tv_data.csv"

    # Hyperparameters
    input_dim = 6  # Number of features
    batch_size = 4
    learning_rate = 0.0005  # Reduced learning rate for better convergence
    num_epochs = 100  # Increased epochs for better learning
    train_split_ratio = 0.8

    # Dataset and DataLoader
    dataset = RedditViewershipDataset(reddit_metrics_file, tv_data_file)
    train_size = int(train_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, Loss Function, Optimizer
    model = RegressionModel(input_dim)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    print("Starting Training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluation
    print("Evaluating Model...")
    evaluate_model(model, val_loader, criterion)

    # Save the model
    torch.save(model.state_dict(), "regression_model.pth")
    print("Model saved to regression_model.pth")

if __name__ == "__main__":
    main()
