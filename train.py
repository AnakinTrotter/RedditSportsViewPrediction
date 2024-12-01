import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
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
        self.data["Total Engagement"] = self.data["Total Comments"] + self.data["Total Scores"]
        self.data["Engagement Per Post"] = self.data["Total Engagement"] / self.data["Total Posts"]

        # Add Sport Type feature
        self.data["Sport Type"] = self.data["Name"].apply(self._extract_sport_type)

        # One-Hot Encode Sport Type
        sport_encoder = OneHotEncoder(sparse_output=False)
        sport_encoded = sport_encoder.fit_transform(self.data[["Sport Type"]])
        sport_encoded_df = pd.DataFrame(sport_encoded, columns=[f"Sport_{s}" for s in sport_encoder.categories_[0]])
        self.data = pd.concat([self.data, sport_encoded_df], axis=1)

        # One-Hot Encode Year
        year_encoder = OneHotEncoder(sparse_output=False)
        year_encoded = year_encoder.fit_transform(self.data[["Year"]])
        year_encoded_df = pd.DataFrame(year_encoded, columns=[f"Year_{int(y)}" for y in year_encoder.categories_[0]])
        self.data = pd.concat([self.data, year_encoded_df], axis=1)

        # Select features and target
        feature_columns = [
            "Total Posts", "Total Comments", "Total Scores",
            "Total Engagement", "Engagement Per Post"
        ] + list(sport_encoded_df.columns) + list(year_encoded_df.columns)

        self.features = self.data[feature_columns].values
        self.target = self.data["Average Viewers (Millions)"].values.reshape(-1, 1)

        # Normalize features using Min-Max Scaling
        scaler = MinMaxScaler()
        self.features = torch.tensor(scaler.fit_transform(self.features), dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

    def _extract_sport_type(self, name):
        """Extract sport type based on keywords in the Name column."""
        if "World Series" in name:
            return "MLB"
        elif "Super Bowl" in name:
            return "NFL"
        elif "NBA" in name:
            return "NBA"
        elif "MLS" in name:
            return "MLS"
        else:
            return "Other"

# Model Definition
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout for regularization
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output is a single regression value
        )
    
    def forward(self, x):
        return self.model(x)

# Training Function
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for features, target in dataloader:
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

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

    return total_loss / len(dataloader), rmse, mae

# Main Function with Cross-Validation
def main():
    # Files
    reddit_metrics_file = "reddit_metrics.csv"
    tv_data_file = "tv_data.csv"

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.001
    weight_decay = 1e-4  # L2 regularization
    num_epochs = 50
    num_folds = 5

    # Dataset
    dataset = RedditViewershipDataset(reddit_metrics_file, tv_data_file)
    input_dim = dataset.features.shape[1]  # Dynamically calculate input dimensions

    # Cross-Validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold + 1}/{num_folds} ---")

        # Create DataLoaders for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Initialize model, loss, and optimizer
        model = RegressionModel(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train model
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        # Evaluate model
        val_loss, rmse, mae = evaluate_model(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        fold_results.append((val_loss, rmse, mae))

    # Aggregate fold results
    avg_loss = np.mean([r[0] for r in fold_results])
    avg_rmse = np.mean([r[1] for r in fold_results])
    avg_mae = np.mean([r[2] for r in fold_results])

    print("\n--- Cross-Validation Results ---")
    print(f"Average Validation Loss (MSE): {avg_loss:.4f}")
    print(f"Average Validation RMSE: {avg_rmse:.4f}")
    print(f"Average Validation MAE: {avg_mae:.4f}")

if __name__ == "__main__":
    main()
