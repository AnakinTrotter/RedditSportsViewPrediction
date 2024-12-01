import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import matplotlib.pyplot as plt

# Constants
INPUT_FILE = "World_Series.csv"
EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
PATIENCE = 10  # Early stopping patience

# Dataset Class for Polynomial Features
class PolyWorldSeriesDataset(Dataset):
    def __init__(self, data, poly_data, target, feature_scaler=None, target_scaler=None):
        self.features = poly_data
        self.target = data[target].values.reshape(-1, 1)

        if feature_scaler:
            self.features = feature_scaler.transform(self.features)
        if target_scaler:
            self.target = target_scaler.transform(self.target)

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# Enhanced Neural Network Model
class EnhancedRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedRegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

# Load and Preprocess Data
data = pd.read_csv(INPUT_FILE)
features = ["Total Posts", "Total Comments", "Total Scores", "Avg Sentiment (TextBlob)", "Avg Sentiment (Vader)"]
target = "Viewers (Millions)"

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[features])
poly_feature_names = poly.get_feature_names_out(features)

# Normalize data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(poly_features)
target_scaler.fit(data[[target]])

# Train/Test Split
train_data, test_data = train_test_split(data, test_size=TEST_SPLIT, random_state=42)
train_poly = poly.transform(train_data[features])
test_poly = poly.transform(test_data[features])

# Create Datasets and DataLoaders
train_dataset = PolyWorldSeriesDataset(train_data, train_poly, target, feature_scaler=feature_scaler, target_scaler=target_scaler)
test_dataset = PolyWorldSeriesDataset(test_data, test_poly, target, feature_scaler=feature_scaler, target_scaler=target_scaler)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model, Loss, Optimizer
model = EnhancedRegressionModel(input_dim=train_poly.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)  # L2 Regularization

# Training Loop with Early Stopping
train_losses = []
test_losses = []
best_loss = float("inf")
patience_counter = 0

print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # Evaluate on Test Data
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            predictions = model(X)
            loss = criterion(predictions, y)
            test_loss += loss.item()
    test_losses.append(test_loss / len(test_loader))

    # Early Stopping
    if test_losses[-1] < best_loss:
        best_loss = test_losses[-1]
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

# Plot Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curves")
plt.grid(True)
plt.show()

# Evaluate Model on Test Set
model.eval()
actuals = []
predictions = []

with torch.no_grad():
    for X, y in test_loader:
        preds = model(X).detach().numpy()
        actuals.extend(target_scaler.inverse_transform(y.numpy()))
        predictions.extend(target_scaler.inverse_transform(preds))

# Calculate Mean Absolute Error
actuals = np.array(actuals).flatten()
predictions = np.array(predictions).flatten()
mae = np.mean(np.abs(actuals - predictions))
print(f"Mean Absolute Error (MAE): {mae:.2f} million viewers")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.7)
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color="red", linestyle="--")
plt.xlabel("Actual Viewership (Millions)")
plt.ylabel("Predicted Viewership (Millions)")
plt.title("Actual vs. Predicted Viewership")
plt.grid(True)
plt.show()

# Residual Analysis
residuals = actuals - predictions
plt.figure(figsize=(10, 6))
plt.scatter(actuals, residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Actual Viewership (Millions)")
plt.ylabel("Residuals")
plt.title("Residuals vs. Actual Viewership")
plt.grid(True)
plt.show()
