import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Trained Model
model_path = "optimized_gbr_model.pkl"
model = joblib.load(model_path)

# Load the Dataset
files = ["World_Series.csv", "Super_Bowl.csv", "NBA_Finals.csv", "Stanley_Cup.csv", "MLS_Cup.csv"]
dfs = []
for file in files:
    df = pd.read_csv(file)
    df['Sport'] = file.split('.')[0]  # Add Sport Column
    dfs.append(df)

# Combine Datasets
data = pd.concat(dfs, ignore_index=True)

# Features and Target
features = ["Total Posts", "Total Comments", "Total Scores", "Avg Sentiment (TextBlob)", "Avg Sentiment (Vader)", "Sport"]
target = "Viewers (Millions)"
data[target] = np.log1p(data[target])

# Extract Features and Target
X = data[features]
y_actual = np.expm1(data[target])  # Reverse log-transform for true values

# Predict Using the Full Pipeline
y_pred = np.expm1(model.predict(X))  # Use the full pipeline for prediction and reverse log-transform predictions

# Evaluate Performance
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} million viewers")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} million viewers")
print(f"R^2 (Coefficient of Determination): {r2:.2f}")

# Save Evaluation Results
results = {
    "Mean Absolute Error (MAE)": mae,
    "Root Mean Squared Error (RMSE)": rmse,
    "R^2 (Coefficient of Determination)": r2
}

results_df = pd.DataFrame([results])
results_df.to_csv("evaluation_results.csv", index=False)
print("Evaluation results saved to 'evaluation_results.csv'.")

# Combine Predictions with Actual Values for Display
comparison_df = pd.DataFrame({
    "Name": data["Name"],  # Use Sport as the game name
    "Predicted Viewership (Millions)": y_pred,
    "Actual Viewership (Millions)": y_actual,
    "Error (Millions)": np.abs(y_actual - y_pred)  # Absolute error
})

# Save the comparison table to CSV
comparison_df.to_csv("predicted_vs_actual_with_error.csv", index=False)
print("Predicted vs Actual values with errors saved to 'predicted_vs_actual_with_error.csv'.")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(y_actual, y_pred, alpha=0.7, label="Predicted vs Actual", color="blue")
plt.plot(
    [min(y_actual), max(y_actual)],
    [min(y_actual), max(y_actual)],
    color="red",
    linestyle="--",
    label="Ideal Fit"
)
plt.xlabel("Actual Viewership (Millions)")
plt.ylabel("Predicted Viewership (Millions)")
plt.title("Actual vs. Predicted Viewership (Full Dataset)")
plt.legend()
plt.savefig("eval_actual_vs_predicted.png", bbox_inches="tight")
plt.show()
