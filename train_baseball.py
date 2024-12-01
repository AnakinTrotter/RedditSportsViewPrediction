import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Set Random Seed for Reproducibility
def set_seed(seed=42):
    np.random.seed(seed)

set_seed(42)

# Constants
INPUT_FILE = "World_Series.csv"
TEST_SPLIT = 0.2

# Load Data
data = pd.read_csv(INPUT_FILE)
features = ["Total Posts", "Total Comments", "Total Scores", "Avg Sentiment (TextBlob)", "Avg Sentiment (Vader)"]
target = "Viewers (Millions)"

# Log-transform the target variable to handle skewness
data[target] = np.log1p(data[target])

# Remove Outliers in the Target Variable
Q1, Q3 = data[target].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data[target] >= lower_bound) & (data[target] <= upper_bound)]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=TEST_SPLIT, random_state=42
)

# Build Pipeline with Gradient Boosting Regressor
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('gbr', GradientBoostingRegressor(random_state=42))
])

# Hyperparameters to tune
param_grid = {
    'gbr__n_estimators': [50, 100, 200],
    'gbr__learning_rate': [0.01, 0.05, 0.1],
    'gbr__max_depth': [2, 3, 4],
    'gbr__subsample': [0.8, 1.0],
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# Save the model
joblib.dump(grid_search.best_estimator_, 'world_series_model.pkl')
print("Model saved to 'world_series_model.pkl'.")

# Evaluate on Test Set
y_pred = grid_search.predict(X_test)

# Inverse log-transform to get actual values
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

# Calculate Metrics
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"Mean Absolute Error (MAE): {mae:.2f} million viewers")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} million viewers")
print(f"R^2 (Coefficient of Determination): {r2:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.7, label="Predicted vs Actual")
plt.plot(
    [min(y_test_actual), max(y_test_actual)],
    [min(y_test_actual), max(y_test_actual)],
    color="red",
    linestyle="--",
    label="Ideal Fit"
)
plt.xlabel("Actual Viewership (Millions)")
plt.ylabel("Predicted Viewership (Millions)")
plt.title("Actual vs. Predicted Viewership")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance Visualization
gbr_model = grid_search.best_estimator_.named_steps['gbr']
feature_importance = gbr_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Gradient Boosting')
plt.grid(True)
plt.show()
