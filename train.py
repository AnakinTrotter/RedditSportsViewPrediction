import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import shap

# Set Random Seed
def set_seed(seed=42):
    np.random.seed(seed)

set_seed(42)

# Load Datasets
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

# Log-transform Target to Handle Skewness
data[target] = np.log1p(data[target])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)

# Preprocessing
categorical_features = ["Sport"]
numerical_features = ["Total Posts", "Total Comments", "Total Scores", "Avg Sentiment (TextBlob)", "Avg Sentiment (Vader)"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('gbr', GradientBoostingRegressor(random_state=42))
])

# Hyperparameter Tuning
param_grid = {
    'gbr__n_estimators': [100, 200, 300],
    'gbr__learning_rate': [0.05, 0.1],
    'gbr__max_depth': [3, 5],
    'gbr__min_samples_split': [2, 5],
    'gbr__subsample': [0.8, 1.0],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# Save the Model
joblib.dump(grid_search.best_estimator_, 'cross_sport_model.pkl')
print("Model saved to 'cross_sport_model.pkl'.")

# Evaluate on Test Set
y_pred = grid_search.predict(X_test)

# Inverse log-transform
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

# Metrics
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
preprocessor_pipeline = grid_search.best_estimator_.named_steps['preprocessor']

numerical_transformed_features = numerical_features
categorical_transformed_features = preprocessor_pipeline.transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = list(numerical_transformed_features) + list(categorical_transformed_features)

feature_importance = gbr_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance Across Sports')
plt.grid(True)
plt.show()

# SHAP Analysis
# Transform Test Data
X_test_transformed = preprocessor_pipeline.transform(X_test)

# Initialize SHAP Explainer
explainer = shap.Explainer(gbr_model, X_test_transformed)

# Generate SHAP Values
shap_values = explainer(X_test_transformed)

# SHAP Summary Plot
shap.summary_plot(shap_values, feature_names=feature_names)
