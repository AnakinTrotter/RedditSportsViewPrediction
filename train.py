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
import matplotlib as mpl

# Set Random Seed
def set_seed(seed=42):
    np.random.seed(seed)

set_seed(42)

# Publication Quality Settings
mpl.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "legend.fontsize": 9,
    "axes.grid": False
})

# Load Datasets
files = ["World_Series.csv", "Super_Bowl.csv", "NBA_Finals.csv", "Stanley_Cup.csv", "MLS_Cup.csv"]
dfs = []
for file in files:
    df = pd.read_csv(file)
    df['Sport'] = file.split('.')[0]  # Add Sport Column
    dfs.append(df)

# Combine Datasets
data = pd.concat(dfs, ignore_index=True)

# Outlier Removal by Sport
def remove_outliers_by_group(df, group_col, columns, factor=3.0):
    for col in columns:
        grouped = df.groupby(group_col)
        Q1 = grouped[col].transform(lambda x: x.quantile(0.25))
        Q3 = grouped[col].transform(lambda x: x.quantile(0.75))
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

numerical_columns = ["Total Posts", "Total Comments", "Total Scores", "Avg Sentiment (TextBlob)", "Avg Sentiment (Vader)"]
data = remove_outliers_by_group(data, 'Sport', numerical_columns)

# Verify all Sports are Present
print("Counts by Sport after outlier removal:")
print(data['Sport'].value_counts())

# Features and Target
features = ["Total Posts", "Total Comments", "Total Scores", "Avg Sentiment (TextBlob)", "Avg Sentiment (Vader)", "Sport"]
target = "Viewers (Millions)"
data[target] = np.log1p(data[target])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

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
    ('gbr', GradientBoostingRegressor(random_state=42, validation_fraction=0.1, n_iter_no_change=10))
])

# Simplified Hyperparameter Tuning
param_grid = {
    'gbr__n_estimators': [100, 200],
    'gbr__learning_rate': [0.05],
    'gbr__max_depth': [3, 5],
    'gbr__min_samples_split': [2, 5],
    'gbr__subsample': [0.8, 1.0],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
joblib.dump(grid_search.best_estimator_, 'optimized_gbr_model.pkl')

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
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(y_test_actual, y_pred_actual, alpha=0.7, label="Predicted vs Actual", color='blue')
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
plt.savefig("actual_vs_predicted.png", bbox_inches="tight")
plt.show()

# SHAP Analysis
gbr_model = grid_search.best_estimator_.named_steps['gbr']

# Properly Transform the Test Data
preprocessor_pipeline = grid_search.best_estimator_.named_steps['preprocessor']
X_test_transformed = preprocessor_pipeline.transform(X_test)

# Feature Names for Transformed Data
numerical_transformed_features = numerical_features
categorical_transformed_features = preprocessor_pipeline.transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = list(numerical_transformed_features) + list(categorical_transformed_features)

# Initialize SHAP Explainer
explainer = shap.Explainer(gbr_model, X_test_transformed)
shap_values = explainer(X_test_transformed, check_additivity=False)

# SHAP Bar Plot of Feature Importance
plt.figure(figsize=(30, 12), dpi=300)  # Increased height for more vertical space
shap.summary_plot(
    shap_values,
    feature_names=feature_names,  # Use full names
    plot_type="bar",
    show=False
)

# Adjust Font Sizes
plt.title("SHAP Feature Importance", fontsize=8)
plt.gca().tick_params(axis='both', which='major', labelsize=8)  # Adjust tick size

# Use Only Tight Layout with Padding
plt.tight_layout(pad=2.0)  # Add padding around the plot to prevent cutoff
plt.savefig("shap_feature_importance_full_titles.png", bbox_inches="tight")
plt.show()

