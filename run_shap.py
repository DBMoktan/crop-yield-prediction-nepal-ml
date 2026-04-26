import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

print("Loading model and preprocessors...")
model = joblib.load("best_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
high_vif_cols = joblib.load("high_vif_cols.pkl")
min_year = joblib.load("min_year.pkl")
numerical_cols = joblib.load("numerical_cols.pkl")
selected_features = joblib.load("selected_features.pkl")

print("Loading data...")
df = pd.read_csv("cleaned_dataset_for_ml.csv")
X = df.drop("yield_kg/ha", axis=1)
y = df["yield_kg/ha"]

# Preprocessing
epsilon = 1e-9
X['temp_range'] = X['max_temp_C'] - X['min_temp_C']
X['rainfall_per_area'] = X['avg_rainfall_mm_per_year'] / (X['Area'] + epsilon)
X['fertilizer_per_area'] = X['fertilizer_in_MT'] / (X['Area'] + epsilon)
X['solar_radiation_per_area'] = X['total_solar_radiation_kWh/m2'] / (X['Area'] + epsilon)
X['par_per_area'] = X['total_PAR_MJ/m2'] / (X['Area'] + epsilon)
X['temp_rainfall_interaction'] = X['avg_temp_C'] * X['avg_rainfall_mm_per_year']
X['log_Area'] = np.log1p(X['Area'])
X['log_fertilizer_in_MT'] = np.log1p(X['fertilizer_in_MT'])
X['years_since_start'] = X['Year'] - min_year

categorical_cols = ['Districts', 'crop_type']
encoded_cats = encoder.transform(X[categorical_cols])
encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_cols, index=X.index)

input_num = X.drop(columns=categorical_cols)
final_df = pd.concat([input_num, encoded_df], axis=1)

final_df.drop(columns=high_vif_cols, inplace=True, errors='ignore')
final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])

# Extract final features as a dataframe to keep column names
final_X_df = final_df[selected_features]

print("Computing SHAP values...")
# Use TreeExplainer on the XGBoost base estimator to avoid Numba errors and speed up analysis
xgb_model = model.named_estimators_['xgb']
explainer = shap.TreeExplainer(xgb_model)
X_explain = final_X_df
shap_values = explainer.shap_values(X_explain)

print("Generating SHAP summary plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_explain, feature_names=selected_features, show=False)
plt.tight_layout()
plt.savefig("plots/shap_summary.png", bbox_inches='tight', dpi=300)
plt.close()

# Calculate mean absolute SHAP values for feature importance
mean_shap_values = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': mean_shap_values
}).sort_values(by='importance', ascending=False)

feature_importance.to_csv("plots/shap_importance.csv", index=False)

print("SHAP analysis complete.")
