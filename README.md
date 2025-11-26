🌾 Cereal Crop Yield Prediction in Nepal using Machine Learning

This repository contains a complete machine learning workflow for predicting cereal crop yields in Nepal using climate, soil, and agricultural production data. The project evaluates multiple ML algorithms, performs advanced feature engineering, and identifies the most influential factors affecting yield across three agro-ecological regions (Terai, Hill, Himalaya) and six major crops (Maize, Wheat, Rice, Barley, Millet, Buckwheat).

📌 Project Overview

Agricultural productivity in Nepal is highly vulnerable to climate variability, limited resources, and geographical diversity. This project builds a data-driven framework to support yield forecasting and decision-making. By integrating multi-year datasets with environmental features, the system predicts yield using several machine learning models and compares their performance to identify the best approach.

🎯 Objectives

Build a supervised ML pipeline for crop yield prediction.

Perform feature engineering to enhance predictive performance.

Compare machine learning models and identify the best performer.

Analyze feature importance for Nepal’s geographical context.

Support policymakers and farmers with data-driven insights.

Methodology
1️⃣ Data Preprocessing

Handling missing values

Removing duplicates

Outlier detection using IQR

Encoding categorical variables

Scaling numerical features

2️⃣ Feature Engineering

Derived features added include:

Temperature range

Rainfall per area

Fertilizer per area

Solar radiation per area

PAR per area

Interaction features (e.g., temp × rainfall)

Log transformations

3️⃣ Model Training

The following models were trained and evaluated:

Linear Regression, Ridge, Lasso

Decision Tree

Random Forest

Gradient Boosting

XGBoost

LightGBM

CatBoost

Support Vector Regressor

Stacking Regressor

4️⃣ Hyperparameter Tuning

Performed using GridSearchCV and K-Fold Cross-Validation.

5️⃣ Model Evaluation

Metrics used:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE

R² Score

📈 Key Results

Stacking Regressor delivered the best overall performance.

Feature engineering and scaling significantly improved model accuracy.

Climate-related variables (temperature, rainfall, solar radiation) were the most influential.

The model achieved R² ≈ 0.83, showing strong predictive capability.

🌱 Agricultural Use Cases

Early yield forecasting

Resource optimization (fertilizer, irrigation)

Climate-risk assessment

District-level agricultural planning

Policy decisions for food security

🧪 Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost, LightGBM, CatBoost

Matplotlib, Seaborn

Joblib

📜 License

This project is licensed under the MIT License, allowing open use, modification, and distribution with attribution.

🤝 Contribution

Contributions, issues, and suggestions are welcome!
Please open an issue or submit a pull request.

👤 Author

DB Moktan
MTech IT, Kathmandu University
Machine Learning & Data Science Enthusiast
LinkedIn: https://www.linkedin.com/in/db-moktan/