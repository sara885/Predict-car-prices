# Used Cars Price Prediction

## Overview
This project predicts the price of used cars based on various features using an XGBoost regression model with hyperparameter tuning. The dataset contains information such as brand, model, mileage, engine size, and other relevant details.

## Dataset
The dataset used contains used car listings with attributes like price, mileage, model year, engine specifications, and more. Outliers in price were removed to improve model performance.

## Features Engineering
- Cleaned and converted mileage and price columns to numeric types.
- Created new features such as:
  - `car_age` (calculated as 2025 - model_year)
  - `engine_size` extracted from engine description
  - `milage_log` (log transformation of mileage)
  - `mileage_per_year` (mileage divided by car age)
  - `price_per_mile` (price divided by mileage)
- Simplified the model column to keep only the top 50 most frequent models.

## Model and Pipeline
- Used `ColumnTransformer` to preprocess categorical features (OneHotEncoding) and numerical features (passed through).
- Trained an `XGBRegressor` model.
- Used `GridSearchCV` for hyperparameter tuning with 3-fold cross-validation.

## Best Hyperparameters Found
```text
{ "n_estimators": 500,
  "learning_rate": 0.05,
  "max_depth": 6,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_alpha": 1,
  "reg_lambda": 5
}
