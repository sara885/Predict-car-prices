import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("/content/sample_data/dtcar/used_cars.csv")

# Clean columns
df['milage'] = df['milage'].str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False).astype(float)
df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

# Remove outliers
df = df[(df['price'] >= 3000) & (df['price'] <= 100000)]

# Car age
df['car_age'] = 2025 - df['model_year']

# Extract engine size
def extract_engine_size(engine_str):
    match = re.search(r'(\d+\.\d+)L', str(engine_str))
    return float(match.group(1)) if match else np.nan

df['engine_size'] = df['engine'].apply(extract_engine_size)
df['engine_size'] = df['engine_size'].fillna(df['engine_size'].median())

# New derived features
df['milage_log'] = np.log1p(df['milage'])
df['mileage_per_year'] = df['milage'] / (df['car_age'] + 1)
df['price_per_mile'] = df['price'] / (df['milage'] + 1)

# Simplify models - keep only top 50 models
top_models = df['model'].value_counts().nlargest(50).index
df['model'] = df['model'].where(df['model'].isin(top_models), other='Other')

# Features
categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'accident', 'clean_title']
numerical_features = ['car_age', 'milage_log', 'engine_size', 'mileage_per_year', 'price_per_mile']

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numerical_features)
])

# Model
xgb = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb)
])

# Features and target
X = df[categorical_features + numerical_features]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter grid
param_grid = {
    'regressor__n_estimators': [300, 500],
    'regressor__learning_rate': [0.03, 0.05],
    'regressor__max_depth': [4, 6],
    'regressor__subsample': [0.8],
    'regressor__colsample_bytree': [0.6, 0.8],
    'regressor__reg_alpha': [1],
    'regressor__reg_lambda': [5]
}

# Training with timing
start = time.time()
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
end = time.time()

# Prediction and evaluation
y_pred_test = grid_search.predict(X_test)
y_pred_train = grid_search.predict(X_train)

print(f"\n✅ Best parameters: {grid_search.best_params_}")
print("\n📊 Performance on **test** data:")
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("R² :", r2_score(y_test, y_pred_test))

print("\n📈 Performance on **training** data:")
print("MAE:", mean_absolute_error(y_train, y_pred_train))
print("MSE:", mean_squared_error(y_train, y_pred_train))
print("R² :", r2_score(y_train, y_pred_train))

print("\n🔍 Comparing the first 5 results:")
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_test}).head()
print(results)

print(f"\n⏱️ Training time: {end - start:.2f} seconds")
