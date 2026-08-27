"""Week 4 - Predictive Modeling and Optimization in Logistics."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the synthetic Week 4 dataset
df = pd.read_csv("synthetic_logistics_week4_dataset.csv")
features = ["region", "transport_mode", "shipment_volume", "distance_km", "weather_risk", "carrier_load_pct", "fuel_price_index"]
target = "delivery_time_days"
X, y = df[features], df[target]
cat_cols = ["region", "transport_mode", "weather_risk"]
num_cols = ["shipment_volume", "distance_km", "carrier_load_pct", "fuel_price_index"]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=7, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=180, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
rows, fitted = [], {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": r2_score(y_test, pred)
    })
    fitted[name] = pipe

results = pd.DataFrame(rows).sort_values("RMSE")
results.to_csv("model_results.csv", index=False)
best_name = results.iloc[0]["Model"]
best_model = fitted[best_name]

# Five-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(best_model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
pd.DataFrame({"fold": range(1, 6), "RMSE": cv_rmse}).to_csv("cross_validation_rmse.csv", index=False)

print(results)
print("Selected model:", best_name)
print("Mean CV RMSE:", cv_rmse.mean())
