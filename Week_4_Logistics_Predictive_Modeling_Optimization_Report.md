# Week 4 — Predictive Modeling and Optimization in Logistics Systems

## 1. Introduction
This project demonstrates how predictive analytics can be applied to logistics operations to forecast delivery time and support better resource-allocation decisions. A reproducible synthetic dataset is used so the complete workflow can be demonstrated without exposing confidential operational data.

## 2. Problem Definition
The objective is to predict `delivery_time_days` for each shipment using region, transport mode, shipment volume, distance, weather risk, carrier load percentage, and fuel-price index. Accurate forecasts can help logistics teams identify high-risk shipments early, allocate suitable transportation capacity, and improve service reliability.

## 3. Data and Features
The synthetic dataset contains shipment-level operational variables. Categorical features are one-hot encoded and missing numeric/categorical values are handled through a Scikit-learn preprocessing pipeline. The target variable is delivery time in days.

## 4. Models
Four regression models are compared:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

An 80/20 train-test split is used for initial evaluation. The models are evaluated using MAE, RMSE, and R². Five-fold cross-validation is then used to check the stability of the selected model.

## 5. Evaluation
The generated experiment selects the model with the lowest test-set RMSE. In the synthetic experiment, Linear Regression produced the lowest test RMSE (approximately **1.057 days**) and an R² of approximately **0.838**. Five-fold cross-validation produced a mean RMSE of approximately **1.022 days**. These figures are illustrative and depend entirely on the synthetic data generation process.

## 6. Optimization Strategy
Prediction is connected to a simplified operational decision layer. Shipments are assigned to Standard Trucks, Express Vans, or Heavy Trucks based on predicted delivery risk, shipment volume, transportation cost, and a simulated budget. The purpose is to demonstrate the transition from a machine-learning prediction to an actionable logistics decision.

A production implementation could replace the heuristic with Vehicle Routing Problem (VRP), linear programming, mixed-integer optimization, or Google OR-Tools while adding real constraints such as vehicle capacity, delivery time windows, depot locations, driver availability, route distance, service penalties, and fuel costs.

## 7. Business Recommendations
1. Use delivery-time predictions as an early-warning mechanism for potentially delayed shipments.
2. Prioritize expedited capacity for high-risk shipments where faster service provides sufficient business value.
3. Match vehicle capacity with shipment volume to reduce inefficient resource use.
4. Monitor prediction error by region and transport mode.
5. Retrain the model when logistics patterns or carrier performance change.
6. Integrate a formal routing/optimization solver when real operational constraints become available.

## 8. Conclusion
The project presents an end-to-end predictive logistics workflow: defining the forecasting problem, preparing features, training multiple machine-learning models, validating performance, and translating predictions into a resource-allocation strategy. The key lesson is that predictive accuracy should ultimately support a measurable operational objective such as lower delivery delays, better capacity utilization, improved service reliability, or controlled transportation cost.

> **Important:** All data, costs, budgets, model results, and optimization assumptions are synthetic and intended for educational demonstration only.

## Python Implementation
The repository includes `predictive_logistics_model.py`, which loads the dataset, preprocesses categorical and numerical features, trains the regression models, calculates MAE/RMSE/R², and performs five-fold cross-validation.
