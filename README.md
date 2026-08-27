# Week 4 - Predictive Modeling and Optimization in Logistics

## Overview
This project applies predictive modeling and optimization techniques to a hypothetical logistics system. The main prediction target is **delivery time in days**, and the resulting predictions are used to support operational resource allocation.

## Dataset
A synthetic dataset of **800 shipment records** was created with:
- region
- transport_mode
- shipment_volume
- distance_km
- weather_risk
- carrier_load_pct
- fuel_price_index
- delivery_time_days (target)
- delay_days
- on_time

The data is synthetic and intended for educational use.

## Predictive Modeling
Models compared:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation metrics:
- MAE
- RMSE
- R²

Validation:
- 80/20 train-test split
- 5-fold cross-validation

The best candidate model is selected using test-set RMSE and then checked using cross-validation.

## Optimization Strategy
A simplified resource-allocation approach assigns shipments to:
- Standard Truck
- Express Van
- Heavy Truck

The decision logic considers predicted delivery time, shipment volume, vehicle cost assumptions, and a budget constraint. The prototype demonstrates how predictive insights can be converted into operational actions.

## Visualizations
- Model comparison by RMSE
- Actual vs predicted delivery time
- 5-fold cross-validation RMSE
- Baseline vs optimized resource-allocation cost

## Repository Structure
```text
Week-4-logistics-data-intern/
├── Week_4_Logistics_Predictive_Modeling_Optimization_Report.docx
├── predictive_logistics_model.py
├── synthetic_logistics_week4_dataset.csv
├── model_results.csv
├── cross_validation_rmse.csv
├── optimization_summary.csv
├── optimized_resource_allocation.csv
├── README.md
├── requirements.txt
├── .gitignore
└── charts/
    ├── 01_model_comparison_rmse.png
    ├── 02_actual_vs_predicted.png
    ├── 03_cross_validation.png
    └── 04_optimization_cost.png
```

## Run
```bash
pip install -r requirements.txt
python predictive_logistics_model.py
```

## Important Note
Model scores and optimization outputs in this repository are based on synthetic data and simulated assumptions. They should not be interpreted as actual logistics performance or guaranteed cost savings.
