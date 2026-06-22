# Predictive Maintenance using AI4I 2020 Dataset

## Project Objective

The objective of this project is to predict machine failures using sensor data collected from industrial machines. The project aims to support predictive maintenance by identifying potential failures before they occur.

## Dataset

AI4I 2020 Predictive Maintenance Dataset

### Features

* Air Temperature [K]
* Process Temperature [K]
* Rotational Speed [rpm]
* Torque [Nm]
* Tool Wear [min]
* Machine Failure

## Work Completed

### Exploratory Data Analysis (EDA)

* Missing value analysis
* Machine failure distribution analysis
* Air temperature distribution analysis
* Process temperature distribution analysis
* Rotational speed distribution analysis
* Torque distribution analysis

### Feature Engineering

Created new features:

* Temperature_Difference = Process Temperature - Air Temperature
* Wear_Torque_Index = Tool Wear × Torque
* AirTemp_RollingMean
* ProcessTemp_RollingMean
* Torque_RollingMean
* ToolWear_RollingMean

## Files

* 01_data_exploration.py
* 02_air_temperature_analysis.py
* 03_process_temperature_analysis.py
* 04_rotational_speed_analysis.py
* 05_torque_analysis.py
* 06_feature_engineering.py

## Current Status

EDA and Initial Feature Engineering completed.

Completed:
* Dataset Understanding
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Rolling Mean Feature Generation

Dataset is prepared for advanced feature engineering and machine learning model development.

## Next Steps

* Contextual Data Fusion
* SMOTE for class balancing
* LightGBM model development
* Model evaluation
* Dashboard creation
