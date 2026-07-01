# Predictive Maintenance using AI4I 2020 Dataset

## Project Objective

The objective of this project is to predict machine failures using sensor data collected from industrial machines. The project aims to support predictive maintenance by identifying potential failures before they occur.

## Dataset

AI4I 2020 Predictive Maintenance Dataset

Dataset Statistics

• Total Records: 10,000
• Original Features: 14
• Target Variable: Machine Failure

Main Features

• Air Temperature [K]
• Process Temperature [K]
• Rotational Speed [rpm]
• Torque [Nm]
• Tool Wear [min]

## Work Completed

### Exploratory Data Analysis (EDA)

• Missing Value Analysis
• Dataset Information
• Machine Failure Distribution
• Air Temperature Distribution
• Process Temperature Distribution
• Rotational Speed Distribution
• Torque Distribution

### Feature Engineering

Created new features:

• Temperature_Difference
• Wear_Torque_Index
• Rolling Mean Features
• Rolling Standard Deviation Features
• Lag Features
• Change Features

### Contextual Data Fusion

Added contextual features:

• Shift
• Production Demand

Created interaction features for improving predictive performance.

### Data Preprocessing

• Label Encoding
• Data Cleaning
• Feature Selection

### Machine Learning

• Train-Test Split
• SMOTE for Class Balancing
• LightGBM Classifier

### Model Evaluation

Evaluated the model using:

• Accuracy
• Precision
• Recall
• F1 Score
• Confusion Matrix
• Feature Importance


## Project Files

01_data_exploration.py

02_air_temperature_analysis.py

03_process_temperature_analysis.py

04_rotational_speed_analysis.py

05_torque_analysis.py

06_feature_engineering.py

07_rolling_features.py

08_lag_features.py

09_change_features.py

10_rolling_std_features.py

11_contextual_features.py

12_contextual_data_fusion.py

13_data_preprocessing.py

14_train_test_split.py

15_smote_balancing.py

16_lightgbm_model.py

17_model_evaluation.py

18_feature_importance.py

19_confusion_matrix_plot.py

## Current Status

Project completed successfully.

Completed:

• Dataset Understanding

• Exploratory Data Analysis

• Feature Engineering

• Contextual Data Fusion

• Data Preprocessing

• Train-Test Split

• SMOTE

• LightGBM Model

• Model Evaluation

• Feature Importance

• Confusion Matrix

## Future Enhancements

• Streamlit Dashboard

• Real-Time IoT Sensor Integration

• Explainable AI (SHAP)

• Cloud Deployment

## Model Performance

Accuracy : 99.3%

Precision : 90.7%

Recall : 99.2%

F1 Score : 94.8%

## Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-learn

LightGBM

Imbalanced-learn (SMOTE)

Git

GitHub


