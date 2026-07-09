# Predictive Maintenance using AI4I 2020 Dataset

# Overview

Predictive maintenance is a data-driven maintenance strategy that predicts machine failures before they occur using machine learning and sensor data. Unlike traditional maintenance methods, which are either reactive (repair after failure) or preventive (maintenance at fixed intervals), predictive maintenance continuously analyzes machine conditions to identify potential failures in advance. This helps industries reduce unexpected equipment breakdowns, minimize maintenance costs, improve production efficiency, and increase machine reliability.

This project develops a predictive maintenance system using the **AI4I 2020 Predictive Maintenance Dataset**. The dataset contains industrial machine sensor readings collected during machine operation. Using these sensor values, the project builds a machine learning model capable of predicting whether a machine is likely to fail.

The project begins with **Exploratory Data Analysis (EDA)** to understand the dataset and analyze machine behavior. Several **feature engineering** techniques are applied to create meaningful features from the original sensor readings. Contextual operational information is then incorporated through **Contextual Data Fusion** to improve prediction quality. After preprocessing and handling class imbalance using **SMOTE**, a **LightGBM Classifier** is trained to predict machine failures. Finally, the model is evaluated using multiple performance metrics, threshold tuning, and noise sensitivity analysis to ensure reliable predictions.

This project demonstrates a complete end-to-end machine learning pipeline for predictive maintenance and serves as a strong foundation for future industrial deployment.

---

# Project Objective

The main objectives of this project are:

- Predict machine failures before they occur.
- Reduce equipment downtime.
- Minimize maintenance costs.
- Improve operational efficiency.
- Analyze industrial sensor data.
- Create meaningful engineered features.
- Handle class imbalance using SMOTE.
- Build an accurate machine learning model.
- Evaluate model performance using multiple evaluation metrics.

---

# Dataset

**Dataset Name:** AI4I 2020 Predictive Maintenance Dataset

The dataset contains sensor readings collected from industrial machines during operation.

### Dataset Statistics

- Total Records : 10,000
- Original Features : 14
- Target Variable : Machine Failure

### Main Features

- UDI
- Product ID
- Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- Machine Failure

---

# Project Workflow

## Step 1 : Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset before model development.

Tasks performed:

- Dataset Information
- Missing Value Analysis
- Machine Failure Distribution
- Air Temperature Distribution
- Process Temperature Distribution
- Rotational Speed Distribution
- Torque Distribution

Purpose:

- Understand the dataset
- Identify missing values
- Analyze sensor distributions
- Understand failure patterns

---

## Step 2 : Feature Engineering

Feature engineering was performed to generate additional informative features from the original sensor data.

### Basic Features

- Temperature_Difference
- Wear_Torque_Index

### Time-Series Features

- Rolling Mean Features
- Rolling Standard Deviation Features
- Lag Features
- Change Features

Purpose:

- Improve model learning
- Capture trends
- Capture historical information
- Represent machine behavior more effectively

---

## Step 3 : Contextual Data Fusion

Additional contextual information was incorporated to improve prediction performance.

Contextual Features:

- Shift
- Production Demand
- Ambient Humidity
- Energy Load Index
- Days Since Maintenance

Interaction Features:

- Torque_Load
- Wear_Demand
- Temperature_Humidity

Purpose:

- Provide operational context
- Improve prediction accuracy
- Capture relationships between operating conditions

---

## Step 4 : Data Preprocessing

Performed preprocessing operations including:

- Data Cleaning
- Label Encoding
- Feature Selection
- Feature Name Standardization

Purpose:

- Prepare clean data for machine learning
- Convert categorical variables into numerical values
- Remove unnecessary information

---

## Step 5 : Train-Test Split

The dataset was divided into:

- Training Data : 80%
- Testing Data : 20%

Purpose:

- Train the model
- Evaluate model performance on unseen data

---

## Step 6 : Class Balancing using SMOTE

The dataset contained significantly fewer machine failure samples than normal machine samples.

SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training dataset to balance the classes.

Purpose:

- Handle class imbalance
- Improve failure prediction
- Prevent model bias toward the majority class

---

## Step 7 : LightGBM Model

Machine failures were predicted using the LightGBM Classifier.

Reason for selecting LightGBM:

- Fast training
- High accuracy
- Efficient with tabular datasets
- Good performance on structured data

---

## Step 8 : Model Evaluation

The trained model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Feature Importance
- Confusion Matrix
- Precision-Recall Curve
- Threshold Tuning
- Noise Sensitivity Analysis

Purpose:

- Measure prediction performance
- Analyze important features
- Evaluate robustness
- Optimize prediction threshold

---

# Project Structure

```
Predictive-Maintenance/

│── data/
│── models/
│── notebooks/
│── results/
│── src/
│── README.md
│── app.py (Future Dashboard)
```

---

# Source Files

```
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

20_precision_recall_curve.py

21_save_model.py

22_threshold_tuning.py

23_noise_sensitivity.py
```

---

# Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 99.3% |
| Precision | 90.7% |
| Recall | 99.2% |
| F1 Score | 94.8% |

---

# Technologies Used

Programming Language

- Python

Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- LightGBM
- Imbalanced-learn (SMOTE)
- Joblib

Development Tools

- VS Code
- Git
- GitHub

---

# Results

The project successfully developed a predictive maintenance model capable of accurately predicting machine failures.

Key achievements:

- Successfully completed Exploratory Data Analysis.
- Generated multiple engineered features.
- Applied contextual data fusion.
- Balanced the dataset using SMOTE.
- Trained a LightGBM classifier.
- Achieved high prediction accuracy.
- Evaluated model using multiple performance metrics.
- Performed Threshold Tuning.
- Performed Noise Sensitivity Analysis.
- Saved the trained model for future use.

---

# Future Enhancements

- Streamlit Dashboard
- Real-Time IoT Sensor Integration
- Explainable AI using SHAP
- Cloud Deployment
- Real-Time Monitoring System

---

# Conclusion

This project demonstrates a complete machine learning workflow for predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset. The workflow begins with exploratory data analysis, followed by feature engineering, contextual data fusion, data preprocessing, train-test splitting, SMOTE-based class balancing, and LightGBM model training. The model was thoroughly evaluated using multiple evaluation techniques, including Precision-Recall Curve, Threshold Tuning, and Noise Sensitivity Analysis. The achieved performance indicates that the proposed approach is capable of accurately predicting machine failures and can serve as a foundation for future industrial predictive maintenance systems.