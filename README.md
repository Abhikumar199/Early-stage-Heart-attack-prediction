#  Predicting the Silent Threat: Early-Stage Heart Attack Prediction using Machine Learning

**Author:** Abhishek Kumar  
**Date:** November 2025  
**Dataset:** CDC Heart Disease Dataset  
**GitHub Repo:** [Early-stage-Heart-attack-prediction](https://github.com/Abhikumar199/Early-stage-Heart-attack-prediction)

---

##  Project Overview

This project focuses on developing a **machine learning-based predictive model** for **early detection of heart attack and cardiovascular disease risk** using the **CDC Heart Disease Dataset**.  

Heart diseases are among the leading causes of death globally, and early prediction can play a crucial role in preventive healthcare. The goal of this project is to build and compare multiple ML models capable of predicting potential heart attack risks based on health, behavioral, and lifestyle features.

---

## Objectives

- Predict early-stage heart disease risk using real-world health data  
- Compare multiple ML algorithms for optimal accuracy, recall, and precision  
- Handle **imbalanced classes** effectively using resampling methods  
- Optimize for **recall**, prioritizing detection of positive (high-risk) cases  
- Provide explainable insights for medical interpretability  

---

##  Dataset Description

**Source:** [CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html)

**Size:** ~300,000+ records  
**Features:** 18 health-related attributes  
**Target:** `HeartDisease` (1 = Heart Disease, 0 = No Heart Disease)

**Key Features:**
- AgeCategory  
- BMI  
- Smoking  
- AlcoholDrinking  
- PhysicalHealth  
- MentalHealth  
- Diabetic  
- SleepTime  
- PhysicalActivity  
- Sex  

---

## 🧩 Workflow

### 1️ Data Preprocessing
- Cleaned and standardized dataset (removed missing values, duplicates)
- Label encoded categorical variables (`Yes/No` → `1/0`, `Male/Female` → `1/0`)
- Handled **class imbalance** using **SMOTE (Synthetic Minority Oversampling Technique)**
- Split data into 80% train and 20% test sets

### 2️ Exploratory Data Analysis (EDA)
- Generated correlation heatmaps and feature distributions  
- Performed **PCA (Principal Component Analysis)** for visualization  
- Observed overlapping clusters, confirming the **non-linear separability** of classes

### 3️ Model Building
Implemented and tuned multiple models:
| Algorithm | Library | Highlights |
|------------|----------|------------|
| Logistic Regression | Scikit-learn | Baseline model for comparison |
| Random Forest | Scikit-learn | Ensemble-based feature selection |
| Support Vector Machine (SVM) | Scikit-learn | High recall and robustness |
| XGBoost | xgboost | Excellent performance on tabular data |
| LightGBM | lightgbm | Fast, memory-efficient |
| CatBoost | catboost | Handles categorical data efficiently |
| Multilayer Perceptron (MLP) | TensorFlow / PyTorch | Captures complex feature relations |

A **Recall-Optimized Ensemble Model (XGBoost + LightGBM + CatBoost)** was created for final evaluation.

### 4️ Model Evaluation
Evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC–AUC Curve**

---

##  Results Summary

| Model | Recall | Accuracy | Precision |
|--------|--------|-----------|------------|
| SVM | **66.08%** | **88.16%** | 20% |
| Ensemble (XGB + LGBM + CatBoost) | **64.3%** | **89%** | 21.4% |
| MLP (Neural Network) | **72.29%** | **74.89%** | 12.69% |
| Logistic Regression | **71.19%** | **84.45%** | 16.68% |
| XGBoost | **61.79%** | **85.33%** | 16% |
| Random Forest | **60%** | **86.19%** | 17% |

 **Final Ensemble Model Performance:**
- **Accuracy:** ~89%  
- **Recall:** ~72%  
- **ROC–AUC:** ≈ 0.90  
- **Precision:** ~21%  

---

## 🧠 Key Insights

- **Top Predictors:** Age Category, Physical Health, Diabetic,has asthma
- Lifestyle and behavioral features significantly influence prediction outcomes  
- Balancing recall is critical — reducing false negatives can save lives  
- PCA showed partial overlap between classes, confirming the need for non-linear models  

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|--------|
| **Languages** | Python |
| **Libraries** | NumPy, Pandas, Scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow, Matplotlib, Seaborn |
| **Preprocessing** | Imbalanced-learn (SMOTE), Scikit-metrics,pytorch,cuda |
| **Development** | Jupyter Notebook, VS Code |
| **Version Control** | Git & GitHub |

---

## 📉 Visualizations

| Plot | Description |
|------|--------------|
| **Correlation Heatmap** | Shows relationships between features |
| **PCA Plot** | Highlights data distribution and separability |
| **ROC Curve** | Evaluates classifier’s performance |
| **Confusion Matrix** | Displays true vs. predicted classes |
| **Precision–Recall Curve** | Measures trade-off between recall and precision |
| **Error Plot** | Highlights false positives and negatives |
  

`
