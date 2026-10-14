# Early-Stage Heart Disease Risk Prediction

A machine-learning project exploring heart-disease classification with the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset. The focus is on building and comparing models on a large, real-world public-health survey dataset, with particular attention to class imbalance, recall, and the trade-off between detecting positive cases and generating false alarms.

This is a research and learning project, not a clinically validated diagnostic tool. The target used in the project is the dataset's `HeartDisease` label; this should not be interpreted as predicting an imminent heart attack.

## Why this project is useful

Unlike a small, clean demonstration dataset, BRFSS is a large observational survey built from self-reported health and lifestyle information. Modeling this type of data raises practical issues that matter in real-world machine learning:

- Imbalanced labels, where overall accuracy can hide poor detection of positive cases.
- Mixed demographic, behavioral, and health-related variables.
- Careful encoding, preprocessing, and train/test separation.
- Comparing model performance using recall, precision, F1-score, ROC-AUC, and accuracy rather than accuracy alone.
- The need to interpret model output cautiously and avoid treating predictive association as causation.

The project explores these issues through exploratory analysis, feature engineering, baseline classifiers, boosting models, and neural-network experiments.

## Dataset

**Source:** [CDC Behavioral Risk Factor Surveillance System (BRFSS), 2024 annual data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)

The project README and report describe the source dataset as containing **300,000+ records** and health-related survey features. The raw full-size dataset is not present in the current repository tree, so users should obtain it from the official source and follow the source's documentation and terms.

The target column used in the project is `HeartDisease`:

- `1`: heart disease reported
- `0`: no heart disease reported

Features used or considered in the project include age category, BMI, smoking, alcohol drinking, physical and mental health, diabetes, sleep time, physical activity, and sex. The precise set of columns depends on the processed dataset used in each notebook.

**Important distinction:** BRFSS is an observational survey. This project models a reported heart-disease outcome from survey variables. It does not establish a clinical diagnosis, infer causality, or directly forecast an acute heart attack.

## Workflow

### 1. Data preprocessing

The notebooks explore cleaning, duplicate and missing-value handling, categorical encoding, feature preparation, and train/test splitting. SMOTE is used in experiments to address class imbalance. Resampling should be applied to training data only, after the split, to prevent leakage into evaluation data.

### 2. Exploratory analysis and feature engineering

The project examines feature distributions and relationships, uses correlation plots and PCA for exploratory visualization, and investigates whether health and lifestyle variables help distinguish the target classes. PCA plots are exploratory; overlap in a projection is not, by itself, proof that a class boundary is nonlinear.

### 3. Models compared

The repository contains experiments involving:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- CatBoost
- Multilayer Perceptron (MLP)
- A boosting ensemble combining XGBoost, LightGBM, and CatBoost

The goal is to compare models and operating trade-offs, especially positive-class recall. Higher recall can reduce missed positive cases, but it may also lead to more false positives.

### 4. Evaluation

Reported evaluation metrics include accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC curves, and precision–recall curves. For imbalanced health labels, review precision and the confusion matrix alongside recall; high accuracy alone is not enough.

## Reported results

The current project README contains the following historical experiment results. These values are reproduced from that README and have not been independently rerun as part of this documentation update.

| Model | Recall | Accuracy | Precision |
| --- | ---: | ---: | ---: |
| SVM | 66.08% | 88.16% | 20.00% |
| XGBoost + LightGBM + CatBoost ensemble | 64.30% | 89.00% | 21.40% |
| MLP | 72.29% | 74.89% | 12.69% |
| Logistic Regression | 71.19% | 84.45% | 16.68% |
| XGBoost | 61.79% | 85.33% | 16.00% |
| Random Forest | 60.00% | 86.19% | 17.00% |

The previous README also reported an ROC-AUC of approximately 0.90 for the final ensemble, but it did not identify the exact model/run and evaluation setup for that figure. Confirm the metric against saved evaluation outputs before comparing it with the table.

The results show an important real-world trade-off: the model with the highest accuracy is not necessarily the one with the highest recall. Precision values are low in these reported experiments, meaning a substantial fraction of positive predictions may be false positives. These outcomes require more validation before practical use.

## Repository structure

```text
Early-stage-Heart-attack-prediction/
├── Analysis/
│   ├── Data_Preprocessing.ipynb
│   ├── EDA.ipynb
│   ├── Feature_Eng.ipynb
│   ├── Model_Combined.ipynb
│   ├── Model_V1.ipynb
│   └── Model_XGB.ipynb
├── Raw_data_preprocessing/
│   └── datap.ipynb
├── catboost_info/
└── README.md
```

The repository includes preprocessing, exploratory analysis, feature engineering, and modeling notebooks. The full raw BRFSS dataset is not currently listed in the repository tree. Some small CSV files in the repository may be placeholders or generated artifacts; check their contents before treating them as complete datasets.

## Getting started

Clone the repository:

```bash
git clone https://github.com/Abhikumar199/Early-stage-Heart-attack-prediction.git
cd Early-stage-Heart-attack-prediction
```

Create and activate a Python environment, then install the libraries required by the notebooks. The repository does not currently provide a verified root-level `requirements.txt`, so check notebook imports and your local environment before choosing package versions.

Common libraries used in the project include:

- Python, NumPy, and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- imbalanced-learn
- XGBoost, LightGBM, and CatBoost
- PyTorch and/or TensorFlow for neural-network experiments

Download the relevant BRFSS data from the official CDC source, prepare the files as expected by the preprocessing notebook, and run the notebooks in sequence. Exact paths and commands depend on the local data layout.

## Methodological notes

For a trustworthy evaluation on imbalanced survey data:

1. Separate training and test data before fitting preprocessing steps or applying SMOTE.
2. Fit encoders, scalers, feature-selection steps, and resampling only on the training fold.
3. Use a stratified split or cross-validation where appropriate, and report how the split was constructed.
4. Select classification thresholds on validation data rather than the test set.
5. Report confusion matrices, precision, recall, F1, ROC-AUC, and precision–recall performance, with class prevalence and the threshold stated.
6. Check for duplicates or closely related records crossing train/test boundaries.
7. Evaluate calibration and stability across splits before making claims about deployment.
8. Document the exact dataset version, feature list, random seed, software versions, and preprocessing choices.

These are recommended reproducibility checks; the existing notebooks should be inspected to confirm which checks are already implemented.

## Limitations

- The current results are historical notebook outputs and require independent reruns for confirmation.
- The full raw dataset is not included in the current repository tree.
- Low reported precision indicates a false-positive burden that needs careful evaluation.
- A random split may not fully represent performance under changes in population or survey collection.
- Self-reported survey variables can contain measurement error and may not represent clinical measurements.
- Feature importance and predictive association do not establish causation.
- No external validation, prospective clinical validation, or real-world deployment assessment is documented here.

## Visualizations

The notebooks are intended to generate exploratory feature plots, a correlation heatmap, PCA visualizations, ROC curves, confusion matrices, and precision–recall curves. Refer to the corresponding notebook outputs for the current figures.

## Intended use

This repository is intended for educational and research use in tabular machine learning, imbalanced classification, and analysis of public-health survey data. It must not be used to diagnose heart disease, predict an imminent heart attack, or guide medical decisions.

## Citation and data terms

Credit the CDC BRFSS dataset and follow the official data documentation and usage terms. If using this project in research or coursework, cite this repository and clearly describe the data version, preprocessing, evaluation method, and model configuration used.
