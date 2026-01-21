# VTE Risk Prediction in ICI-Treated Patients

This repository documents the machine-learning workflow described in the manuscript:
"Machine learning–based prediction of venous thromboembolism in patients treated with immune checkpoint inhibitors."

## Scope
- This repository provides code and documentation to support methodological reproducibility.
- Raw patient-level data are not shared due to ethical and legal restrictions.

## Software and computational environment
The analyses were conducted in a unified and finalized computational environment. Specifically:
- **Statistical analysis** was performed using **SPSS v26.0** (IBM Corp.).
- **Data preprocessing and machine-learning modeling** were implemented in **Python v3.12.7**.

The main Python dependencies included:
- scikit-learn v1.5.2  
- XGBoost v2.1.2  
- CatBoost v1.2.7  
- pandas v2.2.2  
- NumPy v1.26.4  
- matplotlib v3.9.2  
- SHAP v0.45.0  

LASSO regression for preliminary feature screening was performed using **R v4.4.2**, with the following packages:
- glmnet v4.1.10  
- caret v7.0.1  
- dplyr v1.1.4 

## Data preprocessing
- Continuous variables with >15% missing values were excluded.
- Remaining missing values were imputed using mean or median values.
- No explicit outlier removal was performed.
- All preprocessing steps were applied strictly within training folds.

## Feature selection
1. LASSO regression (10-fold CV) was used for preliminary feature screening.
2. Features were further reduced iteratively based on importance stability,
   predictive performance, and clinical relevance.
3. The final model included eight predictors.

## Model development
- Internal cohort: model development and hyperparameter tuning.
- External cohort: independent validation only.
- Stratified 5-fold cross-validation was used.
- Models evaluated: Logistic Regression, Random Forest, Decision Tree,
  Extra Trees, AdaBoost, CatBoost, and XGBoost.

## Threshold selection
- A probability cutoff was identified using the Youden index.
- Decision curve analysis was used to assess clinical utility.


