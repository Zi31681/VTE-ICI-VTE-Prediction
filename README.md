# VTE-ICI-VTE-Prediction
This repository documents the machine-learning workflow described in the manuscript:\
"Machine learning\'96based prediction of venous thromboembolism in patients treated with immune checkpoint inhibitors."\
\
## Scope\
- This repository provides code and documentation to support methodological reproducibility.\
- Raw patient-level data are not shared due to ethical and legal restrictions.\
\
## Data preprocessing\
- Continuous variables with >15% missing values were excluded.\
- Remaining missing values were imputed using mean or median values.\
- No explicit outlier removal was performed.\
- All preprocessing steps were applied strictly within training folds.\
\
## Feature selection\
1. LASSO regression (10-fold CV) was used for preliminary feature screening.\
2. Features were further reduced iteratively based on importance stability,\
   predictive performance, and clinical relevance.\
3. The final model included eight predictors.\
\
## Model development\
- Internal cohort: model development and hyperparameter tuning.\
- External cohort: independent validation only.\
- Stratified 5-fold cross-validation was used.\
- Models evaluated: Logistic Regression, Random Forest, Decision Tree,\
  Extra Trees, AdaBoost, CatBoost, and XGBoost.\
\
## Threshold selection\
- A probability cutoff was identified using the Youden index.\
- Decision curve analysis was used to assess clinical utility.\
