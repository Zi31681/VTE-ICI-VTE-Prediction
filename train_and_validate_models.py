{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 """\
Machine-learning workflow for VTE risk prediction in ICI-treated patients.\
\
This script documents:\
- Model development on the internal cohort\
- Hyperparameter tuning using stratified cross-validation\
- External validation\
\
Raw patient-level data are not provided due to ethical restrictions.\
"""\
\
import numpy as np\
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV\
from sklearn.pipeline import Pipeline\
from sklearn.preprocessing import StandardScaler\
from sklearn.linear_model import LogisticRegression\
from sklearn.ensemble import (\
    RandomForestClassifier,\
    ExtraTreesClassifier,\
    AdaBoostClassifier\
)\
from sklearn.tree import DecisionTreeClassifier\
from xgboost import XGBClassifier\
from catboost import CatBoostClassifier\
\
# ---------------------------------------------------------------------\
# Global settings\
# ---------------------------------------------------------------------\
RANDOM_STATE = 42\
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\
\
# ---------------------------------------------------------------------\
# NOTE:\
# - Hyperparameter tuning is performed ONLY on the internal cohort.\
# - The external cohort is used solely for final evaluation.\
# ---------------------------------------------------------------------\
\
# X_train, y_train = ...\
# X_test, y_test = ...\
\
# =====================================================\
# Logistic Regression\
# =====================================================\
logistic_pipeline = Pipeline([\
    ("scaler", StandardScaler()),\
    ("model", LogisticRegression(\
        solver="lbfgs",\
        max_iter=1000,\
        random_state=RANDOM_STATE\
    ))\
])\
\
logistic_param_grid = \{\
    "model__penalty": ["l2"],\
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],\
    "model__class_weight": ["balanced", None]\
\}\
\
logistic_search = RandomizedSearchCV(\
    estimator=logistic_pipeline,\
    param_distributions=logistic_param_grid,\
    scoring="roc_auc",\
    cv=cv,\
    n_iter=20,\
    random_state=RANDOM_STATE,\
    n_jobs=-1\
)\
\
# =====================================================\
# Random Forest\
# =====================================================\
rf = RandomForestClassifier(random_state=RANDOM_STATE)\
\
rf_param_grid = \{\
    "n_estimators": [100, 200, 500],\
    "max_depth": [10, 20, 30, None],\
    "min_samples_split": [2, 5, 10],\
    "min_samples_leaf": [1, 2, 4],\
    "max_features": ["sqrt", "log2"],\
    "class_weight": ["balanced", "balanced_subsample", None]\
\}\
\
rf_search = RandomizedSearchCV(\
    rf,\
    rf_param_grid,\
    scoring="roc_auc",\
    cv=cv,\
    n_iter=50,\
    random_state=RANDOM_STATE,\
    n_jobs=-1\
)\
\
# =====================================================\
# Decision Tree\
# =====================================================\
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)\
\
dt_param_grid = \{\
    "max_depth": [5, 10, 20, 30, None],\
    "min_samples_split": [2, 5, 10],\
    "min_samples_leaf": [1, 2, 4, 6, 8],\
    "class_weight": ["balanced", None]\
\}\
\
dt_search = RandomizedSearchCV(\
    dt,\
    dt_param_grid,\
    scoring="roc_auc",\
    cv=cv,\
    n_iter=50,\
    random_state=RANDOM_STATE,\
    n_jobs=-1\
)\
\
# =====================================================\
# Extra Trees\
# =====================================================\
et = ExtraTreesClassifier(random_state=RANDOM_STATE)\
\
et_param_grid = \{\
    "n_estimators": [100, 200, 500],\
    "max_depth": [5, 10, 20, 30, None],\
    "min_samples_split": [2, 5, 10],\
    "min_samples_leaf": [1, 2, 4, 6, 8],\
    "max_features": ["sqrt", "log2"],\
    "class_weight": ["balanced", None]\
\}\
\
et_search = RandomizedSearchCV(\
    et,\
    et_param_grid,\
    scoring="roc_auc",\
    cv=cv,\
    n_iter=50,\
    random_state=RANDOM_STATE,\
    n_jobs=-1\
)\
\
# =====================================================\
# XGBoost (final model)\
# =====================================================\
# n_neg = sum(y_train == 0)\
# n_pos = sum(y_train == 1)\
# r = n_neg / n_pos\
\
xgb = XGBClassifier(\
    objective="binary:logistic",\
    random_state=RANDOM_STATE\
)\
\
xgb_param_grid = \{\
    "n_estimators": [100, 200, 500],\
    "max_depth": [1, 3, 5, 10],\
    "learning_rate": [0.01, 0.1, 0.2],\
    "subsample": [0.8, 1.0],\
    "colsample_bytree": [0.8, 1.0],\
    "eval_metric": ["logloss"],\
    # "scale_pos_weight": [0.5 * r, 1.0 * r, 1.5 * r]\
\}\
\
xgb_search = RandomizedSearchCV(\
    xgb,\
    xgb_param_grid,\
    scoring="roc_auc",\
    cv=cv,\
    n_iter=50,\
    random_state=RANDOM_STATE,\
    n_jobs=-1\
)\
\
# =====================================================\
# CatBoost\
# =====================================================\
cat = CatBoostClassifier(\
    loss_function="Logloss",\
    verbose=0,\
    random_state=RANDOM_STATE\
)\
\
cat_param_grid = \{\
    "iterations": [100, 200, 500],\
    "depth": [1, 3, 5, 7, 10],\
    "learning_rate": [0.01, 0.1, 0.2],\
    "auto_class_weights": ["Balanced", "SqrtBalanced", None],\
    "l2_leaf_reg": [1, 3, 5, 7]\
\}\
\
cat_search = RandomizedSearchCV(\
    cat,\
    cat_param_grid,\
    scoring="roc_auc",\
    cv=cv,\
    n_iter=50,\
    random_state=RANDOM_STATE,\
    n_jobs=-1\
)\
\
# =====================================================\
# AdaBoost\
# =====================================================\
ada = AdaBoostClassifier(\
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),\
    algorithm="SAMME.R",\
    random_state=RANDOM_STATE\
)\
\
ada_param_grid = \{\
    "n_estimators": [100, 200, 500],\
    "learning_rate": [0.01, 0.1, 1.0],\
    "estimator__max_depth": [1, 2, 3],\
    "estimator__class_weight": ["balanced"]\
\}\
\
ada_search = RandomizedSearchCV(\
    ada,\
    ada_param_grid,\
    scoring="roc_auc",\
    cv=cv,\
    n_iter=50,\
    random_state=RANDOM_STATE,\
    n_jobs=-1\
)\
}