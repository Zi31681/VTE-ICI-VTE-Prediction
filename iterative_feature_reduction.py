{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 """\
Illustrative implementation of iterative feature reduction analysis\
used to explore the trade-off between model parsimony and performance.\
\
This analysis was conducted on the internal cohort only and served\
as supportive evidence rather than a strict feature selection rule.\
"""\
\
import numpy as np\
from sklearn.metrics import roc_auc_score\
from sklearn.model_selection import StratifiedKFold\
\
def evaluate_feature_subset(model, X, y, features, cv):\
    """\
    Evaluate performance of a given feature subset using cross-validation.\
    """\
    aucs = []\
\
    for train_idx, val_idx in cv.split(X, y):\
        X_tr, X_val = X.iloc[train_idx][features], X.iloc[val_idx][features]\
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]\
\
        model.fit(X_tr, y_tr)\
        y_prob = model.predict_proba(X_val)[:, 1]\
        aucs.append(roc_auc_score(y_val, y_prob))\
\
    return np.mean(aucs)\
\
def iterative_feature_reduction(\
    X,\
    y,\
    ranked_features,\
    model,\
    min_features=8,\
    step=1,\
    n_splits=5,\
    random_state=42\
):\
    """\
    Iteratively remove the least important features and record CV performance.\
    """\
    cv = StratifiedKFold(\
        n_splits=n_splits,\
        shuffle=True,\
        random_state=random_state\
    )\
\
    results = []\
    remaining = ranked_features.copy()\
\
    while len(remaining) >= min_features:\
        mean_auc = evaluate_feature_subset(\
            model, X, y, remaining, cv\
        )\
\
        results.append(\{\
            "n_features": len(remaining),\
            "mean_cv_auc": mean_auc\
        \})\
\
        # Remove least important feature\
        remaining = remaining[:-step]\
\
    return results\
}