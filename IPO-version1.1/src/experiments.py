"""
Experiment campaign: which features + models work best for each task?

Run with:  python src/experiments.py

Tests multiple feature sets (raw, log-transformed, demand-share ratios) x
multiple models (LR, RF, SVM, GBM, XGBoost, Huber) on the OUT-OF-TIME split
(train <= 2023, test 2024-2026) and prints a compact comparison table.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (HuberRegressor, LinearRegression, LogisticRegression)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from data_loader import load_data
from preprocessing import clean_data, drop_unusable_rows, fill_missing_values, winsorize_outliers
from feature_engineering import add_subscription_class, add_all_features

SPLIT_YEAR = 2023

CLS_SETS = {
    "base":        ["Issue_Size(crores)", "Offer Price", "Year"],
    "log":         ["log_Issue_Size", "log_Offer_Price", "Year"],
    "log_noYear":  ["log_Issue_Size", "log_Offer_Price"],
    "log_ratio":   ["log_Issue_Size", "log_Offer_Price", "Year", "Issue_Price_Ratio"],
}

REG_SETS = {
    "base":               ["QIB", "HNI", "RII", "Total", "Issue_Size(crores)", "Offer Price"],
    "log":                ["log_QIB", "log_HNI", "log_RII", "log_Total", "log_Issue_Size", "log_Offer_Price"],
    "log_ratios":         ["log_QIB", "log_HNI", "log_RII", "log_Total", "QIB_share", "RII_share",
                           "HNI_to_QIB", "log_Issue_Size", "log_Offer_Price"],
    "log_ratios_noTotal": ["log_QIB", "log_HNI", "log_RII", "QIB_share", "RII_share",
                           "HNI_to_QIB", "log_Issue_Size", "log_Offer_Price"],
}


def class_weights(y):
    """Balanced sample weights (used by boosting models that lack class_weight)."""
    counts = y.value_counts()
    w = {c: len(y) / (len(counts) * n) for c, n in counts.items()}
    return y.map(w).to_numpy()


def run_classification(df):
    # XGBoost requires integer class labels
    y = df["Subscription_Class"].map({"Low": 0, "Medium": 1, "High": 2})
    rows = []
    for set_name, feats in CLS_SETS.items():
        X = df[feats].dropna()
        idx = X.index
        y_ = y.loc[idx]
        train_mask = df.loc[idx, "Year"] <= SPLIT_YEAR
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y_[train_mask], y_[~train_mask]
        sw = class_weights(y_train)

        models = {
            "LR":      LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
            "RF":      RandomForestClassifier(class_weight="balanced", random_state=42),
            "SVM":     SVC(class_weight="balanced", random_state=42),
            "GBM":     GradientBoostingClassifier(random_state=42),
            "XGB":     XGBClassifier(random_state=42, eval_metric="mlogloss"),
        }
        for mname, model in models.items():
            if mname in ("LR", "SVM"):
                sc = StandardScaler()
                Xtr, Xte = sc.fit_transform(X_train), sc.transform(X_test)
            else:
                Xtr, Xte = X_train, X_test
            if mname in ("GBM", "XGB"):
                model.fit(Xtr, y_train, sample_weight=sw)
            else:
                model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            rows.append({
                "task": "classification", "features": set_name, "model": mname,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
            })
    return pd.DataFrame(rows)


def run_regression(df):
    y = df["Listing Gain"]
    rows = []
    for set_name, feats in REG_SETS.items():
        X = df[feats].dropna()
        idx = X.index
        y_ = y.loc[idx]
        train_mask = df.loc[idx, "Year"] <= SPLIT_YEAR
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y_[train_mask], y_[~train_mask]

        models = {
            "LR":     LinearRegression(),
            "RF":     RandomForestRegressor(random_state=42),
            "GBM":    GradientBoostingRegressor(random_state=42),
            "XGB":    XGBRegressor(random_state=42),
            "Huber":  HuberRegressor(max_iter=3000),
        }
        for mname, model in models.items():
            if mname in ("LR", "Huber"):
                sc = StandardScaler()
                Xtr, Xte = sc.fit_transform(X_train), sc.transform(X_test)
            else:
                Xtr, Xte = X_train, X_test
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            rows.append({
                "task": "regression", "features": set_name, "model": mname,
                "R2": r2_score(y_test, y_pred),
                "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
                "MAE": mean_absolute_error(y_test, y_pred),
            })
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading data and engineering features...")
    df = load_data("data/processed/cleaned_ipo_data.csv")
    df = add_all_features(df)

    print("\n=== CLASSIFICATION (tier) — out-of-time split ===")
    cls = run_classification(df)
    cls = cls.sort_values("balanced_acc", ascending=False)
    print(cls.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== REGRESSION (listing gain) — out-of-time split ===")
    reg = run_regression(df)
    reg = reg.sort_values("R2", ascending=False)
    print(reg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cls.to_csv("reports/figures/experiment_classification.csv", index=False)
    reg.to_csv("reports/figures/experiment_regression.csv", index=False)
    print("\nFull tables saved to reports/figures/experiment_*.csv")


if __name__ == "__main__":
    main()
