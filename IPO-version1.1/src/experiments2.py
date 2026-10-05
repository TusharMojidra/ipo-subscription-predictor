"""
Round 2: tune the winners and test extra ideas.

Run with:  python src/experiments2.py

Classification: GridSearch on log-feature candidates (balanced + unbalanced weights).
Regression:     tune Huber on log features; test adding Year; give XGB a tuned shot.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor

from data_loader import load_data
from feature_engineering import add_all_features

SPLIT_YEAR = 2023

CLS_FEATS = ["log_Issue_Size", "log_Offer_Price", "Year"]
REG_FEATS = ["log_QIB", "log_HNI", "log_RII", "log_Total",
             "log_Issue_Size", "log_Offer_Price"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = add_all_features(load_data("data/processed/cleaned_ipo_data.csv"))

    # ---------------- CLASSIFICATION ----------------
    y = df["Subscription_Class"]
    X = df[CLS_FEATS]
    train_mask = df["Year"] <= SPLIT_YEAR
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(X_train), sc.transform(X_test)

    print("=== CLASSIFICATION: log features, tuned ===")
    candidates = {
        "LR":      (LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
                    {"C": [0.01, 0.1, 1, 10]}, True),
        "SVM":     (SVC(class_weight="balanced", random_state=42),
                    {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear"]}, True),
        "RF":      (RandomForestClassifier(class_weight="balanced", random_state=42),
                    {"n_estimators": [100, 300], "max_depth": [None, 8, 4]}, False),
        "LR_unbal":   (LogisticRegression(max_iter=3000, random_state=42), {"C": [0.01, 0.1, 1]}, True),
        "SVM_unbal":  (SVC(random_state=42), {"C": [0.1, 1, 10]}, True),
    }
    rows = []
    for name, (model, grid, scale) in candidates.items():
        gs = GridSearchCV(model, grid, cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(Xtr_s if scale else X_train, y_train)
        y_pred = gs.predict(Xte_s if scale else X_test)
        rows.append({
            "model": name, "params": str(gs.best_params_),
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------- REGRESSION ----------------
    y = df["Listing Gain"]
    print("\n=== REGRESSION: Huber tuned + extra ideas ===")

    def eval_reg(feats, model, label):
        X_ = df[feats]
        tr, te = df["Year"] <= SPLIT_YEAR, df["Year"] > SPLIT_YEAR
        Xtr, Xte = X_[tr], X_[te]
        ytr, yte = y[tr], y[te]
        if isinstance(model, (HuberRegressor,)):
            s = StandardScaler()
            Xtr, Xte = s.fit_transform(Xtr), s.transform(Xte)
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        print(f"{label:<34} R2={r2_score(yte, p):+.4f}  RMSE={mean_squared_error(yte, p)**0.5:6.2f}  "
              f"MAE={mean_absolute_error(yte, p):6.2f}")

    hub = GridSearchCV(HuberRegressor(max_iter=5000), {"alpha": [1e-4, 1e-3, 1e-2], "epsilon": [1.0, 1.35, 1.7]},
                       cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    hub.fit(StandardScaler().fit_transform(df[REG_FEATS][df["Year"] <= SPLIT_YEAR]),
            y[df["Year"] <= SPLIT_YEAR])
    print(f"Huber+log best params: {hub.best_params_}")
    eval_reg(REG_FEATS, HuberRegressor(max_iter=5000, **hub.best_params_), "Huber log (tuned)")

    eval_reg(REG_FEATS + ["Year"], HuberRegressor(max_iter=5000, **hub.best_params_),
             "Huber log + Year")
    eval_reg(REG_FEATS, HuberRegressor(max_iter=5000, alpha=1e-3, epsilon=1.35),
             "Huber log (mid params)")

    xgb = GridSearchCV(XGBRegressor(random_state=42),
                       {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.03, 0.1]},
                       cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    xgb.fit(df[REG_FEATS][df["Year"] <= SPLIT_YEAR], y[df["Year"] <= SPLIT_YEAR])
    print(f"XGB+log best params: {xgb.best_params_}")
    eval_reg(REG_FEATS, XGBRegressor(random_state=42, **xgb.best_params_), "XGB log (tuned)")


if __name__ == "__main__":
    main()
