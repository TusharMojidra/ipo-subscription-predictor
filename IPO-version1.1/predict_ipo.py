"""
Interactive IPO predictor demo.

Run it with:   python predict_ipo.py

You will be asked for an IPO's details, then the script:
  1) predicts the subscription tier (Low / Medium / High)
  2) predicts the listing gain (%)

The models are trained once and cached to disk (models/ipo_models.joblib).
They are re-trained automatically only when the cleaned dataset changes,
so starting the app is instant after the first run.
"""

import hashlib
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (clean_data, drop_unusable_rows, fill_missing_values,
                               winsorize_outliers)
from src.feature_engineering import add_log_features, add_subscription_class

# Pre-IPO features (log-transformed) for the tier classifier
CLASS_FEATURES = ['log_Issue_Size', 'log_Offer_Price', 'Year']
# Log-transformed subscription features for the gain regressor
GAIN_FEATURES = ['log_QIB', 'log_HNI', 'log_RII', 'log_Total',
                 'log_Issue_Size', 'log_Offer_Price']
CLASS_ORDER = ['Low', 'Medium', 'High']


def load_training_data():
    """Load the cleaned dataset, rebuilding it from the raw CSV if missing."""
    try:
        df = pd.read_csv("data/processed/cleaned_ipo_data.csv")
        if 'Subscription_Class' not in df.columns:
            raise FileNotFoundError
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Cleaned data not found - rebuilding it from the raw CSV...")
        df = pd.read_csv("data/raw/ipo_data.csv")
        df = clean_data(df)
        df = drop_unusable_rows(df)
        df = fill_missing_values(df)
        df = winsorize_outliers(df)
        df = add_subscription_class(df)
        df.to_csv("data/processed/cleaned_ipo_data.csv", index=False)
        return df


def engineer_input(issue_size, offer_price, year, qib, hni, rii, total):
    """Build the same engineered feature row the models were trained on
    (log1p transforms of the raw user inputs)."""
    return {
        'log_Issue_Size': np.log1p(issue_size),
        'log_Offer_Price': np.log1p(offer_price),
        'Year': year,
        'log_QIB': np.log1p(qib),
        'log_HNI': np.log1p(hni),
        'log_RII': np.log1p(rii),
        'log_Total': np.log1p(total),
    }


MODEL_PATH = "models/ipo_models.joblib"


def _data_fingerprint():
    """Hash of the cleaned dataset - used to invalidate cached models."""
    try:
        with open("data/processed/cleaned_ipo_data.csv", "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return "no-clean-file"


def train_models(df, use_cache=True):
    """Train the two predictors on the full cleaned dataset (deployment-style).

    When ``use_cache`` is True the trained models are saved to disk and
    reloaded on the next call, so the app starts instantly. They are only
    retrained when the cleaned dataset changes.
    """
    fp = _data_fingerprint()
    if use_cache and os.path.exists(MODEL_PATH):
        try:
            saved = joblib.load(MODEL_PATH)
            if saved.get("data_fp") == fp:
                return (saved["clf"], saved["scaler_clf"],
                        saved["reg"], saved["scaler_reg"])
        except Exception:
            pass  # corrupt or incompatible cache - just retrain

    df = add_log_features(df)

    # Task A - subscription tier classifier
    # Shallow Random Forest on log features (~61% out-of-time accuracy) and
    # class_weight='balanced' so the rarer classes are not ignored.
    clf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                 class_weight='balanced', random_state=42)
    scaler_clf = StandardScaler()
    clf.fit(scaler_clf.fit_transform(df[CLASS_FEATURES]), df['Subscription_Class'])

    # Task B - listing gain regressor: robust Huber on log features
    # (best out-of-time performer: R2 ~ +0.3 vs -0.2 with raw features)
    reg = GridSearchCV(HuberRegressor(max_iter=5000),
                       {'alpha': [1e-3, 1e-2], 'epsilon': [1.35, 1.7]},
                       cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    scaler_reg = StandardScaler()
    reg.fit(scaler_reg.fit_transform(df[GAIN_FEATURES]), df['Listing Gain'])
    reg = reg.best_estimator_

    # Cache the trained models so the next start is instant
    try:
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        joblib.dump({"clf": clf, "scaler_clf": scaler_clf,
                     "reg": reg, "scaler_reg": scaler_reg,
                     "data_fp": fp}, MODEL_PATH)
    except Exception:
        pass  # saving is optional - the models still work without it

    return clf, scaler_clf, reg, scaler_reg


def ask_number(prompt, default=None):
    """Ask the user for a number, re-asking until they type something valid."""
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  That is not a number. Please type a number, e.g. 500 or 150.25")


def ask_positive(prompt):
    """Like ask_number, but the value must be greater than zero."""
    while True:
        val = ask_number(prompt)
        if val > 0:
            return val
        print("  Please enter a positive number.")


def main():
    print("=" * 60)
    print(" IPO SUBSCRIPTION & LISTING GAIN PREDICTOR")
    print("=" * 60)

    df = load_training_data()
    clf, scaler_clf, reg, scaler_reg = train_models(df)
    print(f"Models trained on {len(df)} historical IPOs (2010-2026).")
    print("Expected accuracy: tier ~61% | gain R2 ~0.30 (error +/-18 points)")
    print()

    print("STEP 1 - PRE-IPO DETAILS (known before the IPO opens)")
    print("----------------------------------------------------")
    issue_size = ask_positive("  Issue size in crores (e.g. 500): ")
    offer_price = ask_positive("  Offer price in Rs (e.g. 150): ")
    year = ask_number("  Year of listing (press Enter for 2026): ", default=2026.0)

    # --- Subscription tier prediction ---
    eng = engineer_input(issue_size, offer_price, year, 1, 1, 1, 1)
    X_clf = scaler_clf.transform(pd.DataFrame(
        [{k: eng[k] for k in CLASS_FEATURES}], columns=CLASS_FEATURES))
    prob_dict = dict(zip(clf.classes_, clf.predict_proba(X_clf)[0]))
    tier = max(prob_dict, key=prob_dict.get)

    print()
    print(f"  Predicted subscription tier: {tier.upper()}")
    for c in CLASS_ORDER:
        print(f"    Chance of {c:<6}: {prob_dict[c] * 100:5.1f}%")

    print()
    print("STEP 2 - SUBSCRIPTION DETAILS (known after bidding closes)")
    print("---------------------------------------------------------")
    qib = ask_positive("  QIB subscription in times (e.g. 45): ")
    hni = ask_positive("  HNI subscription in times (e.g. 40): ")
    rii = ask_positive("  RII subscription in times (e.g. 12): ")
    total = ask_positive("  Total subscription in times (e.g. 25): ")

    # --- Listing gain prediction ---
    eng = engineer_input(issue_size, offer_price, year, qib, hni, rii, total)
    X_reg = scaler_reg.transform(pd.DataFrame([{k: eng[k] for k in GAIN_FEATURES}],
                                              columns=GAIN_FEATURES))
    gain = reg.predict(X_reg)[0]

    print()
    print(f"  Predicted listing gain: {gain:+.1f}%  (typical error +/-18 points)")
    print(f"  So expect roughly: {gain - 18:+.1f}% to {gain + 18:+.1f}%")
    print()
    print("Note: these are rough estimates - stock markets are noisy. Use them")
    print("as a guide, not financial advice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
