import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Pre-IPO available features only (NO subscription ratios - those are the target itself)
FEATURES = ['Issue_Size(crores)', 'Offer Price', 'Year']
TARGET = 'Subscription_Class'
CLASS_ORDER = ['Low', 'Medium', 'High']
SPLIT_YEAR = 2023  # out-of-time split: train on IPOs up to this year, test on later ones


def run_classification(df, outdir="reports/figures", split_year=SPLIT_YEAR):
    """Train Logistic Regression, Random Forest and SVM to predict the
    subscription tier (Low / Medium / High) using an OUT-OF-TIME split:
    models are trained on IPOs up to `split_year` and tested on IPOs after it.

    class_weight='balanced' counteracts the class imbalance so the model
    no longer ignores the rarer classes."""
    os.makedirs(outdir, exist_ok=True)

    data = df.dropna(subset=FEATURES + [TARGET]).copy()
    X = data[FEATURES]
    y = data[TARGET]

    # Out-of-time split - the model must predict IPOs it never saw in time
    train_mask = data['Year'] <= split_year
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # Scaling is required for Logistic Regression and SVM
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=2000, class_weight='balanced',
                                                   random_state=42),
                                {'C': [0.1, 1, 10]}),
        'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42),
                          {'n_estimators': [100, 200], 'max_depth': [None, 10]}),
        'SVM': (SVC(class_weight='balanced', random_state=42),
                {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    }

    results = []
    print("\n" + "=" * 70)
    print("TASK A: SUBSCRIPTION TIER PREDICTION (Classification)")
    print("Target:", TARGET, "| Features:", FEATURES)
    print(f"Train: {len(X_train)} IPOs (year <= {split_year}) | "
          f"Test: {len(X_test)} IPOs (year > {split_year}) - out-of-time split")
    print("=" * 70)

    for name, (model, grid) in models.items():
        gs = GridSearchCV(model, grid, cv=5, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train_s, y_train)

        best = gs.best_estimator_
        y_pred = best.predict(X_test_s)

        results.append({
            'Model': name,
            'Best Params': str(gs.best_params_),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        })

        print(f"\n--- {name} ---")
        print("Best params:", gs.best_params_)
        print(classification_report(y_test, y_pred, target_names=CLASS_ORDER, zero_division=0))

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"{outdir}/cm_{name.replace(' ', '_').lower()}.png", dpi=150)
        plt.close()

    # Summary table
    summary = pd.DataFrame(results).set_index('Model')
    print("\n=== CLASSIFICATION SUMMARY ===")
    print(summary.round(4).to_string())

    # Feature importance from the Random Forest (trained on train split only)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42).fit(X_train_s, y_train)
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nRandom Forest feature importance:")
    print(imp.round(4).to_string())

    summary.to_csv(f"{outdir}/classification_summary.csv")
    return summary
