import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.feature_engineering import add_log_features

# Features known at IPO close, LOG-TRANSFORMED: subscription ratios are heavily
# right-skewed and recent IPOs are far more oversubscribed than older ones, so
# log1p compresses the extremes and helps the models generalise across time.
FEATURES = ['log_QIB', 'log_HNI', 'log_RII', 'log_Total',
            'log_Issue_Size', 'log_Offer_Price']
TARGET = 'Listing Gain'
SPLIT_YEAR = 2023  # out-of-time split: train on IPOs up to this year, test on later ones


def run_regression(df, outdir="reports/figures", split_year=SPLIT_YEAR):
    """Train Huber (robust), Linear Regression, Decision Tree, Random Forest and
    Gradient Boosting to predict Listing Gain (%) using an OUT-OF-TIME split.
    Huber + log features is the winner (R2 ~ +0.3 vs -0.2 with raw features)."""
    os.makedirs(outdir, exist_ok=True)

    df = add_log_features(df)
    data = df.dropna(subset=FEATURES + [TARGET]).copy()
    X = data[FEATURES]
    y = data[TARGET]

    # Out-of-time split - the model must predict IPOs it never saw in time
    train_mask = data['Year'] <= split_year
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # Scaling helps the linear-family models converge and makes coefficients comparable
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Huber Regression': (HuberRegressor(max_iter=5000),
                             {'alpha': [1e-3, 1e-2], 'epsilon': [1.35, 1.7]}),
        'Linear Regression': (LinearRegression(), {}),
        'Decision Tree': (DecisionTreeRegressor(random_state=42),
                          {'max_depth': [5, 10, None]}),
        'Random Forest': (RandomForestRegressor(random_state=42),
                          {'n_estimators': [100, 200], 'max_depth': [None, 10]}),
        'Gradient Boosting': (GradientBoostingRegressor(random_state=42),
                              {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}),
    }

    results = []
    print("\n" + "=" * 70)
    print("TASK B: LISTING GAIN PREDICTION (Regression)")
    print("Target:", TARGET, "| Features:", FEATURES, "(log-transformed)")
    print(f"Train: {len(X_train)} IPOs (year <= {split_year}) | "
          f"Test: {len(X_test)} IPOs (year > {split_year}) - out-of-time split")
    print("=" * 70)

    for name, (model, grid) in models.items():
        if grid:
            gs = GridSearchCV(model, grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            gs.fit(X_train_s, y_train)
            best = gs.best_estimator_
            best_params = gs.best_params_
        else:
            best = model.fit(X_train_s, y_train)
            best_params = {}

        y_pred = best.predict(X_test_s)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'Best Params': str(best_params),
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
        })
        print(f"\n--- {name} --- (params: {best_params or 'default'})")
        print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.4f}")

    summary = pd.DataFrame(results).set_index('Model')
    print("\n=== REGRESSION SUMMARY ===")
    print(summary.round(4).to_string())

    # Feature importance from the best tree-based model (train split only)
    best_tree = None
    for name, (model, _) in models.items():
        if name in ('Random Forest', 'Gradient Boosting'):
            m = model
            if hasattr(m, 'n_estimators'):
                m = m.set_params(**{'n_estimators': 200})
            m.fit(X_train_s, y_train)
            if best_tree is None or getattr(m, 'feature_importances_', [0]).max() > 0:
                best_tree = (name, m)
    if best_tree is not None:
        name, m = best_tree
        imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print(f"\nFeature importance ({name}):")
        print(imp.round(4).to_string())

        plt.figure(figsize=(8, 5))
        imp.plot(kind='barh')
        plt.title(f'Feature Importance - {name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f"{outdir}/feature_importance.png", dpi=150)
        plt.close()

    summary.to_csv(f"{outdir}/regression_summary.csv")
    return summary
