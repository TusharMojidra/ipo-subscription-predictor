import numpy as np
import pandas as pd

SUBSCRIPTION_COLS = ['QIB', 'HNI', 'RII', 'Total']


def add_listing_gain(df):
    """Compute Listing Gain (%) if the column is not already present in the data."""
    if 'Listing Gain' in df.columns:
        return df
    df["Listing Gain"] = ((df["List Price"] - df["Offer Price"]) / df["Offer Price"]) * 100
    return df


def add_subscription_class(df, low=2.0, high=10.0):
    """Categorize the continuous Total subscription into Low / Medium / High classes.

    Low    : Total <  low      (under / lightly subscribed)
    Medium : low <= Total < high
    High   : Total >= high     (hot IPO)
    """
    def classify(x):
        if x < low:
            return 'Low'
        if x < high:
            return 'Medium'
        return 'High'

    df['Subscription_Class'] = df['Total'].apply(classify)
    return df


def add_log_features(df):
    """log1p transforms for the heavily right-skewed subscription and size columns.
    Recent IPOs are far more oversubscribed (median 26x vs 8x before 2024), so the
    log scale compresses those extremes and helps the models generalise across time."""
    for c in SUBSCRIPTION_COLS:
        df[f'log_{c}'] = np.log1p(df[c])
    df['log_Issue_Size'] = np.log1p(df['Issue_Size(crores)'])
    df['log_Offer_Price'] = np.log1p(df['Offer Price'])
    return df


def add_demand_ratios(df):
    """Share-of-demand features (the top predictors in similar published projects)."""
    total_demand = df[SUBSCRIPTION_COLS].sum(axis=1).replace(0, np.nan)
    df['QIB_share'] = df['QIB'] / total_demand
    df['RII_share'] = df['RII'] / total_demand
    df['HNI_to_QIB'] = df['HNI'] / df['QIB'].replace(0, np.nan)
    df['Issue_Price_Ratio'] = df['Issue_Size(crores)'] / df['Offer Price'].replace(0, np.nan)
    return df


def add_all_features(df):
    """Apply every engineered feature (log transforms + demand ratios)."""
    return add_demand_ratios(add_log_features(df))
