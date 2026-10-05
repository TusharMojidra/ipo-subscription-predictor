import pandas as pd
import numpy as np


def remove_duplicates(df):
    return df.drop_duplicates()


def clean_data(df):
    """Basic cleaning: strip column names, parse dates, coerce numerics, drop bad dates."""
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], errors="coerce", format='mixed')
    df['Year'] = df['Date'].dt.year

    numeric_cols = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total',
                    'Offer Price', 'List Price', 'Listing Gain', 'CMP(BSE)', 'CMP(NSE)', 'Current Gains']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df[~df['Date'].isna()]
    return df


def drop_unusable_rows(df):
    """Drop rows that cannot be used for either prediction task:
    - rows missing the regression target (Listing Gain)
    - rows missing all subscription data (QIB/HNI/RII/Total)
    Also fix the data error where List Price == 0 (treat as missing)."""
    df = df[~df['Listing Gain'].isna()].copy()
    sub_cols = ['QIB', 'HNI', 'RII', 'Total']
    df = df[df[sub_cols].notna().all(axis=1)].copy()
    df.loc[df['List Price'] == 0, 'List Price'] = np.nan
    return df.reset_index(drop=True)


def fill_missing_values(df):
    """Impute remaining numeric missing values with the column median.
    (fillna(0) would corrupt price columns, so we use median instead.)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    return df


def winsorize_outliers(df, cols=None, lower=0.01, upper=0.99):
    """Cap extreme outliers at the given percentiles (winsorization)."""
    if cols is None:
        cols = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total']
    for c in cols:
        if c in df.columns:
            lo, hi = df[c].quantile(lower), df[c].quantile(upper)
            df[c] = df[c].clip(lo, hi)
    return df
