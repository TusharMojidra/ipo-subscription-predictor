from src.data_loader import load_data
from src.preprocessing import clean_data, drop_unusable_rows, fill_missing_values, winsorize_outliers
from src.feature_engineering import add_subscription_class
from src.eda import run_eda
from src.models_classification import run_classification
from src.models_regression import run_regression


def main():
    # 1. Load raw data
    df = load_data("data/raw/ipo_data.csv")
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Data cleaning
    df = clean_data(df)
    df = drop_unusable_rows(df)
    df = fill_missing_values(df)
    df = winsorize_outliers(df)
    print(f"After cleaning: {df.shape[0]} rows")

    # 3. Feature engineering - subscription class target
    df = add_subscription_class(df)

    # 4. Save cleaned data
    df.to_csv("data/processed/cleaned_ipo_data.csv", index=False)
    print("Data cleaning completed successfully!")

    # 5. EDA
    run_eda(df)

    # 6. Both prediction tasks
    run_classification(df)
    run_regression(df)

    print("\nPipeline finished. Cleaned data: data/processed/cleaned_ipo_data.csv")
    print("Figures & summaries: reports/figures/")


if __name__ == "__main__":
    main()
