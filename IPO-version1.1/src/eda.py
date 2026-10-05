import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_COLS = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total',
              'Offer Price', 'List Price', 'Listing Gain']


def run_eda(df, outdir="reports/figures"):
    """Generate EDA figures (correlation heatmap, distributions, class balance)
    and print key insights to the console."""
    os.makedirs(outdir, exist_ok=True)

    # 1) Correlation heatmap
    corr = df[MODEL_COLS].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap - IPO Features')
    plt.tight_layout()
    plt.savefig(f"{outdir}/correlation_heatmap.png", dpi=150)
    plt.close()

    # 2) Distributions of the two targets
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['Total'], bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Total Subscription Distribution (times)')
    sns.histplot(df['Listing Gain'], bins=50, kde=True, ax=axes[1])
    axes[1].set_title('Listing Gain Distribution (%)')
    plt.tight_layout()
    plt.savefig(f"{outdir}/distributions.png", dpi=150)
    plt.close()

    # 3) Subscription class balance
    if 'Subscription_Class' in df.columns:
        # same aspect as the correlation heatmap (10, 8) so the two figures
        # align perfectly when rendered side by side in the dashboard
        plt.figure(figsize=(10, 8))
        sns.countplot(x='Subscription_Class', data=df, order=['Low', 'Medium', 'High'])
        plt.title('Subscription Class Balance (Low / Medium / High)')
        plt.tight_layout()
        plt.savefig(f"{outdir}/class_balance.png", dpi=150)
        plt.close()

    # Console summary
    print("\n=== EDA SUMMARY ===")
    print(f"Total IPOs: {len(df)}")
    print(f"Subscription classes:\n{df['Subscription_Class'].value_counts().to_string()}")
    top = corr['Listing Gain'].drop('Listing Gain').sort_values(ascending=False)
    print("\nTop correlations with Listing Gain:")
    print(top.round(3).to_string())
    print(f"\nFigures saved to: {outdir}/")
    return outdir
