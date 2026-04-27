import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_DIR, OUTPUT_DIR


FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def compute_correlation_matrix(df):
    """
    Compute correlation between stocks using daily returns.
    """

    # Pivot table: rows = date, columns = ticker, values = daily returns
    pivot = df.pivot(index="date", columns="ticker", values="ret_1d")

    corr_matrix = pivot.corr()

    return corr_matrix


def plot_correlation_matrix(corr_matrix):
    """
    Plot and save a Seaborn heatmap of stock return correlations.
    """

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"}
    )

    plt.title("Stock Return Correlation Matrix", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig_path = FIGURE_DIR / "correlation_matrix.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved correlation matrix to: {fig_path}")


def main():
    df = pd.read_csv(PROCESSED_DATA_DIR / "model_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])

    corr_matrix = compute_correlation_matrix(df)

    print("\nCorrelation Matrix:\n")
    print(corr_matrix)

    plot_correlation_matrix(corr_matrix)


if __name__ == "__main__":
    main()