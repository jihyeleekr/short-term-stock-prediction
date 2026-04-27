import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from config import PROCESSED_DATA_DIR, OUTPUT_DIR


TARGET_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

HELPER_MAP = {
    "AAPL": ["MSFT", "GOOGL", "NVDA"],
    "MSFT": ["AAPL", "GOOGL", "NVDA"],
    "GOOGL": ["MSFT", "AAPL", "AMZN"],
    "AMZN": ["MSFT", "AAPL", "GOOGL"],
    "META": ["GOOGL", "MSFT", "AAPL"],
}

BASELINE_FEATURES = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ma_5",
    "ma_10",
    "vol_5d",
    "volume_change_1d",
    "spy_ret_1d",
    "spy_ret_5d",
]

FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def time_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """
    Time-based split: earliest 70% train, next 15% validation, final 15% test.
    """
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def normalize_confusion_matrix(y_true, y_pred):
    """
    Compute row-normalized confusion matrix.
    Rows are actual classes, columns are predicted classes.
    Returns both percentages and annotation labels with raw counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)

    cm_percent = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    labels = np.empty_like(cm).astype(str)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm_percent[i, j]:.2f}\n(n={cm[i, j]})"

    return cm_percent, labels


def train_baseline_logistic(model_df: pd.DataFrame, target_stock: str):
    """
    Train baseline logistic regression using only the target stock's own engineered features.
    """
    stock_df = model_df[model_df["ticker"] == target_stock].copy()
    stock_df = stock_df.sort_values("date").dropna().reset_index(drop=True)

    train_df, val_df, test_df = time_split(stock_df)

    X_train = train_df[BASELINE_FEATURES]
    y_train = train_df["target_direction"]

    X_test = test_df[BASELINE_FEATURES]
    y_test = test_df["target_direction"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return y_test, preds


def prepare_cross_asset_dataset(cross_df: pd.DataFrame, target_stock: str, helper_stocks: list):
    """
    Prepare cross-asset dataset for one target stock.
    """
    cols_needed = ["date", target_stock] + helper_stocks
    temp = cross_df[cols_needed].copy()

    temp["target_direction"] = (temp[target_stock].shift(-1) > temp[target_stock]).astype(int)
    temp = temp.dropna().reset_index(drop=True)

    return temp


def train_cross_asset_logistic(cross_df: pd.DataFrame, target_stock: str, helper_stocks: list):
    """
    Train cross-asset logistic regression using helper stock returns.
    """
    dataset = prepare_cross_asset_dataset(cross_df, target_stock, helper_stocks)

    train_df, val_df, test_df = time_split(dataset)

    X_train = train_df[helper_stocks]
    y_train = train_df["target_direction"]

    X_test = test_df[helper_stocks]
    y_test = test_df["target_direction"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return y_test, preds


def train_cross_asset_random_forest(cross_df: pd.DataFrame, target_stock: str, helper_stocks: list):
    """
    Train cross-asset Random Forest using helper stock returns.
    """
    dataset = prepare_cross_asset_dataset(cross_df, target_stock, helper_stocks)

    train_df, val_df, test_df = time_split(dataset)

    X_train = train_df[helper_stocks]
    y_train = train_df["target_direction"]

    X_test = test_df[helper_stocks]
    y_test = test_df["target_direction"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return y_test, preds


def plot_three_confusion_matrices(target_stock: str, results: dict):
    """
    Plot baseline logistic, cross-asset logistic, and cross-asset random forest
    confusion matrices in one row.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    model_names = [
        "Baseline Logistic",
        "Cross-Asset Logistic",
        "Cross-Asset RF",
    ]

    for ax, model_name in zip(axes, model_names):
        y_true, y_pred = results[model_name]
        cm_percent, labels = normalize_confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm_percent,
            annot=labels,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=["Down", "Up"],
            yticklabels=["Down", "Up"],
            cbar=False,
            ax=ax,
            annot_kws={"size": 9},
        )

        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle(f"{target_stock}: Confusion Matrix Comparison", fontsize=15)
    plt.tight_layout()

    save_path = FIGURE_DIR / f"{target_stock}_confusion_matrix_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    model_df = pd.read_csv(PROCESSED_DATA_DIR / "model_dataset.csv")
    model_df["date"] = pd.to_datetime(model_df["date"])

    cross_df = pd.read_csv(PROCESSED_DATA_DIR / "cross_asset_dataset.csv")
    cross_df["date"] = pd.to_datetime(cross_df["date"])

    for target_stock in TARGET_STOCKS:
        print("\n" + "=" * 60)
        print(f"Creating confusion matrix comparison for {target_stock}")
        print("=" * 60)

        helper_stocks = HELPER_MAP[target_stock]

        baseline_result = train_baseline_logistic(model_df, target_stock)
        cross_logistic_result = train_cross_asset_logistic(cross_df, target_stock, helper_stocks)
        cross_rf_result = train_cross_asset_random_forest(cross_df, target_stock, helper_stocks)

        results = {
            "Baseline Logistic": baseline_result,
            "Cross-Asset Logistic": cross_logistic_result,
            "Cross-Asset RF": cross_rf_result,
        }

        plot_three_confusion_matrices(target_stock, results)


if __name__ == "__main__":
    main()