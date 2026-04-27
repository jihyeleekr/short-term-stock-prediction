import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR


METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Separate folder for model comparison plots
MODEL_FIGURE_DIR = FIGURE_DIR / "model_comparison"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


TARGET_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

MODEL_FILES = {
    "Baseline Logistic": "baseline_logistic_metrics.json",
    "Cross-Asset Logistic": "cross_asset_metrics.json",
    "Cross-Asset Random Forest": "cross_asset_rf_metrics.json",
    "Cross-Asset Gradient Boosting": "cross_asset_gradient_boosting_metrics.json",
}

MODEL_ORDER = [
    "Baseline Logistic",
    "Cross-Asset Logistic",
    "Cross-Asset Random Forest",
    "Cross-Asset Gradient Boosting",
]


def load_json(file_path):
    """
    Load one JSON metrics file.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def build_model_comparison_table():
    """
    Build a long-format comparison table from saved model metric JSON files.

    Output columns:
    stock, model, accuracy, f1, roc_auc
    """
    rows = []

    for model_name, file_name in MODEL_FILES.items():
        file_path = METRICS_DIR / file_name

        if not file_path.exists():
            print(f"Warning: missing metrics file for {model_name}: {file_path}")
            continue

        results = load_json(file_path)

        for stock in TARGET_STOCKS:
            if stock not in results:
                print(f"Warning: {stock} missing from {model_name}")
                continue

            stock_results = results[stock]

            if "test" not in stock_results:
                print(f"Warning: test metrics missing for {model_name} - {stock}")
                continue

            test_metrics = stock_results["test"]

            rows.append(
                {
                    "stock": stock,
                    "model": model_name,
                    "accuracy": test_metrics["accuracy"],
                    "f1": test_metrics["f1"],
                    "roc_auc": test_metrics["roc_auc"],
                }
            )

    comparison_df = pd.DataFrame(rows)

    return comparison_df


def plot_model_comparison_seaborn(comparison_df, metric_name):
    """
    Plot one metric comparison across models and target stocks.
    metric_name should be: accuracy, f1, or roc_auc.
    """
    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=comparison_df,
        x="stock",
        y=metric_name,
        hue="model",
        order=TARGET_STOCKS,
        hue_order=MODEL_ORDER,
    )

    plt.title(f"Model Comparison: Test {metric_name.replace('_', ' ').title()}")
    plt.xlabel("Target Stock")
    plt.ylabel(f"Test {metric_name.replace('_', ' ').title()}")
    plt.ylim(0, 1)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    save_path = MODEL_FIGURE_DIR / f"model_comparison_test_{metric_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_model_comparison_subplots(comparison_df):
    """
    One figure with 3 subplots:
    Test Accuracy / Test F1 / Test ROC-AUC.
    This is useful for final report.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    metric_info = [
        ("accuracy", "Test Accuracy"),
        ("f1", "Test F1 Score"),
        ("roc_auc", "Test ROC-AUC"),
    ]

    for ax, (metric_name, title) in zip(axes, metric_info):
        sns.barplot(
            data=comparison_df,
            x="stock",
            y=metric_name,
            hue="model",
            order=TARGET_STOCKS,
            hue_order=MODEL_ORDER,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("Target Stock")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)

        if ax != axes[0]:
            ax.get_legend().remove()

    axes[0].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.suptitle("Overall Model Comparison on Test Set", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "model_comparison_test_metrics_subplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_average_model_performance(comparison_df):
    """
    Plot average test performance across all target stocks for each model.
    """
    avg_df = (
        comparison_df.groupby("model", as_index=False)[["accuracy", "f1", "roc_auc"]]
        .mean()
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    metric_info = [
        ("accuracy", "Average Test Accuracy"),
        ("f1", "Average Test F1 Score"),
        ("roc_auc", "Average Test ROC-AUC"),
    ]

    for ax, (metric_name, title) in zip(axes, metric_info):
        sns.barplot(
            data=avg_df,
            x="model",
            y=metric_name,
            order=MODEL_ORDER,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Average Test Performance by Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "average_test_performance_by_model_subplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    comparison_df = build_model_comparison_table()

    if comparison_df.empty:
        print("No model comparison data found. Run the model training scripts first.")
        return

    comparison_path = METRICS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print(f"Saved comparison table to: {comparison_path}")
    print("\nModel Comparison Table:")
    print(comparison_df)

    # Individual metric plots
    plot_model_comparison_seaborn(comparison_df, "accuracy")
    plot_model_comparison_seaborn(comparison_df, "f1")
    plot_model_comparison_seaborn(comparison_df, "roc_auc")

    # Combined subplot figures
    plot_model_comparison_subplots(comparison_df)
    plot_average_model_performance(comparison_df)


if __name__ == "__main__":
    main()