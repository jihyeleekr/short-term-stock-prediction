import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR


METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_FILES = {
    "Baseline Logistic": "baseline_logistic_metrics.json",
    "Cross-Asset Logistic": "cross_asset_metrics.json",
    "Cross-Asset Random Forest": "cross_asset_rf_metrics.json",
    "Cross-Asset Gradient Boosting": "cross_asset_gradient_boosting_metrics.json",
}


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def build_train_val_test_table():
    """
    Build a long-format dataframe with train, validation, and test metrics
    for each model and each target stock.
    """
    rows = []

    for model_name, file_name in MODEL_FILES.items():
        file_path = METRICS_DIR / file_name

        if not file_path.exists():
            print(f"Warning: missing metrics file for {model_name}: {file_path}")
            continue

        results = load_json(file_path)

        for stock, stock_results in results.items():
            for split in ["train", "validation", "test"]:
                if split not in stock_results:
                    print(f"Warning: {split} missing for {model_name} - {stock}")
                    continue

                metrics = stock_results[split]

                rows.append(
                    {
                        "model": model_name,
                        "stock": stock,
                        "split": split.capitalize(),
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "roc_auc": metrics["roc_auc"],
                    }
                )

    return pd.DataFrame(rows)


def plot_train_val_test_by_model(df, metric_name):
    """
    For each model, plot Train vs Validation vs Test for each stock.
    """
    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=df,
        x="stock",
        y=metric_name,
        hue="split",
    )

    plt.title(f"Train vs Validation vs Test: {metric_name.replace('_', ' ').title()}")
    plt.xlabel("Target Stock")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.ylim(0, 1)
    plt.legend(title="Data Split")
    plt.tight_layout()

    save_path = FIGURE_DIR / f"train_val_test_{metric_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_train_val_test_for_each_model(df, metric_name):
    """
    Save one plot per model.
    This is useful for checking overfitting/underfitting model-by-model.
    """
    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name].copy()

        plt.figure(figsize=(10, 5))

        sns.barplot(
            data=model_df,
            x="stock",
            y=metric_name,
            hue="split",
            order=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            hue_order=["Train", "Validation", "Test"],
        )

        plt.title(f"{model_name}: Train vs Validation vs Test {metric_name.replace('_', ' ').title()}")
        plt.xlabel("Target Stock")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.ylim(0, 1)
        plt.legend(title="Data Split")
        plt.tight_layout()

        clean_model_name = model_name.lower().replace(" ", "_").replace("-", "_")
        save_path = FIGURE_DIR / f"{clean_model_name}_train_val_test_{metric_name}.png"

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")


def plot_average_metric_by_model(df, metric_name):
    """
    Plot average train/validation/test score for each model.
    This gives a clean summary for overfitting/underfitting.
    """
    avg_df = (
        df.groupby(["model", "split"], as_index=False)[metric_name]
        .mean()
    )

    plt.figure(figsize=(12, 5))

    sns.barplot(
        data=avg_df,
        x="model",
        y=metric_name,
        hue="split",
        hue_order=["Train", "Validation", "Test"],
    )

    plt.title(f"Average Train vs Validation vs Test {metric_name.replace('_', ' ').title()} by Model")
    plt.xlabel("Model")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend(title="Data Split")
    plt.tight_layout()

    save_path = FIGURE_DIR / f"average_train_val_test_{metric_name}_by_model.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    comparison_df = build_train_val_test_table()

    output_csv = METRICS_DIR / "train_val_test_comparison.csv"
    comparison_df.to_csv(output_csv, index=False)
    print(f"Saved table: {output_csv}")

    print("\nTrain / Validation / Test Comparison:")
    print(comparison_df)

    for metric in ["accuracy", "f1", "roc_auc"]:
        plot_train_val_test_for_each_model(comparison_df, metric)
        plot_average_metric_by_model(comparison_df, metric)


if __name__ == "__main__":
    main()