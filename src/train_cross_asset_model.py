import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from config import PROCESSED_DATA_DIR, OUTPUT_DIR


TARGET_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

HELPER_MAP = {
    "AAPL": ["MSFT", "GOOGL", "NVDA"],
    "MSFT": ["AAPL", "GOOGL", "NVDA"],
    "GOOGL": ["MSFT", "AAPL", "AMZN"],
    "AMZN": ["MSFT", "AAPL", "GOOGL"],
    "META": ["GOOGL", "MSFT", "AAPL"],
}

# Smaller C = stronger regularization.
# This helps reduce overfitting.
C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

MODEL_FIGURE_DIR = FIGURE_DIR / "cross_asset_logistic"

MODEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def time_split(df, train_ratio=0.7, val_ratio=0.15):
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


def safe_roc_auc(y_true, probs):
    """
    ROC-AUC requires both classes to be present.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, probs)


def evaluate_model(model, X, y, split_name):
    """
    Evaluate classifier with accuracy, F1, and ROC-AUC.
    """
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": safe_roc_auc(y, probs),
    }

    print(f"\n{split_name} metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def prepare_target_dataset(df, target_stock, helper_stocks):
    """
    Build dataset for one target stock.
    Predict next-day direction of target_stock using helper stocks' same-day returns.
    """
    cols_needed = ["date", target_stock] + helper_stocks
    temp = df[cols_needed].copy()

    temp["target_direction"] = (
        temp[target_stock].shift(-1) > temp[target_stock]
    ).astype(int)

    temp = temp.dropna().reset_index(drop=True)

    return temp


def build_logistic_model(c_value):
    """
    Build logistic regression pipeline with scaling and L2 regularization.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def plot_model_comparison(results_dict):
    """
    Save Seaborn bar plot of test accuracy across target stocks.
    """
    rows = []

    for stock, results in results_dict.items():
        rows.append(
            {
                "stock": stock,
                "accuracy": results["test"]["accuracy"],
            }
        )

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=plot_df,
        x="stock",
        y="accuracy",
        order=TARGET_STOCKS,
    )

    plt.title("Tuned Cross-Asset Logistic Test Accuracy by Target Stock")
    plt.xlabel("Target Stock")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()

    save_path = MODEL_FIGURE_DIR / "cross_asset_model_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_regularization_results(all_results):
    """
    Plot validation ROC-AUC for different C values.
    This helps show how regularization affects performance.
    """
    rows = []

    for stock, stock_results in all_results.items():
        for c_value, c_results in stock_results["all_c_results"].items():
            rows.append(
                {
                    "stock": stock,
                    "C": float(c_value),
                    "validation_roc_auc": c_results["validation"]["roc_auc"],
                    "test_roc_auc": c_results["test"]["roc_auc"],
                    "train_test_gap": (
                        c_results["train"]["roc_auc"]
                        - c_results["test"]["roc_auc"]
                    ),
                }
            )

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        data=plot_df,
        x="C",
        y="validation_roc_auc",
        hue="stock",
        marker="o",
    )

    plt.xscale("log")
    plt.title("Cross-Asset Logistic Regularization Tuning")
    plt.xlabel("C Value (smaller = stronger regularization)")
    plt.ylabel("Validation ROC-AUC")
    plt.ylim(0, 1)
    plt.tight_layout()

    save_path = MODEL_FIGURE_DIR / "cross_asset_logistic_regularization_tuning.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_overfit_underfit_subplots(results_dict):
    """
    One figure with 3 subplots:
    Accuracy / F1 / ROC-AUC.
    Each subplot shows Train vs Validation vs Test across all 5 stocks.
    """
    rows = []

    for stock, stock_results in results_dict.items():
        for split in ["train", "validation", "test"]:
            rows.append(
                {
                    "stock": stock,
                    "split": split.capitalize(),
                    "accuracy": stock_results[split]["accuracy"],
                    "f1": stock_results[split]["f1"],
                    "roc_auc": stock_results[split]["roc_auc"],
                }
            )

    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metric_info = [
        ("accuracy", "Accuracy"),
        ("f1", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
    ]

    for ax, (metric_col, metric_title) in zip(axes, metric_info):
        sns.barplot(
            data=plot_df,
            x="stock",
            y=metric_col,
            hue="split",
            order=TARGET_STOCKS,
            hue_order=["Train", "Validation", "Test"],
            ax=ax,
        )

        ax.set_title(metric_title)
        ax.set_xlabel("Target Stock")
        ax.set_ylabel(metric_title)
        ax.set_ylim(0, 1)

        if ax != axes[0]:
            ax.get_legend().remove()

    axes[0].legend(title="Data Split")

    plt.suptitle(
        "Cross-Asset Logistic Regression: Train vs Validation vs Test",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "cross_asset_logistic_overfit_underfit_subplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_overall_test_results_subplots(results_dict):
    """
    One figure with 3 subplots:
    Accuracy / F1 / ROC-AUC.
    Test performance only, across all 5 stocks.
    """
    rows = []

    for stock, stock_results in results_dict.items():
        rows.append(
            {
                "stock": stock,
                "accuracy": stock_results["test"]["accuracy"],
                "f1": stock_results["test"]["f1"],
                "roc_auc": stock_results["test"]["roc_auc"],
            }
        )

    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metric_info = [
        ("accuracy", "Test Accuracy"),
        ("f1", "Test F1 Score"),
        ("roc_auc", "Test ROC-AUC"),
    ]

    for ax, (metric_col, metric_title) in zip(axes, metric_info):
        sns.barplot(
            data=plot_df,
            x="stock",
            y=metric_col,
            order=TARGET_STOCKS,
            ax=ax,
        )

        ax.set_title(metric_title)
        ax.set_xlabel("Target Stock")
        ax.set_ylabel(metric_title)
        ax.set_ylim(0, 1)

    plt.suptitle("Cross-Asset Logistic Regression: Overall Test Results", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "cross_asset_logistic_overall_test_results_subplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_confusion_matrices_subplots(predictions_dict):
    """
    One combined confusion matrix figure with 5 stocks in subplots.
    Uses test-set confusion matrices.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, stock in enumerate(TARGET_STOCKS):
        ax = axes[idx]

        y_test = predictions_dict[stock]["y_test"]
        test_preds = predictions_dict[stock]["test_preds"]

        cm = confusion_matrix(y_test, test_preds)

        row_sums = cm.sum(axis=1, keepdims=True)

        cm_percent = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )

        annot_labels = np.empty_like(cm).astype(str)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_labels[i, j] = f"{cm_percent[i, j]:.2f}\n(n={cm[i, j]})"

        sns.heatmap(
            cm_percent,
            annot=annot_labels,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=["Down", "Up"],
            yticklabels=["Down", "Up"],
            cbar=(idx == 0),
            ax=ax,
        )

        ax.set_title(stock)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    # Turn off the unused 6th subplot
    axes[-1].axis("off")

    plt.suptitle(
        "Cross-Asset Logistic Regression: Test Confusion Matrices",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "cross_asset_logistic_confusion_matrices_subplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    dataset_path = PROCESSED_DATA_DIR / "cross_asset_dataset.csv"
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    all_results = {}
    all_predictions = {}

    for target_stock in TARGET_STOCKS:
        print("\n" + "=" * 70)
        print(f"Tuning cross-asset logistic model for {target_stock}")
        print("=" * 70)

        helper_stocks = HELPER_MAP[target_stock]
        print(f"Using helper stocks: {helper_stocks}")

        modeling_df = prepare_target_dataset(df, target_stock, helper_stocks)

        train_df, val_df, test_df = time_split(modeling_df)

        X_train = train_df[helper_stocks]
        y_train = train_df["target_direction"]

        X_val = val_df[helper_stocks]
        y_val = val_df["target_direction"]

        X_test = test_df[helper_stocks]
        y_test = test_df["target_direction"]

        best_c = None
        best_val_auc = -1
        best_model = None
        all_c_results = {}

        for c_value in C_VALUES:
            print("\n" + "-" * 50)
            print(f"Training {target_stock} with C = {c_value}")
            print("-" * 50)

            model = build_logistic_model(c_value)
            model.fit(X_train, y_train)

            train_metrics = evaluate_model(model, X_train, y_train, "Train")
            val_metrics = evaluate_model(model, X_val, y_val, "Validation")
            test_metrics = evaluate_model(model, X_test, y_test, "Test")

            train_test_gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]

            print(f"Train-Test ROC-AUC Gap: {train_test_gap:.4f}")

            all_c_results[str(c_value)] = {
                "helper_stocks": helper_stocks,
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "train_test_roc_auc_gap": train_test_gap,
            }

            # Choose best C using validation ROC-AUC only.
            # Do not choose based on test because that would overfit to test set.
            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_c = c_value
                best_model = model

        print("\n" + "*" * 50)
        print(f"Best C for {target_stock}: {best_c}")
        print(f"Best validation ROC-AUC: {best_val_auc:.4f}")
        print("*" * 50)

        final_train_metrics = evaluate_model(best_model, X_train, y_train, "Final Train")
        final_val_metrics = evaluate_model(best_model, X_val, y_val, "Final Validation")
        final_test_metrics = evaluate_model(best_model, X_test, y_test, "Final Test")

        final_train_test_gap = (
            final_train_metrics["roc_auc"] - final_test_metrics["roc_auc"]
        )

        # Store test predictions for combined confusion matrix subplot.
        # This does NOT save individual stock confusion matrices.
        test_preds = best_model.predict(X_test)

        all_predictions[target_stock] = {
            "y_test": y_test.to_numpy(),
            "test_preds": test_preds,
        }

        all_results[target_stock] = {
            "helper_stocks": helper_stocks,
            "best_c": best_c,
            "train": final_train_metrics,
            "validation": final_val_metrics,
            "test": final_test_metrics,
            "train_test_roc_auc_gap": final_train_test_gap,
            "all_c_results": all_c_results,
        }

    output_json = METRICS_DIR / "cross_asset_metrics.json"

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved metrics to: {output_json}")

    plot_model_comparison(all_results)
    plot_regularization_results(all_results)
    plot_overfit_underfit_subplots(all_results)
    plot_overall_test_results_subplots(all_results)
    plot_confusion_matrices_subplots(all_predictions)


if __name__ == "__main__":
    main()