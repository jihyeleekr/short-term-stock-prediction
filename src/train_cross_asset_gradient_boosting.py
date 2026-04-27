import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
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


# More regularized Gradient Boosting parameter grid.
# Goal: reduce overfitting by using shallow trees, low learning rate,
# fewer estimators, and larger leaf sizes.
GB_PARAM_GRID = [
    {
        "n_estimators": 25,
        "learning_rate": 0.03,
        "max_depth": 1,
        "min_samples_leaf": 30,
        "subsample": 0.7,
    },
    {
        "n_estimators": 50,
        "learning_rate": 0.02,
        "max_depth": 1,
        "min_samples_leaf": 30,
        "subsample": 0.7,
    },
    {
        "n_estimators": 50,
        "learning_rate": 0.03,
        "max_depth": 1,
        "min_samples_leaf": 50,
        "subsample": 0.7,
    },
    {
        "n_estimators": 75,
        "learning_rate": 0.02,
        "max_depth": 1,
        "min_samples_leaf": 50,
        "subsample": 0.7,
    },
    {
        "n_estimators": 50,
        "learning_rate": 0.01,
        "max_depth": 1,
        "min_samples_leaf": 50,
        "subsample": 0.8,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.01,
        "max_depth": 1,
        "min_samples_leaf": 50,
        "subsample": 0.8,
    },
]


FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

# Save this model's plots in a separate folder
MODEL_FIGURE_DIR = FIGURE_DIR / "gradient_boosting"

MODEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


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


def safe_roc_auc(y_true, probs):
    """
    ROC-AUC requires both classes to be present.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, probs)


def evaluate_classifier(model, X, y):
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

    return metrics, preds, probs


def prepare_target_dataset(df: pd.DataFrame, target_stock: str, helper_stocks: list):
    """
    Build dataset for one target stock.
    Predict next-day direction of target_stock using helper stock returns.
    """
    cols_needed = ["date", target_stock] + helper_stocks
    temp = df[cols_needed].copy()

    temp["target_direction"] = (
        temp[target_stock].shift(-1) > temp[target_stock]
    ).astype(int)

    temp = temp.dropna().reset_index(drop=True)

    return temp


def select_best_gradient_boosting_model(X_train, y_train, X_val, y_val):
    """
    Try multiple Gradient Boosting parameter settings.
    Select the best model using validation ROC-AUC.
    Does not save or plot tuning config results.
    """
    best_model = None
    best_params = None
    best_val_auc = -np.inf
    best_gap = None

    for idx, params in enumerate(GB_PARAM_GRID):
        config_name = f"config_{idx + 1}"

        model = GradientBoostingClassifier(
            **params,
            random_state=42,
        )

        model.fit(X_train, y_train)

        train_metrics, _, _ = evaluate_classifier(model, X_train, y_train)
        val_metrics, _, _ = evaluate_classifier(model, X_val, y_val)

        train_val_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]

        print("\n" + "-" * 60)
        print(f"{config_name}: {params}")
        print(f"Train ROC-AUC: {train_metrics['roc_auc']:.4f}")
        print(f"Val ROC-AUC:   {val_metrics['roc_auc']:.4f}")
        print(f"Train-Val Gap: {train_val_gap:.4f}")

        if (
            val_metrics["roc_auc"] > best_val_auc
            or (
                val_metrics["roc_auc"] == best_val_auc
                and (best_gap is None or train_val_gap < best_gap)
            )
        ):
            best_val_auc = val_metrics["roc_auc"]
            best_gap = train_val_gap
            best_params = params
            best_model = model

    return best_model, best_params, best_val_auc, best_gap


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
        "Cross-Asset Gradient Boosting: Train vs Validation vs Test",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "gradient_boosting_overfit_underfit_subplots.png"
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

    plt.suptitle("Cross-Asset Gradient Boosting: Overall Test Results", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "gradient_boosting_overall_test_results_subplots.png"
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

    axes[-1].axis("off")

    plt.suptitle(
        "Cross-Asset Gradient Boosting: Test Confusion Matrices",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "gradient_boosting_confusion_matrices_subplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_feature_importance_subplots(feature_importance_dict):
    """
    One combined feature importance figure with 5 stocks in subplots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, stock in enumerate(TARGET_STOCKS):
        ax = axes[idx]

        importance_df = feature_importance_dict[stock].sort_values(
            "importance", ascending=False
        )

        sns.barplot(
            data=importance_df,
            x="importance",
            y="feature",
            ax=ax,
        )

        ax.set_title(stock)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Helper Stock")

    axes[-1].axis("off")

    plt.suptitle(
        "Cross-Asset Gradient Boosting: Feature Importance by Stock",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = MODEL_FIGURE_DIR / "gradient_boosting_feature_importance_subplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    dataset_path = PROCESSED_DATA_DIR / "cross_asset_dataset.csv"
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    all_results = {}
    all_predictions = {}
    all_feature_importances = {}

    for target_stock in TARGET_STOCKS:
        print("\n" + "=" * 70)
        print(f"Tuning Gradient Boosting model for {target_stock}")
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

        best_model, best_params, best_val_auc, best_gap = (
            select_best_gradient_boosting_model(
                X_train,
                y_train,
                X_val,
                y_val,
            )
        )

        print("\n" + "*" * 60)
        print(f"Best params for {target_stock}: {best_params}")
        print(f"Best validation ROC-AUC: {best_val_auc:.4f}")
        print(f"Best train-validation ROC-AUC gap: {best_gap:.4f}")
        print("*" * 60)

        train_metrics, train_preds, train_probs = evaluate_classifier(
            best_model, X_train, y_train
        )
        val_metrics, val_preds, val_probs = evaluate_classifier(
            best_model, X_val, y_val
        )
        test_metrics, test_preds, test_probs = evaluate_classifier(
            best_model, X_test, y_test
        )

        train_test_gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]

        print("\nFinal Train metrics:", train_metrics)
        print("Final Validation metrics:", val_metrics)
        print("Final Test metrics:", test_metrics)
        print(f"Final Train-Test ROC-AUC Gap: {train_test_gap:.4f}")

        all_results[target_stock] = {
            "helper_stocks": helper_stocks,
            "best_params": best_params,
            "best_validation_roc_auc": best_val_auc,
            "best_train_validation_roc_auc_gap": best_gap,
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "train_test_roc_auc_gap": train_test_gap,
        }

        all_predictions[target_stock] = {
            "y_test": y_test.to_numpy(),
            "test_preds": test_preds,
        }

        importance_df = pd.DataFrame(
            {
                "feature": helper_stocks,
                "importance": best_model.feature_importances_,
            }
        )

        all_feature_importances[target_stock] = importance_df

    output_json = METRICS_DIR / "cross_asset_gradient_boosting_metrics.json"

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved metrics to: {output_json}")

    plot_overfit_underfit_subplots(all_results)
    plot_overall_test_results_subplots(all_results)
    plot_confusion_matrices_subplots(all_predictions)
    plot_feature_importance_subplots(all_feature_importances)


if __name__ == "__main__":
    main()