import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import PROCESSED_DATA_DIR, OUTPUT_DIR


TARGET_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"
INTERACTIVE_DIR = FIGURE_DIR / "interactive"

INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)


MODEL_FILES = {
    "Baseline Logistic": "baseline_logistic_metrics.json",
    "Cross-Asset Logistic": "cross_asset_metrics.json",
    "Cross-Asset Random Forest": "cross_asset_rf_metrics.json",
    "Cross-Asset Gradient Boosting": "cross_asset_gradient_boosting_metrics.json",
}


def plot_interactive_stock_prices(df: pd.DataFrame):
    """
    Interactive line chart for stock closing prices.
    """
    plot_df = df[df["ticker"].isin(TARGET_STOCKS)].copy()

    fig = px.line(
        plot_df,
        x="date",
        y="close",
        color="ticker",
        title="Interactive Stock Closing Prices Over Time",
        labels={
            "date": "Date",
            "close": "Close Price",
            "ticker": "Ticker",
        },
    )

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
    )

    save_path = INTERACTIVE_DIR / "interactive_stock_prices.html"
    fig.write_html(save_path)

    print(f"Saved: {save_path}")


def plot_interactive_returns(df: pd.DataFrame):
    """
    Interactive daily returns chart.
    """
    plot_df = df[df["ticker"].isin(TARGET_STOCKS)].copy()

    fig = px.line(
        plot_df,
        x="date",
        y="ret_1d",
        color="ticker",
        title="Interactive Daily Returns Over Time",
        labels={
            "date": "Date",
            "ret_1d": "Daily Return",
            "ticker": "Ticker",
        },
    )

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
    )

    save_path = INTERACTIVE_DIR / "interactive_daily_returns.html"
    fig.write_html(save_path)

    print(f"Saved: {save_path}")


def plot_interactive_volatility(df: pd.DataFrame):
    """
    Interactive 5-day rolling volatility chart.
    """
    plot_df = df[df["ticker"].isin(TARGET_STOCKS)].copy()

    fig = px.line(
        plot_df,
        x="date",
        y="vol_5d",
        color="ticker",
        title="Interactive 5-Day Rolling Volatility Over Time",
        labels={
            "date": "Date",
            "vol_5d": "5-Day Rolling Volatility",
            "ticker": "Ticker",
        },
    )

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
    )

    save_path = INTERACTIVE_DIR / "interactive_volatility.html"
    fig.write_html(save_path)

    print(f"Saved: {save_path}")


def plot_interactive_correlation_heatmap(df: pd.DataFrame):
    """
    Interactive correlation heatmap using daily returns.
    """
    pivot = df.pivot(index="date", columns="ticker", values="ret_1d")
    corr_matrix = pivot.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Interactive Stock Return Correlation Matrix",
        labels={
            "color": "Correlation",
        },
    )

    fig.update_layout(
        template="plotly_white",
    )

    save_path = INTERACTIVE_DIR / "interactive_correlation_matrix.html"
    fig.write_html(save_path)

    print(f"Saved: {save_path}")


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def build_model_comparison_table():
    """
    Build long-format model comparison table using test metrics.
    """
    rows = []

    for model_name, file_name in MODEL_FILES.items():
        file_path = METRICS_DIR / file_name

        if not file_path.exists():
            print(f"Warning: missing metrics file: {file_path}")
            continue

        results = load_json(file_path)

        for stock in TARGET_STOCKS:
            if stock not in results:
                continue

            test_metrics = results[stock]["test"]

            rows.append(
                {
                    "stock": stock,
                    "model": model_name,
                    "accuracy": test_metrics["accuracy"],
                    "f1": test_metrics["f1"],
                    "roc_auc": test_metrics["roc_auc"],
                }
            )

    return pd.DataFrame(rows)


def plot_interactive_model_comparison(comparison_df: pd.DataFrame):
    """
    Interactive model comparison plot.
    User can choose Accuracy, F1, or ROC-AUC using dropdown.
    """
    metrics = {
        "Accuracy": "accuracy",
        "F1 Score": "f1",
        "ROC-AUC": "roc_auc",
    }

    fig = go.Figure()

    for metric_label, metric_col in metrics.items():
        for model_name in comparison_df["model"].unique():
            model_df = comparison_df[comparison_df["model"] == model_name]

            fig.add_trace(
                go.Bar(
                    x=model_df["stock"],
                    y=model_df[metric_col],
                    name=model_name,
                    visible=(metric_label == "Accuracy"),
                    text=model_df[metric_col].round(3),
                    textposition="auto",
                )
            )

    buttons = []
    num_models = comparison_df["model"].nunique()

    for i, metric_label in enumerate(metrics.keys()):
        visible = [False] * (len(metrics) * num_models)

        start = i * num_models
        end = start + num_models

        for j in range(start, end):
            visible[j] = True

        buttons.append(
            {
                "label": metric_label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"Interactive Model Comparison: Test {metric_label}",
                        "yaxis": {"title": f"Test {metric_label}", "range": [0, 1]},
                    },
                ],
            }
        )

    fig.update_layout(
        title="Interactive Model Comparison: Test Accuracy",
        xaxis_title="Target Stock",
        yaxis_title="Test Accuracy",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.02,
                "xanchor": "left",
                "y": 1.0,
                "yanchor": "top",
            }
        ],
        legend_title="Model",
    )

    save_path = INTERACTIVE_DIR / "interactive_model_comparison.html"
    fig.write_html(save_path)

    print(f"Saved: {save_path}")


def main():
    dataset_path = PROCESSED_DATA_DIR / "model_dataset.csv"

    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    plot_interactive_stock_prices(df)
    plot_interactive_returns(df)
    plot_interactive_volatility(df)
    plot_interactive_correlation_heatmap(df)

    comparison_df = build_model_comparison_table()

    if not comparison_df.empty:
        plot_interactive_model_comparison(comparison_df)
    else:
        print("No model comparison data found. Run model scripts first.")


if __name__ == "__main__":
    main()