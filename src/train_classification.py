import json
from pathlib import Path

# Training
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from config import PROCESSED_DATA_DIR, OUTPUT_DIR

# Directory for saving figures
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

"""
Logistic Regress: https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/
"""


def time_split(df: pd.DataFrame, train_ratio = 0.7, val_ratio = 0.15):
  """
  Time-based split: earlest 70% train, next 15% validation, final 15% test
  Train with past data to predidct more recent data.
  """

  df = df.sort_values("date").reset_index(drop=True)

  n = len(df)
  train_end = int(n * train_ratio)
  val_end = int(n * (train_ratio + val_ratio))

  train_df  = df.iloc[:train_end]
  val_df = df.iloc[train_end:val_end]
  test_df = df.iloc[val_end:]

  return train_df, val_df, test_df

def evaluate_classifier(model, X, y, split_name: str):
  preds = model.predict(X)
  probs = model.predict_proba(X)[:, 1]

  metrics = {
    "accuracy": accuracy_score(y, preds),
    "f1": f1_score(y, preds),
    "roc_auc": roc_auc_score(y, probs)
  }

  print(f"\n{split_name} metrics:")
  for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

  print("\nClassification Report:")                
  print(classification_report(y, preds))

  return metrics

# Visualizations
def plot_price(df, ticker="AAPL"):
  df_ticker = df[df["ticker"] == ticker]
  
  plt.figure()
  plt.plot(df_ticker["date"], df_ticker["close"])
  plt.title(f"{ticker} Price Over Time")
  plt.xlabel("date")
  plt.ylabel("close price")
  plt.xticks(rotation=45)
  plt.tight_layout()

  plt.savefig(FIGURE_DIR / f"{ticker}_price.png")
  plt.show()

# Returns and volatility
def plot_returns_volatility(df, ticker="AAPL"):
    df_ticker = df[df["ticker"] == ticker]

    plt.figure()
    plt.plot(df_ticker["date"], df_ticker["ret_1d"])
    plt.title(f"{ticker} Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / f"{ticker}_returns.png")
    plt.show()

    plt.figure()
    plt.plot(df_ticker["date"], df_ticker["vol_5d"])
    plt.title(f"{ticker} 5-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / f"{ticker}_volatility.png")
    plt.show()

# Confusion matrix
def plot_confusion_matrix(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()

    plt.savefig(FIGURE_DIR / "confusion_matrix.png")
    plt.show()

# ROC curve and AUC
def plot_roc_curve(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(FIGURE_DIR / "roc_curve.png")
    plt.show()

def main():
  dataset_path = PROCESSED_DATA_DIR / "model_dataset.csv"
  df = pd.read_csv(dataset_path)
  df["date"] = pd.to_datetime(df["date"])

  feature_cols = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ma_5",
    "ma_10",
    "vol_5d",
    "volume_change_1d",
    "spy_ret_1d",
    "spy_ret_5d"
  ]

  target_col = "target_direction"

  train_df, val_df, test_df = time_split(df)

  X_train = train_df[feature_cols]
  y_train = train_df[target_col]

  X_val = val_df[feature_cols]
  y_val = val_df[target_col]

  X_test = test_df[feature_cols]
  y_test = test_df[target_col]

  model = LogisticRegression(max_iter=1000)
  model.fit(X_train, y_train)

  val_metrics = evaluate_classifier(model, X_val, y_val, "Validation")
  test_metrics = evaluate_classifier(model, X_test, y_test, "Test")

  metrics_dir = OUTPUT_DIR / "metrics"
  metrics_dir.mkdir(parents=True, exist_ok=True)

  for ticker in ["AAPL", "AMZN", "FB", "GOOGL", "SPY", "MSFT"]:
    plot_price(df, ticker)
    plot_returns_volatility(df, ticker)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)


  with open(metrics_dir / "validation_metrics.json", "w") as f:
    json.dump(
      {
        "validation": val_metrics,
        "test": test_metrics,
      },
      f,
      indent=2,
    )
if __name__ == "__main__":
  main()
        