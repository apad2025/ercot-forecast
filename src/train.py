import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent

COMBINED_PATH = ROOT / "data/processed/ercot_with_weather.csv"
FEATURES_PATH = ROOT / "data/processed/features.csv"
PREDICTION_PATH = ROOT / "data/processed/predictions.csv"
REPORTS = ROOT / "reports"

TARGET = "ERCOT"
SPLIT_DATE = "2025-01-01"

# stats
def mae(y_true, y_prediction):
    return float(np.mean(np.abs(y_true - y_prediction)))

def mape(y_true, y_prediction):
    return float(np.mean(np.abs((y_true - y_prediction) / y_true)) * 100)

def report(name, y_true, y_prediction):
    return {
        "model": name,
        "MAE (MW)": mae(y_true, y_prediction),
        "MAPE (%)": mape(y_true, y_prediction)
    }

def main():
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"], index_col="timestamp")

    train = df.loc[df.index < SPLIT_DATE]
    test = df.loc[df.index >= SPLIT_DATE]
    print(f"train: {train.shape} ({train.index.min()} to {train.index.max()})")
    print(f"test: {test.shape} ({test.index.min()} to {test.index.max()})")

    feature_cols = [c for c in df.columns if c != TARGET]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    # calculate percentage shift from 2024 to 2025
    original = pd.read_csv(COMBINED_PATH, parse_dates=["timestamp"], index_col="timestamp")
    mean_2024 = original.loc["2024", TARGET].mean()
    mean_2025 = original.loc["2025", TARGET].mean()
    shift = mean_2025 - mean_2024
    pct = (shift / mean_2024) * 100

    print(f"\nShift from 2024 to 2025: {shift:+,.0f} MW ({pct:+.1f}%)")

    # use persistence as baselines
    results = [
        report("persistence_24h", y_test, X_test["load_lag_24"]),
        report("persistence_168h", y_test, X_test["load_lag_168"])
    ]

    # model
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        random_state=0,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    results.append(report("lightgbm", y_test, y_prediction))

    # print report
    summary = pd.DataFrame(results)
    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # write hourly predictions
    prediction_df = pd.DataFrame({
        "actual": y_test,
        "persistence_24h": X_test["load_lag_24"].values,
        "persistence_168h": X_test["load_lag_168"].values,
        "lightgbm": y_prediction
    }, index=y_test.index)
    prediction_df.to_csv(PREDICTION_PATH)
    print(f"\nwrote {PREDICTION_PATH}")

    # plot one summer and one winter week
    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    for ax, (start, end, title) in zip(axes, [
        ("2025-08-04", "2025-08-10", "Summer test week, Aug 4-10, 2025"),
        ("2025-01-13", "2025-01-19", "Winter test week, Jan 13-19, 2025")
    ]):
        week = prediction_df.loc[start:end] / 1000
        ax.plot(week.index, week["actual"], lw=2.0, label="actual", color="black")
        ax.plot(week.index, week["lightgbm"], lw=1.2, label="lightgbm", color="tab:blue")
        ax.plot(week.index, week["persistence_168h"], lw=0.8, ls="--", label="persistence 168h", color="tab:gray")
        ax.set_ylabel("Load (GW)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORTS / "actual_vs_predicted.png", dpi=120)
    plt.close()

    # residual diagnostic with error by hour of day
    prediction_df["residual"] = prediction_df["actual"] - prediction_df["lightgbm"]
    prediction_df["abs_residual"] = prediction_df["residual"].abs()
    prediction_df["hour"] = prediction_df.index.hour
    by_hour = prediction_df.groupby("hour")["abs_residual"].mean()

    plt.figure(figsize=(9, 4))
    plt.bar(by_hour.index, by_hour.values, color="tab:blue", alpha=0.8)
    plt.xlabel("Hour of day")
    plt.ylabel("Mean |error| (MW)")
    plt.title("LightGBM error by hour of day (against 2025 set)")
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(REPORTS / "residuals_by_hour.png", dpi=120)
    plt.close()

    # feature importance
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=feature_cols).sort_values()
    imp_percent = imp / imp.sum() * 100
    plt.figure(figsize=(8, 5))
    plt.barh(imp_percent.index, imp_percent.values, color="tab:blue", alpha=0.8)
    plt.xlabel("Feature importance as a share of total gain (%)")
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(REPORTS / "feature_importance.png", dpi=120)
    plt.close()

    print(f"wrote {REPORTS}/actual_vs_predicted.png, residuals_by_hour.png, feature_importance.png")

    # bias by load decile
    decile = pd.qcut(y_test, 10, labels=False)
    bias = (
        pd.Series(y_test.values - prediction_df["lightgbm"].values, index=decile).groupby(level=0).mean()
    )

    plt.figure(figsize=(10, 4.5))
    plt.plot(bias.index, bias.values, marker="o", lw=1.5, color="tab:blue")
    plt.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    plt.xlabel("Load decile (0 = lowest 10% of hours, 9 = peak 10%)")
    plt.ylabel("Mean signed error (MW)\n(+ = underpredict, - = overpredict)")
    plt.title("LightGBM bias by load decile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "bias_by_decile.png", dpi=120)
    plt.close()

    print(f"wrote {REPORTS}/bias_by_decile.png")


if __name__ == "__main__":
    main()