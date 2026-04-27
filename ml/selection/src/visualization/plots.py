import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

PLOT_DPI = 300
PLOT_FIGSIZE = (10, 6)
PLOT_FONTSIZE = 12
EPA_R2_TARGET = 0.70
EPA_RMSE_TARGET = 7.0

plt.rcParams.update({
    "font.size": PLOT_FONTSIZE,
    "font.family": "sans-serif",
    "figure.figsize": PLOT_FIGSIZE,
    "figure.dpi": PLOT_DPI,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_comparison_table(results_df, title, save_path):
    fig, ax = plt.subplots(figsize=(10, max(3, len(results_df) * 0.6 + 1.5)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    cols = list(results_df.columns)
    cell_text = results_df.values.tolist()

    cell_colors = []
    for _, row in results_df.iterrows():
        row_colors = []
        for col in cols:
            if col == "EPA_Pass":
                row_colors.append("#c8e6c9" if row[col] else "#ffcdd2")
            elif col == "R²":
                row_colors.append("#c8e6c9" if row[col] >= EPA_R2_TARGET else "#fff9c4")
            elif col == "RMSE":
                row_colors.append("#c8e6c9" if row[col] <= EPA_RMSE_TARGET else "#fff9c4")
            else:
                row_colors.append("#ffffff")
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text, colLabels=cols,
        cellColours=cell_colors, colColours=["#bbdefb"] * len(cols),
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(y_true, y_pred, model_name, save_path):
    fig, ax = plt.subplots()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="none")

    lims = [
        min(min(y_true), min(y_pred)) * 0.9,
        max(max(y_true), max(y_pred)) * 1.1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("True PM2.5 (ug/m3)")
    ax.set_ylabel("Predicted PM2.5 (ug/m3)")
    ax.set_title(f"{model_name}: True vs Predicted")
    ax.text(0.05, 0.92, f"R2 = {r2:.4f}\nRMSE = {rmse:.2f} ug/m3",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history_dict, model_name, save_path):
    fig, ax = plt.subplots()

    train_loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_loss, label="Validation Loss", linewidth=1.5)

    best_epoch = np.argmin(val_loss) + 1
    ax.axvline(x=best_epoch, color="red", linestyle=":", linewidth=1,
               label=f"Best Epoch ({best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"{model_name}: Training & Validation Loss")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(feature_names, importances, title, save_path):
    indices = np.argsort(importances)[::-1][:15]
    top_names = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top_names) * 0.4)))
    y_pos = np.arange(len(top_names))

    ax.barh(y_pos, top_importances[::-1], color="#4a90d9", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(dates, y_true, y_pred, model_name, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, y_true, label="Reference", linewidth=1.5, color="#2196F3")
    ax.plot(dates, y_pred, label=f"Predicted ({model_name})", linewidth=1.5,
            color="#FF5722", alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title(f"{model_name}: Time Series Comparison (Test Set)")
    ax.legend()
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(y_true, y_pred, model_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(y_true, bins=30, alpha=0.7, color="#2196F3", edgecolor="black", linewidth=0.5)
    axes[0].axvline(np.mean(y_true), color="red", linestyle="--",
                    label=f"Mean={np.mean(y_true):.1f}")
    axes[0].set_title("Reference PM2.5 Distribution")
    axes[0].set_xlabel("PM2.5 (ug/m3)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(y_pred, bins=30, alpha=0.7, color="#FF5722", edgecolor="black", linewidth=0.5)
    axes[1].axvline(np.mean(y_pred), color="red", linestyle="--",
                    label=f"Mean={np.mean(y_pred):.1f}")
    axes[1].set_title(f"{model_name} Predicted Distribution")
    axes[1].set_xlabel("PM2.5 (ug/m3)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.suptitle(f"PM2.5 Distribution: Reference vs {model_name}", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name, save_path):
    residuals = y_true - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors="none")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Predicted PM2.5 (ug/m3)")
    ax.set_ylabel("Residual (ug/m3)")
    ax.set_title(f"{model_name}: Residual Plot")

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_horizon(horizon_rmses, save_path):
    fig, ax = plt.subplots()

    model_names = list(horizon_rmses.keys())
    n_models = len(model_names)
    days = np.arange(1, 8)
    width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (name, rmses) in enumerate(horizon_rmses.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(days + offset, rmses, width, label=name, color=colors[i], edgecolor="none")

    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("RMSE (ug/m3)")
    ax.set_title("RMSE vs Forecast Horizon")
    ax.set_xticks(days)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_example(y_true, y_pred, model_name, sample_idx, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))

    days = np.arange(1, len(y_true) + 1)
    ax.plot(days, y_true, "o-", label="Actual", color="#2196F3", linewidth=2, markersize=6)
    ax.plot(days, y_pred, "s--", label=f"Predicted ({model_name})", color="#FF5722",
            linewidth=2, markersize=6)

    ax.set_xlabel("Forecast Day")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title(f"7-Day Forecast Example (Sample {sample_idx + 1})")
    ax.set_xticks(days)
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_pm25_histograms(df_raw, df_standard, save_dir="outputs/diagrams"):
    os.makedirs(save_dir, exist_ok=True)
    column = "PM2.5"
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df_raw[column].dropna(), kde=True, bins=30)
    plt.title(f'Original: {column}')

    plt.subplot(1, 2, 2)
    sns.histplot(df_standard[column].dropna(), kde=True, bins=30, color='green')
    plt.title(f'StandardScaled: {column}')

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{column}_standard_scaling.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Histogram saved to: {save_path}")
