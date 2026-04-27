"""Extract and plot feature importance from the best calibration model."""

import os
import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "virtual_colocation", "output", "best_calibration_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FEATURE_LABELS = [
    "PM2.5 raw",
    "Humidity",
    "Temperature",
    "Pressure",
    "Heat Index",
    "Hour of Day",
    "Month",
    "Day of Week",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("Loading calibration model...")
    model = joblib.load(MODEL_PATH)
    model_name = type(model).__name__
    print(f"Model type: {model_name}")

    if not hasattr(model, "feature_importances_"):
        print(f"ERROR: {model_name} does not have feature_importances_. "
              "Only tree-based models (RF, XGBoost, DTR) support this.")
        return

    importances = model.feature_importances_

    if len(importances) != len(FEATURE_LABELS):
        print(f"WARNING: Expected {len(FEATURE_LABELS)} features, got {len(importances)}")

    # Print table
    print(f"\n{'Feature':<20} {'Importance':>12}")
    print("-" * 34)
    pairs = sorted(zip(FEATURE_LABELS, importances), key=lambda x: x[1], reverse=True)
    for label, imp in pairs:
        print(f"{label:<20} {imp:>12.4f}")
    print("-" * 34)
    print(f"{'Total':<20} {sum(importances):>12.4f}")

    # Plot
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#378ADD", edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importance score")
    ax.set_title(f"Feature importance for PM2.5 calibration ({model_name})")

    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
