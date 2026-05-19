import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, TARGET, DATE_COL, LEAKAGE_COLS


def _savefig(name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------------
# 1. target distribution
# ------------------------------------------------------------------------

def plot_target_distribution(df: pd.DataFrame):
    counts = df[TARGET].value_counts()
    labels = ["No delay (≤15 min)", "Delay (>15 min)"]
    colors = ["#4C9BE8", "#E8614C"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)
    axes[0].set_title("Target Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")

    # pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Class Ratio", fontweight="bold")

    plt.suptitle(
        f"Severe class imbalance: {counts[1]/len(df):.1%} positive class  →  "
        "accuracy alone is misleading",
        fontsize=10, color="darkred"
    )
    plt.tight_layout()
    _savefig("target_distribution")
    plt.show()


# ------------------------------------------------------------------------
# 2. missing values heatmap
# ------------------------------------------------------------------------

def plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(7, 3))
    missing_pct = (missing / len(df) * 100).round(1)
    bars = ax.barh(missing.index, missing_pct, color="#E8A24C")
    for bar, pct in zip(bars, missing_pct):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{pct}%", va="center", fontsize=10)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column", fontweight="bold")
    plt.tight_layout()
    _savefig("missing_values")
    plt.show()


# ------------------------------------------------------------------------
# 3. leakage correlation plot
# ------------------------------------------------------------------------

def plot_leakage_correlations(df: pd.DataFrame):
    """
    Shows correlation of leakage candidates vs. clean features with the target.
    Dramatically illustrates WHY we must drop those columns.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corrwith(df[TARGET]).drop(TARGET, errors="ignore")
    corr = corr.sort_values(key=abs, ascending=False)

    colors = [
        "#E8614C" if col in LEAKAGE_COLS else "#4C9BE8"
        for col in corr.index
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with target")
    ax.set_title("Feature Correlations with Target\n(red = leakage columns — must be removed)",
                 fontweight="bold")

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#E8614C", label="Leakage — excluded"),
        Patch(facecolor="#4C9BE8", label="Clean feature — kept"),
    ]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    _savefig("leakage_correlation")
    plt.show()


# ------------------------------------------------------------------------
# 4. numeric feature distributions by target
# ------------------------------------------------------------------------

def plot_numeric_distributions(df: pd.DataFrame, cols: list = None):
    """
    KDE / boxplot for each numeric feature, split by target class.
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
        cols = [c for c in cols if c not in [TARGET] + LEAKAGE_COLS]

    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        for label, grp in df.groupby(TARGET):
            axes[i].hist(
                grp[col].dropna(), bins=30, alpha=0.5,
                label="No delay" if label == 0 else "Delay",
                density=True
            )
        axes[i].set_title(col, fontsize=9)
        axes[i].legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Target Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("numeric_distributions")
    plt.show()


# ------------------------------------------------------------------------
# 5. delays over time
# ------------------------------------------------------------------------

def plot_delays_over_time(df: pd.DataFrame):
    """
    Delay rate per month and per hour of day.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["month"] = df[DATE_COL].dt.month
    df["hour"] = df[DATE_COL].dt.hour

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # by month
    monthly = df.groupby("month")[TARGET].mean() * 100
    axes[0].bar(monthly.index, monthly.values, color="#4C9BE8", edgecolor="white")
    axes[0].set_title("Delay Rate by Month", fontweight="bold")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Delay rate (%)")
    axes[0].set_xticks(monthly.index)

    # by hour
    hourly = df.groupby("hour")[TARGET].mean() * 100
    axes[1].plot(hourly.index, hourly.values, marker="o", color="#E8614C", linewidth=2)
    axes[1].fill_between(hourly.index, hourly.values, alpha=0.15, color="#E8614C")
    axes[1].set_title("Delay Rate by Hour of Day", fontweight="bold")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Delay rate (%)")

    plt.suptitle("Temporal Patterns in Delays", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("delays_over_time")
    plt.show()


# ------------------------------------------------------------------------
# 6. correlation heatmap (clean features only)
# ------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    clean = [c for c in num_cols if c not in LEAKAGE_COLS]

    fig, ax = plt.subplots(figsize=(14, 10))
    corr_matrix = df[clean].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=False,
        cmap="coolwarm", center=0, ax=ax,
        linewidths=0.3, cbar_kws={"shrink": 0.7}
    )
    ax.set_title("Correlation Matrix — Clean Features", fontweight="bold")
    plt.tight_layout()
    _savefig("correlation_heatmap")
    plt.show()