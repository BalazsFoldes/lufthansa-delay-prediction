import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.config import FIGURES_DIR, RESULTS_DIR


# -------------------------------------------------------------------------
# 1. single-model evaluation
# -------------------------------------------------------------------------

def evaluate_model(model, X, y, name: str, threshold: float = 0.50):
    """
    Evaluate one model on one dataset split at a fixed decision threshold.
    """
    y_prob = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else np.zeros(len(y))
    )
 
    if y_prob.sum() > 0:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
 
    return {
        "model": name,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, y_prob) if y_prob.sum() > 0 else 0.5, 4),
        "threshold": round(threshold, 2),
        "_y_pred": y_pred,
        "_y_prob": y_prob,
    }



# -------------------------------------------------------------------------
# 2. threshold search(runs on the VALIDATION set only)
# -------------------------------------------------------------------------
 
def find_best_threshold(model, X_val, y_val, name: str) -> float:
    """
    Sweep thresholds 0.05 → 0.50 on the VALIDATION set and return the
    value that maximises F1.
    """
    if not hasattr(model, "predict_proba"):
        return 0.50
 
    y_prob = model.predict_proba(X_val)[:, 1]
 
    best_thresh, best_f1 = 0.50, 0.0
 
    for t in np.arange(0.05, 0.55, 0.05):
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_val, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
 
    best_thresh = round(float(best_thresh), 2)
    print(f"[{name}] best threshold on val set: {best_thresh} (val F1={best_f1:.4f})")
    return best_thresh


# -------------------------------------------------------------------------
# 3. evaluate all models on test set
# -------------------------------------------------------------------------

def evaluate_all(fitted_models: dict, X_val,  X_val_sc,  y_val, X_test, X_test_sc, y_test) -> pd.DataFrame:
    """
    For every model:
      1. Find the F1-maximising threshold on the VALIDATION set.
      2. Evaluate on the TEST set using that threshold.
 
    This two-step approach ensures the test set is never used for any
    tuning decision — the threshold included.
    """
    rows = []
    preds = {}
 
    print("\n--- threshold search on validation set ---")
    for name, model in fitted_models.items():
        needs_scaling = name in ["Logistic Regression"]
 
        #find best threshold on the val set
        X_v = X_val_sc if needs_scaling else X_val
        best_thresh = find_best_threshold(model, X_v, y_val, name)
 
        #evaluate on the test set with that threshold
        X_t = X_test_sc if needs_scaling else X_test
        result = evaluate_model(model, X_t, y_test, name, threshold=best_thresh)
 
        preds[name] = (result.pop("_y_pred"), result.pop("_y_prob"))
        rows.append(result)
 
    df_results = pd.DataFrame(rows).set_index("model")
    return df_results, preds


# -------------------------------------------------------------------------
# 4. save results as a CSV / JSON
# -------------------------------------------------------------------------

def save_results(df_results: pd.DataFrame, filename: str = "model_comparison.csv"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    df_results.to_csv(path)
    print(f"[save_results] Results saved to {path}")


# -------------------------------------------------------------------------
# 5. visualisations
# -------------------------------------------------------------------------

def _savefig(name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved: {path}")


def plot_model_comparison(df_results: pd.DataFrame):
    """
    Grouped bar chart comparing all models across key metrics.
    """
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df_plot = df_results[metrics].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_plot))
    width = 0.15

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df_plot[metric], width, label=metric)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_plot.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0.96, color="grey", linestyle="--", linewidth=0.8, label="Majority baseline accuracy")
    ax.set_ylabel("Score")
    plt.tight_layout()
    _savefig("model_comparison")
    plt.close()


def plot_roc_curves(fitted_models: dict, preds: dict, y_test):
    """
    ROC curves for all models on a single plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (_, y_prob) in preds.items():
        if y_prob.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig("roc_curves")
    plt.close()


def plot_confusion_matrices(preds: dict, y_test):
    """
    Grid of confusion matrices for all models.
    """
    n = len(preds)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (name, (y_pred, _)) in enumerate(preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", ax=axes[i],
            cmap="Blues", cbar=False,
            xticklabels=["No delay", "Delay"],
            yticklabels=["No delay", "Delay"],
        )
        axes[i].set_title(name, fontsize=9, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig("confusion_matrices")
    plt.close()


def plot_feature_importance(fitted_models: dict, feature_names: list, top_n: int = 15):
    """
    Feature importance for tree-based models (Random Forest, XGBoost, LightGBM).
    Side-by-side bars so we can compare which features each model values.
    """
    tree_models = ["Random Forest", "XGBoost", "LightGBM"]
    available = [m for m in tree_models if m in fitted_models]

    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    for ax, name in zip(axes, available):
        model = fitted_models[name]
        importances = model.feature_importances_
        idx = np.argsort(importances)[-top_n:]
        ax.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color="steelblue",
        )
        ax.set_title(f"{name}\nTop {top_n} Features", fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance")

    plt.suptitle("Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("feature_importances")
    plt.close()


def print_summary_table(df_results: pd.DataFrame):
    print("\n" + "=" * 75)
    print("MODEL COMPARISON — TEST SET")
    print("=" * 75)
    print(df_results.to_string())
    print("=" * 75)
    best = df_results["f1"].idxmax()
    print(f"\n  → Best model by F1: {best}  (F1={df_results.loc[best,'f1']:.4f}"
      f"/ threshold={df_results.loc[best,'threshold']})")
    print()