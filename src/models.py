from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_STATE, POS_CLASS_RATIO


# model registry
# ---------------------------------------------------------------------------
# each entry: (model_object, needs_scaling)
#   needs_scaling=True  → use the StandardScaler-transformed X
#   needs_scaling=False → use raw X (tree-based models)
#
# Why these models?
# -----------------
# DummyClassifier  : true baseline; always predicts majority class (no delay).
#                    Any real model must beat this — with 96% class-0 rate it
#                    achieves 96% accuracy, which puts accuracy into perspective.
#
# LogisticRegression: linear baseline; interpretable coefficients; requires
#                    scaling; uses class_weight to handle imbalance.
#
# DecisionTreeClassifier: non-linear baseline; human-readable rules;
#                    no scaling needed; prone to overfitting → controlled via
#                    max_depth.
#
# RandomForestClassifier: ensemble of trees; robust to noise; provides
#                    reliable feature importances; handles imbalance via
#                    class_weight.
#
# XGBClassifier   : gradient boosting; typically best on tabular data;
#                   scale_pos_weight corrects for class imbalance.
#
# LGBMClassifier  : similar to XGBoost but faster; natively handles missing
#                   values (relevant because visibility and maintenance cols
#                   have ~8–12% missing). is_unbalance flag used.

MODELS = {
    "Dummy (majority)": (
        DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        False,
    ),
    "Logistic Regression": (
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        True,   # needs scaling
    ),
    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        False,
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        False,
    ),
    "XGBoost": (
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=POS_CLASS_RATIO,   # handles 96/4 imbalance
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
        False,
    ),
    "LightGBM": (
        LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
        False,
    ),
}


# training function ---------------------------------

def train_all(
    X_train, X_val,
    X_train_sc, X_val_sc,
    y_train, y_val,
):
    """
    train all models in the registry and return fitted model objects.

    parameters
    ----------
    X_train / X_val       : unscaled feature DataFrames (for tree models)
    X_train_sc / X_val_sc : scaled feature DataFrames   (for linear models)
    y_train / y_val       : target Series

    Returns
    -------
    fitted_models : dict  {model_name: fitted_model_object}
    """
    fitted_models = {}

    for name, (model, needs_scaling) in MODELS.items():
        Xtr = X_train_sc if needs_scaling else X_train
        print(f"  Training: {name}...")
        model.fit(Xtr, y_train)
        fitted_models[name] = model

    print(f"\n[train_all] Trained {len(fitted_models)} models.")
    return fitted_models