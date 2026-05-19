import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import NUMERIC_COLS, CATEGORICAL_COLS, DATE_COL, TARGET, RANDOM_STATE


# 1. feature engineering
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Instead we extract cyclical and categorical signals:
      - month        : seasonal patterns (winter delays, summer traffic)
      - day_of_week  : weekend vs weekday operational load
      - is_weekend   : binary flag
      - hour_sin/cos : encode hour cyclically so 23 and 0 are 'close'
    """
    if DATE_COL not in df.columns:
        return df

    dt = df[DATE_COL]
    df = df.copy()

    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek  # 0 = monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    #cyclical encoding of hour
    hour = dt.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df = df.drop(columns=[DATE_COL])
    return df


# 2. imputation
def impute(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
):
    """
    Fit imputers on TRAIN only, then transform val and test.

    Why fit on train only?
    -----------------------
    Fitting on the full dataset would let information from val/test leak into
    the imputed values (e.g. the median would shift). This is a subtle but
    real form of data leakage.
    """
    num_cols = [c for c in NUMERIC_COLS if c in train.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in train.columns]

    num_imputer = SimpleImputer(strategy=numeric_strategy)
    cat_imputer = SimpleImputer(strategy=categorical_strategy)

    #fit on train, transform all splits
    train[num_cols] = num_imputer.fit_transform(train[num_cols])
    val[num_cols] = num_imputer.transform(val[num_cols])
    test[num_cols] = num_imputer.transform(test[num_cols])

    if cat_cols:
        train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
        val[cat_cols] = cat_imputer.transform(val[cat_cols])
        test[cat_cols] = cat_imputer.transform(test[cat_cols])

    print(f"[impute] Numeric imputer (strategy={numeric_strategy}) fitted on {len(train):,} rows.")
    return train, val, test


# 3. encoding

def encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    One-hot encode categorical columns.
    Fitted on X_train only; unknown categories in val/test are ignored.
    """
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    if not cat_cols:
        return X_train, X_val, X_test

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[cat_cols])

    def _apply(df):
        encoded = encoder.transform(df[cat_cols])
        feature_names = encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        return pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    X_train = _apply(X_train)
    X_val = _apply(X_val)
    X_test = _apply(X_test)

    print(f"[encode] One-hot encoded {len(cat_cols)} columns → {len(encoder.get_feature_names_out()):} dummies created.")
    return X_train, X_val, X_test


# 4. scaling  (only needed for Logistic Regression)

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    StandardScaler fitted on X_train only.
    Tree-based models (RF, XGBoost, LightGBM) do not require scaling;
    this is applied separately for Logistic Regression.
    """
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_val_sc = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns, index=X_val.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    return X_train_sc, X_val_sc, X_test_sc, scaler


# 5. full preprocessing pipeline (convenience wrapper)

def full_preprocess(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """
    End-to-end preprocessing:
      1. Add time features
      2. Separate X / y
      3. Impute
      4. Encode categoricals
      5. Return both raw and scaled versions of X

    Returns
    -------
    X_train, X_val, X_test             : DataFrames (unscaled, for tree models)
    X_train_sc, X_val_sc, X_test_sc    : DataFrames (scaled, for linear models)
    y_train, y_val, y_test             : Series
    """
    # 1. time features
    train = add_time_features(train)
    val = add_time_features(val)
    test = add_time_features(test)

    # 2. separate target
    y_train = train.pop(TARGET)
    y_val = val.pop(TARGET)
    y_test = test.pop(TARGET)

    # 3. impute (fit on train only)
    train, val, test = impute(train, val, test)

    # 4. encode
    X_train, X_val, X_test = encode_categoricals(train, val, test)

    # 5. scale
    X_train_sc, X_val_sc, X_test_sc, _ = scale_features(X_train, X_val, X_test)

    print(f"[full_preprocess] Final feature matrix: {X_train.shape[1]} features.")
    return (
        X_train, X_val, X_test,
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
    )