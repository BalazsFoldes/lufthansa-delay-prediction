import pandas as pd
import numpy as np
from src.config import (
    DATA_PATH, DATE_COL, TARGET, LEAKAGE_COLS,
    NUMERIC_COLS, CATEGORICAL_COLS
)


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"[load_raw] Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df


def report_leakage(df: pd.DataFrame) -> None:
    print("\n=== Leakage diagnostic ===")
    for col in LEAKAGE_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            overlap = df.groupby(col)[TARGET].mean().round(3).to_dict()
            print(f"{col} (cat) → delay rate per category: {overlap}")
        else:
            try:
                corr = df[col].corr(df[TARGET])
                print(f"{col} (num) → corr with target: {corr:+.4f}")
            except Exception:
                print(f"{col} → could not compute correlation")
    print()


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[drop_leakage] Dropped {len(cols_to_drop)} leakage columns: {cols_to_drop}")
    return df


def report_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\n=== Missing values ===")
    if missing.empty:
        print("Nothing found.")
    else:
        for col, n in missing.items():
            print(f"{col}: {n} ({n / len(df):.1%})")
    print()
    return missing


def time_split(df: pd.DataFrame, test_month: int = 5, val_month: int = 4):
    """
    Time-based train / validation / test split.

    Why time-based and not random?
    A random split would allow the model to see future flights during
    training — impossible in real deployment. We hold out the most recent
    data to simulate a realistic production scenario.

      train : jan – mar 2024
      val   : apr 2024
      test  : may 2024
    """
    month = df[DATE_COL].dt.month
    train = df[month < val_month].copy()
    val = df[month == val_month].copy()
    test = df[month == test_month].copy()

    print(f"[time_split] Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"Train target rate: {train[TARGET].mean():.2%}")
    print(f"Val target rate: {val[TARGET].mean():.2%}")
    print(f"Test target rate: {test[TARGET].mean():.2%}")
    return train, val, test


def get_feature_matrix(df: pd.DataFrame):
    drop = [DATE_COL, TARGET] + [c for c in LEAKAGE_COLS if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    y = df[TARGET]
    return X, y


if __name__ == "__main__":
    df = load_raw()
    report_leakage(df)
    df = drop_leakage(df)
    report_missing(df)
    train, val, test = time_split(df)