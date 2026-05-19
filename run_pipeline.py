import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_raw, report_leakage, drop_leakage, report_missing, time_split
from src.preprocessing import full_preprocess
from src.models import train_all, MODELS
from src.evaluation import (
    evaluate_all, save_results,
    plot_model_comparison, plot_roc_curves,
    plot_confusion_matrices, plot_feature_importance,
    print_summary_table,
)


def main():
    print("=" * 65)
    print("  Aviation Delay Prediction — Full Pipeline")
    print("=" * 65)

    #1. load
    print("[1/7] Loading data...")
    df = load_raw()

    #2. leakage report
    print("[2/7] Leakage report...")
    report_leakage(df)

    #3. drop leakage columns
    print("[3/7] Dropped leakage columns...")
    df = drop_leakage(df)

    #4. missing value report
    print("[4/7] Missing value report...")
    report_missing(df)

    #time-based split
    train, val, test = time_split(df)

    #preprocessing
    (
        X_train, X_val, X_test,
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
    ) = full_preprocess(train, val, test)

    #5. train
    print("\n--- [5/7] Training models ---")
    fitted_models = train_all(X_train, X_val, X_train_sc, X_val_sc, y_train, y_val)

    #6. evaluate
    print("\n--- [6/7] Evaluating on test set ---")
    df_results, preds = evaluate_all(
    fitted_models,
    X_val, X_val_sc, y_val,
    X_test, X_test_sc, y_test
    )
    print_summary_table(df_results)

    #save results
    save_results(df_results)

    #7. plots
    print("\n--- [7/7] Generating plots ---")
    feature_names = list(X_train.columns)
    plot_model_comparison(df_results)
    plot_roc_curves(fitted_models, preds, y_test)
    plot_confusion_matrices(preds, y_test)
    plot_feature_importance(fitted_models, feature_names)

    print("\n✓ Pipeline complete. Results in outputs/")


if __name__ == "__main__":
    main()