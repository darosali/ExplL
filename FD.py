import polars as pl
import numpy as np
import shap
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score, roc_curve, auc, roc_auc_score, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from utils1 import *
import argparse

def parameters_tuning(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2, 0.5],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 6, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.75, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.75, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5]
    }
    kfolds = StratifiedKFold(n_splits=5, shuffle=False)
    random_search = RandomizedSearchCV(model, param_grid, n_iter=50, scoring='f1', cv=kfolds, verbose=2, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    with open("output/best_params.txt", 'w') as f:
        f.write(str(best_params))
    return best_model, best_params

def transform_data(df):

    df = df.with_columns([
        pl.col('error').fill_null("No error"),
        pl.col('merchant.state').fill_null("Unknown"),
    ])
    print("done2")
    # Apply log to amount_usd data
    df = df.with_columns((pl.col('amount_usd')+1).log().alias('amount_usd_log'))
    print("done3")
    # Add a relative spend column (amount_usd / credit_limit)
    # df = df.with_columns((pl.col('amount_usd') / (pl.col('credit_limit') + 1e-5)).alias('relative_spend'))
    # Extract day of the week from date column to a seperate column
    df = df.with_columns(pl.col('date').dt.weekday().alias('weekday'))
    print("done4")
    # Create age buckets (pro zajimovost)
    df = df.with_columns(
        pl.when(pl.col('customer.age') < 18).then(0)
        .when((pl.col('customer.age') >= 18) & (pl.col('customer.age') < 25)).then(1)
        .when((pl.col('customer.age') >= 25) & (pl.col('customer.age') < 35)).then(2)
        .when((pl.col('customer.age') >= 35) & (pl.col('customer.age') < 45)).then(3)
        .when((pl.col('customer.age') >= 45) & (pl.col('customer.age') < 55)).then(4)
        .when((pl.col('customer.age') >= 55) & (pl.col('customer.age') < 65)).then(5)
        .otherwise(6).alias('age_bucket')
    )

    # Binary encoding for is_fraud, direction, same_state, and same_city
    df = df.with_columns((pl.when(pl.col('direction') == "outbound").then(1).otherwise(0).alias('direction_bin'),
                    pl.col('same_state').cast(pl.Int8),
                    pl.col('same_city').cast(pl.Int8),
                    pl.when(pl.col('is_fraud') == "Yes").then(1).otherwise(0).alias('is_fraud_bin')))
    # One-hot encoding
    df = df.to_dummies(columns=['card.chip', 'card.type', 'card.brand'])

    # Frequency encoding (https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.dataframe.group_by.GroupBy.agg.html)
    for column in ['error', 'card.mcc', 'merchant.state', 'merchant.city']:
        freq_df = df.groupby(column).agg(pl.count().alias(f'{column}_freq'))
        df = df.join(freq_df, on=column, how='left')

    return df

def f1_smart(y_true, y_pred) -> tuple[float, float]:
    """
    Smart calculation of F1 score that should be fast.

    Returns `max_f1, best_threshold`.
    """
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    max_f1 = 2 * fs[res_idx]
    best_threshold = (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2
    return max_f1, best_threshold


def evaluate_performance(model, X_train, X_val, X_test, y_train, y_val, y_test):
    output_lines = []

    def evaluate_predictions(X, y, label, threshold=None):
        y_probs = model.predict_proba(X)[:, 1]
        y_pred = (y_probs >= threshold).astype(int) if threshold else model.predict(X)
        return f"{label}:\n" + print_results(y_pred, y)

    # Without threshold optimization
    output_lines.append("Results without threshold optimization\n")
    output_lines.append(evaluate_predictions(X_train, y_train, "Train data"))
    output_lines.append(evaluate_predictions(X_val, y_val, "Val data"))
    output_lines.append(evaluate_predictions(X_test, y_test, "Test data"))

    # With threshold optimization
    y_probs_val = model.predict_proba(X_val)[:, 1]
    _, best_threshold = f1_smart(y_val, y_probs_val)
    output_lines.append(f"\nBest threshold: {best_threshold:.4f}\n")

    output_lines.append("Results with threshold optimization\n")
    output_lines.append(evaluate_predictions(X_train, y_train, "Train data", best_threshold))
    output_lines.append(evaluate_predictions(X_val, y_val, "Val data", best_threshold))
    output_lines.append(evaluate_predictions(X_test, y_test, "Test data", best_threshold))
    y_probs = model.predict_proba(X_test)[:, 1]

    with open('output/performance_evaluation.txt', 'w') as f:
        f.writelines(output_lines)

    return (y_probs >= best_threshold).astype(int), y_probs, best_threshold


def print_results(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)[1]
    recall = recall_score(y_true, y_pred, average=None)[1]
    f1 = f1_score(y_true, y_pred, average=None)[1]

    results = (
        f"Confusion Matrix:\n{cm}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
    )
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Run xgboost with custom hyperparameters')
    parser.add_argument('--tuning', type=int, default=0, help='Flag indicating tuning needed')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=6, help='Maximum depth')
    parser.add_argument('--min_child_weight', type=int, default=1, help='Minimum child weight')
    parser.add_argument('--subsample', type=float, default=1, help='Subsample ratio of the training instances')
    parser.add_argument('--colsample_bytree', type=float, default=1, help='Subsample ratio of the training instances')
    parser.add_argument('--gamma', type=float, default=0, help='Gamma')

    return vars(parser.parse_args())

def load_data(filename):
    df = pl.scan_parquet(filename).collect()
    # df = pl.scan_parquet(filename).filter(pl.col("date") >= datetime(2018, 1, 1)).collect()
    # df = pl.scan_parquet(filename).filter(pl.col("date") >= datetime(2018, 6, 1)).collect()
    df = transform_data(df)
    return df

def split_data(df):
    train_df = df.filter(pl.col("date") < datetime(2018, 1, 1))
    val_df = df.filter((pl.col("date") >= datetime(2018, 1, 1))
                       & (pl.col("date") < datetime(2019, 1, 1)))
    test_df = df.filter((pl.col("date") >= datetime(2019, 1, 1)))
    # train_df = df.filter(pl.col('date') < datetime(2018, 12, 1))
    # val_df = df.filter((pl.col('date') >= datetime(2018, 12, 1)) & (pl.col('date') < datetime(2018, 12, 15)))
    # test_df = df.filter((pl.col('date') >= datetime(2018, 12, 15)) & (pl.col('date') < datetime(2019, 1, 1)))
    # train_df = df.filter(pl.col('date') < datetime(2018, 10, 1))
    # val_df = df.filter((pl.col('date') >= datetime(2018, 10, 1)) & (pl.col('date') < datetime(2018, 11, 15)))
    # test_df = df.filter((pl.col('date') >= datetime(2018, 11, 15)) & (pl.col('date') < datetime(2019, 1, 1)))
    return train_df, val_df, test_df

def run_xgb(X_train, X_val, X_test, y_train, y_val, y_test, params, weights=False):
    scale_pos_weight = None
    if weights:
        scale_pos_weight = int(np.sqrt(((y_train == 0).sum() / (y_train == 1).sum())))
    xgb_model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train, y_train)
    y_pred_test_threshold, y_probs_test, best_threshold = (
        evaluate_performance(xgb_model, X_train, X_val, X_test, y_train, y_val, y_test)
    )
    indices_TP = np.where((y_pred_test_threshold == 1) & (y_test == 1))[0]
    indices_FN = np.where((y_pred_test_threshold == 0) & (y_test == 1))[0]

    return indices_TP, indices_FN, y_probs_test, best_threshold, xgb_model

if __name__ == "__main__":

    xgb_params = parse_args()

    # Load and split data
    df = load_data(r'ibm_fraud_processed.pq')
    train_df, val_df, test_df = split_data(df)

    feature_columns = ['amount_usd', 'score', 'total_debt', 'credit_limit', 'weekday', 'same_state', 'same_city',
                         'error_freq', 'card.mcc_freq', 'card.chip_Chip Transaction',
                         'card.chip_Online Transaction', 'card.chip_Swipe Transaction', 'merchant.state_freq',
                         'merchant.city_freq', 'is_known_merchant', 'seen_count_last_2_days', 'seen_count_last_7_days',
                         'seen_count_mcc_last_7_days', 'seen_count_last_30_days']
    target_column = 'is_fraud_bin'
    X_train, y_train = train_df.select(feature_columns).to_numpy(), train_df.select(target_column).to_numpy().flatten()
    X_val, y_val = val_df.select(feature_columns).to_numpy(), val_df.select(target_column).to_numpy().flatten()
    X_test, y_test = test_df.select(feature_columns).to_numpy(), test_df.select(target_column).to_numpy().flatten()

    if xgb_params['tuning'] == 1:
        print("bebr")
        parameters_tuning(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
        exit(0)
    # X_train, y_train = oversample(X_train, y_train)

    # Run XGBoost
    indices_TP, indices_FN, y_probs_test, best_threshold, xgb_model = run_xgb(X_train, X_val, X_test, y_train, y_val, y_test, xgb_params)

    save_prc(y_test, y_probs_test)
    save_roc_auc_curve(y_test, y_probs_test)

    true_positive_instances = test_df[indices_TP]
    false_negative_instances = test_df[indices_FN]

    true_positive_instances.write_parquet('true_positive_instances.pq')
    false_negative_instances.write_parquet('false_negative_instances.pq')

    fraudulent_transactions = test_df[indices_TP]

    # Get model native feature importance
    importance = xgb_model.get_booster().get_score(importance_type='weight')
    #show_importance(importance, feature_columns)

    # Get shap explanations
    X_shap = fraudulent_transactions.select(feature_columns).to_numpy()
    #get_shap_explanations(xgb_model, X_train, X_shap, feature_columns, 'shap_importance2.png')
    X_shap = X_test
    #get_shap_explanations(xgb_model, X_train, X_shap, feature_columns, 'shap_importance_test_data2.png')
    y_pred = (y_probs_test >= best_threshold).astype(int)
    save_confusion_matrix(y_test, y_pred)