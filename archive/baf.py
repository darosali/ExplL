from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score, roc_curve, auc, roc_auc_score, PrecisionRecallDisplay, precision_recall_curve, make_scorer
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from utils_baf import *
from custom_scorer_module import tpr_at_fixed_fpr_scorer
import argparse


def parameters_tuning(X_train, y_train):

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2, 0.5],
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 6, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.75, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.75, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5]
    }
    #my_scorer = make_scorer(tpr_at_fixed_fpr_scorer, greater_is_better=True)
    kfolds = StratifiedKFold(n_splits=5, shuffle=False)
    random_search = RandomizedSearchCV(model, param_grid, n_iter=250, scoring=tpr_at_fixed_fpr_scorer, cv=kfolds, verbose=3, n_jobs=-1, random_state=42)
    print("Here3")
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    with open("output/best_params_baf.txt", 'w') as f:
        f.write(str(best_params))
    return best_model, best_params

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
    df = pl.scan_csv(filename).collect()
    df = transform_data_baf(df)
    return df

def split_data(df):
    df_train = df.filter(pl.col('month') < 6)
    df_val = df.filter((pl.col('month') >= 6) & (pl.col('month') < 7))
    df_test = df.filter((pl.col('month') >= 7))
    return df_train, df_val, df_test

def run_xgb(X_train, X_test, y_train, y_test, params, weights=False):
    scale_pos_weight = None
    if weights:
        scale_pos_weight = int(np.sqrt(((y_train == 0).sum() / (y_train == 1).sum())))
    xgb_model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train, y_train)
    y_pred_test_threshold, y_probs_test, best_threshold = (
        evaluate_performance(xgb_model, X_train, X_test, y_train, y_test)
    )
    indices_TP = np.where((y_pred_test_threshold == 1) & (y_test == 1))[0]
    indices_FN = np.where((y_pred_test_threshold == 0) & (y_test == 1))[0]

    return indices_TP, indices_FN, y_probs_test, best_threshold, xgb_model


if __name__ == '__main__':

    params = parse_args()
    df = load_data("Variant III.csv")
    feature_columns = df.columns
    df_train, df_val, df_test = split_data(df)

    target_column = 'fraud_bool'
    y_train = df_train.select(target_column).to_numpy().flatten()
    X_train = df_train.drop(target_column).to_numpy()
    y_test = df_test.select(target_column).to_numpy().flatten()
    X_test = df_test.drop(target_column).to_numpy()

    if params['tuning'] == 1:
        print("bebr")
        parameters_tuning(X_train, y_train)
        exit(0)
    print("Here")
    params.pop('tuning', None)
    indices_TP, indices_FN, y_probs_test, best_threshold, xgb_model = run_xgb(X_train, X_test, y_train, y_test, params)

    save_prc(y_test, y_probs_test)
    save_roc_auc_curve(y_test, y_probs_test)

    importance = xgb_model.get_booster().get_score(importance_type='weight')

    X_shap = X_test
    print("Here2")
    #get_shap_explanations(xgb_model, X_train, X_shap, feature_columns, 'shap_importance_test_data.png')
    y_pred = (y_probs_test >= best_threshold).astype(int)
    save_confusion_matrix(y_test, y_pred)


