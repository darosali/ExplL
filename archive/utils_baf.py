import polars as pl
from matplotlib import pyplot as plt
import numpy as np
from lightgbm import early_stopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score, roc_curve, PrecisionRecallDisplay, precision_recall_curve, auc
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
import xgboost as xgb
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def transform_data_baf(df):
    df = df.to_dummies(columns=['device_os', 'source', 'housing_status', 'employment_status', 'payment_type'])
    return df


def run_xgb(X_train, X_test, y_train, y_test, weight=None):
    # X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    # X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)
    print(X_train.shape)
    scale_pos_weight = None
    if weight:
        scale_pos_weight = int(np.sqrt(((y_train == 0).sum() / (y_train == 1).sum())))
        print(f"Fraud class weight: {scale_pos_weight}")
    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    xgb_model.fit(X_train, y_train)
    y_probs = xgb_model.predict_proba(X_test)[:, 1]
    tpr, best_threshold = tpr_at_fixed_fpr(y_test, y_probs)
    print(f"TPR {tpr} for threshold {best_threshold}")
    y_pred, y_probs, threshold = evaluate_performance(xgb_model, X_train, X_test, y_train, y_test)
    indices = np.where((y_pred == 1) & (y_test == 1))[0]

    return indices, y_probs, xgb_model


def run_xgb_rus(X_train, X_val, X_test, y_train, y_val, y_test):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0,
                                  subsample=1.0, n_estimators=200, min_child_weight=3, max_depth=6, learning_rate=0.5, gamma=0.1, colsample_bytree=0.6)
    X_train_resampled, y_train_resampled = undersample(X_train, y_train)
    xgb_model.fit(X_train_resampled, y_train_resampled)
    _, _ = evaluate_performance(xgb_model, X_train, X_val, X_test, y_train, y_val, y_test)
    # y_pred = xgb_model.predict(X_test)
    # print("Training data results:")
    # print_results(y_pred, y_test)

def run_xgb_smote(X_train, X_val, X_test, y_train, y_val, y_test):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0,
                                  subsample=1.0, n_estimators=200, min_child_weight=3, max_depth=6, learning_rate=0.5, gamma=0.1, colsample_bytree=0.6)
    X_train_resampled, y_train_resampled = oversample(X_train, y_train)
    xgb_model.fit(X_train_resampled, y_train_resampled)
    _, _ = evaluate_performance(xgb_model, X_train, X_val, X_test, y_train, y_val, y_test)


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

def undersample(X, y):
    desired_proportion = 0.3
    fraud_samples = (y==1).sum()
    total_samples = int(fraud_samples / desired_proportion)
    undersampler = RandomUnderSampler(sampling_strategy={0: total_samples - fraud_samples, 1: fraud_samples}, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    return X_resampled, y_resampled

def oversample(X, y):
    desired_proportion = 0.3
    legit_samples = (y == 0).sum()
    fraud_samples = int(legit_samples * desired_proportion)
    smote = SMOTE(sampling_strategy={0: legit_samples, 1: fraud_samples}, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def tpr_at_fixed_fpr(y_true, y_scores, fpr_target=0.05):
    fprs, tprs, thresholds = roc_curve(y_true, y_scores)
    #tpr_at_fpr = tpr[np.where(fpr <= fpr_target)[0][-1]] # Last TPR before exceeding target FPR
    threshold = np.min(thresholds[fprs == max(fprs[fprs < 0.05])])
    recall = np.max(tprs[fprs == max(fprs[fprs < 0.05])])
    return recall, threshold


def evaluate_performance(model, X_train, X_test, y_train, y_test):
    output_lines = []

    def evaluate_predictions(X, y, label, threshold=None):
        y_probs = model.predict_proba(X)[:, 1]
        y_pred = (y_probs >= threshold).astype(int) if threshold else model.predict(X)
        return f"{label}:\n" + print_results(y_pred, y, y_probs)

    # Without threshold optimization
    output_lines.append("Results without threshold optimization\n")
    output_lines.append(evaluate_predictions(X_train, y_train, "Train data"))
    output_lines.append(evaluate_predictions(X_test, y_test, "Test data"))

    # With threshold optimization
    y_probs = model.predict_proba(X_test)[:, 1]
    tpr, best_threshold = tpr_at_fixed_fpr(y_test, y_probs)
    output_lines.append(f"\nBest threshold: {best_threshold:.4f}\n")

    output_lines.append("Results with threshold optimization\n")
    output_lines.append(evaluate_predictions(X_train, y_train, "Train data", best_threshold))
    output_lines.append(evaluate_predictions(X_test, y_test, "Test data", best_threshold))

    with open('output/performance_evaluation.txt', 'w') as f:
        f.writelines(output_lines)

    return (y_probs >= best_threshold).astype(int), y_probs, best_threshold


def print_results(y_pred, y_true, y_proba):
    tpr, _ = tpr_at_fixed_fpr(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)[1]
    recall = recall_score(y_true, y_pred, average=None)[1]
    f1 = f1_score(y_true, y_pred, average=None)[1]

    results = (
        f"TPR at 5% FPR: {tpr:.4f}\n"
        f"Confusion Matrix:\n{cm}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
    )
    return results

def save_prc(y_test, y_probs_test, filename="output/PR_curve.png"):
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs_test)
    display = PrecisionRecallDisplay(precision=precision, recall=recall)

    display.plot()
    plt.title(f"Precision-Recall Curve")
    plt.savefig(filename)
    plt.close()


def save_roc_auc_curve(y_test, y_probs, filename="output/roc_auc_curve.png"):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def show_importance(importance, features, filename='output/feature_importances_xgb.txt'):
    output_lines = []
    # Create a list of (feature_name, importance_value) tuples
    feature_importances = [(feature, importance.get(f"f{i}", 0)) for i, feature in enumerate(features)]
    # Sort the list by importance values in descending order
    feature_importances_sorted = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Store the sorted feature importances
    output_lines = [f"{feature}: {importance_value:.4f}\n" for feature, importance_value in feature_importances_sorted]

    # Write feature importances to a file
    with open(filename, 'w') as f:
        f.writelines(output_lines)

def get_shap_explanations(model, X_train, X_shap, feature_names, filename='output/shap_importance.png'):
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer.shap_values(X_shap)

    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    # Save the summary plot to a file
    plt.savefig(filename)
    plt.close()

def save_confusion_matrix(y_true, y_pred, labels=None, filename="output/confusion_matrix.png"):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        colorbar=False
    )

    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()