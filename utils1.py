import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score, roc_curve, auc, roc_auc_score, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
import shap
import polars as pl
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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

def f1_importance(model, feature_names, X_train, X_test, y_train, y_test):
    for feature in feature_names:

        feature_index = feature_names.index(feature)

        X_train_mod = np.delete(X_train, feature_index, axis=1)
        X_test_mod = np.delete(X_test, feature_index, axis=1)

        model.fit(X_train_mod, y_train)
        y_pred_mod = model.predict(X_test_mod)
        
        # Calculate F1 score
        f1_mod = f1_score(y_test, y_pred_mod)

        print(f"F1 score after dropping {feature}: {f1_mod}")


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


def show_fraud_cases(df: pl.DataFrame, timestamps: list) -> pl.DataFrame:
    timestamps_series = pl.Series(timestamps)
    filtered_df = df.filter((pl.col("timestamp").is_in(timestamps_series)) & (pl.col("is_fraud") == "Yes"))
    return filtered_df

def find_records_around_timestamp(df: pl.DataFrame, target_timestamp: int, customer_id: int, N: int) -> pl.DataFrame:
    customer_df = df.filter(pl.col("customer.id") == customer_id)
    target_index = customer_df.get_column("timestamp").to_list().index(target_timestamp)
    start_index = max(0, target_index - N)
    end_index = min(len(customer_df), target_index + N + 1)
    result_df = customer_df[start_index:end_index]
    return result_df