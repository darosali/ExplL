import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import shap
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve, PrecisionRecallDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


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

def sample_shap_data(X, y, n_samples=2000, fraud_ratio=0.3):

    fraud_indices = np.where(y == 1)[0]
    normal_indices = np.where(y == 0)[0]

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    fraud_sample_indices = np.random.choice(fraud_indices, n_fraud, replace=False)
    normal_sample_indices = np.random.choice(normal_indices, n_normal, replace=False)

    selected_indices = np.concatenate([fraud_sample_indices, normal_sample_indices])
    np.random.shuffle(selected_indices)

    X_shap = X[selected_indices]
    y_shap = y[selected_indices]

    return X_shap, y_shap

def get_shap_explanations(model, X_train, X_shap, feature_names, filename='output/shap_importance.png'):

    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(X_shap, torch.Tensor):
        X_shap = torch.tensor(X_shap, dtype=torch.float32)

    model.eval()

    explainer = shap.DeepExplainer(model, X_train)
    shap_values = explainer.shap_values(X_shap)

    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss for binary classification.
        Args:
            alpha (float): Weighting factor for the rare class (default: 0.25)
            gamma (float): Focusing parameter (default: 2)
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute BCE Loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute probabilities
        pt = torch.sigmoid(inputs)
        pt = pt * targets + (1 - pt) * (1 - targets)  # Get the probability for the true class

        # Apply the focal loss factor
        focal_factor = (1 - pt) ** self.gamma
        loss = self.alpha * focal_factor * BCE_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def scale_data(df: pl.DataFrame) -> pl.DataFrame:
    
    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()
    # minmax_scaler = MinMaxScaler()

    df = df.with_columns([
        pl.col('error').fill_null("No error"),
        pl.col('merchant.state').fill_null("Unknown"),
    ])

    df = df.with_columns((
                          pl.col('same_state').cast(pl.Int8),
                          pl.col('same_city').cast(pl.Int8),
                          pl.when(pl.col('is_fraud') == "Yes").then(1).otherwise(0).alias('is_fraud_bin')))

    # One-hot encoding
    df = df.to_dummies(columns=['card.chip', 'card.type', 'card.brand'])

    # Frequency encoding
    for column in ['error', 'card.mcc', 'merchant.state', 'merchant.city']:
        freq_df = df.group_by(column).agg(pl.len().alias(f'{column}_freq'))
        df = df.join(freq_df, on=column, how='left')

    df_dict = df.to_dict(as_series=False)
    print("Encodings done")
    # Log-transform + StandardScaler
    log_standard_cols = ['amount_usd']
    for col in log_standard_cols:
        df_dict[col] = np.log1p(df_dict[col])
        df_dict[col] = standard_scaler.fit_transform(np.array(df_dict[col]).reshape(-1, 1)).flatten()

    # StandardScaler
    standard_cols = ['customer.age', 'score']
    for col in standard_cols:
        df_dict[col] = standard_scaler.fit_transform(np.array(df_dict[col]).reshape(-1, 1)).flatten()

    # Log-transform + RobustScaler
    log_robust_cols = ['total_debt', 'credit_limit', 'error_freq', 'card.mcc_freq', 'merchant.state_freq', 'merchant.city_freq']
    for col in log_robust_cols:
        df_dict[col] = np.log1p(df_dict[col])
        df_dict[col] = robust_scaler.fit_transform(np.array(df_dict[col]).reshape(-1, 1)).flatten()

    # RobustScaler
    robust_cols = [
        'seen_count_last_2_days', 'seen_count_mcc_last_2_days',
        'seen_count_last_7_days', 'seen_count_mcc_last_7_days',
        'seen_count_last_30_days', 'seen_count_mcc_last_30_days',
        'log_timediff'
    ]
    for col in robust_cols:
        df_dict[col] = robust_scaler.fit_transform(np.array(df_dict[col]).reshape(-1, 1)).flatten()

    df_scaled = pl.DataFrame(df_dict)

    return df_scaled

def transform_exp1(df):

    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()

    df = df.with_columns([
        pl.col('error').fill_null("No error"),
        pl.col('merchant.state').fill_null("Unknown"),
    ])

    df = df.with_columns((
        pl.col('same_state').cast(pl.Int8),
        pl.col('same_city').cast(pl.Int8),
        pl.when(pl.col('is_fraud') == "Yes").then(1).otherwise(0).alias('is_fraud_bin')))

    # One-hot encoding
    df = df.to_dummies(columns=['error', 'card.mcc', 'card.chip', 'card.type', 'card.brand', 'merchant.state'])

    # Frequency encoding
    for column in ['merchant.city']:
        freq_df = df.group_by(column).agg(pl.len().alias(f'{column}_freq'))
        df = df.join(freq_df, on=column, how='left')

    df_dict = df.to_dict(as_series=False)

    # Log-transform + StandardScaler
    log_standard_cols = ['amount_usd']
    for col in log_standard_cols:
        df_dict[col] = np.log1p(df_dict[col])
        df_dict[col] = standard_scaler.fit_transform(np.array(df_dict[col]).reshape(-1, 1)).flatten()

    # StandardScaler
    standard_cols = ['customer.age', 'score']
    for col in standard_cols:
        df_dict[col] = standard_scaler.fit_transform(np.array(df_dict[col]).reshape(-1, 1)).flatten()

    # Log-transform + RobustScaler
    log_robust_cols = ['total_debt', 'credit_limit', 'merchant.city_freq']
    for col in log_robust_cols:
        df_dict[col] = np.log1p(df_dict[col])
        df_dict[col] = robust_scaler.fit_transform(np.array(df_dict[col]).reshape(-1, 1)).flatten()

    # RobustScaler
    robust_cols = [
        'seen_count_last_2_days', 'seen_count_mcc_last_2_days',
        'seen_count_last_7_days', 'seen_count_mcc_last_7_days',
        'seen_count_last_30_days', 'seen_count_mcc_last_30_days',
        'log_timediff'
    ]
    for col in robust_cols:
        df_dict[col] = robust_scaler.fit_transform(np.array(df_dict[col]).reshape(-1, 1)).flatten()

    print("done for good")
    df_scaled = pl.DataFrame(df_dict)
    df_scaled.write_parquet('ibm_fraud_transformed2.pq')

    print(len(df_scaled.columns))

    return df_scaled

if __name__ == "__main__":
    df = pl.read_parquet(r'ibm_fraud_processed.pq')
    df = transform_exp1(df)
    print(df.columns)