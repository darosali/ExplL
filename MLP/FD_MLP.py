import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from datetime import datetime
from utils2 import *
import argparse


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):

        np.random.seed(1234)
        torch.manual_seed(1234)
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

class FraudDataset(Dataset):
    def __init__(self, data, labels):
        """
        data: numpy array of shape (num_samples, num_features)
        labels: numpy array of shape (num_samples,) or (num_samples, 1)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MLP(nn.Module):
    def __init__(self, input_size=383, hidden_units=[94, 26, 53, 20], output_size=1, activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_units[0]))
        layers.append(activation_fn)

        # Hidden layers
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(activation_fn)

        # Output layer
        layers.append(nn.Linear(hidden_units[-1], output_size))


        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a fraud detection MLP model.")

    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[16, 16], help="Sizes of hidden layers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer.")
    parser.add_argument("--activation_fn", type=str, default="ReLU", choices=["ReLU", "ELU", "Tanh", "LeakyReLU"],
                        help="Activation function (default: ReLU).")

    return vars(parser.parse_args())

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

def load_data_mlp(filename):
    df = pl.scan_parquet(filename).filter(pl.col("date") >= datetime(2018, 1, 1)).collect()
    # df = scale_data_Standart_Scaler(df)
    df = scale_data(df)
    return df

def load_data(filename):
    #df = pl.scan_parquet(filename).collect()
    # df = pl.scan_parquet(filename).filter(pl.col("date") >= datetime(2018, 1, 1)).collect()
    # df = pl.scan_parquet(filename).filter(pl.col("date") >= datetime(2018, 6, 1)).collect()
    df = pl.scan_parquet(filename).filter(pl.col("date") >= datetime(2018, 1, 1)).collect()
    # df = transform_data(df)
    df = scale_data(df)
    return df

def split_data(df):
    # train_df = df.filter(pl.col("date") < datetime(2018, 1, 1))
    # val_df = df.filter((pl.col("date") >= datetime(2018, 1, 1))
    #                    & (pl.col("date") < datetime(2019, 1, 1)))
    # test_df = df.filter((pl.col("date") >= datetime(2019, 1, 1)))

    # train_df = df.filter(pl.col('date') < datetime(2018, 11, 24))
    # val_df = df.filter((pl.col('date') >= datetime(2018, 11, 24)) & (pl.col('date') < datetime(2018, 12, 10)))
    # test_df = df.filter((pl.col('date') >= datetime(2018, 12, 10)) & (pl.col('date') < datetime(2019, 1, 1)))

    train_df = df.filter(pl.col('date') < datetime(2018, 10, 1))
    val_df = df.filter((pl.col('date') >= datetime(2018, 10, 1)) & (pl.col('date') < datetime(2018, 11, 15)))
    test_df = df.filter((pl.col('date') >= datetime(2018, 11, 15)) & (pl.col('date') < datetime(2019, 1, 1)))
    return train_df, val_df, test_df

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.float().squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        # Calculate average loss over the epoch
        train_loss /= len(train_loader.dataset)

        # Validation phase
        # model.eval()
        # val_loss = 0.0

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

def evaluate_model(model, test_loader, file_path="evaluation_metrics.txt"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            preds = torch.sigmoid(outputs).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    all_preds_bin = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, all_preds_bin)
    precision = precision_score(all_labels, all_preds_bin)
    recall = recall_score(all_labels, all_preds_bin)
    f1 = f1_score(all_labels, all_preds_bin)
    roc_auc = roc_auc_score(all_labels, all_preds_bin)
    conf_matrix = confusion_matrix(all_labels, all_preds_bin)

    with open(file_path, "a") as f:
        f.write(f"0.9 0.1 sample weights + BCE loss + w_decay 0.001. 2018-2019 data + new feature columns\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {roc_auc:.4f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n\n")

    _, best_threshold = f1_smart(all_labels, all_preds)
    all_preds_bin_threshold = (all_preds > best_threshold).astype(int)

    accuracy = accuracy_score(all_labels, all_preds_bin_threshold)
    precision = precision_score(all_labels, all_preds_bin_threshold)
    recall = recall_score(all_labels, all_preds_bin_threshold)
    f1 = f1_score(all_labels, all_preds_bin_threshold)
    roc_auc = roc_auc_score(all_labels, all_preds_bin_threshold)
    conf_matrix = confusion_matrix(all_labels, all_preds_bin_threshold)

    # Store metrics in the file
    with open(file_path, "a") as f:
        f.write(f"With threshold optimization\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {roc_auc:.4f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n\n\n")

if __name__ == '__main__':

    args = get_args()

    ACTIVATION_FUNCTIONS = {
        "ReLU": torch.nn.ReLU(),
        "ELU": torch.nn.ELU(),
        "Tanh": torch.nn.Tanh(),
        "LeakyReLU": torch.nn.LeakyReLU(),
    }
    activation_fn = ACTIVATION_FUNCTIONS[args["activation_fn"]]

    #df = load_data_mlp(r'E:\bachelor\bachelorP\ibm_fraud_processed.pq')
    df = load_data_mlp(r'ibm_fraud_processed.pq')
    train_df, val_df, test_df = split_data(df)

    # Choose features and convert to np arrays
    feature_columns = ['amount_usd', 'score', 'total_debt', 'credit_limit', 'same_state', 'same_city',
                       'error_freq', 'card.mcc_freq','card.brand_Amex', 'card.brand_Discover',
                       'card.brand_Mastercard', 'card.brand_Visa', 'card.type_Credit', 'card.type_Debit',
                       'card.type_Debit (Prepaid)', 'card.chip_Chip Transaction',
                       'card.chip_Online Transaction', 'card.chip_Swipe Transaction', 'merchant.state_freq',
                       'merchant.city_freq', 'is_known_merchant', 'seen_count_last_2_days', 'seen_count_mcc_last_2_days',
                       'seen_count_last_7_days', 'seen_count_mcc_last_7_days', 'seen_count_last_30_days',
                       'seen_count_mcc_last_30_days', 'log_timediff']

    #feature_columns = ['amount_usd', 'score', 'total_debt', 'credit_limit', 'same_state', 'same_city',
    #                     'error_freq', 'card.mcc_freq', 'card.chip_Chip Transaction',
    #                     'card.chip_Online Transaction', 'card.chip_Swipe Transaction', 'merchant.state_freq',
    #                     'merchant.city_freq', 'is_known_merchant', 'seen_count_last_2_days', 'seen_count_last_7_days',
    #                     'seen_count_mcc_last_7_days', 'seen_count_last_30_days']
    target_column = 'is_fraud_bin'
    X_train, y_train = train_df.select(feature_columns).to_numpy(), train_df.select(target_column).to_numpy().flatten()
    X_val, y_val = val_df.select(feature_columns).to_numpy(), val_df.select(target_column).to_numpy().flatten()
    X_test, y_test = test_df.select(feature_columns).to_numpy(), test_df.select(target_column).to_numpy().flatten()

    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    test_dataset = FraudDataset(X_test, y_test)

    # Manually defining class weights
    class_weights = {0: 0.9, 1: 0.1}
    sample_weights = np.array([class_weights[int(label)] for label in y_train.flatten()])
    sample_weights = torch.from_numpy(sample_weights).float()
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    sampler = CustomWeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2)
    print("Data has been loaded")

    input_size = len(feature_columns)
    hidden_layers = args["hidden_layers"]
    model = MLP(input_size=input_size, hidden_units=hidden_layers, activation_fn=activation_fn)

    # Use BCEWithLogitsLoss since it combines sigmoid and binary cross-entropy
    pos_weight = np.sqrt((y_train == 0).sum()/((y_train == 1).sum()))
    print(pos_weight)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    criterion = FocalLoss(alpha=0.25, gamma=2)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    print("Training model")
    train_model(model, train_loader, criterion, optimizer, epochs=args["epochs"])
    print("Model eval")
    evaluate_model(model, test_loader)