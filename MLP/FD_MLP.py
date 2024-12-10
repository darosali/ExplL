import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from datetime import datetime
from utils2 import *
import argparse

# https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
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
    def __init__(self, input_size, hidden_units=[16, 16], output_size=1, activation_fn=nn.ReLU()):
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
    parser = argparse.ArgumentParser(description="Fraud detection MLP model.")

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
    #Load already scaled and transformed data
    #df = pl.scan_parquet(filename).collect()
    # Data from 2018 to 2019
    df = pl.scan_parquet(filename).filter((pl.col("date") >= datetime(2018, 1, 1))
                                          & (pl.col("date") < datetime(2019, 1, 1))).collect()
    # df = scale_data(df)
    return df

def split_data(df):
    # train_df = df.filter(pl.col("date") < datetime(2018, 1, 1))
    # val_df = df.filter((pl.col("date") >= datetime(2018, 1, 1))
    #                    & (pl.col("date") < datetime(2019, 1, 1)))
    # test_df = df.filter((pl.col("date") >= datetime(2019, 1, 1)))

    # Split for one year
    # train_df = df.filter(pl.col('date') < datetime(2018, 10, 1))
    # val_df = df.filter((pl.col('date') >= datetime(2018, 10, 1))
    #                    & (pl.col('date') < datetime(2018, 11, 15)))
    # test_df = df.filter((pl.col('date') >= datetime(2018, 11, 15)))

    # Customer split
    train_df = df.filter(pl.col("customer.id") <= 1200)
    val_df = df.filter((pl.col("customer.id") > 1200) & (pl.col("customer.id") <= 1600))
    test_df = df.filter(pl.col("customer.id") > 1600)

    return train_df, val_df, test_df

def train_model(model, test_loader, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.float().squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels.float().squeeze())
                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_loader.dataset)

        # Follow progress
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # Eval after each epoch
        # evaluate_model(model, test_loader, train_loader, val_loader, "performance1.txt")

    # print("Saving model...")
    # torch.save(model.state_dict(), "model_weights.pth")
    # torch.save(optimizer.state_dict(), "optimizer_state.pth")


def evaluate_model(model, test_loader, train_loader, val_loader=None, file_path="evaluation_metrics1.txt"):
    def compute_predictions(loader):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in loader:
                outputs = model(features)
                preds = torch.sigmoid(outputs).squeeze()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_preds), np.array(all_labels)

    def compute_metrics(all_preds, all_labels, threshold, data_type):

        all_preds_bin = (all_preds >= threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds_bin)
        precision = precision_score(all_labels, all_preds_bin)
        recall = recall_score(all_labels, all_preds_bin)
        f1 = f1_score(all_labels, all_preds_bin)
        roc_auc = roc_auc_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds_bin)

        with open(file_path, "a") as f:
            f.write(f"Evaluation Metrics ({data_type}, Threshold {threshold:.4f})\n")
            f.write("=" * 50 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"AUC-ROC: {roc_auc:.4f}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n\n")

    def find_best_threshold(val_loader):
        if val_loader is None:
            return 0.5

        print("Finding best threshold using validation data...")
        val_preds, val_labels = compute_predictions(val_loader)
        _, best_threshold = f1_smart(val_labels, val_preds)
        print(f"Best threshold found: {best_threshold:.4f}")
        return best_threshold
    model.eval()
    with open(file_path, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")

    best_threshold = find_best_threshold(val_loader)

    print("Evaluating on training data...")
    train_preds, train_labels = compute_predictions(train_loader)
    compute_metrics(train_preds, train_labels, threshold=0.5, data_type="Training Data")
    compute_metrics(train_preds, train_labels, threshold=best_threshold, data_type="Training Data")

    if val_loader:
        print("Evaluating on validation data...")
        val_preds, val_labels = compute_predictions(val_loader)
        compute_metrics(val_preds, val_labels, threshold=0.5, data_type="Validation Data")
        compute_metrics(val_preds, val_labels, threshold=best_threshold, data_type="Validation Data")

    print("Evaluating on test data...")
    test_preds, test_labels = compute_predictions(test_loader)
    compute_metrics(test_preds, test_labels, threshold=0.5, data_type="Test Data")
    compute_metrics(test_preds, test_labels, threshold=best_threshold, data_type="Test Data")

    return test_preds, (test_preds >= best_threshold).astype(int)


if __name__ == '__main__':

    args = get_args()

    ACTIVATION_FUNCTIONS = {
        "ReLU": torch.nn.ReLU(),
        "ELU": torch.nn.ELU(),
        "Tanh": torch.nn.Tanh(),
        "LeakyReLU": torch.nn.LeakyReLU(),
    }
    activation_fn = ACTIVATION_FUNCTIONS[args["activation_fn"]]

    df = load_data_mlp(r'ibm_fraud_transformed2.pq')
    columns_to_drop = ['timestamp', 'customer.id', 'direction', 'amount_signed', 'date', 'customer.name', 'customer.address', 'customer.city',
                       'customer.state', 'customer.gender', 'latitude', 'longitude', 'card.id', 'card.number', 'customer.zip', 'card.expiry_date',
                       'card.cvv', 'num_cards', 'merchant.name', 'merchant.city', 'is_fraud', 'log_timediff']
    train_df, val_df, test_df = split_data(df)
    train_df = train_df.drop(columns_to_drop)
    val_df = val_df.drop(columns_to_drop)
    test_df = test_df.drop(columns_to_drop)

    # feature_columns = ['amount_usd', 'score', 'total_debt', 'credit_limit', 'same_state', 'same_city',
    #                    'error_freq', 'card.mcc_freq','card.brand_Amex', 'card.brand_Discover',
    #                    'card.brand_Mastercard', 'card.brand_Visa', 'card.type_Credit', 'card.type_Debit',
    #                    'card.type_Debit (Prepaid)', 'card.chip_Chip Transaction',
    #                    'card.chip_Online Transaction', 'card.chip_Swipe Transaction', 'merchant.state_freq',
    #                    'merchant.city_freq', 'is_known_merchant', 'seen_count_last_2_days', 'seen_count_mcc_last_2_days',
    #                    'seen_count_last_7_days', 'seen_count_mcc_last_7_days', 'seen_count_last_30_days',
    #                    'seen_count_mcc_last_30_days', 'log_timediff']

    target_column = 'is_fraud_bin'
    #X_train, y_train = train_df.select(feature_columns).to_numpy(), train_df.select(target_column).to_numpy().flatten()
    #X_val, y_val = val_df.select(feature_columns).to_numpy(), val_df.select(target_column).to_numpy().flatten()
    #X_test, y_test = test_df.select(feature_columns).to_numpy(), test_df.select(target_column).to_numpy().flatten()
    y_train = train_df.select(target_column).to_numpy().flatten()
    y_val = val_df.select(target_column).to_numpy().flatten()
    y_test = test_df.select(target_column).to_numpy().flatten()
    train_df = train_df.drop(target_column)
    val_df = val_df.drop(target_column)
    test_df = test_df.drop(target_column)
    #print(len(train_df.columns))
    #print(train_df.columns)
    X_train = train_df.to_numpy()
    X_val = val_df.to_numpy()
    X_test = test_df.to_numpy()

    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    test_dataset = FraudDataset(X_test, y_test)

    # class weights
    class_weights = {0: 0.7, 1: 0.3}
    sample_weights = np.array([class_weights[int(label)] for label in y_train.flatten()])
    sample_weights = torch.from_numpy(sample_weights).float()
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    sampler = CustomWeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=2)
    print("Data has been loaded")

    input_size = X_train.shape[1]
    hidden_layers = args["hidden_layers"]
    model = MLP(input_size=input_size, hidden_units=hidden_layers, activation_fn=activation_fn)

    # Pos weight for BCEWithLogitsLoss
    pos_weight = np.sqrt((y_train == 0).sum()/((y_train == 1).sum()))
    print(pos_weight)
    # Can either use BCE or Focal Loss
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    criterion = FocalLoss(alpha=0.25, gamma=20)
    optimizer = optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    print("Training model...")
    train_model(model, test_loader, train_loader, val_loader, criterion, optimizer, epochs=args["epochs"])
    print("Model eval...")
    probs, preds = evaluate_model(model, test_loader, train_loader, val_loader, "performance1.txt")

    # Get plots and SHAP explanations
    save_prc(y_test, probs)
    save_roc_auc_curve(y_test, probs)
    save_confusion_matrix(y_test, preds)

    # X_shap_tr, y_shap_tr = sample_shap_data(X_train, y_train, n_samples=2000, fraud_ratio=0.3)
    # print(X_shap_tr.shape)
    # X_shap = X_test
    # print(X_shap.shape)
    # get_shap_explanations(model, X_shap_tr, X_shap, train_df.columns)