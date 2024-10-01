import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
import xgboost as xgb
import lightgbm as lgb



def transform_data(df):
    # Remove raws where direction is not null
    df = df.filter(pl.col('direction').is_not_null())
    # Fill in the null values in 'error'
    df = df.with_columns(pl.col('error').fill_null("No error"))
    # Apply log to amount_usd data
    df = df.with_columns((pl.col('amount_usd')+1).log().alias('amount_usd_log'))
    # Add a relative spend column (amount_usd / credit_limit)
    df = df.with_columns((pl.col('amount_usd') / (pl.col('credit_limit') + 1e-5)).alias('relative_spend'))
    # Extract day of the week from date column to a seperate column
    df = df.with_columns(pl.col('date').dt.weekday().alias('weekday'))
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
    # Binary encoding for direction, same_state, and same_city
    df = df.with_columns((pl.when(pl.col('direction') == "outbound").then(1).otherwise(0).alias('direction_bin'),
                    pl.col('same_state').cast(pl.Int8),
                    pl.col('same_city').cast(pl.Int8)))
    # One-hot encoding
    df = df.to_dummies(columns=['card.chip', 'card.type', 'card.brand'])
    # df = df.drop(["card.chip", "card.type", "card.brand"])
    # Frequency encoding (https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.dataframe.group_by.GroupBy.agg.html)
    error_freq = df.group_by('error').agg(
        pl.len().alias('error_frequency')  # Count the occurrences
    )
    df = df.join(error_freq, on='error', how='left')
    card_mcc_freq = df.group_by('card.mcc').agg(
        pl.len().alias('mcc_frequency')  # Count the occurrences
    )
    df = df.join(card_mcc_freq, on='card.mcc', how='left')

    return df

def test_rf(X, y, best_threshold):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    weights = {0: 0.5, 1: 411}
    rf_model = RandomForestClassifier(class_weight=weights, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print("Results without threshold optimization")
    print_results(y_pred, y_test)
    if best_threshold:
        y_probs = rf_model.predict_proba(X_test)[:, 1]
        y_pred_opt = (y_probs >= best_threshold).astype(int)
        print("Results with threshold optimization")
        print_results(y_pred_opt, y_test)

def find_threshold(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    best_threshold = optimize_threshold(y_test, y_probs)
    y_pred = (y_probs >= best_threshold).astype(int)
    # print_results(y_pred, y_test)
    return best_threshold

def test_xgb(X, y, weight):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scale_pos_weight = None
    if weight:
        scale_pos_weight = 20
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("XGBoost Results:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def test_lightgbm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    # weights = {0: class_weights[0], 1: class_weights[1]}
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("LightGBM Results:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def test_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_reg_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    log_reg_model.fit(X_train, y_train)
    y_pred = log_reg_model.predict(X_test)
    print("Logistic Regression Results:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def test_perceptron(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    perceptron_model = Perceptron(class_weight='balanced', random_state=42, max_iter=1000)
    perceptron_model.fit(X_train, y_train)
    y_pred = perceptron_model.predict(X_test)
    print("Perceptron Results:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def test_rf_hp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    weights = {0: 1, 1: 10}
    rf_model = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=1,
        n_jobs=-1,
        scoring='f1_macro',
        random_state=42
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print("Best hyperparameters:", best_params)
    best_rf = random_search.best_estimator_
    y_pred_best_rf = best_rf.predict(X_test)
    print("Random Forest Classifier Results with Best Hyperparameters:")
    print_results(y_pred_best_rf, y_test)


def optimize_threshold(y_true, y_probs):
    thresholds = np.arange(0.0, 1.01, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    # Find the best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Best Threshold: {best_threshold:.2f}, Best F1 Score: {best_f1:.4f}")

    return best_threshold

def print_results(y_pred, y_test):
    print(confusion_matrix(y_test, y_pred))
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")