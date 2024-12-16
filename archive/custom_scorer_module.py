from sklearn.metrics import roc_curve
import numpy as np

def tpr_at_fixed_fpr_scorer(estimator, X, y):
    y_probs = estimator.predict_proba(X)[:, 1]
    fprs, tprs, thresholds = roc_curve(y, y_probs)
    recall = np.max(tprs[fprs == max(fprs[fprs < 0.05])])
    return recall