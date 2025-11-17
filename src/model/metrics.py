from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score

import numpy as np


# Compute standard regression metrics for HuggingFace Trainer evaluation
def compute_regression_metrics(eval_pred):
    # Unpacking
    preds, labels = eval_pred

    # Remove all singleton dimensions
    preds = preds.squeeze()
    labels = labels.squeeze()

    # Metrics
    rmse = root_mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# Compute standard classification metrics for HuggingFace Trainer evaluation
def compute_classification_metrics(eval_pred):
    # Unpacking
    logits, labels = eval_pred

    # Select the index of the highest logit score
    preds = np.argmax(logits, axis=-1)

    # Metrics
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
