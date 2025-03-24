from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd

def evaluation(predictions, labels):
    # input: numpy arrays
    predictions_flat = predictions.ravel()
    labels_flat = labels.ravel()
    
    accuracy = accuracy_score(labels_flat, predictions_flat)
    print(f"Accuracy: {accuracy:.4f}")

    # 计算 Weighted-F1、Precision、Recall
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(labels_flat, predictions_flat, average='weighted')
    print(f"Weighted-F1 Score: {weighted_f1:.4f}")
    print(f"Weighted-Precision: {weighted_precision:.4f}")
    print(f"Weighted-Recall: {weighted_recall:.4f}")
