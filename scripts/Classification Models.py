## Part 2 ##
## Import Libraries ##
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

## Load data ##
heart = fetch_openml(name="heart-disease", version=1, as_frame=True)
df = heart.frame

## Drop missing target values ##
df = df.dropna(subset=["target"])

## Convert to binary: 0 = no disease, 1 = has disease ##
df["target"] = df["target"].astype(int)
df["target"] = (df["target"] > 0).astype(int)  # binarize

## Features and target ##
X = df.drop(columns=["target"])
y = df["target"]

## One-hot encoding (if categorical columns present)  ##
X = pd.get_dummies(X, drop_first=True)

## Split the data ##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

penalties = ['l2', 'None']
solvers = ['lbfgs', 'saga']
log_results = []

for penalty in penalties:
    for solver in solvers:
        try:
            model = LogisticRegression(penalty=penalty, solver=solver, max_iter=5000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                'penalty': penalty,
                'solver': solver,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auroc': roc_auc_score(y_test, y_prob),
                'auprc': average_precision_score(y_test, y_prob)
            }
            log_results.append(metrics)
        except Exception as e:
            print(f"Skipped: penalty={penalty}, solver={solver} -> {e}")

log_df = pd.DataFrame(log_results)
print("\nLogistic Regression Results:\n", log_df.sort_values(by='auroc', ascending=False))

knn_results = []
for k in [1, 5, 10]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'n_neighbors': k,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_prob),
        'auprc': average_precision_score(y_test, y_prob)
    }
    knn_results.append(metrics)

knn_df = pd.DataFrame(knn_results)
print("\nKNN Results:\n", knn_df.sort_values(by='auroc', ascending=False))

## Finding best logistic regression configuration ##
best_log = log_df.sort_values(by='auroc', ascending=False).iloc[0]
best_log_model = LogisticRegression(penalty=best_log['penalty'], solver=best_log['solver'], max_iter=1000)
best_log_model.fit(X_train, y_train)
log_probs = best_log_model.predict_proba(X_test)[:, 1]

## Finding best k-NN configuration ##
best_knn = knn_df.sort_values(by='auroc', ascending=False).iloc[0]
best_knn_model = KNeighborsClassifier(n_neighbors=int(best_knn['n_neighbors']))
best_knn_model.fit(X_train, y_train)
knn_probs = best_knn_model.predict_proba(X_test)[:, 1]

## AUROC plot ##
fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label=f"LogReg (AUROC={best_log['auroc']:.2f})")
plt.plot(fpr_knn, tpr_knn, label=f"k-NN (AUROC={best_knn['auroc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

## AUPRC plot ##
prec_log, recall_log, _ = precision_recall_curve(y_test, log_probs)
prec_knn, recall_knn, _ = precision_recall_curve(y_test, knn_probs)

plt.figure(figsize=(8,6))
plt.plot(recall_log, prec_log, label=f"LogReg (AUPRC={best_log['auprc']:.2f})")
plt.plot(recall_knn, prec_knn, label=f"k-NN (AUPRC={best_knn['auprc']:.2f})")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
