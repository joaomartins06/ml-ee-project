import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

SEED = 42


class DeepNeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def train_baseline_models(X_train, y_train, X_test):

    log_reg = LogisticRegression(max_iter=200, random_state=SEED)
    log_reg.fit(X_train, y_train)
    y_pred_logreg = log_reg.predict(X_test)

    svm = SVC(kernel="linear", C=1.0, random_state=SEED)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    return {
        "Multi-Class Logistic Regression": y_pred_logreg,
        "Linear SVM": y_pred_svm,
    }
