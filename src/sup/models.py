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


#works on the raw pixels not on PCA features
class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),              
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),              #
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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