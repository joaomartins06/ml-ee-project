from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42


def preprocess_and_pca(X_train, X_val, X_test, n_components=50, whiten=True):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components, whiten=whiten, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Retained variance with {n_components} components: {explained_var:.2f}%")

    return X_train_pca, X_val_pca, X_test_pca, pca, scaler
