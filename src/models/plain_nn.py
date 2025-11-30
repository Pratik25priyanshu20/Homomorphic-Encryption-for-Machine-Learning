# src/models/plain_nn.py

"""
NumPy-based 1-hidden-layer Neural Network
Compatible with Homomorphic Encryption (HE)

This model:
- Uses simple matrix operations
- Trains with gradient descent
- Saves weights W1, b1, W2, b2 in HE-friendly format
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time


class PlainNeuralNetwork:
    def __init__(self, input_dim, hidden_dim=8, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((hidden_dim,))

        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1,))

    # ------------------------------------------------------------------
    # ACTIVATIONS
    # ------------------------------------------------------------------
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # ------------------------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------------------------
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        out = self.sigmoid(z2)
        return z1, a1, out

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train(self, X, y, epochs=200, batch_size=32, verbose=True):
        n = X.shape[0]
        y = y.reshape(-1, 1)

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuf, y_shuf = X[indices], y[indices]

            for i in range(0, n, batch_size):
                Xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]

                # Forward pass
                z1, a1, out = self.forward(Xb)

                # Backpropagation
                dz2 = (out - yb)
                dW2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * self.relu_deriv(z1)
                dW1 = Xb.T @ dz1
                db1 = np.sum(dz1, axis=0)

                # Update weights
                lr = self.learning_rate
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            if verbose and (epoch+1) % 20 == 0:
                _, _, out_all = self.forward(X)
                loss = np.mean((out_all - y)**2)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

        return True

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    def evaluate(self, X, y):
        y = y.reshape(-1)
        start = time.time()

        _, _, out = self.forward(X)
        preds = (out.reshape(-1) > 0.5).astype(int)

        elapsed = (time.time() - start) * 1000  # ms

        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "f1": f1_score(y, preds),
            "roc_auc": roc_auc_score(y, out.reshape(-1)),
            "inference_time_ms": elapsed
        }

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }, path)
        print(f"💾 Saved HE-friendly NN weights: {path}")

    def load(self, path):
        data = joblib.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        print(f"📂 Loaded NN weights: {path}")