import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from pathlib import Path


class HEFriendlyNN(nn.Module):
    """
    HE-Friendly Neural Network
    Architecture:
        Input 13 → Hidden 4 → Output 1
    Nonlinear activation:
        Square (x^2) which is very HE-friendly (low multiplicative depth)
    """

    def __init__(self, input_dim=13, hidden_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = lambda x: x * x  # square activation

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.activation(z1)
        out = self.fc2(a1)
        return out  # raw score (we convert later)


class HEFriendlyNNTrainer:

    def __init__(self, input_dim=13, hidden_dim=4, lr=0.001):
        self.model = HEFriendlyNN(input_dim, hidden_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=40, batch_size=32):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        n = len(X_train)
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(n)
            losses = []

            for i in range(0, n, batch_size):
                idx = permutation[i:i + batch_size]
                batch_x = X_train[idx]
                batch_y = y_train[idx]

                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            if epoch % 10 == 0:
                avg_loss = np.mean(losses)
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        print("✓ Training complete")

    def evaluate(self, X_test, y_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            logits = self.model(X_test)
            preds = (torch.sigmoid(logits) > 0.5).float()

        accuracy = (preds == y_test).float().mean().item()
        print(f"Validation Accuracy: {accuracy*100:.2f}%")

        return accuracy

    def save_parameters(self, path="models/plaintext/nn_he_parameters.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        params = {
            "W1": self.model.fc1.weight.detach().numpy(),
            "b1": self.model.fc1.bias.detach().numpy(),
            "W2": self.model.fc2.weight.detach().numpy(),
            "b2": self.model.fc2.bias.detach().numpy(),
        }

        joblib.dump(params, path)
        print(f"💾 Saved HE-friendly parameters → {path}")