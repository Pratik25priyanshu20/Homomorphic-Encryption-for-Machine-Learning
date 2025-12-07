import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib


# ============================================================
#   HE-SAFE NEURAL NETWORK FOR CKKS ENCRYPTED INFERENCE
# ============================================================
# Matches EXACTLY the architecture used on the encrypted server.
#
#   INPUT (13)
#      ↓
#   Linear(13 → 3)
#      ↓
#   Activation:  0.5*z + 0.5      (HE-friendly)
#      ↓
#   Linear(3 → 1)
#      ↓
#   Output: raw logit
#
# ============================================================


class HEFriendlyNN(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # shape: (3,13)
        self.fc2 = nn.Linear(hidden_dim, 1)          # shape: (1,3)

    def forward(self, x):
        # z1 = W1*x + b1
        z1 = self.fc1(x)

        # HE-FRIENDLY ACTIVATION: 0.5*z + 0.5
        h = 0.5 * z1 + 0.5

        # Output layer
        logit = self.fc2(h)
        return logit


# ============================================================
#   TRAINER WITH STRICT HE CONSTRAINTS
# ============================================================

class HEFriendlyNNTrainer:

    def __init__(self, input_dim=13, hidden_dim=3, lr=0.0005, clip_value=0.25):
        self.model = HEFriendlyNN(input_dim, hidden_dim)
        self.lr = lr
        self.clip_value = clip_value

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    # ---------------------------
    #      TRAINING LOOP
    # ---------------------------
    def train(self, X_train, y_train, X_val, y_val, epochs=40, batch_size=32):

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for xb, yb in loader:
                self.optimizer.zero_grad()

                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()

                # 🔥 CRITICAL: weight clipping BEFORE update
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)
                self.optimizer.step()

                # 🔥 Clip weights themselves AFTER update
                with torch.no_grad():
                    for p in self.model.parameters():
                        p.clamp_(-self.clip_value, self.clip_value)

                running_loss += loss.item()

            # -------------------------
            # Validation
            # -------------------------
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = self.criterion(val_logits, y_val)
                preds = (torch.sigmoid(val_logits) >= 0.5).float()
                val_acc = (preds == y_val).float().mean().item()

            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"Loss: {running_loss/len(loader):.4f}  "
                  f"Val Loss: {val_loss:.4f}  "
                  f"Val Acc: {val_acc:.4f}")

    # ---------------------------
    #       EVALUATION
    # ---------------------------
    def evaluate(self, X_test, y_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            return (preds == y_test).float().mean().item()

    # ---------------------------
    #      SAVE PARAMETERS
    # ---------------------------
    def save_parameters(self, path):
        W1 = self.model.fc1.weight.detach().numpy()
        b1 = self.model.fc1.bias.detach().numpy()

        W2 = self.model.fc2.weight.detach().numpy()
        b2 = self.model.fc2.bias.detach().numpy()

        params = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

        joblib.dump(params, path)
        print(f"💾 Saved HE parameters → {path}")