# src/models/encrypted_lr.py

"""
Encrypted Logistic Regression 
SERVER RETURNS ENCRYPTED LOGIT
CLIENT APPLIES TRUE SIGMOID
"""

from pathlib import Path
from typing import List

import joblib
import numpy as np
import tenseal as ts
import time


class EncryptedLogisticRegression:

    def __init__(self, context: ts.Context):
        self.context = context
        self.weights = None
        self.bias = None
        self.is_loaded = False

    # =========================================================
    # LOAD MODEL
    # =========================================================
    def load_plaintext_model(self, model_path: str):
        print(f"\n📂 Loading LR model from {model_path}")

        data = joblib.load(model_path)
        model = data["model"]

        self.weights = model.coef_[0]
        self.bias = float(model.intercept_[0])

        self.is_loaded = True

        print("   ✓ Loaded LR parameters")
        print(f"   ✓ Weights: {len(self.weights)} features")
        print(f"   ✓ Bias: {self.bias:.4f}")

    # =========================================================
    # ENCRYPTED LINEAR TERM (NO SIGMOID)
    # =========================================================
    def encrypted_linear(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        """
        Compute encrypted linear logit: w·x + b
        """
        enc_mul = enc_x * self.weights.tolist()

        if hasattr(enc_mul, "sum"):
            enc_dot = enc_mul.sum()
        else:
            vals = enc_mul.decrypt()
            enc_dot = ts.ckks_vector(self.context, [float(np.sum(vals))])

        return enc_dot + self.bias

    # =========================================================
    # PUBLIC API
    # =========================================================
    def predict_encrypted_logit(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        """
        Return encrypted logit — NOT sigmoid.
        """
        return self.encrypted_linear(enc_x)

    def predict_batch_encrypted(self, enc_list: List[ts.CKKSVector]):
        return [self.predict_encrypted_logit(enc) for enc in enc_list]

    # Save/Load weights only
    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weights": self.weights, "bias": self.bias}, filepath)
        print(f"💾 Saved encrypted LR params → {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.weights = data["weights"]
        self.bias = float(data["bias"])
        self.is_loaded = True
        print(f"📂 Loaded encrypted LR params from {filepath}")