# src/models/encrypted_lr.py

import joblib
import numpy as np
import tenseal as ts
from pathlib import Path

SAFE_SCALE = 1e-6  # strong shrink to avoid CKKS blowup

class EncryptedLogisticRegression:

    def __init__(self, context: ts.Context):
        self.context = context
        self.weights = None
        self.bias = None

    def load_he_parameters(self, model_path: str):
        """
        Load HE-normalized LR parameters and shrink to avoid scale explosion.
        """
        print(f"\n📂 Loading HE LR parameters from {model_path}")
        data = joblib.load(model_path)
        w = np.array(data["weights"], dtype=float)
        b = float(data["bias"])

        # Additional shrink to keep logits tiny under CKKS
        self.weights = w * SAFE_SCALE / 10.0  # extra shrink
        self.bias = b * SAFE_SCALE / 10.0     # extra shrink

        print("   ✓ Loaded HE LR parameters (safe scaled)")

    # ============================================================
    def predict_encrypted_logit(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        """
        Compute encrypted logit: w·x + b without extra rescaling.
        """
        enc_mul = enc_x * self.weights.tolist()
        enc_out = enc_mul.sum()
        enc_out = enc_out + self.bias
        return enc_out

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weights": self.weights, "bias": self.bias}, filepath)
