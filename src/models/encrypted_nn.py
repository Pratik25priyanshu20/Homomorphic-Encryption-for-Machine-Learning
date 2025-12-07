# src/models/encrypted_nn.py

import numpy as np
import joblib
import tenseal as ts

SAFE_SCALE = 1e-6  # strong shrink to avoid CKKS blowup

class EncryptedNeuralNetwork:

    def __init__(self, context: ts.Context):
        self.context = context
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loaded = False

    def load_parameters(self, filepath: str):
        """
        Load HE-safe parameters (already normalized when saved).
        """
        print(f"\n📂 Loading HE-SAFE NN parameters from {filepath}")
        params = joblib.load(filepath)

        self.W1 = np.array(params["W1"], dtype=float)
        self.b1 = np.array(params["b1"], dtype=float)
        self.W2 = np.array(params["W2"], dtype=float)
        self.b2 = np.array(params["b2"], dtype=float)

        self.loaded = True
        print(f"   ✓ Shapes: W1={self.W1.shape}, W2={self.W2.shape}")

    # ============================================================
    # ACTIVATION (requires rescale)
    # ============================================================
    @staticmethod
    def activation(enc_vec: ts.CKKSVector):
        return enc_vec * 0.5 + 0.5

    # ============================================================
    # LINEAR LAYER
    # ============================================================
    def encrypted_linear_vector(self, enc_x, W, b):
        outputs = []

        for i in range(W.shape[0]):
            row = W[i].tolist()

            tmp = enc_x * row
            dot = tmp.sum()
            out = dot + float(b[i])
            outputs.append(out)

        return outputs

    def encrypted_linear_from_list(self, enc_list, W, b):
        outputs = []

        for i in range(W.shape[0]):
            acc = None
            for j, w in enumerate(W[i].tolist()):
                term = enc_list[j] * float(w)
                acc = term if acc is None else (acc + term)

            outputs.append(acc + float(b[i]))

        return outputs

    # ============================================================
    def predict_logit_encrypted(self, enc_x):
        if not self.loaded:
            raise RuntimeError("NN parameters not loaded")

        # Layer 1
        h1 = self.encrypted_linear_vector(enc_x, self.W1, self.b1)
        h1 = [self.activation(v) for v in h1]

        # Output layer
        out = self.encrypted_linear_from_list(h1, self.W2, self.b2)

        return out[0]
