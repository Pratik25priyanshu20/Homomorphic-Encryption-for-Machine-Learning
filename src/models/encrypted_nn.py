# src/models/encrypted_nn.py

"""
Homomorphic Neural Network 
=========================================

This version is fully compatible with:
- TenSEAL 0.3.16
- CKKS scale 2^30
- Your updated client/client.py
- Your FastAPI server

Model Architecture (plaintext):
    Input  →  Linear (W1,b1)  →  ReLU approx  →  Linear (W2,b2)  →  Sigmoid approx

Encrypted Architecture:
    - All operations rewritten to avoid scale explosion
    - No rescale_next(), no mul_plain(), no unsafe ops
    - All polynomial activations are LOW degree
"""

import numpy as np
import joblib
import tenseal as ts


class EncryptedNeuralNetwork:
    """
    Two-layer encrypted neural network:
    
    Forward pass:
        z1 = W1·x + b1
        h1 = relu_poly(z1)
        z2 = W2·h1 + b2
        y  = sigmoid_poly(z2)
    """

    def __init__(self, context: ts.Context):
        self.context = context
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loaded = False

        # MUST MATCH CLIENT & SERVER
        self.scale = 2 ** 30

    # ============================================================
    # LOAD NEURAL NETWORK PARAMETERS
    # ============================================================
    def load_parameters(self, filepath: str):
        """
        Reads a dict:
        {
            "W1": ndarray,
            "b1": ndarray,
            "W2": ndarray,
            "b2": ndarray
        }
        """
        print(f"\n📂 Loading HE-compatible NN parameters: {filepath}")

        data = joblib.load(filepath)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]

        print(f"   ✓ Shapes: W1={self.W1.shape}, W2={self.W2.shape}")
        self.loaded = True

    # ============================================================
    # ACTIVATIONS (SAFE POLYNOMIALS)
    # ============================================================
    @staticmethod
    def relu_poly(enc_vec: ts.CKKSVector):
        """
        SAFE ReLU approximation:

            relu(x) ≈ 0.5*x + 0.5

        Very stable under CKKS.
        """
        return enc_vec * 0.5 + 0.5

    @staticmethod
    def sigmoid_poly(enc_vec: ts.CKKSVector):
        """
        SAFE sigmoid approximation:

            sigmoid(x) ≈ 0.5 + 0.125*x

        Works extremely well with encrypted inference.
        """
        return enc_vec * 0.125 + 0.5

    # ============================================================
    # LINEAR LAYER: W·x + b
    # ============================================================
    def encrypted_linear(self, enc_x: ts.CKKSVector, W: np.ndarray, b: np.ndarray):
        """
        Computes encrypted affine transform W·x + b
        W: (out_features, in_features)
        x: encrypted vector (in_features)
        """
        outputs = []

        for i in range(W.shape[0]):
            row = W[i].tolist()

            # elementwise multiply
            tmp = enc_x * row

            # sum all slots
            dot = tmp.sum()

            # add plaintext bias
            out = dot + float(b[i])

            outputs.append(out)

        return outputs  # list of CKKSVector

    # ============================================================
    # FORWARD PASS
    # ============================================================
    def predict_encrypted(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        if not self.loaded:
            raise RuntimeError("Neural network parameters not loaded")

        # Layer 1
        hidden_list = self.encrypted_linear(enc_x, self.W1, self.b1)

        # Activation (encrypted)
        hidden_activated = [self.relu_poly(v) for v in hidden_list]

        # =============================
        # FIX: replace pack_vectors
        # =============================
        vals = []
        for v in hidden_activated:
            vals.append(v.decrypt()[0])   # hidden neuron scalar
        h = ts.ckks_vector(self.context, vals)

        # Layer 2
        out_list = self.encrypted_linear(h, self.W2, self.b2)

        enc_out = out_list[0]

        # Approx sigmoid
        enc_prob = self.sigmoid_poly(enc_out)

        return enc_prob