# client/client.py

import time
import joblib
import numpy as np
import requests
import tenseal as ts

CKKS_SCALE = 2 ** 30    # MUST match server + NN + LR


class PrivateMLClient:
    """
    Client-side HE module.
    Handles:
        - Preprocessing (StandardScaler)
        - CKKS context reconstruction (client owns secret key)
        - Encrypting feature vectors
        - Sending HE requests
        - Decrypting HE predictions
    """

    def __init__(self, server_url: str, preprocessor_path: str):
        self.server_url = server_url.rstrip("/")
        self.scaler = joblib.load(preprocessor_path)  # raw StandardScaler
        print(f"📂 Loaded scaler from {preprocessor_path}")

        self.context: ts.Context | None = None

    # ============================================================
    # Initialize CKKS context
    # ============================================================
    def initialize(self):
        print("\nInitializing HE client...")

        # -------------------------------
        # 1. Health check
        # -------------------------------
        r = requests.get(f"{self.server_url}/health")
        if r.status_code != 200:
            raise RuntimeError("❌ Server not healthy")

        # -------------------------------
        # 2. Request public HE parameters
        # -------------------------------
        r = requests.get(f"{self.server_url}/context")
        if r.status_code != 200:
            raise RuntimeError("❌ Failed to fetch CKKS parameters")

        params = r.json()
        poly = params["poly_modulus_degree"]
        bits = params["coeff_mod_bit_sizes"]

        # -------------------------------
        # 3. Rebuild CKKS context (client)
        # -------------------------------
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly,
            coeff_mod_bit_sizes=bits,
        )

        # client generates SECRET keys
        ctx.generate_galois_keys()
        ctx.generate_relin_keys()

        # FIXED SCALE (same as server)
        ctx.global_scale = CKKS_SCALE
        ctx.auto_relin = True

        self.context = ctx

        print("✓ CKKS context reconstructed")
        print(f"✓ Using global_scale = {CKKS_SCALE}")
        print("✓ Ready for encrypted inference")

    # ============================================================
    # Preprocessing
    # ============================================================
    def preprocess(self, features: list) -> np.ndarray:
        """
        Scale input using the same StandardScaler used during training.
        """
        arr = np.array(features).reshape(1, -1)
        return self.scaler.transform(arr)[0]

    # ============================================================
    # Encrypt features
    # ============================================================
    def encrypt(self, features: np.ndarray) -> ts.CKKSVector:
        """
        Encrypt preprocessed features using CKKS.
        """
        return ts.ckks_vector(self.context, features.tolist())

    # ============================================================
    # Send encrypted inference request
    # ============================================================
    def send_request(self, enc_vec: ts.CKKSVector, model: str):
        """
        Sends encrypted vector → receives encrypted result.
        """
        endpoint = "/predict/lr" if model == "lr" else "/predict/nn"
        url = f"{self.server_url}{endpoint}"

        payload = {
            "encrypted_data": list(enc_vec.serialize())
        }

        t0 = time.time()
        r = requests.post(url, json=payload)
        elapsed_ms = (time.time() - t0) * 1000

        if r.status_code != 200:
            raise RuntimeError(f"❌ Server error: {r.status_code} → {r.text}")

        enc_bytes = bytes(r.json()["encrypted_prediction"])
        enc_pred = ts.ckks_vector_from(self.context, enc_bytes)

        return enc_pred, elapsed_ms

    # ============================================================
    # Decrypt + rescale
    # ============================================================
    def decrypt(self, enc_pred: ts.CKKSVector) -> float:
        """
        Decrypts encrypted prediction and rescales the value.
        """
        raw_val = float(enc_pred.decrypt()[0])

        # Convert from CKKS scale → actual probability
        prob = raw_val / CKKS_SCALE

        # Clip noise
        prob = max(0.0, min(1.0, prob))
        return prob

    # ============================================================
    # Full pipeline
    # ============================================================
    def predict(self, features: list, model: str):
        """
        Runs the entire encrypted inference:
            1. scale
            2. encrypt
            3. send
            4. decrypt
        """
        Xp = self.preprocess(features)
        enc_x = self.encrypt(Xp)
        enc_pred, ms = self.send_request(enc_x, model)
        prob = self.decrypt(enc_pred)

        return {
            "probability": prob,
            "prediction": int(prob > 0.5),
            "inference_time_ms": ms
        }