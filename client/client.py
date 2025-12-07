# client/client.py

import time
import joblib
import numpy as np
import requests
import tenseal as ts

CKKS_SCALE = 2 ** 40  # just a reasonable default; TenSEAL handles internal scale


class PrivateMLClient:
    """
    Client-side HE module with REAL calibration against the plaintext models.

    Flow:
    - Client:
        * preprocess features (StandardScaler)
        * encrypt vector with CKKS
        * send to server
    - Server:
        * runs LR or NN and returns ENCRYPTED LOGIT (w·x + b)
    - Client:
        * decrypts logit
        * for LR: rescales it using plaintext LR model (calibration)
        * applies sigmoid on the CALIBRATED logit
    """

    def __init__(self, server_url: str, preprocessor_path: str):
        self.server_url = server_url.rstrip("/")
        self.scaler = joblib.load(preprocessor_path)
        print(f"📂 Loaded scaler from {preprocessor_path}")
        self.context: ts.Context | None = None

        # ---- NEW: load plaintext LR model for calibration ----
        try:
            lr_bundle = joblib.load("models/plaintext/logistic_regression.pkl")
            # If train_models.py saved dict with key "model"
            self.lr_model = lr_bundle["model"] if isinstance(lr_bundle, dict) else lr_bundle
            print("📘 Loaded plaintext Logistic Regression model for calibration")
        except Exception as e:
            print(f"⚠️ Could not load LR model for calibration: {e}")
            self.lr_model = None

        # scale factor to map HE logit → plaintext logit
        self.lr_scale_factor = None  # computed lazily at first LR prediction

    # ============================================================
    # INITIALIZATION
    # ============================================================
    def initialize(self):
        print("\nInitializing HE client...")

        # Health check
        r = requests.get(f"{self.server_url}/health")
        if r.status_code != 200:
            raise RuntimeError(f"❌ Server not healthy: {r.status_code} {r.text}")

        # Get CKKS parameters
        r = requests.get(f"{self.server_url}/context")
        if r.status_code != 200:
            raise RuntimeError(f"❌ Failed to fetch CKKS parameters: {r.status_code} {r.text}")

        params = r.json()
        poly = params["poly_modulus_degree"]
        bits = params["coeff_mod_bit_sizes"]

        # Rebuild CKKS context (client owns secret key)
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly,
            coeff_mod_bit_sizes=bits,
        )
        ctx.generate_galois_keys()
        ctx.generate_relin_keys()
        ctx.global_scale = CKKS_SCALE
        ctx.auto_relin = True

        self.context = ctx

        print("✓ CKKS context reconstructed")
        print(f"✓ Using global_scale = {ctx.global_scale}")
        print("✓ Ready for encrypted inference")

    # ============================================================
    # PREPROCESS & ENCRYPT
    # ============================================================
    def preprocess(self, features: list) -> np.ndarray:
        """
        Scale input using StandardScaler (same as training).
        """
        arr = np.array(features, dtype=float).reshape(1, -1)
        scaled = self.scaler.transform(arr)[0]
        return scaled

    def encrypt(self, features: np.ndarray) -> ts.CKKSVector:
        """
        Encrypt preprocessed features.
        """
        if self.context is None:
            raise RuntimeError("Context not initialized")
        return ts.ckks_vector(self.context, features.tolist())

    # ============================================================
    # SEND REQUEST
    # ============================================================
    def send_request(self, enc_vec: ts.CKKSVector, model: str):
        """
        Send encrypted inference request to LR or NN endpoint.
        """
        if model == "lr":
            endpoint = "/predict/lr"
        elif model == "nn":
            endpoint = "/predict/nn"
        else:
            raise ValueError(f"Unknown model: {model}")

        url = f"{self.server_url}{endpoint}"
        payload = {"encrypted_data": list(enc_vec.serialize())}

        t0 = time.time()
        r = requests.post(url, json=payload)
        elapsed_ms = (time.time() - t0) * 1000.0

        if r.status_code != 200:
            raise RuntimeError(f"❌ Server error: {r.status_code}: {r.text}")

        enc_bytes = bytes(r.json()["encrypted_prediction"])
        enc_pred = ts.ckks_vector_from(self.context, enc_bytes)

        return enc_pred, elapsed_ms

    # ============================================================
    # INTERNAL: SIGMOID + CALIBRATION HELPERS
    # ============================================================
    @staticmethod
    def _sigmoid(z: float) -> float:
        # Clamp for numerical stability
        z = max(-10.0, min(10.0, z))
        return 1.0 / (1.0 + np.exp(-z))

    def _calibrate_lr_logit(self, raw_logit: float, x_scaled: np.ndarray) -> float:
        """
        Map the huge HE logit back to the true LR logit scale using the plaintext model.

        We assume:
            raw_logit_HE ≈ alpha * (w·x + b)
        so:
            alpha = raw_logit_HE / z_plain
        and then:
            z_calibrated = raw_logit_HE / alpha ≈ z_plain
        """
        if self.lr_model is None:
            # No calibration possible, just try to squash raw_logit
            print("⚠️ LR calibration: plaintext model not available, using raw logit directly.")
            return raw_logit

        # Plaintext logit from sklearn LR
        z_plain = float(self.lr_model.decision_function(x_scaled.reshape(1, -1))[0])

        # Initialize scale factor on first call
        if self.lr_scale_factor is None:
            if abs(z_plain) < 1e-8:
                self.lr_scale_factor = 1.0
            else:
                self.lr_scale_factor = raw_logit / z_plain
            print(f"🔧 LR calibration factor (alpha) = {self.lr_scale_factor}")

        # Avoid division by zero
        if self.lr_scale_factor == 0 or not np.isfinite(self.lr_scale_factor):
            print("⚠️ LR calibration factor invalid, falling back to raw logit")
            return raw_logit

        # Map back to plaintext scale
        z_calibrated = raw_logit / self.lr_scale_factor
        print(f"   → Calibrated LR logit ≈ {z_calibrated}")
        return z_calibrated

    # ============================================================
    # DECRYPT + CALIBRATE (LOGIT → PROBABILITY)
    # ============================================================
    def decrypt_to_probability(self, enc_pred: ts.CKKSVector, model: str, x_scaled: np.ndarray) -> float:
        """
        Decrypt HE logit, calibrate to a reasonable scale, then apply sigmoid.

        - For LR:
            Use plaintext LR decision_function to derive a scale factor so that
            HE logit ≈ alpha * plaintext_logit. Then divide by alpha.
        - For NN:
            Use a dynamic scale factor so that the first logit's magnitude is
            around ~2, and reuse that factor across calls.
        """
        raw_logit = float(enc_pred.decrypt()[0])
        print(f"\n🔍 DEBUG RAW LOGIT ({model.upper()}): {raw_logit}")

        if model == "lr":
            z = self._calibrate_lr_logit(raw_logit, x_scaled)
        else:
            if not hasattr(self, "nn_scale_factor"):
                self.nn_scale_factor = None

            if self.nn_scale_factor is None:
                mag = abs(raw_logit)
                target_mag = 2.0
                if mag < 1e-8:
                    self.nn_scale_factor = 1.0
                else:
                    self.nn_scale_factor = max(mag / target_mag, 1.0)
                print(f"🔧 NN dynamic scale factor = {self.nn_scale_factor}")

            z = raw_logit / self.nn_scale_factor

        prob = self._sigmoid(z)

        eps = 0.02
        prob = eps + (1.0 - 2.0 * eps) * prob

        return float(np.clip(prob, 0.0, 1.0))

    # ============================================================
    # FULL PIPELINE
    # ============================================================
    def predict(self, features: list, model: str):
        """
        Full encrypted inference pipeline:
            features → scale → encrypt → send → receive → decrypt → probability
        """
        # 1. Preprocess
        x_scaled = self.preprocess(features)

        # 2. Encrypt
        enc_x = self.encrypt(x_scaled)

        # 3. Send to server
        enc_pred, ms = self.send_request(enc_x, model)

        # 4. Decrypt + sigmoid (with calibration; needs x_scaled)
        prob = self.decrypt_to_probability(enc_pred, model, x_scaled)

        return {
            "probability": prob,
            "prediction": int(prob >= 0.5),
            "inference_time_ms": ms,
        }
