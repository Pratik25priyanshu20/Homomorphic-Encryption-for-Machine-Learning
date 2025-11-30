# scripts/run_client_demo.py

import argparse
import time
import numpy as np
import joblib
from pathlib import Path
import requests
import tenseal as ts

CKKS_SCALE = 2 ** 30   # MUST MATCH: server, LR, NN, client


# ============================================================
# Load StandardScaler
# ============================================================
def load_scaler():
    path = "models/plaintext/preprocessor.pkl"
    scaler = joblib.load(path)
    print(f"\n📂 Loaded scaler from {path}")
    return scaler


# ============================================================
# Client wrapper for this demo script only
# ============================================================
class DemoClient:
    def __init__(self, server_url, scaler):
        self.server_url = server_url.rstrip("/")
        self.scaler = scaler
        self.context = None

    # --------------------------------------------------------
    # Fetch HE params + reconstruct CKKS context
    # --------------------------------------------------------
    def init_context(self):
        print("\nInitializing client...")

        # Health
        r = requests.get(f"{self.server_url}/health")
        if r.status_code != 200:
            raise RuntimeError("❌ Server not healthy")

        # Public params
        r = requests.get(f"{self.server_url}/context")
        if r.status_code != 200:
            raise RuntimeError("❌ Failed to fetch CKKS parameters")

        data = r.json()
        poly = data["poly_modulus_degree"]
        bits = data["coeff_mod_bit_sizes"]

        # Recreate CKKS context (client owns secret key)
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

        print("✓ Client CKKS context reconstructed")
        print(f"✓ Using global_scale = {CKKS_SCALE}")
        print("✓ Ready for encrypted inference")

    # --------------------------------------------------------
    # Encrypt vector
    # --------------------------------------------------------
    def encrypt(self, x_scaled: np.ndarray):
        return ts.ckks_vector(self.context, x_scaled.tolist())

    # --------------------------------------------------------
    # Send encrypted inference request
    # --------------------------------------------------------
    def send(self, enc_vec, model):
        endpoint = "/predict/lr" if model == "lr" else "/predict/nn"
        url = f"{self.server_url}{endpoint}"

        payload = {"encrypted_data": list(enc_vec.serialize())}

        t0 = time.time()
        r = requests.post(url, json=payload)
        elapsed_ms = (time.time() - t0) * 1000.0

        if r.status_code != 200:
            raise RuntimeError(f"❌ Server error {r.status_code}: {r.text}")

        enc_out = ts.ckks_vector_from(self.context, bytes(r.json()["encrypted_prediction"]))
        return enc_out, elapsed_ms


# ============================================================
# Helper printing
# ============================================================
def header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True, choices=["lr", "nn"])
    args = parser.parse_args()

    print("\n======================================================================")
    print("PRIVATE ENCRYPTED ML CLIENT DEMO")
    print("======================================================================")

    # --------------------------------------------------------
    # STEP 1 — Load scaler
    # --------------------------------------------------------
    scaler = load_scaler()

    # --------------------------------------------------------
    # STEP 2 — Init HE client
    # --------------------------------------------------------
    client = DemoClient(args.server, scaler)
    client.init_context()

    print(f"Connected to server: {args.server}")
    print("Encryption context established.")

    # --------------------------------------------------------
    # STEP 3 — Use Model
    # --------------------------------------------------------
    header(f"STEP 2 — Using Model: {args.model.upper()}")
    print(f"Selected Model: {args.model.upper()}")

    # --------------------------------------------------------
    # Patients
    # --------------------------------------------------------
    header("STEP 3 — Encrypted Predictions")

    patients = {
        "Patient A": np.array([62, 1, 0, 130, 263, 0, 1, 97, 1, 1.2, 0, 1, 3]),
        "Patient B": np.array([45, 0, 1, 120, 180, 0, 1, 150, 0, 0.2, 1, 0, 2]),
        "Patient C": np.array([58, 1, 2, 140, 200, 0, 0, 120, 0, 0.4, 1, 1, 1]),
    }

    inference_times = []

    for name, x in patients.items():
        print(f"\n{name}")
        print("-" * 30)

        # 1) Preprocess
        x_scaled = scaler.transform([x])[0]

        # 2) Encrypt
        enc_x = client.encrypt(x_scaled)

        # 3) Send encrypted
        try:
            enc_pred, ms = client.send(enc_x, args.model)
        except Exception as e:
            print(f"Prediction failed: {e}")
            continue

        inference_times.append(ms)

        # 4) Decrypt + RESCALE fix
        raw_val = float(enc_pred.decrypt()[0])
        prob = raw_val / CKKS_SCALE

        # Clip to [0,1]
        prob = max(0.0, min(1.0, prob))

        # Class
        pred_class = int(prob > 0.5)

        # Risk label
        if 0.45 <= prob <= 0.55:
            risk = "MEDIUM"
        elif pred_class == 1:
            risk = "HIGH"
        else:
            risk = "LOW"

        print(f"Probability: {prob:.4f}")
        print(f"Predicted Class: {pred_class}")
        print(f"Risk Level: {risk}")
        print(f"Inference Time: {ms:.2f} ms")

    # --------------------------------------------------------
    # STEP 4 — Summary
    # --------------------------------------------------------
    header("STEP 4 — Summary")

    if inference_times:
        avg_ms = sum(inference_times) / len(inference_times)
        print(f"Total Predictions: {len(inference_times)}")
        print(f"Average Inference Time: {avg_ms:.2f} ms")
    else:
        print("No successful predictions.")

    print(f"\nModel Used: {args.model.upper()}")
    print("\nDemo complete.\n")


if __name__ == "__main__":
    main()