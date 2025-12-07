# scripts/evaluate_encrypted_vs_plain.py

"""
Evaluate Plaintext vs Encrypted Inference

This script:
  - Loads the heart disease dataset
  - Uses the SAME StandardScaler as training
  - Compares plaintext vs encrypted predictions for LR/NN
  - Prints:
        * accuracy (plaintext vs label)
        * accuracy (encrypted vs label)
        * agreement between plaintext & encrypted
        * mean absolute error of probabilities
        * avg encrypted inference time

Usage:
    python scripts/evaluate_encrypted_vs_plain.py --server http://localhost:8000 --model lr --n 50
    python scripts/evaluate_encrypted_vs_plain.py --server http://localhost:8000 --model nn --n 50
"""

import argparse
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import tenseal as ts
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# allow importing from src/
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import HeartDiseasePreprocessor
from src.data.he_preprocessing import he_preprocess
CKKS_SCALE = 2 ** 20  # this matches what you saw printed (1048576)
NN_TEMP = 4.0        # temperature for NN logit calibration


# ============================================================
# Simple HE client (like DemoClient, but reused here)
# ============================================================
class EvalClient:
    def __init__(self, server_url, scaler):
        self.server_url = server_url.rstrip("/")
        self.scaler = scaler
        self.context = None

    def init_context(self):
        print("\n[HE] Initializing evaluation client...")

        # Health check
        r = requests.get(f"{self.server_url}/health")
        if r.status_code != 200:
            raise RuntimeError("❌ Server not healthy")

        # Fetch public HE params
        r = requests.get(f"{self.server_url}/context")
        if r.status_code != 200:
            raise RuntimeError("❌ Failed to fetch CKKS parameters")

        data = r.json()
        poly = data["poly_modulus_degree"]
        bits = data["coeff_mod_bit_sizes"]

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

        print("✓ HE context ready for evaluation")

    def encrypt(self, x_scaled: np.ndarray):
        return ts.ckks_vector(self.context, x_scaled.tolist())

    def send(self, enc_vec, model: str):
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
# Helper
# ============================================================
def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def make_plain_he_predictor(params: dict):
    """
    Returns a callable that computes plaintext probabilities using the
    same HE-friendly architecture (shallow, client-side sigmoid):
        z1 = W1·x + b1
        a1 = 0.5*z1 + 0.5
        z2 = W2·a1 + b2    # raw logit
        prob = sigmoid(z2)
    """
    W1 = np.array(params["W1"])
    b1 = np.array(params["b1"])
    W2 = np.array(params["W2"])
    b2 = np.array(params["b2"])

    def predict(x_scaled_row: np.ndarray) -> float:
        z1 = W1.dot(x_scaled_row) + b1
        a1 = 0.5 * z1 + 0.5
        z2 = W2.dot(a1) + b2
        prob = sigmoid(float(z2[0]))
        return max(0.0, min(1.0, prob))

    return predict


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True, choices=["lr", "nn"])
    parser.add_argument("--n", type=int, default=50, help="Number of test samples to evaluate")
    args = parser.parse_args()

    print("\n======================================================================")
    print(" EVALUATION: Plaintext vs Encrypted Inference")
    print("======================================================================")
    print(f"Server: {args.server}")
    print(f"Model:  {args.model.upper()}")
    print(f"Samples to evaluate: {args.n}")

    # --------------------------------------------------------
    # 1. Load dataset (RAW features)
    # --------------------------------------------------------
    pre = HeartDiseasePreprocessor()
    df = pre.load_data("data/raw/heart_disease.csv")  # same file you use elsewhere

    # Raw X / y (no scaling here)
    df = df.dropna()
    X_raw = df.drop(columns=["target"])
    y = df["target"].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # restrict to first N
    X_eval_raw = X_test_raw.iloc[:args.n].copy()
    y_eval = y_test.iloc[:args.n].copy()

    print(f"Prepared evaluation set with {len(X_eval_raw)} samples.")

    # --------------------------------------------------------
    # 2. Load scaler + plaintext models
    # --------------------------------------------------------
    scaler_path = "models/plaintext/preprocessor.pkl"
    scaler = joblib.load(scaler_path)
    print(f"✓ Loaded scaler from {scaler_path}")

    if args.model == "lr":
        # sklearn logistic regression (fitted on SCALED features)
        lr_bundle = joblib.load("models/plaintext/logistic_regression.pkl")
        lr_model = lr_bundle["model"]
        print("✓ Loaded plaintext Logistic Regression model")
    else:
        # HE-friendly NN parameters (same as encrypted NN architecture)
        nn_params_path = "models/plaintext/nn_he_parameters.pkl"
        nn_params = joblib.load(nn_params_path)
        print(f"✓ Loaded HE-friendly NN parameters from {nn_params_path}")
        plain_he_predict = make_plain_he_predictor(nn_params)

    # --------------------------------------------------------
    # 3. Init HE evaluation client
    # --------------------------------------------------------
    client = EvalClient(args.server, scaler)
    client.init_context()

    # --------------------------------------------------------
    # 4. Loop over samples and compare
    # --------------------------------------------------------
    pt_probs = []
    he_probs = []
    pt_labels = []
    he_labels = []
    latencies = []

    for idx, (i, row) in enumerate(X_eval_raw.iterrows(), start=1):
        x_raw = row.values.reshape(1, -1)

        # Scale (same scaler for both plaintext + encrypted)
        x_scaled = scaler.transform(x_raw)

        # ----- Plaintext prediction -----
        if args.model == "lr":
            pt_prob = float(lr_model.predict_proba(x_scaled)[0, 1])
        else:
            pt_prob = float(plain_he_predict(x_scaled[0]))

        pt_label = int(pt_prob >= 0.5)

        # ----- Encrypted prediction -----
        if args.model == "lr":
            enc_x = client.encrypt(x_scaled[0])
        else:
            enc_x = client.encrypt(he_preprocess(x_scaled[0]))
        try:
            enc_out, ms = client.send(enc_x, args.model)
        except Exception as e:
            print(f"[{idx}] HE prediction failed: {e}")
            continue

        latencies.append(ms)

        raw_logit = float(enc_out.decrypt()[0])
        temp = 8.0 if args.model == "nn" else 1.0
        z = max(-10.0, min(10.0, raw_logit / temp))
        he_prob = 1.0 / (1.0 + math.exp(-z))
        eps = 0.02
        he_prob = eps + (1.0 - 2.0 * eps) * he_prob
        he_prob = max(0.0, min(1.0, he_prob))
        he_label = int(he_prob >= 0.5)

        pt_probs.append(pt_prob)
        he_probs.append(he_prob)
        pt_labels.append(pt_label)
        he_labels.append(he_label)

        print(f"[{idx:02d}] y_true={int(y_eval.iloc[idx-1])} | "
              f"PT_prob={pt_prob:.4f} HE_prob={he_prob:.4f} | "
              f"PT={pt_label} HE={he_label} | {ms:.1f} ms")

    # --------------------------------------------------------
    # 5. Metrics
    # --------------------------------------------------------
    if not pt_probs:
        print("\nNo successful HE predictions, cannot compute metrics.")
        return

    y_true = y_eval.iloc[:len(pt_probs)].values

    pt_acc = accuracy_score(y_true, pt_labels)
    he_acc = accuracy_score(y_true, he_labels)
    agreement = accuracy_score(pt_labels, he_labels)
    mae = mean_absolute_error(pt_probs, he_probs)
    avg_ms = sum(latencies) / len(latencies)

    print("\n======================================================================")
    print(" METRICS")
    print("======================================================================")
    print(f"Plaintext Accuracy      : {pt_acc*100:.2f}%")
    print(f"Encrypted Accuracy      : {he_acc*100:.2f}%")
    print(f"PT vs HE label agreement: {agreement*100:.2f}%")
    print(f"Mean Abs Error (probs)  : {mae:.4f}")
    print(f"Avg HE inference time   : {avg_ms:.2f} ms")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
