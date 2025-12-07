"""
Calibrate HE logits to match plaintext logits (LR + NN).
Generates linear correction:
    plain_logit ≈ a * he_logit + b
"""

import numpy as np
import joblib
from pathlib import Path
import random

import requests
import tenseal as ts

from src.data.preprocessor import HeartDiseasePreprocessor
from src.models.neural_network import NeuralNetworkModel
from src.models.logistic_regression import LogisticRegressionModel
from src.data.he_preprocessing import he_preprocess


SERVER = "http://localhost:8000"
SAMPLES = 300  # number of calibration points

CKKS_SCALE = 2 ** 40


# -----------------------------------------
# Client loader
# -----------------------------------------
def load_context():
    r = requests.get(f"{SERVER}/context")
    data = r.json()

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=data["poly_modulus_degree"],
        coeff_mod_bit_sizes=data["coeff_mod_bit_sizes"],
    )
    ctx.global_scale = CKKS_SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.auto_relin = True
    return ctx


def encrypt(ctx, x):
    return ts.ckks_vector(ctx, x.tolist())


def send(enc_x, model):
    endpoint = "/predict/lr" if model == "lr" else "/predict/nn"
    r = requests.post(f"{SERVER}{endpoint}", json={"encrypted_data": list(enc_x.serialize())})
    enc_out = ts.ckks_vector_from(ctx, bytes(r.json()["encrypted_prediction"]))
    return float(enc_out.decrypt()[0])


# -----------------------------------------
# TRAIN LR + NN IN PLAINTEXT
# -----------------------------------------
def train_plain_models():
    pre = HeartDiseasePreprocessor()
    df = pre.load_data("data/raw/heart_disease.csv")
    X_train, X_test, y_train, y_test = pre.prepare_data(df)

    scaler = joblib.load("models/plaintext/preprocessor.pkl")

    # LR
    lr = LogisticRegressionModel()
    lr.train(X_train, y_train)

    # NN
    nn = NeuralNetworkModel(input_dim=X_train.shape[1], hidden_dims=(16,16))
    nn.train(X_train, y_train, epochs=50, batch_size=32)

    return scaler, lr, nn, X_test


# -----------------------------------------
# MAIN CALIBRATION PROCESS
# -----------------------------------------
if __name__ == "__main__":

    print("\n============================")
    print("📏 CALIBRATING HE MODELS...")
    print("============================")

    scaler, lr, nn, X_data = train_plain_models()
    ctx = load_context()

    he_logits_lr = []
    pt_logits_lr = []

    he_logits_nn = []
    pt_logits_nn = []

    for i in range(SAMPLES):
        x = X_data[random.randint(0, len(X_data)-1)]
        x_scaled = scaler.transform([x])[0]

        # plaintext logits
        pt_lr = float(lr.model.decision_function([x_scaled])[0])
        pt_nn = float(nn.forward_single(x_scaled))

        # LR encrypted
        enc_x_lr = encrypt(ctx, x_scaled)
        he_lr = send(enc_x_lr, "lr")

        # NN encrypted
        x_he = he_preprocess(x_scaled)
        enc_x_nn = encrypt(ctx, x_he)
        he_nn = send(enc_x_nn, "nn")

        he_logits_lr.append(he_lr)
        pt_logits_lr.append(pt_lr)

        he_logits_nn.append(he_nn)
        pt_logits_nn.append(pt_nn)

    # Fit affine transforms
    a_lr, b_lr = np.polyfit(he_logits_lr, pt_logits_lr, 1)
    a_nn, b_nn = np.polyfit(he_logits_nn, pt_logits_nn, 1)

    print("\nCalibration Results:")
    print(f"LR → a={a_lr}, b={b_lr}")
    print(f"NN → a={a_nn}, b={b_nn}")

    joblib.dump({"a": a_lr, "b": b_lr}, "models/plaintext/calibration_lr.pkl")
    joblib.dump({"a": a_nn, "b": b_nn}, "models/plaintext/calibration_nn.pkl")

    print("\nSaved:")
    print(" → models/plaintext/calibration_lr.pkl")
    print(" → models/plaintext/calibration_nn.pkl")
    print("\n🎉 Calibration complete!\n")