# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import tenseal as ts

from src.encryption.context import EncryptionContextManager
from src.models.encrypted_lr import EncryptedLogisticRegression
from src.models.encrypted_nn import EncryptedNeuralNetwork


app = FastAPI(
    title="Private ML Server",
    description="Privacy-Preserving Machine Learning using CKKS Homomorphic Encryption",
    version="2.0.0",
)


# ============================================================
# Request model
# ============================================================
class PredictionRequest(BaseModel):
    encrypted_data: list  # serialized CKKSVector → list[int]


# ============================================================
# Global objects
# ============================================================
context: ts.Context | None = None

lr_model: EncryptedLogisticRegression | None = None
nn_model: EncryptedNeuralNetwork | None = None


# ============================================================
# STARTUP: Initialize server, load context, load models
# ============================================================
@app.on_event("startup")
def initialize_server():
    global context, lr_model, nn_model

    print("\n====================================================")
    print("🚀 PRIVATE ML SERVER — STARTING")
    print("====================================================\n")

    # -----------------------------------------------------
    # 1. Load client-provided CKKS context + evaluation keys
    # -----------------------------------------------------
    print("🔐 Loading shared CKKS context + eval keys...")
    ctx_path = Path("models/encrypted/context.bin")
    galois_path = Path("models/encrypted/galois.bin")
    relin_path = Path("models/encrypted/relin.bin")

    try:
        with open(ctx_path, "rb") as f:
            context_bytes = f.read()
        ctx = ts.context_from(context_bytes)

        if galois_path.exists():
            with open(galois_path, "rb") as f:
                ctx.load_galois_keys(f.read())
        if relin_path.exists():
            with open(relin_path, "rb") as f:
                ctx.load_relin_keys(f.read())

        ctx.auto_relin = True
        context = ctx
        print(f"✓ Loaded CKKS context from {ctx_path}")
        if galois_path.exists():
            print(f"✓ Loaded Galois keys from {galois_path}")
        if relin_path.exists():
            print(f"✓ Loaded Relinearization keys from {relin_path}")
        print("")
    except Exception as e:
        raise RuntimeError(f"Failed to load shared CKKS context/keys: {e}")

    # -----------------------------------------------------
    # 2. Load Logistic Regression
    # -----------------------------------------------------
    print("📘 Loading Logistic Regression model...")
    lr_model = EncryptedLogisticRegression(context)
    # Prefer HE-normalized parameters if present; otherwise fallback.
    he_lr_path = "models/plaintext/lr_he_parameters.pkl"
    if Path(he_lr_path).exists():
        lr_model.load_he_parameters(he_lr_path)
    else:
        lr_model.load_plaintext_model("models/plaintext/logistic_regression.pkl")
    print("✓ Logistic Regression ready.\n")

    # -----------------------------------------------------
    # 3. Load Neural Network
    # -----------------------------------------------------
    print("🤖 Loading Encrypted Neural Network...")
    nn_model = EncryptedNeuralNetwork(context)
    nn_model.load_parameters("models/plaintext/nn_he_parameters.pkl")
    print("✓ Neural Network ready.\n")

    print("🎉 SERVER INITIALIZED — READY TO SERVE ENCRYPTED INFERENCE!\n")


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# PUBLIC CONTEXT INFO FOR CLIENTS
# ============================================================
@app.get("/context")
def get_public_context():
    """
    Returns ONLY the public parameters needed to build a CKKS context:
        - poly_modulus_degree
        - coeff_mod_bit_sizes

    SECRET KEY is never included.
    """
    try:
        mgr = EncryptionContextManager()
        info = mgr.get_context_info()
        return {
            "poly_modulus_degree": info["poly_modulus_degree"],
            "coeff_mod_bit_sizes": info["coeff_mod_bit_sizes"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Logistic Regression Inference
# ============================================================
@app.post("/predict/lr")
def predict_lr(request: PredictionRequest):
    if lr_model is None:
        raise HTTPException(500, "LR model not initialized")

    try:
        enc_bytes = bytes(request.encrypted_data)
        enc_x = ts.ckks_vector_from(context, enc_bytes)
    except Exception as e:
        raise HTTPException(400, f"Invalid encrypted input: {e}")

    try:
        # Return encrypted logit; client applies sigmoid
        enc_out = lr_model.predict_encrypted_logit(enc_x)

        pred_bytes = list(enc_out.serialize())
        return {"encrypted_prediction": pred_bytes}

    except Exception as e:
        raise HTTPException(500, detail=f"Inference error: {e}")

# ============================================================
# Neural Network Inference
# ============================================================
@app.post("/predict/nn")
def predict_nn(request: PredictionRequest):
    if nn_model is None:
        raise HTTPException(500, "NN model not initialized")

    # Convert client-sent bytes → encrypted vector
    try:
        enc_bytes = bytes(request.encrypted_data)
        enc_x = ts.ckks_vector_from(context, enc_bytes)
    except Exception as e:
        raise HTTPException(400, f"Invalid encrypted input: {e}")

    # Run encrypted NN inference
    try:
        enc_out = nn_model.predict_logit_encrypted(enc_x)
        pred_bytes = list(enc_out.serialize())
        return {"encrypted_prediction": pred_bytes}
    except Exception as e:
        raise HTTPException(500, detail=f"Inference error: {e}")


# ============================================================
# Root
# ============================================================
@app.get("/")
def root():
    return {"message": "Private HE-ML Server Running"}
