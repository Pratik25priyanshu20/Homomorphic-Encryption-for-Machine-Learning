# 🔒 Homomorphic Encryption ML (CKKS) — FastAPI • TenSEAL • PyTorch • Streamlit

End-to-end, privacy-preserving ML for heart-disease prediction on encrypted data. The client encrypts locally, the server computes on ciphertext (CKKS), and only the client can decrypt the result.

Table of contents
- About
- Why it matters
- Problem Statement — The €2B GDPR Challenge
- Quick architecture overview
- Quickstart
- How it works
- Files & layout
- HE / model notes
- Troubleshooting & tips
- Data
- Privacy guarantee
- Status & next steps
- Contributing & license

---

## About
This repository demonstrates practical homomorphic-encrypted inference using the CKKS scheme (TenSEAL). It includes:
- a FastAPI server that accepts encrypted feature vectors and returns encrypted logits,
- a Python client SDK that builds CKKS context, encrypts inputs, sends requests and decrypts predictions,
- training scripts for HE-friendly logistic regression and shallow neural networks,
- calibration/evaluation scripts and a Streamlit dashboard.

The goal: enable cloud inference without ever exposing plaintext data (client holds the secret key).

---

## Why it matters
- GDPR-grade privacy: secret key remains client-side; server only sees ciphertexts.
- Accurate & usable: encrypted predictions align with plaintext after calibration.
- Real-world-ready stack: FastAPI, Python SDK, PyTorch, TenSEAL, Streamlit.
- Latency: typical CPU latency for encrypted LR/NN is ~0.7–1.1s per inference in demo settings (tunable).

## Problem Statement — The €2B GDPR Challenge
- GDPR Article 32 → data must be encrypted  
- ML inference → normally needs plaintext  
- Sharing medical/financial data → violates GDPR Article 9  
**Result:** €2B+ in trapped data value every year (no safe way to run models in the cloud).

Real-World Impact  
- 🏥 Hospitals cannot use cloud AI without exposing patient data  
- 🏦 Banks cannot outsource fraud detection without sharing transactions  
- 🛡️ Insurers cannot evaluate risk using external ML providers

This project solves this by running ML directly on encrypted data with CKKS: the server never decrypts, the client holds the only secret key, and predictions remain accurate enough for real use.

---

## Quick architecture overview
Client (private key)
  - Preprocess features → scale → encrypt (CKKS)
  - Send ciphertext to server → receive encrypted logit → decrypt → calibrate → sigmoid → probability

Server (no secret key)
  - Load CKKS public/context + relin/galois (eval keys)
  - Run encrypted LR/NN on ciphertexts
  - Return encrypted logit

Diagram (conceptual)
HOMOMORPHIC_ENCRYPTION_ML/
├── api/ (FastAPI encrypted inference)
├── client/ (Python SDK, encryption + decrypt)
├── dashboard/ (Streamlit UI)
├── src/ (encrypted models, preprocessing)
├── scripts/ (train / evaluate / calibrate / demo)
└── models/ (plaintext and encrypted artifacts)

---

## Quickstart

Prereqs
- Python 3.10+
- Install: `pip install -r requirements.txt`
- TenSEAL 0.3.x-compatible environment (check your platform for wheels)

1) Train models (plaintext + HE-safe export)
```bash
python scripts/train_models.py      # trains LR and exports lr_he_parameters.pkl
python scripts/train_he_nn.py       # trains shallow HE NN; exports nn_he_parameters.pkl
```

2) Prepare CKKS artifacts (client-side generation)
Place these files where the server loads them:
```
models/encrypted/context.bin
models/encrypted/galois.bin
models/encrypted/relin.bin
```
Important: these must be generated with matching poly_modulus_degree / coeff_mod_bit_sizes / global_scale. Server loads them — it does not generate secret keys.

3) Start the server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Run client demo (CLI)
```bash
python scripts/run_client_demo.py --server http://localhost:8000 --model lr
python scripts/run_client_demo.py --server http://localhost:8000 --model nn
```

5) (Optional) Calibrate logits → probabilities
Calibration adjusts the decrypted (but scaled) logits so that decrypted encrypted outputs align with plaintext model outputs.
```bash
python scripts/calibrate_models.py
# produces models/plaintext/calibration_lr.pkl and calibration_nn.pkl
```
If these exist, the client SDK auto-loads them.

6) Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Use the sidebar to connect to a server, select LR/NN, run encrypted predictions and inspect analytics.

7) Compare encrypted vs plaintext
```bash
python scripts/evaluate_encrypted_vs_plain.py --server http://localhost:8000 --model lr --n 50
python scripts/evaluate_encrypted_vs_plain.py --server http://localhost:8000 --model nn --n 50
```

8) Tests
```bash
pytest tests/unit/test_encryption.py
```

---

## How it works (detail)
- Client builds CKKS context locally (public params + secret key) and optionally requests server `/context` (public params) to ensure param alignment.
- Features are preprocessed and scaled to fixed-point using a SAFE_SCALE constant to avoid CKKS overflow.
- Client encrypts feature vector(s) and sends ciphertext(s) to the server via REST endpoints.
- Server evaluates encrypted linear layers and polynomial-style activations supported under CKKS (NN uses HE-friendly approximations).
- Server returns an encrypted scalar logit (ciphertext) to client.
- Client decrypts the logit, applies affine calibration (if available) and the sigmoid (in plaintext) to produce the final probability.

Notes:
- Sigmoid is applied client-side because CKKS evaluation of a numerically-stable sigmoid is expensive; instead we use calibration + sigmoid in plaintext.
- Models are trained with HE-safe parameter scaling (weights are shrunk using SAFE_SCALE during export).

---

## HE / model notes
- Scheme: CKKS (TenSEAL 0.3.x), aiming for 128-bit security.
- Demo params: poly_modulus_degree = 8192 (or 16384 for deeper nets).
- Example coeff_mod_bit_sizes: [60, 40, 40, 40, 60]; typical global_scale = 2**40.
- Keep client/server CKKS parameters identical.
- Exported HE parameters include weight scaling to avoid overflow; client calibration corrects for that shrinkage.
- Models provided:
  - scikit-learn logistic regression (LR)
  - PyTorch shallow HE-friendly neural network:
    - linear layers
    - approximate activation: 0.5*x + 0.5*x^3 or custom polynomial approximations
- TenSEAL multi-vector packing is used where appropriate.

---

## Files & layout (key files)
- `api/main.py` — FastAPI server; encrypted inference endpoints & context loading.
- `client/client.py` — PrivateMLClient SDK; context build, encrypt, request, decrypt + calibration.
- `scripts/train_models.py` — Train LR & export HE parameters for LR.
- `scripts/train_he_nn.py` — Train HE-compatible NN & export HE parameters for NN.
- `scripts/run_client_demo.py` — Demo CLI for LR/NN encrypted inference.
- `scripts/evaluate_encrypted_vs_plain.py` — Script that compares plaintext and encrypted outputs.
- `scripts/calibrate_models.py` — Fits affine corrections for logits (encrypted → plaintext alignment).
- `dashboard/app.py` — Streamlit UI for predictions and analytics.
- `src/models/encrypted_lr.py`, `src/models/encrypted_nn.py` — CKKS inference implementations.
- `src/data/preprocessor.py`, `src/data/he_preprocessing.py` — Scaling & HE-safe preprocessing utilities.

---

## Troubleshooting & tips
- Context mismatch → server `/context` may return 500 or server may error on evaluation:
  - Ensure `context.bin`, `galois.bin`, `relin.bin` exist and were generated using the same CKKS params (poly_modulus_degree, coeff_mod_bit_sizes, global_scale).
- Extreme 0/1 probabilities:
  - Retrain models (`train_models.py`, `train_he_nn.py`) then run `calibrate_models.py`.
  - Confirm calibration PKLs are present under `models/plaintext/`.
- Latency:
  - Expected ~0.7–1.1s per encrypted inference on CPU for demo models.
  - Use larger poly_modulus_degree and deeper coeff mods only if you need more precision, at the cost of latency and memory.
- Security:
  - Secret key MUST stay client-side. Server should only have context/eval keys.
  - Do not commit secret keys to the repo.
- Debugging:
  - Use small synthetic inputs to validate pipeline step-by-step: preprocess → plaintext model inference → HE-encrypt → server inference → decrypt → compare.

---

## Data
Uses `data/raw/heart_disease.csv` (UCI Heart Disease dataset). Ensure it exists at that path. Scripts assume the same column names used by the preprocessor in `src/data/preprocessor.py`.

---

## Privacy guarantee
- Server computes on encrypted vectors only.
- Server never sees raw patient features.
- Only the client holds secret keys and decrypts predictions.
- Approach aligns with GDPR Articles 9, 25, 32, 35 when deployed correctly.

---

## Status
- Scripts, dashboard and basic encrypted inference pipeline are working in the demo environment.
- Encrypted outputs depend on CKKS params and calibration. If you change CKKS parameters, regenerate context files and retrain/recalibrate models.

---

## Next steps & suggestions
- Add CI that runs training & end-to-end encrypted inference on a small synthetic dataset to catch regressions.
- Add end-to-end integration tests that simulate client/server interaction with generated CKKS contexts.
- Provide Docker images for server and client to simplify deployment.
- Add docs for generating CKKS artifacts and secure distribution of eval keys.

---

## Contributing
Contributions are welcome. Please open issues/PRs for bugs, feature requests, or improvements. When contributing HE parameters or sample contexts, never include secret keys.

---

## License
Specify your license here (e.g., MIT). Remove or change as appropriate.
