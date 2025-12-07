# 🔒 Homomorphic Encryption ML (CKKS) — FastAPI • TenSEAL • PyTorch • Streamlit

End-to-end, privacy-preserving ML for heart-disease prediction on encrypted data. The client encrypts locally, the server computes on ciphertext (CKKS), and only the client can decrypt the result.

### Why it matters
- **GDPR-grade privacy:** Client holds the secret key; server sees only random-looking ciphertexts.
- **Real models, real latency:** Encrypted LR/NN with ~0.7–1.1s CPU latency in current runs (tuneable).
- **Accuracy retained:** Encrypted predictions align with plaintext after calibration (see calibration step).
- **Production shape:** FastAPI server, Python SDK, Streamlit dashboard, calibration & eval scripts.


## Problem & Solution (quick story)
**The dilemma:** Sensitive medical/financial data must stay encrypted (GDPR, HIPAA, etc.), but ML inference normally needs plaintext.  
**This system:** Uses CKKS HE so the server never decrypts. Client-owned keys, encrypted features, encrypted logits back, plaintext never leaves the client. Works today with LR + shallow NN, with only modest latency overhead on CPU.

---
## Problem Statement — The €2B GDPR Challenge
- GDPR Article 32 → data must be encrypted  
- ML inference → normally needs plaintext  
- Sharing medical/financial data → violates GDPR Article 9  
**Result:** €2B+ in trapped data value every year (no safe way to run models in the cloud).

Real-World Impact  
- 🏥 Hospitals cannot use cloud AI without exposing patient data  
- 🏦 Banks cannot outsource fraud detection without sharing transactions  
- 🛡️ Insurers cannot evaluate risk using external ML providers  

**This project solves this** by running ML directly on encrypted data with CKKS: the server never decrypts, the client holds the only secret key, and predictions remain accurate enough for real use.




---
## What’s included
- FastAPI server that loads client-provided CKKS context + eval keys and runs encrypted LR/NN.
- Python client SDK (`client/client.py`) handling preprocess → encrypt → send → decrypt + calibration.
- Training scripts for plaintext + HE-friendly models.
- Evaluation and calibration scripts to align encrypted logits with plaintext.
- Streamlit dashboard for encrypted predictions, analytics, and tech details.

---
## Quickstart
Prereqs: Python 3.10+, `pip install -r requirements.txt`

1) Train and export HE-safe params  
```bash
python scripts/train_models.py      # trains LR/NN, saves lr_he_parameters.pkl
python scripts/train_he_nn.py       # trains shallow HE NN, saves nn_he_parameters.pkl
```

2) Provide CKKS artifacts (client side)  
Generate and place these for the server to load:  
```
models/encrypted/context.bin
models/encrypted/galois.bin
models/encrypted/relin.bin
```
(Ensure they are created with matching poly_modulus_degree / coeff_mod_bit_sizes / global_scale; server only loads them, does not generate keys.)

3) Start the server  
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Run a demo (CLI)  
```bash
python scripts/run_client_demo.py --server http://localhost:8000 --model lr
python scripts/run_client_demo.py --server http://localhost:8000 --model nn
```

5) (Optional) Calibrate logits → probs  
```bash
python scripts/calibrate_models.py
# saves models/plaintext/calibration_lr.pkl and calibration_nn.pkl
```
Client auto-loads these if present.

6) Streamlit dashboard  
```bash
streamlit run dashboard/app.py
```
Connect via sidebar, choose LR/NN, run encrypted predictions, view risk gauge + analytics.

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
## How it works
- Client rebuilds CKKS context from `/context` API (public params), encrypts scaled features, sends ciphertext.
- Server uses the shared context/keys (no secret key) to run LR/NN and returns an encrypted logit.
- Client decrypts logit, applies calibration (affine or dynamic scaling) → sigmoid → probability.

---
## Problem & Solution (quick story)
**The dilemma:** Sensitive medical/financial data must stay encrypted (GDPR, HIPAA, etc.), but ML inference normally needs plaintext.  
**This system:** Uses CKKS HE so the server never decrypts. Client-owned keys, encrypted features, encrypted logits back, plaintext never leaves the client. Works today with LR + shallow NN, with only modest latency overhead on CPU.

---
## Problem Statement — The €2B GDPR Challenge
- GDPR Article 32 → data must be encrypted  
- ML inference → normally needs plaintext  
- Sharing medical/financial data → violates GDPR Article 9  
**Result:** €2B+ in trapped data value every year (no safe way to run models in the cloud).

Real-World Impact  
- 🏥 Hospitals cannot use cloud AI without exposing patient data  
- 🏦 Banks cannot outsource fraud detection without sharing transactions  
- 🛡️ Insurers cannot evaluate risk using external ML providers  

**This project solves this** by running ML directly on encrypted data with CKKS: the server never decrypts, the client holds the only secret key, and predictions remain accurate enough for real use.

---

HOMOMORPHIC_ENCRYPTION_ML/
│
├── api/                     # FastAPI encrypted inference server
├── client/                  # Python SDK (encrypt → infer → decrypt)
├── dashboard/               # Streamlit UI
├── src/
│   ├── models/              # Encrypted LR + NN implementations
│   ├── data/                # Preprocessing & scaling
│   └── utils/
│
├── scripts/
│   ├── train_models.py
│   ├── train_he_nn.py
│   ├── run_client_demo.py
│   ├── evaluate_encrypted_vs_plain.py
│   ├── calibrate_models.py
│
├── models/
│   ├── plaintext/           # Scaler, LR, NN, HE parameters
│   └── encrypted/           # context.bin, galois.bin, relin.bin
│
└── data/
    └── raw/heart_disease.csv
---
## HE / Model notes
- Scheme: CKKS (TenSEAL 0.3.x), 128-bit security.  
- Typical params: poly_modulus_degree 8192/16384 (demo vs deeper), coeff_mod_bit_sizes similar to `[60, 40, 40, 40, 60]` variants, global_scale often `2**40`. Keep client/server aligned.
- Models: scikit-learn LR; PyTorch shallow HE-friendly NN (linear + 0.5x+0.5 activation).
- HE exports: weights shrunk (SAFE_SCALE) to avoid CKKS overflow; client calibration compensates.

---
## Project layout (key files)
- `api/main.py` — FastAPI server; loads shared context/keys; encrypted LR/NN endpoints.
- `client/client.py` — PrivateMLClient SDK; encryption, request, decrypt + calibration.
- `scripts/train_models.py` — Train LR/NN; export `lr_he_parameters.pkl`.
- `scripts/train_he_nn.py` — Train HE-friendly NN; export `nn_he_parameters.pkl`.
- `scripts/run_client_demo.py` — CLI demo for LR/NN.
- `scripts/evaluate_encrypted_vs_plain.py` — Plain vs encrypted metrics.
- `scripts/calibrate_models.py` — Fit affine corrections for logits.
- `dashboard/app.py` — Streamlit dashboard (Prediction, Analytics, Technical).
- `src/models/encrypted_lr.py`, `src/models/encrypted_nn.py` — CKKS inference.
- `src/data/preprocessor.py`, `src/data/he_preprocessing.py` — Scaling + HE-safe preprocessing.

---
## Tips / Troubleshooting
- Context mismatch → 500 on `/context`: ensure `context.bin/galois.bin/relin.bin` exist and match client params.
- Extreme 0/1 probs: rerun training (`train_models.py`, `train_he_nn.py`), then `calibrate_models.py`; ensure calibration PKLs exist.
- Latency: expect ~0.7–1.1s per encrypted inference on CPU; NN deeper variants may be slower.
- Secret key stays client-side. Server only holds public/eval keys and encrypted models.

---
## Data
Uses `data/raw/heart_disease.csv` (UCI Heart Disease). Ensure it exists in that path.

---

🛡️ Privacy Guarantee

✔ Server computes on encrypted vectors
✔ Server never sees patient features
✔ Only client holds secret key
✔ Encrypted predictions returned to client
✔ Fully aligns with GDPR Articles 9, 25, 32, 35
---
## Status
Scripts and dashboard run; encrypted outputs depend on current HE params and calibration. If you change CKKS parameters, regenerate context files and retrain/reevaluate.***
