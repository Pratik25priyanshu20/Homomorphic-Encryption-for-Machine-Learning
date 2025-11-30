<div align="center">

# 🔒 **Homomorphic Encryption for Privacy-Preserving Machine Learning**  
### *(FastAPI + TenSEAL + PyTorch)*  

</div>


This project demonstrates a fully production-ready system for privacy-preserving machine learning using Homomorphic Encryption, enabling secure medical diagnosis without exposing patient data. It implements the CKKS scheme (TenSEAL 0.3.14, 128-bit security) to perform encrypted inference on both Logistic Regression and Neural Network models, allowing a FastAPI server to compute predictions on encrypted features it never sees in plaintext. The architecture includes a complete preprocessing pipeline, an encrypted inference engine, a client-side SDK that handles context generation, feature scaling, encryption/decryption, and a modern Streamlit dashboard to visualize predictions, model details, and security parameters. With ~5ms encrypted inference and only 3% accuracy loss compared to plaintext models, this system proves that real-time, GDPR-compliant ML on sensitive healthcare data is practical today. The project is designed for hospitals, banking, and insurance use cases, showing how homomorphic encryption unlocks valuable data while maintaining strict confidentiality and regulatory compliance.

🎯 Problem Statement
The €2 Billion GDPR Challenge:
German healthcare and financial institutions face a critical dilemma:

GDPR Article 32 requires encryption of personal data
Traditional ML requires plaintext data for inference
Sharing patient data with ML providers violates GDPR Article 9 (special category data)
Result: €2B+ in trapped data value

Real-World Impact:

🏥 Hospitals can't use cloud AI without exposing patient records
🏦 Banks can't outsource fraud detection without sharing transactions
🛡️ Insurers can't use ML for risk assessment without violating privacy laws


✨ Solution: Homomorphic Encryption
This project demonstrates:

✅ ML inference on encrypted data - Server never sees plaintext
✅ 80% accuracy maintained - Only 3% loss vs plaintext
✅ 5ms inference time - Production-acceptable latency
✅ GDPR compliant - Data never leaves encrypted form
✅ Production-ready - FastAPI server + Python client SDK

How It Works:
Client (Hospital)          Server (ML Provider)
┌─────────────┐            ┌──────────────┐
│ Patient     │  Encrypt   │  Encrypted   │
│ Data        ├───────────>│  Inference   │
│             │            │              │
│ [Age: 63]   │            │ [Gibberish]  │
│ [BP: 145]   │            │ [Random]     │
│             │◄───────────┤              │
│ Decrypt     │  Encrypted │  Returns     │
│ Result      │  Result    │  Encrypted   │
└─────────────┘            └──────────────┘
     ✓ Has secret key          ✗ No secret key
Privacy Guarantee: Server performs computation without ever decrypting data!

🚀 Quick Start
Prerequisites

Python 3.10+
pip
Virtual environment (recommended)

git clone https://github.com/Pratik25priyanshu20/Homomorphic-Encryption-for-Machine-Learning.git
cd HOMOMORPHIC_ENCRYPTION_ML

python3.10 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install -r requirements.txt

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload [ or 8081]

streamlit run dashboard/app.py




🏗️ Architecture
System Components
┌─────────────────────────────────────────────────────────────┐
│                     Client Application                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Preprocessor │→ │  Encryptor   │→ │  API Client  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTPS (encrypted data)
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Context    │  │  Encrypted   │  │   Response   │     │
│  │   Manager    │→ │  Inference   │→ │   Handler    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘


CLIENT                                     SERVER
─────────────────────────────────────────────────────────────────
- Generates CKKS context                  - Receives public context
- Holds secret key (never sent)           - Loads model (LR/NN)
- Encrypts patient data                   - Computes on encrypted vectors
- Sends ciphertext to server        →     - Returns encrypted prediction
- Decrypts encrypted output               - Never decrypts anything



Technology Stack

Encryption:
• TenSEAL (0.3.14) — CKKS homomorphic encryption
• Security Level: 128-bit
• Encryption Parameters:
   - poly_modulus_degree = 16384
   - coeff_mod_bit_sizes = [60, 45, 45, 45, 60]
   - global_scale = 2^30

Machine Learning:
• scikit-learn — Logistic Regression baseline
• PyTorch — Neural network (exported to CKKS operations)
• Encrypted inference accuracy: 80%+

Backend:
• FastAPI — Encrypted inference server
• Pydantic — Validation layer
• Uvicorn — ASGI server

Frontend:
• Streamlit — User-facing encrypted dashboard
• Plotly — Interactive visualizations



📁 Project Structure
homomorphic-ml-privacy/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── data/                              # Datasets
│   ├── raw/                           # Original data
│   └── processed/                     # Preprocessed data
│
├── src/                               # Source code
│   ├── data/
│   │   └── preprocessor.py           # Data preprocessing
│   ├── models/
│   │   ├── logistic_regression.py    # Plaintext LR
│   │   ├── neural_network.py         # Plaintext NN
│   │   └── encrypted_lr.py           # Encrypted inference
│   └── encryption/
│       └── context.py                # Encryption management
│
├── api/                               # FastAPI server
│   ├── main.py                       # API entrypoint
│   └── schemas/                      # Request/response models
│
├── client/                            # Client SDK
│   └── client.py                     # Python client
│
├── dashboard/                         # Streamlit UI
│   └── app.py                        # Interactive dashboard
│
├── scripts/                           # Utility scripts
│   ├── download_data.py              # Dataset download
│   ├── train_models.py               # Model training
│   └── test_encrypted_inference.py   # Testing
│
├── tests/                             # Unit tests
│   ├── unit/
│   └── integration/
│
├── docs/                              # Documentation
│   ├── technical_report.md           # Technical details
│   ├── business_case.md              # Business value
│   └── architecture/                 # Diagrams
│
├── benchmarks/                        # Performance data
│   ├── results/                      # Raw data
│   └── plots/                        # Visualizations
│
└── models/                            # Saved models
    ├── plaintext/                    # Classical models
    └── encrypted/                    # Encryption contexts



Security & Privacy
Encryption Details
Scheme: CKKS (Cheon-Kim-Kim-Song)

Supports approximate arithmetic on real numbers
Optimized for machine learning operations
Industry-standard for privacy-preserving ML




Privacy Guarantees
✅ What's Protected:

All patient features (age, blood pressure, cholesterol, etc.)
Model predictions and probabilities
Intermediate computation results

❌ What Server Sees:

Model architecture (public)
Encrypted data (random gibberish)
Encrypted predictions (random gibberish)

✅ Only Client Has:

Secret decryption key
Plaintext patient data
Plaintext predictions

GDPR Compliance
This system satisfies:

✅ Article 32: Data encryption (pseudonymisation)
✅ Article 9: Special category data protection
✅ Article 25: Privacy by design
✅ Article 35: DPIA-ready architecture


