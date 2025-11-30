<div align="center">

# 🔒 **Homomorphic Encryption for Privacy-Preserving Machine Learning**  
### *(FastAPI • TenSEAL • PyTorch • Streamlit)*  

</div>

This project demonstrates a **production-ready, end-to-end system for privacy-preserving machine learning** using **Homomorphic Encryption (HE)**.  
Using the **CKKS scheme** (TenSEAL 0.3.14, 128-bit security), the server can run **Logistic Regression** and **Neural Network** inference **directly on encrypted data** — **without ever decrypting it**.

The client encrypts patient features locally, sends ciphertexts to the server, and receives encrypted predictions back. Only the client has the secret key; the server sees nothing but random encrypted numbers.

With **~5ms encrypted inference latency** and only **3% accuracy loss**, this system proves that **real-time GDPR-compliant medical ML** is possible today.

It includes:

- A complete **CKKS encryption pipeline**
- A **FastAPI encrypted ML inference server**
- A Python **client SDK** (encryption → inference → decryption)
- A modern **Streamlit dashboard** for model insights
- Full **technical documentation**, **business case**, and **benchmarks**

This project is designed for **hospitals**, **banks**, and **insurance companies** looking to unlock sensitive data **without violating privacy laws**.

---

## 🎯 Problem Statement  
### **The €2 Billion GDPR Challenge**

German healthcare and financial institutions face a critical dilemma:

- GDPR Article 32 → **data must be encrypted**
- ML inference → **requires plaintext**
- Sharing patient financial or medical data → **violates GDPR Article 9**
- Result → **€2B+ in trapped data value every year**

### Real-World Impact  
- 🏥 **Hospitals** cannot use cloud AI without exposing patient data  
- 🏦 **Banks** cannot outsource fraud detection without sharing transactions  
- 🛡️ **Insurers** cannot evaluate risk using external ML providers  

This project solves this.

---

## ✨ Solution: Homomorphic Encryption

This system demonstrates:

- ✅ Machine learning **directly on encrypted data**
- ✅ **Server never sees plaintext**
- ✅ **Client holds the only secret key**
- ✅ **80%+ accuracy** on encrypted NN & LR
- ✅ **5ms encrypted inference** (fast enough for production)
- ✅ Fully **GDPR compliant** (Articles 9, 25, 32, 35)

---

## 🔐 How It Works



**Privacy Guarantee:**  
✔ Server performs computation  
✘ Server cannot decrypt anything  
✔ Client decrypts final result only  

---

## 🚀 Quick Start

### 1. Clone the Repo
```bash
git clone https://github.com/Pratik25priyanshu20/Homomorphic-Encryption-for-Machine-Learning.git
cd HOMOMORPHIC_ENCRYPTION_ML

python3.10 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt


uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

streamlit run dashboard/app.py



 Architecture

CLIENT (Hospital)
──────────────────────────────────────────────
• Preprocess data
• Generate CKKS keys
• Encrypt patient features
• Invoke FastAPI server
• Decrypt encrypted predictions

SERVER (ML Provider)
──────────────────────────────────────────────
• Loads encrypted context (no secret key)
• Loads LR/NN plaintext model
• Computes on encrypted vectors
• Returns encrypted outputs




Technology Stack

🔒 Encryption
	•	TenSEAL (CKKS, 128-bit security)
	•	poly_modulus_degree: 16384
	•	coeff_mod_bit_sizes: [60, 45, 45, 45, 60]
	•	global_scale: 2^30

🤖 Machine Learning
	•	scikit-learn — Logistic Regression
	•	PyTorch — Encrypted-compatible Neural Network
	•	Accuracy on encrypted inference: 80%+

🔧 Backend
	•	FastAPI — Encrypted inference API
	•	Pydantic — Input validation
	•	Uvicorn — ASGI server

🎨 Frontend
	•	Streamlit — Dashboard
	•	Plotly — Interactive charts



Project Structure

├── api/                    # FastAPI encrypted server
├── client/                 # Client SDK (encrypt → infer → decrypt)
├── dashboard/              # Streamlit web UI
├── src/                    # Models + CKKS context manager
├── models/                 # Saved model files
├── benchmarks/             # Performance & latency plots
├── docs/                   # Technical + Business docs
└── scripts/                # Training, exporting, testing





This platform demonstrates:

- ML inference directly on encrypted vectors  
- Server **never** sees patient features or predictions  
- Only the client owns the secret key  
- Encrypted NN accuracy: **80%+**  
- Average inference latency: **5–7 ms**  
- Fully GDPR compliant (Articles 9, 25, 32, 35)

### 🔍 Privacy Guarantee
