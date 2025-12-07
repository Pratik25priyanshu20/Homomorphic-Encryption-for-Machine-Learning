import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from client.client import PrivateMLClient  # noqa: E402


st.set_page_config(page_title="HE‑ML | Home", layout="wide")

st.title("🏥 Fully Encrypted Medical Prediction")
st.subheader("GDPR-Compliant AI using Homomorphic Encryption")

st.markdown("""
Welcome to **HE-ML**, a privacy-preserving medical prediction system powered by **CKKS Homomorphic Encryption**.  
All computations — including model inference — happen **on encrypted data**.
""")

st.info("Use the left navigation to run predictions, view benchmarks, or inspect technical details.")

# ---------------------------------------------------------
# Connection panel (shared across pages via session_state)
# ---------------------------------------------------------
if "client" not in st.session_state:
    st.session_state.client = None
if "connected" not in st.session_state:
    st.session_state.connected = False

st.markdown("### 🔌 Connect to your HE server")
server_url = st.text_input("Server URL", value="http://localhost:8000")

if st.button("Connect"):
    try:
        st.session_state.client = PrivateMLClient(server_url, "models/plaintext/preprocessor.pkl")
        st.session_state.client.initialize()
        st.session_state.connected = True
        st.success("Connected & CKKS context initialized.")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.session_state.connected = False

status = "🟢 Connected" if st.session_state.get("connected") else "🔴 Not connected"
st.markdown(f"**Status:** {status}")

st.markdown("---")
st.markdown("#### What’s inside?")
st.markdown("""
- **Encrypted Logistic Regression & Neural Network** endpoints  
- **Client-side decryption + calibration** for stable probabilities  
- **Benchmarks** for latency and probability quality  
- **Context & HE parameters** for technical inspection  
""")
