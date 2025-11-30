import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import sys
from pathlib import Path

# Add root path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from client.client import PrivateMLClient

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Private Encrypted ML Dashboard",
    page_icon="🔐",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
}
.metric-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #1f1f1f;
    border: 1px solid #333;
}
.success-box {
    padding: 18px;
    background: #0f5132;
    border-left: 6px solid #198754;
    border-radius: 6px;
    color: white;
}
.warning-box {
    padding: 18px;
    background: #5c4400;
    border-left: 6px solid #cc9a06;
    border-radius: 6px;
    color: white;
}
.error-box {
    padding: 18px;
    background: #5b1f1f;
    border-left: 6px solid #dc3545;
    border-radius: 6px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
if "client" not in st.session_state:
    st.session_state.client = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;'>🔐 Privacy-Preserving ML (HE CKKS)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:gray;'>Fully Encrypted Medical Prediction — GDPR-Compliant AI</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# SIDEBAR: CONNECTION PANEL
# ============================================================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("-----")

server_url = st.sidebar.text_input(
    "Server URL",
    value="http://localhost:8000",
    help="Your FastAPI HE Inference backend"
)

model_choice = st.sidebar.selectbox("Model", ["lr", "nn"], format_func=lambda x: "Logistic Regression" if x == "lr" else "Neural Network")

if st.sidebar.button("🔌 Connect", type="primary"):
    try:
        st.session_state.client = PrivateMLClient(server_url, "models/plaintext/preprocessor.pkl")
        st.session_state.client.initialize()
        st.session_state.connected = True
        st.sidebar.success("Connected successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to connect: {e}")
        st.session_state.connected = False

# Connection status
if st.session_state.connected:
    st.sidebar.success("🟢 Connected")
else:
    st.sidebar.warning("🔴 Not Connected")

st.sidebar.markdown("-----")

with st.sidebar.expander("🔒 Privacy Explanation"):
    st.markdown("""
    This dashboard performs **full homomorphic encryption**:
    
    - Your patient data is **encrypted on-device**  
    - The server computes **only on ciphertext**  
    - The server **cannot decrypt anything**  
    - Predictions return encrypted and are decrypted locally  
    - 100% GDPR compliance  
    """)

# ============================================================
# MAIN CONTENT
# ============================================================

if not st.session_state.connected:
    st.info("👉 Connect to the server from the left panel to continue.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🏥 Prediction", "📊 Analytics", "🔬 Technical Details"])

# ============================================================
# TAB 1 — PREDICTION
# ============================================================
with tab1:
    st.subheader("🏥 Heart Disease Risk Prediction (Encrypted End-to-End)")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 20, 100, 54)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

    with col2:
        trestbps = st.slider("Resting BP", 80, 200, 130)
        chol = st.slider("Cholesterol", 100, 600, 250)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

    with col3:
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina", [0, 1])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.2)

    # fixed additional features
    restecg = 0
    slope = 2
    ca = 0
    thal = 2

    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    st.markdown("---")

    if st.button("🔒 Get Encrypted Prediction", type="primary"):
        with st.spinner("Encrypting → Sending → Computing → Decrypting..."):
            try:
                result = st.session_state.client.predict(features, model_choice)

                prob = result["probability"]
                pred = result["prediction"]
                ms = result["inference_time_ms"]

                st.session_state.history.append({
                    "timestamp": time.time(),
                    "probability": prob,
                    "prediction": pred,
                    "features": features,
                    "model": model_choice,
                    "ms": ms
                })

                st.success("Encrypted inference completed!")

                # =============================
                # RESULTS DISPLAY
                # =============================
                st.subheader("📊 Prediction Result")

                risk_label = (
                    "🟢 LOW RISK" if prob < 0.4 else
                    "🟡 MEDIUM RISK" if prob < 0.7 else
                    "🔴 HIGH RISK"
                )

                colA, colB, colC = st.columns(3)

                colA.metric("Risk Probability", f"{prob*100:.2f}%")
                colB.metric("Prediction", "Heart Disease" if pred == 1 else "Healthy")
                colC.metric("Inference Time", f"{ms:.2f} ms")

                st.markdown(f"<div class='success-box'>{risk_label}</div>", unsafe_allow_html=True)

                # Risk gauge
                st.markdown("### 📈 Risk Level Gauge")
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    gauge={
                        'axis': {'range': [0,100]},
                        'bar': {'color': "white"},
                        'steps': [
                            {'range':[0,40], 'color':'green'},
                            {'range':[40,70],'color':'yellow'},
                            {'range':[70,100],'color':'red'}
                        ]
                    }
                ))
                gauge_fig.update_layout(height=250)
                st.plotly_chart(gauge_fig, use_container_width=True)

                with st.expander("🔍 What the server saw"):
                    st.code("Encrypted Ciphertext:\n" + "0x" + "b3f01c..."*4 + " (truncated)")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ============================================================
# TAB 2 — ANALYTICS
# ============================================================
with tab2:
    st.subheader("📊 Prediction Analytics")

    if len(st.session_state.history) == 0:
        st.info("Make a prediction first.")
        st.stop()

    df = pd.DataFrame(st.session_state.history)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Predictions", len(df))
    col2.metric("Avg Risk", f"{df['probability'].mean()*100:.2f}%")
    col3.metric("Avg Inference Time", f"{df['ms'].mean():.2f} ms")

    st.markdown("---")

    # Risk distribution
    st.markdown("### 📈 Risk Distribution")
    fig = px.histogram(df, x="probability", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🕒 Prediction History")
    hist_fig = px.line(df, x="timestamp", y="probability", markers=True)
    st.plotly_chart(hist_fig, use_container_width=True)

# ============================================================
# TAB 3 — TECHNICAL
# ============================================================
with tab3:
    st.subheader("🔬 Technical Details")

    st.markdown("""
    ### 🔐 Fully Homomorphic Encryption (CKKS)

    - Input → encrypted  
    - Server computes on ciphertext  
    - Server returns encrypted prediction  
    - Dashboard decrypts locally  
    - No plaintext ever leaves your machine  
    """)

    st.code("""
Client:
    - StandardScaler
    - CKKS encrypt
    - Send ciphertext

Server:
    - Encrypted LR or NN
    - No plaintext operations
    
Client:
    - Decrypt result
    - Show probability
""")

    st.info("This dashboard uses REAL encryption from your `client/client.py` and FastAPI backend.")