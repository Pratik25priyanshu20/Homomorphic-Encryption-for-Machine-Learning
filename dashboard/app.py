import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from pathlib import Path
import sys

# Allow imports from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from client.client import PrivateMLClient  # noqa: E402

st.set_page_config(
    page_title="Private Encrypted ML Dashboard",
    page_icon="🔐",
    layout="wide",
)

# -----------------------------------------------------------
# Session state
# -----------------------------------------------------------
if "client" not in st.session_state:
    st.session_state.client = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "history" not in st.session_state:
    st.session_state.history = []


def connect(server_url: str):
    try:
        client = PrivateMLClient(server_url, str(ROOT / "models/plaintext/preprocessor.pkl"))
        client.initialize()
        st.session_state.client = client
        st.session_state.connected = True
        st.sidebar.success("Connected successfully!")
    except Exception as e:
        st.session_state.connected = False
        st.sidebar.error(f"Failed to connect: {e}")


def risk_gauge(prob: float):
    """
    prob: value between 0 and 1
    """
    value = int(prob * 100)
    if value < 33:
        color, label = "#2ecc71", "LOW"
    elif value < 66:
        color, label = "#f1c40f", "MEDIUM"
    else:
        color, label = "#e74c3c", "HIGH"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%"},
        title={'text': f"Risk Level: {label}", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "#145A32"},
                {'range': [33, 66], 'color': "#7D6608"},
                {'range': [66, 100], 'color': "#641E16"},
            ],
        }
    ))
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=20),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white",
    )
    return fig


# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
server_url = st.sidebar.text_input("Server URL", value="http://localhost:8000")
model_choice = st.sidebar.selectbox("Model", ["lr", "nn"], format_func=lambda x: "Logistic Regression" if x == "lr" else "Neural Network")

if st.sidebar.button("🔌 Connect", type="primary"):
    connect(server_url)

if st.session_state.connected:
    st.sidebar.success("🟢 Connected")
else:
    st.sidebar.warning("🔴 Not Connected")

with st.sidebar.expander("🔒 Privacy Explanation"):
    st.markdown("""
    This dashboard performs **full homomorphic encryption**:

    - Your patient data is **encrypted on-device**  
    - The server computes **only on ciphertext**  
    - The server **cannot decrypt anything**  
    - Predictions return encrypted and are decrypted locally  
    """)

# -----------------------------------------------------------
# Header & Tabs
# -----------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🔐 Privacy-Preserving ML (HE CKKS)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:gray;'>Fully Encrypted Medical Prediction — GDPR-Compliant AI</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🏥 Prediction", "📊 Analytics", "🔬 Technical Details"])

# -----------------------------------------------------------
# Tab 1: Prediction
# -----------------------------------------------------------
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
        if not st.session_state.connected or st.session_state.client is None:
            st.error("Please connect to the server first.")
        else:
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

                    st.markdown("### 📊 Risk Level Gauge")
                    st.plotly_chart(risk_gauge(prob), use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -----------------------------------------------------------
# Tab 2: Analytics
# -----------------------------------------------------------
with tab2:
    st.subheader("📊 Prediction Analytics")

    if len(st.session_state.history) == 0:
        st.info("Make a prediction first.")
    else:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(st.session_state.history)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(df))
        col2.metric("Avg Risk", f"{df['probability'].mean()*100:.2f}%")
        col3.metric("Avg Inference Time", f"{df['ms'].mean():.2f} ms")

        st.markdown("---")
        st.markdown("### 📈 Risk Distribution")
        fig = px.histogram(df, x="probability", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🕒 Prediction History")
        hist_fig = px.line(df, x="timestamp", y="probability", markers=True)
        st.plotly_chart(hist_fig, use_container_width=True)

# -----------------------------------------------------------
# Tab 3: Technical
# -----------------------------------------------------------
with tab3:
    st.subheader("🔬 Technical Details")
    st.write("""
    - CKKS encryption  
    - Polynomial modulus degree  
    - Coefficient modulus sizes  
    - Relinearization & Galois keys  
    - Affine calibration for encrypted logits  
    """)

