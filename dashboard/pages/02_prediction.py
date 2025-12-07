import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from client.client import PrivateMLClient  # noqa: E402

st.set_page_config(
    page_title="Encrypted Medical Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Navigation via query param
# -----------------------------
params = st.query_params
current_page = params.get("page", ["home"])[0]


def goto(page: str):
    st.query_params["page"] = page


# -----------------------------
# Sidebar navigation
# -----------------------------
with st.sidebar:
    st.title("🔐 Encrypted ML")
    if st.button("🏠 Home"):
        goto("home")
    if st.button("🧬 Prediction"):
        goto("prediction")
    if st.button("📈 Benchmarks"):
        goto("bench")
    if st.button("⚙️ Technical"):
        goto("tech")


# -----------------------------
# HOME
# -----------------------------
if current_page == "home":
    st.title("Home")

    st.markdown("""
    ### Next-Generation Privacy-Preserving Machine Learning  

    This dashboard demonstrates a **fully homomorphic-encrypted medical inference** system using CKKS:  

    - 🔐 Encrypted Logistic Regression  
    - 🧠 Encrypted Neural Network  
    - ✨ Client-side encryption & decryption  
    - 🛡 Server never sees raw patient data  
    """)

    cols = st.columns(4)
    cols[0].metric("Typical Latency", "0.8 – 1.1 sec")
    cols[1].metric("Models", "LR + NN")
    cols[2].metric("HE Scheme", "CKKS")
    cols[3].metric("Client Data Exposure", "0 bytes")


# -----------------------------
# PREDICTION
# -----------------------------
if current_page == "prediction":
    st.title("🧬 Heart Disease Risk Prediction (Encrypted End-to-End)")

    # Load HE client once
    @st.cache_resource
    def load_he():
        client = PrivateMLClient(
            server_url="http://localhost:8000",
            preprocessor_path=str(ROOT / "models/plaintext/preprocessor.pkl")
        )
        client.initialize()
        return client

    he_client = load_he()

    st.success("Connected to HE server — encrypted context initialized!")

    st.subheader("Patient Features")
    c1, c2, c3 = st.columns(3)

    age = c1.number_input("Age", 20, 90, 54)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    chol = c3.slider("Cholesterol", 100, 400, 250)

    risk_button = st.button("🔒 Run Encrypted Prediction")

    if risk_button:
        with st.spinner("Encrypting → sending → decrypting..."):

            sex_val = 1 if sex == "Male" else 0
            features = [age, sex_val, chol, 130]  # simplified example

            result = he_client.predict(features, "lr")

            prob = result["probability"]
            pred = result["prediction"]
            ms = result["inference_time_ms"]

            st.header("Prediction Result")
            st.metric("Probability", f"{prob*100:.1f}%")
            st.metric("Risk Level", "HIGH" if pred else "LOW")
            st.metric("Inference Time", f"{ms:.2f} ms")

            # Risk gauge
            def risk_gauge(p):
                val = int(p * 100)
                if val < 33:
                    color, label = "#2ecc71", "LOW"
                elif val < 66:
                    color, label = "#f1c40f", "MEDIUM"
                else:
                    color, label = "#e74c3c", "HIGH"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    number={'suffix': "%"},
                    title={'text': f"Risk Level: {label}", 'font': {'size': 22}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 33], 'color': "#145A32"},
                            {'range': [33, 66], 'color': "#7D6608"},
                            {'range': [66, 100], 'color': "#641E16"}
                        ],
                    }
                ))
                fig.update_layout(
                    margin=dict(l=40, r=40, t=50, b=20),
                    height=320,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color="white"
                )
                return fig

            st.header("📊 Risk Level Gauge")
            st.plotly_chart(risk_gauge(prob), use_container_width=True)


# -----------------------------
# BENCHMARKS
# -----------------------------
if current_page == "bench":
    st.title("📈 Performance Benchmarks")
    st.write("This page will show latency charts, model comparisons, and encrypted inference overhead.")


# -----------------------------
# TECHNICAL
# -----------------------------
if current_page == "tech":
    st.title("⚙️ Technical Details")
    st.write("""
    - CKKS encryption  
    - Polynomial modulus degree  
    - Coefficient modulus sizes  
    - Relinearization & Galois keys  
    - Affine calibration for encrypted logits  
    """)
