import streamlit as st
import requests

st.title("⚙️ Technical Details")
st.write("Architecture, HE parameters, and context information.")

server_url = st.text_input("Server URL", value="http://localhost:8000")

if st.button("Fetch Context"):
    try:
        r = requests.get(f"{server_url}/context")
        ctx_info = r.json()
        st.json(ctx_info)
        st.success("Context fetched.")
    except Exception as e:
        st.error(f"Unable to fetch context: {e}")

st.markdown("---")
st.markdown("### Architecture Overview")
st.markdown("""
- Client builds CKKS context, encrypts feature vector.
- Server runs homomorphic inference (LR or NN) and returns encrypted logit.
- Client decrypts and applies calibration → probability.
""")

st.markdown("### Current HE Parameters (expected)")
st.markdown("""
- CKKS scale: 2^40  
- poly_modulus_degree: 8192 or 16384 (depending on profile)  
- coeff_mod_bit_sizes: e.g., [60, 40, 40, 40, 60]  
""")
