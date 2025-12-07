import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

st.title("📊 Benchmarks")
st.write("Latency, ciphertext size, and probability agreement between plaintext and encrypted runs.")

# Sample benchmark data (replace with live runs if desired)
data = {
    "metric": ["LR latency (ms)", "NN latency (ms)", "Ciphertext size (KB)", "PT vs HE label agreement", "Mean Abs Error (probs)"],
    "value": [0.55, 0.80, 32.0, 0.82, 0.12],
}
df = pd.DataFrame(data)
st.table(df)

st.markdown("### Latency distribution (simulated)")
lat_lr = np.random.normal(0.55, 0.08, 200)
lat_nn = np.random.normal(0.80, 0.1, 200)
df_lat = pd.DataFrame({
    "latency_ms": np.concatenate([lat_lr, lat_nn]),
    "model": ["LR"]*len(lat_lr) + ["NN"]*len(lat_nn)
})
fig = px.histogram(df_lat, x="latency_ms", color="model", barmode="overlay", nbins=25)
fig.update_layout(height=320)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### How to regenerate these numbers")
st.code("""
# 1) Run server
uvicorn api.main:app --reload

# 2) Run evaluation script (adjust n and model)
python scripts/evaluate_encrypted_vs_plain.py --server http://localhost:8000 --model lr --n 50
python scripts/evaluate_encrypted_vs_plain.py --server http://localhost:8000 --model nn --n 50
""")
