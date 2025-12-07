# scripts/export_nn_for_encryption.py

"""
Export PyTorch NN → NumPy weights for Encrypted Inference
"""

import torch
import joblib
import numpy as np
from pathlib import Path

from src.models.neural_network import NeuralNetworkModel


def export_nn_parameters(model_path: str, output_path: str):
    print("\n======================================================")
    print("🔄 EXPORTING PYTORCH NN WEIGHTS FOR ENCRYPTED INFERENCE")
    print("======================================================")

    # Load PyTorch model
    print(f"\n📂 Loading trained model from: {model_path}")
    dummy_model = NeuralNetworkModel(input_dim=13)  # input_dim=13 fixed for dataset
    dummy_model.load(model_path)
    model = dummy_model.model
    model.eval()

    # Extract weights
    W1 = model.fc1.weight.detach().cpu().numpy()
    b1 = model.fc1.bias.detach().cpu().numpy()
    W2 = model.fc2.weight.detach().cpu().numpy()
    b2 = model.fc2.bias.detach().cpu().numpy()
    W3 = model.fc3.weight.detach().cpu().numpy()
    b3 = model.fc3.bias.detach().cpu().numpy().item()

    print("\n📐 Extracted layer shapes:")
    print(f"W1: {W1.shape}")
    print(f"b1: {b1.shape}")
    print(f"W2: {W2.shape}")
    print(f"b2: {b2.shape}")
    print(f"W3: {W3.shape}")
    print(f"b3: {type(b3)}")

    # Save dictionary
    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(params, output_path)

    print(f"\n💾 Saved HE-ready parameters to: {output_path}")
    print("======================================================")


if __name__ == "__main__":
    export_nn_parameters(
        model_path="models/plaintext/neural_network.pth",
        output_path="models/plaintext/nn_parameters.pkl"
    )
