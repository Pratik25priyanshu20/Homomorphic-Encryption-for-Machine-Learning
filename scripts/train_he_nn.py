"""
Train HE-Friendly Neural Network
--------------------------------
This script trains the shallow CKKS-compatible NN:

    Input: 13 features
    Hidden: 4 neurons
    Activation: square(x)  (HE-safe)
    Output: 1 neuron (logit)

It loads:
    models/plaintext/preprocessor.pkl
    data/raw/heart_disease.csv

And saves:
    models/plaintext/nn_he_parameters.pkl
"""

import sys
from pathlib import Path

# add src to the python path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data.preprocessor import HeartDiseasePreprocessor
from src.models.neural_network_he import HEFriendlyNNTrainer


def main():
    print("\n======================================================")
    print("      🧠 TRAINING HE-FRIENDLY NEURAL NETWORK")
    print("======================================================\n")

    # ------------------------------------------------------
    # Load & preprocess dataset
    # ------------------------------------------------------
    pre = HeartDiseasePreprocessor()
    df = pre.load_data("data/raw/heart_disease.csv")

    X_train, X_test, y_train, y_test = pre.prepare_data(df)

    # save scaler (already done previously; we re-save to be safe)
    pre.save("models/plaintext/preprocessor.pkl")

    # ------------------------------------------------------
    # Train HE-Friendly NN
    # ------------------------------------------------------
    trainer = HEFriendlyNNTrainer(
        input_dim=X_train.shape[1],
        hidden_dim=4,
        lr=0.001
    )

    trainer.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=40,
        batch_size=32
    )

    acc = trainer.evaluate(X_test, y_test)
    print(f"\nFinal Validation Accuracy: {acc*100:.2f}%\n")

    # ------------------------------------------------------
    # Save HE-ready parameters
    # ------------------------------------------------------
    save_path = "models/plaintext/nn_he_parameters.pkl"
    trainer.save_parameters(save_path)

    print("\n======================================================")
    print("   🎉 HE-Friendly Neural Network Training Complete")
    print("======================================================\n")


if __name__ == "__main__":
    main()