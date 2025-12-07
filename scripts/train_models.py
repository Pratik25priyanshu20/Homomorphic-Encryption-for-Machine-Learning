# scripts/train_models.py
"""
Complete Model Training Script (FINAL FIXED VERSION)

This version:
✔ Uses updated HeartDiseasePreprocessor
✔ Saves ONLY the scaler (no dict)
✔ Trains LR and NN
✔ Saves LR in .pkl and NN weights for encrypted inference
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import HeartDiseasePreprocessor
import numpy as np
import joblib
from src.models.logistic_regression import LogisticRegressionModel
from src.models.neural_network import NeuralNetworkModel
import pandas as pd


def main():

    print("\n" + "="*70)
    print(" "*10 + "🏥 HEART DISEASE — BASELINE MODEL TRAINING")
    print("="*70)

    # ===============================================================
    # STEP 1 — PREPROCESS
    # ===============================================================
    print("\n📊 STEP 1 — DATA PREPROCESSING")
    pre = HeartDiseasePreprocessor()

    df = pre.load_data("data/raw/heart_disease.csv")
    X_train, X_test, y_train, y_test = pre.prepare_data(df)

    # SAVE ONLY THE FITTED SCALER  ← CRITICAL FIX
    pre.save("models/plaintext/preprocessor.pkl")

    # ===============================================================
    # STEP 2 — LOGISTIC REGRESSION
    # ===============================================================
    print("\n🤖 STEP 2 — TRAINING LOGISTIC REGRESSION")

    lr = LogisticRegressionModel()
    lr.train(X_train, y_train)
    lr_metrics = lr.evaluate(X_test, y_test)
    lr.save("models/plaintext/logistic_regression.pkl")

    # Export SAFE LR params for encrypted inference
    w = lr.model.coef_[0].astype(np.float64)
    b = float(lr.model.intercept_[0])

    SAFE_SCALE = 1e-6
    w = w * SAFE_SCALE
    b = b * SAFE_SCALE

    joblib.dump({"weights": w.flatten(), "bias": b}, "models/plaintext/lr_he_parameters.pkl")
    print("   ✓ Saved HE-safe LR parameters → models/plaintext/lr_he_parameters.pkl")

    print(f"   ✓ LR Accuracy: {lr_metrics['accuracy']*100:.2f}%")

    # ===============================================================
    # STEP 3 — NEURAL NETWORK
    # ===============================================================
    print("\n🧠 STEP 3 — TRAINING NEURAL NETWORK")

    nn = NeuralNetworkModel(
        input_dim=X_train.shape[1],
        hidden_dims=(16, 16),
        learning_rate=0.001
    )

    nn.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=100,
        batch_size=32,
        verbose=True
    )

    nn_metrics = nn.evaluate(X_test, y_test)

    # SAVE NN PARAMETERS FOR ENCRYPTED INFERENCE
    nn.save_parameters("models/plaintext/nn_parameters.pkl")

    print(f"   ✓ NN Accuracy: {nn_metrics['accuracy']*100:.2f}%")

    # ===============================================================
    # DONE
    # ===============================================================
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE")
    print("="*70)
    print("Saved:")
    print("   ✓ models/plaintext/preprocessor.pkl")
    print("   ✓ models/plaintext/logistic_regression.pkl")
    print("   ✓ models/plaintext/nn_parameters.pkl")


if __name__ == "__main__":
    main()
