"""
HE-SAFE Neural Network Training
-------------------------------
This version is FIXED for homomorphic encryption:

✔ Stable architecture (13 → 8 → 1)
✔ Linear activation 0.5*x ONLY (no +0.5 shift)
✔ Weight clipping to prevent CKKS explosion
✔ Small LR to avoid gradient blow-up
✔ Produces HE-safe weights guaranteed < 0.25
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.data.preprocessor import HeartDiseasePreprocessor
from src.models.neural_network_he import HEFriendlyNNTrainer


def main():
    print("\n======================================================")
    print("   🧠 TRAINING FIXED HE-FRIENDLY NEURAL NETWORK")
    print("======================================================\n")

    pre = HeartDiseasePreprocessor()
    df = pre.load_data("data/raw/heart_disease.csv")
    X_train, X_test, y_train, y_test = pre.prepare_data(df)

    pre.save("models/plaintext/preprocessor.pkl")

    # ★ NEW PARAMETERS ★
    trainer = HEFriendlyNNTrainer(
        input_dim=X_train.shape[1],
        hidden_dim=8,          # bigger = stability
        lr=0.0005,             # slower = stable grads
        clip_value=0.25        # keeps CKKS weights safe
    )

    trainer.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=20,             # more epochs = divergence
        batch_size=32
    )

    acc = trainer.evaluate(X_test, y_test)
    print(f"\nFinal Validation Accuracy: {acc*100:.2f}%\n")

    trainer.save_parameters("models/plaintext/nn_he_parameters.pkl")

    print("\n======================================================")
    print("   🎉 HE-Friendly Neural Network Training Complete")
    print("======================================================\n")


if __name__ == "__main__":
    main()