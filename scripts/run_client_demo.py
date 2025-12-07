# scripts/run_client_demo.py

import argparse
from client.client import PrivateMLClient


def header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True, choices=["lr", "nn"])
    args = parser.parse_args()

    print("\n======================================================================")
    print("      🔒 PRIVATE ENCRYPTED ML CLIENT DEMO")
    print("======================================================================")

    # --------------------------------------------------------
    # Init client
    # --------------------------------------------------------
    client = PrivateMLClient(
        server_url=args.server,
        preprocessor_path="models/plaintext/preprocessor.pkl",
    )
    client.initialize()

    print(f"\nConnected to server: {args.server}")
    print("Encryption context established.")

    header(f"STEP 2 — Using Model: {args.model.upper()}")
    print(f"Selected Model: {args.model.upper()}")

    # --------------------------------------------------------
    # Patients
    # --------------------------------------------------------
    header("STEP 3 — Encrypted Predictions")

    patients = {
        "Patient A": [62, 1, 0, 130, 263, 0, 1, 97, 1, 1.2, 0, 1, 3],
        "Patient B": [45, 0, 1, 120, 180, 0, 1, 150, 0, 0.2, 1, 0, 2],
        "Patient C": [58, 1, 2, 140, 200, 0, 0, 120, 0, 0.4, 1, 1, 1],
    }

    inference_times = []

    for name, feats in patients.items():
        print(f"\n{name}")
        print("-" * 30)

        try:
            result = client.predict(feats, model=args.model)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            continue

        prob = result["probability"]
        pred = result["prediction"]
        ms = result["inference_time_ms"]
        inference_times.append(ms)

        if 0.4 <= prob <= 0.6:
            risk = "MEDIUM"
        elif pred == 1:
            risk = "HIGH"
        else:
            risk = "LOW"

        print(f"Probability: {prob:.4f}")
        print(f"Predicted Class: {pred}")
        print(f"Risk Level: {risk}")
        print(f"Inference Time: {ms:.2f} ms")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    header("STEP 4 — Summary")

    if inference_times:
        avg_ms = sum(inference_times) / len(inference_times)
        print(f"Total Predictions: {len(inference_times)}")
        print(f"Average Inference Time: {avg_ms:.2f} ms")
    else:
        print("No successful predictions.")

    print(f"\nModel Used: {args.model.upper()}")
    print("\nDemo complete.\n")


if __name__ == "__main__":
    main()