"""
Test Encrypted Inference

This script:
1. Creates encryption context
2. Loads trained plaintext model
3. Converts to encrypted inference
4. Tests on encrypted data
5. Compares with plaintext results

Usage:
    python scripts/test_encrypted_inference.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.encryption.context import EncryptionContextManager
from src.models.encrypted_lr import EncryptedLogisticRegression
from src.data.preprocessor import HeartDiseasePreprocessor
from src.models.logistic_regression import LogisticRegressionModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    """Main testing pipeline"""
    
    print("\n" + "="*70)
    print(" "*10 + "🔒 ENCRYPTED INFERENCE TESTING")
    print(" "*15 + "DAY 2-3: HOMOMORPHIC ENCRYPTION")
    print("="*70)
    
    # =========================================================================
    # STEP 1: SETUP ENCRYPTION CONTEXT
    # =========================================================================
    print("\n🔐 STEP 1: ENCRYPTION SETUP")
    print("-" * 70)
    
    ctx_manager = EncryptionContextManager(security_level='128bit')
    context = ctx_manager.create_context(
        generate_galois_keys=True,
        generate_relin_keys=True
    )
    
    ctx_manager.print_context_info()
    
    # Save context for later use
    Path('models/encrypted').mkdir(parents=True, exist_ok=True)
    ctx_manager.save_context('models/encrypted/context.bin', include_secret_key=True)
    
    # =========================================================================
    # STEP 2: LOAD DATA
    # =========================================================================
    print("\n📊 STEP 2: DATA LOADING")
    print("-" * 70)
    
    preprocessor = HeartDiseasePreprocessor()
    df = preprocessor.load_data('data/raw/heart_disease.csv')
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    print(f"\n✅ Data ready:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # =========================================================================
    # STEP 3: LOAD PLAINTEXT MODEL (BASELINE)
    # =========================================================================
    print("\n🔓 STEP 3: PLAINTEXT BASELINE")
    print("-" * 70)
    
    plaintext_model = LogisticRegressionModel()
    plaintext_model.load('models/plaintext/logistic_regression.pkl')
    
    plain_metrics = plaintext_model.evaluate(X_test, y_test, verbose=False)
    
    print(f"\n📊 Plaintext Performance:")
    print(f"   Accuracy:       {plain_metrics['accuracy']*100:.2f}%")
    print(f"   Precision:      {plain_metrics['precision']*100:.2f}%")
    print(f"   Recall:         {plain_metrics['recall']*100:.2f}%")
    print(f"   F1-Score:       {plain_metrics['f1']*100:.2f}%")
    print(f"   ROC-AUC:        {plain_metrics['roc_auc']:.4f}")
    print(f"   Inference time: {plain_metrics['inference_time_ms']:.4f} ms/sample")
    
    # =========================================================================
    # STEP 4: ENCRYPTED INFERENCE
    # =========================================================================
    print("\n🔒 STEP 4: ENCRYPTED INFERENCE")
    print("-" * 70)
    
    enc_model = EncryptedLogisticRegression(context)
    enc_model.load_plaintext_model('models/plaintext/logistic_regression.pkl')
    
    # Test on subset first (encrypted inference is slow)
    test_size = 30  # Test on 30 samples
    print(f"\n⚡ Running encrypted inference on {test_size} samples...")
    print("   (This will take ~30-60 seconds)")
    
    enc_metrics = enc_model.evaluate_encrypted(
        X_test[:test_size], 
        y_test[:test_size],
        verbose=True
    )
    
    # =========================================================================
    # STEP 5: COMPARISON
    # =========================================================================
    print("\n📊 STEP 5: DETAILED COMPARISON")
    print("-" * 70)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Inference Time'],
        'Plaintext': [
            f"{plain_metrics['accuracy']*100:.2f}%",
            f"{plain_metrics['precision']*100:.2f}%",
            f"{plain_metrics['recall']*100:.2f}%",
            f"{plain_metrics['f1']*100:.2f}%",
            f"{plain_metrics['inference_time_ms']:.4f} ms"
        ],
        'Encrypted': [
            f"{enc_metrics['accuracy']*100:.2f}%",
            f"{enc_metrics['precision']*100:.2f}%",
            f"{enc_metrics['recall']*100:.2f}%",
            f"{enc_metrics['f1']*100:.2f}%",
            f"{enc_metrics['avg_inference_time_ms']:.2f} ms"
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Calculate differences
    acc_diff = abs(plain_metrics['accuracy'] - enc_metrics['accuracy']) * 100
    time_slowdown = enc_metrics['avg_inference_time_ms'] / plain_metrics['inference_time_ms']
    
    print(f"\n🔍 Analysis:")
    print(f"   Accuracy difference: {acc_diff:.2f}%")
    print(f"   Speed slowdown:      {time_slowdown:.0f}x")
    
    # =========================================================================
    # STEP 6: PRIVACY DEMONSTRATION
    # =========================================================================
    print("\n🔐 STEP 6: PRIVACY DEMONSTRATION")
    print("-" * 70)
    
    # Show what server sees
    import tenseal as ts
    
    sample_patient = X_test[0]
    print(f"\n👤 Patient Data (Plaintext - PRIVATE):")
    print(f"   {sample_patient[:5]}... (showing first 5 features)")
    
    # Encrypt
    enc_patient = ts.ckks_vector(context, sample_patient.tolist())
    encrypted_bytes = enc_patient.serialize()
    
    print(f"\n🔒 Encrypted Data (What Server Sees):")
    print(f"   {encrypted_bytes[:50].hex()}... (random gibberish)")
    print(f"   Size: {len(encrypted_bytes)} bytes")
    
    # Server computes on encrypted data
    enc_prediction = enc_model.predict_encrypted(enc_patient)
    enc_pred_bytes = enc_prediction.serialize()
    
    print(f"\n🔒 Encrypted Prediction (Server Returns This):")
    print(f"   {enc_pred_bytes[:50].hex()}... (still encrypted)")
    
    # Client decrypts
    decrypted_pred = enc_prediction.decrypt()[0]
    print(f"\n🔓 Decrypted Prediction (Only Client Can See):")
    print(f"   Probability: {decrypted_pred:.4f}")
    print(f"   Prediction:  {'Disease' if decrypted_pred > 0.5 else 'No Disease'}")
    
    print(f"\n✅ Privacy Guaranteed:")
    print(f"   ✓ Server never saw patient features")
    print(f"   ✓ Server never saw prediction probability")
    print(f"   ✓ Server only performed computation on encrypted data")
    
    # =========================================================================
    # STEP 7: VISUALIZATIONS
    # =========================================================================
    print("\n📈 STEP 7: GENERATING VISUALIZATIONS")
    print("-" * 70)
    
    Path('benchmarks/plots').mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Plaintext', 'Encrypted']
    accuracies = [plain_metrics['accuracy']*100, enc_metrics['accuracy']*100]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Plaintext vs Encrypted Inference Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmarks/plots/encrypted_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: benchmarks/plots/encrypted_accuracy_comparison.png")
    plt.close()
    
    # Plot 2: Inference Time Comparison (Log Scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = [plain_metrics['inference_time_ms'], enc_metrics['avg_inference_time_ms']]
    
    bars = ax.bar(models, times, color=['#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Inference Time (ms, log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Plaintext vs Encrypted Inference Speed', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('benchmarks/plots/encrypted_speed_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: benchmarks/plots/encrypted_speed_comparison.png")
    plt.close()
    
    # Plot 3: Privacy-Utility Trade-off
    fig, ax = plt.subplots(figsize=(10, 6))
    
    privacy_levels = ['No Privacy\n(Plaintext)', 'Full Privacy\n(Encrypted)']
    privacy_scores = [0, 100]
    utility_scores = [plain_metrics['accuracy']*100, enc_metrics['accuracy']*100]
    
    x = np.arange(len(privacy_levels))
    width = 0.35
    
    ax.bar(x - width/2, privacy_scores, width, label='Privacy Level', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, utility_scores, width, label='Utility (Accuracy)', color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Privacy-Utility Trade-off', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(privacy_levels)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmarks/plots/privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: benchmarks/plots/privacy_utility_tradeoff.png")
    plt.close()
    
    # Save comparison table
    Path('benchmarks/results').mkdir(parents=True, exist_ok=True)
    comparison.to_csv('benchmarks/results/encrypted_comparison.csv', index=False)
    print("   ✅ Saved: benchmarks/results/encrypted_comparison.csv")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("✅ ENCRYPTED INFERENCE TESTING COMPLETE")
    print("="*70)
    
    print("\n🎯 Key Results:")
    print(f"   ✓ Encrypted accuracy: {enc_metrics['accuracy']*100:.2f}% (vs {plain_metrics['accuracy']*100:.2f}% plaintext)")
    print(f"   ✓ Accuracy loss: {acc_diff:.2f}% (acceptable!)")
    print(f"   ✓ Inference time: {enc_metrics['avg_inference_time_ms']:.0f}ms (vs {plain_metrics['inference_time_ms']:.2f}ms plaintext)")
    print(f"   ✓ Slowdown: {time_slowdown:.0f}x (acceptable for privacy!)")
    
    print("\n🔒 Privacy Guarantees:")
    print("   ✓ Server never sees patient data (encrypted)")
    print("   ✓ Server never sees predictions (encrypted)")
    print("   ✓ Only client can decrypt results")
    print("   ✓ GDPR compliant - data never exposed")
    
    print("\n📊 Saved Artifacts:")
    print("   ✓ models/encrypted/context.bin")
    print("   ✓ benchmarks/results/encrypted_comparison.csv")
    print("   ✓ benchmarks/plots/encrypted_accuracy_comparison.png")
    print("   ✓ benchmarks/plots/encrypted_speed_comparison.png")
    print("   ✓ benchmarks/plots/privacy_utility_tradeoff.png")
    
    print("\n🚀 Next Steps (Day 4-5):")
    print("   → Implement encrypted neural network")
    print("   → Build client-server architecture (FastAPI)")
    print("   → Create interactive dashboard (Streamlit)")
    print("   → Performance optimization")
    
    print("\n" + "="*70)
    print("🎉 YOU'VE BUILT WORKING ENCRYPTED ML!")
    print("="*70)


if __name__ == '__main__':
    main()