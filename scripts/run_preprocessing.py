from src.data.preprocessor import HeartDiseasePreprocessor

def main():
    print("\n🚀 Running preprocessing pipeline...")

    pre = HeartDiseasePreprocessor()

    # Load dataset
    df = pre.load_data("data/raw/heart_disease.csv")

    # Prepare data
    X_train, X_test, y_train, y_test = pre.prepare_data(df)

    # Save processor
    pre.save("models/plaintext/preprocessor.pkl")

    print("\n🎉 Preprocessing complete!")
    print("   ➤ X_train:", X_train.shape)
    print("   ➤ X_test :", X_test.shape)

if __name__ == "__main__":
    main()