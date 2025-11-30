"""
Data Preprocessing Module for Heart Disease Dataset

Handles:
- Missing value cleaning
- Feature scaling (StandardScaler)
- Train-test splitting
- Saving/loading the scaler correctly for client-side preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import joblib
from pathlib import Path


class HeartDiseasePreprocessor:
    """
    Preprocessor for Heart Disease Dataset.

    IMPORTANT CHANGE (Fix):
    ------------------------
    We now save ONLY the fitted StandardScaler to disk.
    This ensures the client loads a real scaler with .transform()
    instead of a dict (which caused previous errors).
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = None
        self.is_fitted = False

    # ==================================================================
    # LOAD DATA
    # ==================================================================
    def load_data(self, filepath: str = 'data/raw/heart_disease.csv') -> pd.DataFrame:
        df = pd.read_csv(filepath)
        print(f"📂 Loaded {len(df)} rows from {filepath}")
        return df

    # ==================================================================
    # MISSING VALUE HANDLING
    # ==================================================================
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna()
        removed = before - len(df)

        if removed > 0:
            print(f"🧹 Removed {removed} rows with missing values")
        return df

    # ==================================================================
    # DATA VALIDATION
    # ==================================================================
    def validate_data(self, df: pd.DataFrame) -> bool:
        required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'target']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        if not df['target'].isin([0, 1]).all():
            raise ValueError("Target column must contain only 0 or 1")

        print("✅ Data validation passed")
        return True

    # ==================================================================
    # FULL PREPROCESSING PIPELINE
    # ==================================================================
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        print("\n" + "="*60)
        print("🔧 DATA PREPROCESSING PIPELINE")
        print("="*60)

        df = self.handle_missing_values(df)
        self.validate_data(df)

        X = df.drop(columns=[target_col])
        y = df[target_col].values
        self.feature_names = X.columns.tolist()

        print("\n📊 Features:")
        for i, f in enumerate(self.feature_names, 1):
            print(f"   {i}. {f}")

        print("\n⚖️  Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True

        print(f"\n✂️  Splitting dataset (test_size={self.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples:  {len(X_test)}")

        print("\n✅ Preprocessing COMPLETE")
        print("="*60)

        return X_train, X_test, y_train, y_test

    # ==================================================================
    # TRANSFORM NEW DATA (used by API/server and client)
    # ==================================================================
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted — run prepare_data() first")
        return self.scaler.transform(X)

    # ==================================================================
    # SAVE ONLY THE SCALER  ← FIXED
    # ==================================================================
    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save only the scaler
        joblib.dump(self.scaler, filepath)
        print(f"💾 Scaler saved to: {filepath}")

    # ==================================================================
    # LOAD ONLY THE SCALER (STATIC-METHOD)  ← FIXED
    # ==================================================================
    @staticmethod
    def load(filepath: str):
        scaler = joblib.load(filepath)
        print(f"📂 Scaler loaded from: {filepath}")
        return scaler


# Self-test (run directly)
if __name__ == '__main__':
    proc = HeartDiseasePreprocessor()
    df = proc.load_data()
    X_train, X_test, y_train, y_test = proc.prepare_data(df)
    proc.save('models/plaintext/preprocessor.pkl')
    print("\n🎉 Preprocessor ready!")