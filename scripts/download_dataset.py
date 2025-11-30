"""
Dataset Download Script
Downloads Heart Disease UCI dataset for private ML project

Usage:
    python scripts/download_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def download_heart_disease():
    """
    Download and prepare Heart Disease UCI dataset
    
    Dataset Info:
        - Source: UCI Machine Learning Repository
        - Samples: 303 patients
        - Features: 13 (age, sex, chest pain, blood pressure, etc.)
        - Target: Binary (heart disease present/absent)
    """
    print("=" * 60)
    print("📥 DOWNLOADING HEART DISEASE UCI DATASET")
    print("=" * 60)
    
    # Define URL and column names
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    columns = [
        'age',       # Age in years
        'sex',       # Sex (1 = male; 0 = female)
        'cp',        # Chest pain type (1-4)
        'trestbps',  # Resting blood pressure (mm Hg)
        'chol',      # Serum cholesterol (mg/dl)
        'fbs',       # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        'restecg',   # Resting ECG results (0-2)
        'thalach',   # Maximum heart rate achieved
        'exang',     # Exercise induced angina (1 = yes; 0 = no)
        'oldpeak',   # ST depression induced by exercise
        'slope',     # Slope of peak exercise ST segment
        'ca',        # Number of major vessels colored by fluoroscopy (0-3)
        'thal',      # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
        'target'     # Heart disease diagnosis (0 = no, 1-4 = yes)
    ]
    
    try:
        # Create data directory
        data_dir = Path('data/raw')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n📡 Fetching data from UCI repository...")
        
        # Download data
        df = pd.read_csv(url, names=columns, na_values='?')
        
        print(f"✅ Downloaded {len(df)} records with {len(columns)} columns")
        
        # Convert multi-class target to binary (0 = no disease, 1+ = disease)
        print("\n🔄 Converting target to binary classification...")
        df['target'] = (df['target'] > 0).astype(int)
        
        # Basic data info
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Features: {len(df.columns) - 1}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        
        # Target distribution
        target_counts = df['target'].value_counts()
        print(f"\n🎯 Target Distribution:")
        print(f"   No Disease (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
        print(f"   Disease (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
        
        # Save to CSV
        output_path = data_dir / 'heart_disease.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n💾 Saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Display sample
        print("\n👀 First 3 rows:")
        print(df.head(3).to_string())
        
        # Feature summary
        print("\n📈 Feature Summary:")
        print(df.describe().to_string())
        
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: jupyter notebook")
        print("  2. Open: notebooks/01_data_exploration.ipynb")
        print("  3. Start exploring the data!")
        
        return df
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify UCI repository is accessible")
        print("  3. Try alternative download method")
        sys.exit(1)

def verify_download():
    """Verify downloaded dataset"""
    data_path = Path('data/raw/heart_disease.csv')
    
    if not data_path.exists():
        print("❌ Dataset not found! Run download first.")
        return False
    
    try:
        df = pd.read_csv(data_path)
        
        # Basic checks
        assert len(df) > 0, "Dataset is empty"
        assert 'target' in df.columns, "Target column missing"
        assert len(df.columns) == 14, "Incorrect number of columns"
        
        print("✅ Dataset verification passed!")
        print(f"   Location: {data_path}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        return False

if __name__ == '__main__':
    # Download dataset
    df = download_heart_disease()
    
    # Verify
    print("\n🧪 Running verification...")
    verify_download()