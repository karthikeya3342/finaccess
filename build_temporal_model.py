import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False

def train_temporal_model(csv_path: str):
    """
    TASK 1: TEMPORAL FEATURE ENGINEERING (SLIDING WINDOW)
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Sort the entire DataFrame by Loan_ID to simulate chronological sequence
    if 'Loan_ID' in df.columns:
        df = df.sort_values(by="Loan_ID").reset_index(drop=True)
    else:
        print("Warning: 'Loan_ID' not found. Cannot sort chronologically.")
        
    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Impute numerical with median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)
        else:
            # Impute categorical with mode
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

    # 2. Create "Lag Features" for macro sequence (LoanAmount of previous 5 applicants)
    if 'LoanAmount' in df.columns:
        loan_amount_mean = df['LoanAmount'].mean()
        for i in range(1, 6):
            lag_col = f'Lag_LoanAmount_{i}'
            df[lag_col] = df['LoanAmount'].shift(i)
            # 3. Fill missing lag values (first 5 rows) with column mean (or 0)
            df[lag_col] = df[lag_col].fillna(loan_amount_mean)
            
    # Prepare target variable (Risk of Default/Rejection -> N=1, Y=0)
    target_col = 'Loan_Status'
    if target_col in df.columns:
        # Map 'N' (Rejection) to 1, 'Y' (Approval) to 0
        df[target_col] = df[target_col].map({'N': 1, 'Y': 0}).fillna(0).astype(int)
    
    # 4. Encode categorical features and scale continuous features
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col not in ['Loan_ID', target_col]]
    
    # Label Encoding for categoricals
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    # Feature columns subset (exclude ID and target)
    features_to_exclude = ['Loan_ID', target_col]
    feature_columns = [col for col in df.columns if col not in features_to_exclude]
    
    # Scaling continuous features
    scaler = StandardScaler()
    # Scale all feature columns to ensure XGB or RF works on normalized data
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    X = df[feature_columns]
    y = df[target_col] if target_col in df.columns else None

    """
    TASK 2: MODEL ARCHITECTURE
    """
    if y is not None:
        if HAS_XGBOOST:
            print("Initializing XGBClassifier...")
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,  # Multithreaded execution
                random_state=42
            )
        else:
            print("XGBoost not available. Falling back to RandomForestClassifier...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                n_jobs=-1,
                random_state=42
            )
            
        print("Training model...")
        model.fit(X, y)
        
        """
        TASK 3: EXPORT
        """
        # 1. Export model
        model_filename = 'temporal_xgb_model.pkl'
        joblib.dump(model, model_filename)
        print(f"Model exported to {model_filename}")
        
        # We also export the scaler/encoders just in case it's needed downstream
        joblib.dump({'scaler': scaler, 'encoders': label_encoders}, 'preprocessing_pipeline.pkl')
        
        # 2. Export exact feature names used
        features_filename = 'feature_columns.json'
        with open(features_filename, 'w') as f:
            json.dump(feature_columns, f, indent=4)
        print(f"Feature names exported to {features_filename}")
        
        return model, feature_columns
    else:
        print("Target variable not found, skipping training.")
        return None, feature_columns

"""
TASK 3: INFERENCE INTERFACE
"""
def predict_temporal_risk(feature_array: np.ndarray, model) -> float:
    """
    Highly optimized for CPU inference in multithreaded FastAPI server.
    Expects a NumPy array (1D or 2D) containing the exact features (scaled/encoded)
    in the order specified by feature_columns.json.
    
    Returns:
        float: Probability representing the risk of default/rejection (class 1).
    """
    # Ensure 2D shape for inference (1 sample, n features)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)
        
    # Using predict_proba to get probabilities. 
    # Index 1 corresponds to "Class 1" which we mapped to Default/Rejection.
    prob = model.predict_proba(feature_array)
    risk_score = float(prob[0][1])
    
    return risk_score

if __name__ == "__main__":
    # Simulate execution on the provided dataset
    csv_file = "train_u6lujuX_CVtuZ9i.csv"  # The file found in the workspace
    
    model, features = train_temporal_model(csv_file)
    
    if model:
        print("\nTesting inference interface...")
        # Create a mock encoded & scaled feature array of correct length
        mock_input = np.random.randn(1, len(features))
        
        risk = predict_temporal_risk(mock_input, model)
        print(f"Mock Applicant Risk Score: {risk:.4f} ({risk * 100:.2f}% chance of rejection)")
