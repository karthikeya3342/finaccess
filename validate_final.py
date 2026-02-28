import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def validate_final():
    # 1. FILE DEPENDENCY CHECKS
    required_files = [
        "temporal_xgb_model.pkl",
        "preprocessing_pipeline.pkl",
        "gcn_scores.json",
        "Dataset 2.csv"
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Missing REQUIRED file! '{f}' not found in the current directory.")
            return

    # Load artifacts
    print("[*] Loading models and dictionaries...")
    xgb_model = joblib.load("temporal_xgb_model.pkl")
    
    pipeline = joblib.load("preprocessing_pipeline.pkl")
    scaler = pipeline.get("scaler")
    encoders = pipeline.get("encoders", {})
        
    with open("gcn_scores.json", "r") as f:
        gcn_scores = json.load(f)

    # 2. LOAD DATASET
    print("[*] Processing Dataset 2.csv...")
    df = pd.read_csv("Dataset 2.csv")
    
    # Target column mapping
    target_col = "Loan_Status"
    if target_col not in df.columns:
        # Fallback to last column if Loan_Status isn't found
        target_col = df.columns[-1]
        
    df[target_col] = df[target_col].map({'Y': 1, 'N': 0})
    y_true = df[target_col].values
    loan_ids = df['Loan_ID'].values

    # 3. ENGINEER LAGS
    df = df.sort_values(by="Loan_ID").reset_index(drop=True)
    mean_loan_amt = df['LoanAmount'].mean()
    
    for n in range(1, 6):
        col_name = f"Lag_LoanAmount_{n}"
        df[col_name] = df['LoanAmount'].shift(n).fillna(mean_loan_amt)

    # Fill NaNs for base features
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    # 4. APPLY ENCODERS
    cat_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_columns:
        if col in df.columns and col in encoders:
            enc = encoders[col]
            # Use a lambda to map any value not in encoder.classes_ to encoder.classes_[0]
            df[col] = df[col].map(lambda s: s if s in enc.classes_ else enc.classes_[0])
            df[col] = enc.transform(df[col])

    # 5. MATCH COLUMNS
    exact_columns = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        'Credit_History', 'Property_Area', 'Lag_LoanAmount_1', 'Lag_LoanAmount_2', 
        'Lag_LoanAmount_3', 'Lag_LoanAmount_4', 'Lag_LoanAmount_5'
    ]
    
    X = df[exact_columns]

    # 6. PREDICT & FUSE
    # Scale Data
    X_scaled = scaler.transform(X)
    
    # Predict XGB Probabilities
    xgb_probs = xgb_model.predict_proba(X_scaled)[:, 1] # Probability of class 1
    
    # Calculate Final Risk
    final_risks = []
    for idx, row in df.iterrows():
        l_id = row['Loan_ID']
        gcn_score = gcn_scores.get(str(l_id), 0.5) # Fallback 0.5 if not in graph
        xgb_prob = xgb_probs[idx]
        
        final_risk = (0.4 * gcn_score) + (0.6 * xgb_prob)
        final_risks.append(final_risk)
        
    final_risks = np.array(final_risks)

    # 7 & 8. CALCULATE METRICS
    # Logic A: Prediction = 1 if Final_Risk < 0.5 else 0
    preds_A = (final_risks < 0.5).astype(int)
    acc_A = accuracy_score(y_true, preds_A)
    f1_A = f1_score(y_true, preds_A)

    # Logic B: Prediction = 1 if Final_Risk >= 0.5 else 0
    preds_B = (final_risks >= 0.5).astype(int)
    acc_B = accuracy_score(y_true, preds_B)
    f1_B = f1_score(y_true, preds_B)

    print("\n[+] Validation Metrics Computed")
    print(f"Logic A (Risk < 0.5)  -> Acc: {acc_A:.4f}  |  F1: {f1_A:.4f}")
    print(f"Logic B (Risk >= 0.5) -> Acc: {acc_B:.4f}  |  F1: {f1_B:.4f}")
    
    print("\n" + "="*40)
    if acc_A > acc_B:
        print(f"--- LOGIC A WINS! ---")
        print(f"Final Accuracy: {acc_A:.4f}")
        print(f"Final F1 Score: {f1_A:.4f}")
    else:
        print(f"--- LOGIC B WINS! ---")
        print(f"Final Accuracy: {acc_B:.4f}")
        print(f"Final F1 Score: {f1_B:.4f}")
    print("="*40)

if __name__ == "__main__":
    validate_final()
