import joblib
import json
import pandas as pd
import numpy as np

print("📊 Initializing AI Fairness Audit...")

# 1. Load ML Assets
try:
    assets = joblib.load('preprocessing_pipeline.pkl')
    SCALER = assets['scaler']
    ENCODERS = assets['encoders']
    XGB_MODEL = joblib.load('temporal_xgb_model.pkl')
    with open("gcn_scores.json", "r") as f:
        gcn_scores = json.load(f)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# 2. Load Raw Data
df = pd.read_csv("Dataset 2.csv")

# Save a copy of the raw demographics BEFORE we encode them into numbers
df_raw_demographics = df[['Gender', 'Property_Area']].copy()

# 3. Quick Feature Engineering (Lags)
df = df.sort_values('Loan_ID').reset_index(drop=True)
for i in range(1, 6):
    df[f'Lag_LoanAmount_{i}'] = df['LoanAmount'].shift(i)
    df[f'Lag_LoanAmount_{i}'] = df[f'Lag_LoanAmount_{i}'].fillna(df['LoanAmount'].mean())

# 4. Apply Encoders securely
for col, encoder in ENCODERS.items():
    if col in df.columns:
        df[col] = df[col].map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
        df[col] = encoder.transform(df[col])

# 5. Extract exact 16 columns for inference
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_History', 'Property_Area', 'Lag_LoanAmount_1', 'Lag_LoanAmount_2', 
            'Lag_LoanAmount_3', 'Lag_LoanAmount_4', 'Lag_LoanAmount_5']

X = df[features]
X_scaled = SCALER.transform(X)

# 6. Generate Fusion Predictions
xgb_probs = XGB_MODEL.predict_proba(X_scaled)[:, 1]

predictions = []
for idx, row in df.iterrows():
    loan_id = row['Loan_ID']
    gcn_score = gcn_scores.get(loan_id, 0.5) # default to 0.5 if not in graph
    temporal_score = xgb_probs[idx]
    
    # Using the Harmonic Mean Fusion we validated
    final_risk = (2 * gcn_score * temporal_score) / (gcn_score + temporal_score)
    
    # Logic A: Approved if Risk < 0.5
    decision = "Approved" if final_risk < 0.5 else "Rejected"
    predictions.append(decision)

# 7. Attach predictions to the raw demographic data
df_raw_demographics['System_Decision'] = predictions

# ==========================================
# ⚖️ FAIRNESS AUDIT: GENDER
# ==========================================
print("=" * 40)
print(" ⚖️  FAIRNESS AUDIT: GENDER (MALE VS FEMALE)")
print("=" * 40)

df_gender = df_raw_demographics.dropna(subset=['Gender'])

male_total    = len(df_gender[df_gender['Gender'] == 'Male'])
male_approved = len(df_gender[(df_gender['Gender'] == 'Male') & (df_gender['System_Decision'] == 'Approved')])
male_approval_rate = male_approved / male_total if male_total > 0 else 0

female_total    = len(df_gender[df_gender['Gender'] == 'Female'])
female_approved = len(df_gender[(df_gender['Gender'] == 'Female') & (df_gender['System_Decision'] == 'Approved')])
female_approval_rate = female_approved / female_total if female_total > 0 else 0

print(f"Male Approval Rate:   {male_approval_rate*100:.1f}% ({male_approved}/{male_total})")
print(f"Female Approval Rate: {female_approval_rate*100:.1f}% ({female_approved}/{female_total})")

if male_approval_rate > 0:
    disparate_impact_gender = female_approval_rate / male_approval_rate
    print(f"\nDisparate Impact Ratio (Gender): {disparate_impact_gender:.3f}")
    if disparate_impact_gender >= 0.80:
        print("✅ PASSED: The system satisfies the EEOC 80% Rule for Gender Parity.")
    else:
        print("⚠️  WARNING: Potential gender bias detected.")

print("=" * 40 + "\n")

# ==========================================
# ⚖️ FAIRNESS AUDIT: PROPERTY AREA (URBAN vs RURAL)
# ==========================================
print("=" * 40)
print(" ⚖️  FAIRNESS AUDIT: PROPERTY AREA (URBAN VS RURAL)")
print("=" * 40)

df_area = df_raw_demographics.dropna(subset=['Property_Area'])

urban_total    = len(df_area[df_area['Property_Area'] == 'Urban'])
urban_approved = len(df_area[(df_area['Property_Area'] == 'Urban') & (df_area['System_Decision'] == 'Approved')])
urban_rate     = urban_approved / urban_total if urban_total > 0 else 0

rural_total    = len(df_area[df_area['Property_Area'] == 'Rural'])
rural_approved = len(df_area[(df_area['Property_Area'] == 'Rural') & (df_area['System_Decision'] == 'Approved')])
rural_rate     = rural_approved / rural_total if rural_total > 0 else 0

semi_total    = len(df_area[df_area['Property_Area'] == 'Semiurban'])
semi_approved = len(df_area[(df_area['Property_Area'] == 'Semiurban') & (df_area['System_Decision'] == 'Approved')])
semi_rate     = semi_approved / semi_total if semi_total > 0 else 0

print(f"Urban     Approval Rate: {urban_rate*100:.1f}% ({urban_approved}/{urban_total})")
print(f"Semiurban Approval Rate: {semi_rate*100:.1f}% ({semi_approved}/{semi_total})")
print(f"Rural     Approval Rate: {rural_rate*100:.1f}% ({rural_approved}/{rural_total})")

# Disparate Impact: Rural (unprivileged) vs Urban (privileged)
if urban_rate > 0:
    di_rural_vs_urban = rural_rate / urban_rate
    print(f"\nDisparate Impact Ratio (Rural vs Urban): {di_rural_vs_urban:.3f}")
    if di_rural_vs_urban >= 0.80:
        print("✅ PASSED: Rural applicants meet the EEOC 80% Rule vs Urban applicants.")
    else:
        print("⚠️  WARNING: Potential geographic bias detected — rural applicants under-served.")

if urban_rate > 0:
    di_semi_vs_urban = semi_rate / urban_rate
    print(f"Disparate Impact Ratio (Semiurban vs Urban): {di_semi_vs_urban:.3f}")
    if di_semi_vs_urban >= 0.80:
        print("✅ PASSED: Semiurban applicants meet the EEOC 80% Rule vs Urban applicants.")
    else:
        print("⚠️  WARNING: Semiurban applicants may be disadvantaged vs Urban.")

print("=" * 40 + "\n")
