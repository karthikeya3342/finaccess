"""
FinAccess — Clean Pipeline Refit & Official Holdout Evaluation
==============================================================
1. Loads Dataset 2.csv and performs a strict 80/20 split (random_state=42).
2. Refits the Transductive GCN on the 80% training split  → gcn_scores.json
3. Refits the Temporal XGBoost on the same 80% split     → temporal_xgb_model.pkl
                                                           → preprocessing_pipeline.pkl
4. Evaluates the blended prediction on the 20% test set.
5. Outputs Accuracy, F1-Score, and Confusion Matrix.
"""

import json
import os
import joblib
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

CSV_PATH = "Dataset 2 (1).csv"

# ──────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────
CAT_COLS = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Credit_History', 'Property_Area']

XGB_FEATURE_COLS = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area',
    'Lag_LoanAmount_1', 'Lag_LoanAmount_2', 'Lag_LoanAmount_3',
    'Lag_LoanAmount_4', 'Lag_LoanAmount_5'
]

def impute_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs: median for numeric, mode for categorical."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
    return df

def engineer_gcn_features(df: pd.DataFrame) -> tuple:
    """Return (X_cont_scaled, scaler, loan_ids, y_tensor)."""
    df = df.copy()
    base = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for c in base:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    df['TotalIncome']     = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log((df['TotalIncome'] + 1).astype(float))
    df['LoanAmount_log']  = np.log((df['LoanAmount'] + 1).astype(float))
    df['EMI']             = np.where(df['Loan_Amount_Term'] == 0, 0,
                                     df['LoanAmount'] / df['Loan_Amount_Term'])
    df['BalanceIncome']   = df['TotalIncome'] - (df['EMI'] * 1000)

    cont_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                 'Loan_Amount_Term', 'TotalIncome', 'TotalIncome_log',
                 'LoanAmount_log', 'EMI', 'BalanceIncome']

    scaler   = StandardScaler()
    X_cont   = scaler.fit_transform(df[cont_cols])
    loan_ids = df['Loan_ID'].values

    y = None
    if 'Loan_Status' in df.columns:
        y_raw = df['Loan_Status'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
        y = torch.tensor(y_raw.values, dtype=torch.long)
    return X_cont, scaler, loan_ids, y


# ──────────────────────────────────────────────────────────────
# STEP 1 — DATA SPLIT
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading and Splitting Dataset 2.csv (80/20)")
print("=" * 60)

df_full = pd.read_csv(CSV_PATH)
df_full = impute_df(df_full)

train_df, test_df = train_test_split(df_full, test_size=0.20,
                                     random_state=42, stratify=df_full['Loan_Status'])
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"  Train rows : {len(train_df)}")
print(f"  Test  rows : {len(test_df)}")


# ──────────────────────────────────────────────────────────────
# STEP 2 — GCN REFIT (train split only)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Refitting Transductive GCN on 80% training split")
print("=" * 60)

class RiskGCN(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch)
        self.conv2 = GCNConv(hidden_ch, out_ch)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index, edge_weight)

def build_graph_data(df_split: pd.DataFrame) -> Data:
    """Build a PyG Data object from a DataFrame split."""
    X_cont, _, loan_ids, y = engineer_gcn_features(df_split)

    # Categorical one-hot
    cat_fill = df_split[CAT_COLS].copy()
    for col in CAT_COLS:
        cat_fill[col] = cat_fill[col].fillna(cat_fill[col].mode()[0]
                                             if not cat_fill[col].mode().empty else "Unknown")
    X_cat = pd.get_dummies(cat_fill, drop_first=True).values.astype(float)
    X     = torch.tensor(np.hstack([X_cont, X_cat]), dtype=torch.float32)

    # KNN edges (k=5) on continuous features
    A       = kneighbors_graph(X_cont, n_neighbors=5, mode='distance', include_self=False)
    A_coo   = A.tocoo()
    ei      = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
    ew      = torch.tensor(1.0 / (A_coo.data + 1e-8), dtype=torch.float32)

    data            = Data(x=X, edge_index=ei, edge_attr=ew, y=y)
    data.loan_ids   = loan_ids
    return data

# Build full graph (transductive learning)
full_data = build_graph_data(df_full)

num_nodes  = full_data.x.shape[0]

# Create masks based on train/test split Loan_IDs
train_ids = set(train_df['Loan_ID'].values)
tr_mask = torch.tensor([lid in train_ids for lid in full_data.loan_ids], dtype=torch.bool)
val_mask = ~tr_mask  # Test set is the validation mask for GCN

full_data.train_mask = tr_mask
full_data.val_mask = val_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
full_data = full_data.to(device)

gcn_model = RiskGCN(full_data.x.shape[1], 32, 2).to(device)
opt = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)

gcn_model.train()
for epoch in range(1, 201):
    opt.zero_grad()
    out  = gcn_model(full_data.x, full_data.edge_index, full_data.edge_attr)
    loss = F.cross_entropy(out[full_data.train_mask], full_data.y[full_data.train_mask])
    loss.backward(); opt.step()
    if epoch % 50 == 0:
        gcn_model.eval()
        with torch.no_grad():
            val_out  = gcn_model(full_data.x, full_data.edge_index, full_data.edge_attr)
            val_pred = val_out[full_data.val_mask].argmax(1)
            val_acc  = (val_pred == full_data.y[full_data.val_mask]).float().mean().item()
        print(f"  GCN Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val (Test) Acc: {val_acc:.4f}")
        gcn_model.train()

# Export GCN probabilities for the ENTIRE graph (train + test)
gcn_model.eval()
with torch.no_grad():
    full_probs = F.softmax(gcn_model(full_data.x, full_data.edge_index, full_data.edge_attr), dim=1)
    # Risk = P(class 1) — the rejection probability
    risk       = full_probs[:, 1].cpu().numpy()

gcn_scores = {lid: float(r) for lid, r in zip(full_data.loan_ids, risk)}
with open("gcn_scores.json", "w") as f:
    json.dump(gcn_scores, f, indent=4)
print(f"\n  [+] gcn_scores.json written ({len(gcn_scores)} entries - ALL DATA)")


# ──────────────────────────────────────────────────────────────
# STEP 3 — TEMPORAL XGBOOST REFIT (train split only)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Hyperparameter Tuning & Optimal Threshold Finding")
print("=" * 60)

def prepare_xgb_df(df: pd.DataFrame,
                   label_encoders: dict = None,
                   scaler: StandardScaler = None,
                   fit: bool = True):
    """
    Encode categoricals, build lag features, scale.
    If fit=True  → fit new encoders/scaler and return them.
    If fit=False → transform using supplied encoders/scaler.
    """
    df = df.sort_values('Loan_ID').reset_index(drop=True).copy()

    # Lag features
    loan_amt_mean = df['LoanAmount'].mean()
    for n in range(1, 6):
        df[f'Lag_LoanAmount_{n}'] = df['LoanAmount'].shift(n).fillna(loan_amt_mean)

    df = impute_df(df)

    # Outlier clipping (99th percentile) to improve robustness
    if fit:
        df['ApplicantIncome'] = df['ApplicantIncome'].clip(upper=df['ApplicantIncome'].quantile(0.99))
        df['CoapplicantIncome'] = df['CoapplicantIncome'].clip(upper=df['CoapplicantIncome'].quantile(0.99))
        df['LoanAmount'] = df['LoanAmount'].clip(upper=df['LoanAmount'].quantile(0.99))

    # Feature Engineering for XGBoost (matching GCN's power)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log((df['TotalIncome'] + 1).astype(float))
    df['LoanAmount_log'] = np.log((df['LoanAmount'] + 1).astype(float))
    df['EMI'] = np.where(df['Loan_Amount_Term'] == 0, 0, df['LoanAmount'] / df['Loan_Amount_Term'])
    df['BalanceIncome'] = df['TotalIncome'] - (df['EMI'] * 1000)

    if fit:
        label_encoders = {}
        for col in CAT_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
    else:
        for col in CAT_COLS:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))

    X = df[XGB_FEATURE_COLS]

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, label_encoders, scaler, df.get('Loan_ID', None)

# Map target: N=1 (default), Y=0 (to predict risk)
train_df['_y'] = train_df['Loan_Status'].map({'N': 1, 'Y': 0}).fillna(0).astype(int)

X_train_scaled, le_dict, xgb_scaler, train_loan_ids_raw = prepare_xgb_df(train_df, fit=True)
y_train = train_df['_y'].values

# Apply scale_pos_weight natively to XGBoost to heavily penalize missing a Rejection
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_ratio = float(num_neg / num_pos) if num_pos > 0 else 1.0

# 1. Hyperparameter Tuning using RandomizedSearchCV
print(f"  [*] Running GridSearchCV for XGBoost hyperparameters (scale_pos_weight={scale_ratio:.2f})...")
param_dist = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

base_xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_ratio, random_state=42)
random_search = RandomizedSearchCV(base_xgb, param_distributions=param_dist, 
                                   n_iter=20, cv=5, scoring='f1_macro', n_jobs=-1, random_state=42)
random_search.fit(X_train_scaled, y_train)

xgb_model = random_search.best_estimator_
print(f"  [+] Best XGB Params: {random_search.best_params_}")

# 2. Extract OOF predictions for Fusion
print("  [*] Computing Out-Of-Fold (OOF) predictions to tune Fusion Threshold...")
oof_xgb_probs = cross_val_predict(xgb_model, X_train_scaled, y_train, cv=5, method='predict_proba')[:, 1]

# Align GCN scores for the train set
train_loan_ids_sorted = train_df.sort_values('Loan_ID')['Loan_ID'].values
gcn_arr_train = np.array([gcn_scores.get(lid, 0.5) for lid in train_loan_ids_sorted])

# 3. Deterministic Weighted Average Fusion (30% GCN / 70% XGBoost)
print("  [*] Applying Weighted Average Fusion (30% Graph / 70% Temporal)...")
oof_fusion_probs = (0.3 * gcn_arr_train) + (0.7 * oof_xgb_probs)

# Tune optimal threshold on the FUSION probabilities to maximize F1 (Macro)
best_thresh = 0.5
best_f1 = 0.0
for th in np.arange(0.2, 0.8, 0.01):
    preds = (oof_fusion_probs >= th).astype(int)              # Class 1 (Risk) prediction
    current_f1 = f1_score(y_train, preds, average='macro')
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_thresh = th

print(f"  [+] Tuned Optimal Fusion Threshold: {best_thresh:.3f} (OOF Macro F1: {best_f1:.4f})")

# Persist artefacts
joblib.dump(xgb_model, "temporal_xgb_model.pkl")
joblib.dump({'scaler': xgb_scaler, 'encoders': le_dict}, "preprocessing_pipeline.pkl")
with open("feature_columns.json", "w") as f:
    json.dump(XGB_FEATURE_COLS, f, indent=4)

with open("optimal_threshold.json", "w") as f:
    json.dump({"threshold": float(best_thresh)}, f, indent=4)
    
print("  [+] temporal_xgb_model.pkl, optimal_threshold.json, preprocessing_pipeline.pkl written")


# ──────────────────────────────────────────────────────────────
# STEP 4 — HOLDOUT EVALUATION (20% test set)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Evaluating Learnable Fusion on 20% unseen test set")
print("=" * 60)

# FIXED mapping: N=1 (Rejected/Risk), Y=0 (Approved/Safe) to match Train exactly
test_df['_y'] = test_df['Loan_Status'].map({'N': 1, 'Y': 0}).fillna(0).astype(int)
y_test        = test_df['_y'].values

# XGBoost probabilities on test set
X_test_scaled, _, _, _ = prepare_xgb_df(test_df, label_encoders=le_dict,
                                         scaler=xgb_scaler, fit=False)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]   # P(default/rejection)

# GCN scores (O(1) lookup from JSON; fallback 0.5 for unseen IDs)
test_loan_ids = test_df.sort_values('Loan_ID')['Loan_ID'].values
gcn_arr = np.array([gcn_scores.get(lid, 0.5) for lid in test_loan_ids])

# Weighted Average Fusion prediction
final_risk_probs = (0.3 * gcn_arr) + (0.7 * xgb_probs)

print("\n  [DEBUG] Raw Probabilities (First 10):")
print(f"  GCN: {gcn_arr[:10]}")
print(f"  XGB: {xgb_probs[:10]}")
print(f"  WA_FUSION: {final_risk_probs[:10]}")
print(f"  BEST_THRESH: {best_thresh}")

# Predict Class 1 (Risk) if probability >= tuned best_thresh
preds = (final_risk_probs >= best_thresh).astype(int)

# Invert for reporting purely so Y=Approved is treated as the "positive" class for F1
# We want F1 for Approved correctly predicted. Since 0 = Approved, we invert 0->1, 1->0
y_test_approved = (y_test == 0).astype(int)
preds_approved = (preds == 0).astype(int)

acc  = accuracy_score(y_test_approved, preds_approved)
f1   = f1_score(y_test_approved, preds_approved)
cm   = confusion_matrix(y_test_approved, preds_approved)

print("\n  Official Competition Metrics (20% Hold-Out Set)")
print("  " + "-" * 46)
print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
print(f"  F1-Score (Approved) : {f1:.4f}")
print("\n  Confusion Matrix (rows=Actual, cols=Predicted):")
print(f"  Labels   : [Rejected=0, Approved=1]")
print(f"  {cm}")
print("\n  Breakdown:")
tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives  (Correctly Rejected) : {tn}")
print(f"  False Positives (Wrongly Approved)   : {fp}")
print(f"  False Negatives (Wrongly Rejected)   : {fn}")
print(f"  True Positives  (Correctly Approved) : {tp}")
print("=" * 60)
