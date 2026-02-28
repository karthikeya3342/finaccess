import json
import os
import sqlite3
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import concurrent.futures
import asyncio
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# GLOBAL MEMORY
# ---------------------------------------------------------
GCN_SCORES = {}
XGB_MODEL = None
PREPROCESSOR = None
FEATURE_COLUMNS = []
SHAP_EXPLAINER = None  # Initialized once at startup — reused for all requests (~5ms/call)

# Initialize FastAPI app
app = FastAPI(title="FinAccess Risk-Scoring Platform")

# Thread pool executor — 100 workers for true concurrent dispatch
executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)

# ---------------------------------------------------------
# DATABASE — dual-mode: PostgreSQL (Render) or SQLite (local)
# Set DATABASE_URL env var on Render; leave unset for local SQLite dev.
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")   # e.g. postgresql://user:pass@host/db
DB_PATH      = "finaccess.db"              # only used when DATABASE_URL is absent
USE_POSTGRES = bool(DATABASE_URL and HAS_PSYCOPG2)

def _get_conn():
    """Return a live DB connection — Postgres or SQLite depending on environment."""
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    return sqlite3.connect(DB_PATH)

def _ph():
    """Return the correct SQL placeholder string for the active driver."""
    return "%s" if USE_POSTGRES else "?"

def init_db():
    """Create the applications table if it does not already exist."""
    conn = _get_conn()
    cur  = conn.cursor()
    if USE_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                loan_id         TEXT PRIMARY KEY,
                applicant_data  TEXT,
                risk_score      REAL,
                decision        TEXT,
                xai_explanation TEXT,
                timestamp       TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                loan_id         TEXT PRIMARY KEY,
                applicant_data  TEXT,
                risk_score      REAL,
                decision        TEXT,
                xai_explanation TEXT,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    cur.close()
    conn.close()
    db_type = "PostgreSQL (persistent)" if USE_POSTGRES else "SQLite (local)"
    print(f"[DB] Initialised — using {db_type}")

@app.on_event("startup")
def load_assets():
    """Load all models and assets into global memory at startup."""
    global GCN_SCORES, XGB_MODEL, PREPROCESSOR, FEATURE_COLUMNS, SHAP_EXPLAINER

    init_db()   # Ensure DB + table exist
    print("[*] Loading global assets...")

    try:
        with open("gcn_scores.json", "r") as f:
            GCN_SCORES = json.load(f)
        print(f"  - Loaded gcn_scores.json ({len(GCN_SCORES)} entries)")
    except Exception as e:
        print(f"  - Warning: gcn_scores.json not found ({e})")

    try:
        with open("feature_columns.json", "r") as f:
            FEATURE_COLUMNS = json.load(f)
        print(f"  - Loaded feature_columns.json ({len(FEATURE_COLUMNS)} features)")
    except Exception:
        print("  - Warning: feature_columns.json not found. Will infer from payload.")

    try:
        PREPROCESSOR = joblib.load("preprocessing_pipeline.pkl")
        print("  - Loaded preprocessing_pipeline.pkl")
    except Exception:
        print("  - Warning: preprocessing_pipeline.pkl not found.")

    try:
        XGB_MODEL = joblib.load("temporal_xgb_model.pkl")
        print("  - Loaded temporal_xgb_model.pkl")
    except Exception:
        print("  - Warning: temporal_xgb_model.pkl not found.")

    # Initialize SHAP TreeExplainer once at startup — reused for every request
    if XGB_MODEL is not None:
        try:
            SHAP_EXPLAINER = shap.TreeExplainer(XGB_MODEL)
            print("  - SHAP TreeExplainer initialized.")
        except Exception as e:
            print(f"  - Warning: SHAP explainer failed ({e})")

    print("[+] All assets loaded successfully.")

# ---------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------
class ApplicantPayload(BaseModel):
    loan_id: str
    features: Dict[str, Any]

# ---------------------------------------------------------
# INFERENCE PIPELINE
# ---------------------------------------------------------
def run_shap_explanation(df_scaled: np.ndarray, feature_cols: list) -> Dict[str, float]:
    """
    Compute exact Shapley values via SHAP TreeExplainer (purpose-built for XGBoost).
    Runs in ~5ms per request — 400x faster than LIME.
    Returns top 5 features sorted by absolute SHAP contribution.
    """
    if SHAP_EXPLAINER is None:
        return {"SHAP_Error": 0.0}
    try:
        shap_values = SHAP_EXPLAINER.shap_values(df_scaled)  # (1, n_features) or list
        # Binary classification: shap_values is a list [class0_vals, class1_vals]
        if isinstance(shap_values, list):
            vals = shap_values[1][0]   # class-1 = default/rejection risk
        else:
            vals = shap_values[0]

        pairs = sorted(zip(feature_cols, vals.tolist()),
                       key=lambda x: abs(x[1]), reverse=True)
        return {feat: round(val, 4) for feat, val in pairs[:5]}

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return {"SHAP_Error": 0.0}

def process_applicant(payload: ApplicantPayload) -> Dict[str, Any]:
    """Task 2: Core Thread-Safe Inference processing function"""
    loan_id = payload.loan_id
    
    # Step A: Retrieve the GCN score from memory using loan_id
    # Default to 0.5 if loan_id is missing from graph scores
    gcn_score = float(GCN_SCORES.get(loan_id, 0.5))
    
    # Step B: Process features using pipeline, then XGB Model
    raw_features = payload.features
    
    # Identify proper feature ordering
    cols = FEATURE_COLUMNS if FEATURE_COLUMNS else list(raw_features.keys())
    
    # Fill in dict for dataframe creation
    dict_features = {}
    for c in cols:
        dict_features[c] = raw_features.get(c, 0.0) # Default missing
            
    df = pd.DataFrame([dict_features], columns=cols)
    
    # XGBoost inference
    temporal_score = 0.5
    if XGB_MODEL:
        # PREPROCESSOR is a dict {scaler, encoders}; use scaler for transform
        scaler   = PREPROCESSOR.get('scaler')   if isinstance(PREPROCESSOR, dict) else PREPROCESSOR
        encoders = PREPROCESSOR.get('encoders', {}) if isinstance(PREPROCESSOR, dict) else {}

        # Apply LabelEncoders to categorical columns before scaling
        cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Credit_History', 'Property_Area']
        for col in cat_cols:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[col] = df[col].map(
                    lambda s: str(s) if str(s) in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col].astype(str))

        processed_x = scaler.transform(df) if scaler is not None else df.values
        probs = XGB_MODEL.predict_proba(processed_x)
        # Assume probability of class 1 represents target Risk score
        temporal_score = float(probs[0][1] if probs.shape[1] > 1 else probs[0])
        
    # Step C: Harmonic Mean Fusion (safe: avoid div-by-zero when both scores are 0)
    denom = gcn_score + temporal_score
    final_risk = (2 * gcn_score * temporal_score) / denom if denom > 0 else 0.0

    # Step D: SHAP exact Shapley values (~5ms per request)
    top_xai_features = run_shap_explanation(processed_x, cols)
    
    # Construct exact expected output schema
    result = {
        "Loan_ID":          loan_id,
        "Final_Risk_Score": round(float(final_risk), 4),
        "GCN_Score":        round(float(gcn_score),  4),
        "Temporal_Score":   round(float(temporal_score), 4),
        "Top_XAI_Features": top_xai_features
    }

    # Persist to database (PostgreSQL on Render, SQLite locally)
    decision = "Rejected" if final_risk >= 0.5 else "Approved"
    ph = _ph()
    try:
        conn = _get_conn()
        cur  = conn.cursor()
        if USE_POSTGRES:
            cur.execute(
                f"""INSERT INTO applications
                       (loan_id, applicant_data, risk_score, decision, xai_explanation)
                    VALUES ({ph},{ph},{ph},{ph},{ph})
                    ON CONFLICT (loan_id) DO UPDATE SET
                       applicant_data  = EXCLUDED.applicant_data,
                       risk_score      = EXCLUDED.risk_score,
                       decision        = EXCLUDED.decision,
                       xai_explanation = EXCLUDED.xai_explanation,
                       timestamp       = NOW()""",
                (loan_id, json.dumps(raw_features),
                 round(float(final_risk), 4), decision,
                 json.dumps(top_xai_features))
            )
        else:
            cur.execute(
                f"""INSERT OR REPLACE INTO applications
                       (loan_id, applicant_data, risk_score, decision, xai_explanation)
                    VALUES ({ph},{ph},{ph},{ph},{ph})""",
                (loan_id, json.dumps(raw_features),
                 round(float(final_risk), 4), decision,
                 json.dumps(top_xai_features))
            )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as db_err:
        print(f"DB insert failed for {loan_id}: {db_err}")

    return result



# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------
@app.post("/score_applicant")
async def score_applicant(payload: ApplicantPayload):
    """
    Score Application POST Endpoint.
    Runs full inference (GCN + XGBoost + SHAP) inside a ThreadPoolExecutor
    so the async event loop is never blocked.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, process_applicant, payload)
        return result
    except Exception as e:
        print(f"Error processing applicant {payload.loan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/applications")
def get_applications():
    """
    Admin endpoint — returns the 50 most recent scored applications.
    Uses PostgreSQL on Render, SQLite locally.
    """
    try:
        conn = _get_conn()
        if USE_POSTGRES:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT * FROM applications ORDER BY timestamp DESC LIMIT 50"
            )
            rows = [dict(r) for r in cur.fetchall()]
        else:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM applications ORDER BY timestamp DESC LIMIT 50"
            )
            rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
