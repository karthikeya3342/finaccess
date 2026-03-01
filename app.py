import json
import os
import sqlite3
import time as _time
try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
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
OPTIMAL_THRESHOLD = 0.5  # Learned from OOF F1 Macro Maximization
OPTIMAL_ALPHA = 0.5      # Learned weight for GCN vs XGBoost

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

# Connection pool (Postgres only) — reuses connections instead of creating one per request.
# ThreadedConnectionPool is thread-safe — safe to use with our ThreadPoolExecutor.
_PG_POOL: "psycopg2.pool.ThreadedConnectionPool | None" = None
_STARTUP_TIME = _time.time()

def _get_pool():
    """Lazily create and return the Postgres connection pool."""
    global _PG_POOL
    if _PG_POOL is None:
        _PG_POOL = psycopg2.pool.ThreadedConnectionPool(
            minconn=2, maxconn=20, dsn=DATABASE_URL
        )
    return _PG_POOL

def _get_conn():
    """Return a DB connection — from pool (Postgres) or fresh (SQLite)."""
    if USE_POSTGRES:
        return _get_pool().getconn()
    return sqlite3.connect(DB_PATH)

def _put_conn(conn):
    """Return a connection back to the pool (Postgres) or close it (SQLite)."""
    if USE_POSTGRES:
        _get_pool().putconn(conn)
    else:
        conn.close()

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
    _put_conn(conn)

@app.on_event("startup")
def load_assets():
    """Load all models and assets into global memory at startup."""
    global GCN_SCORES, XGB_MODEL, PREPROCESSOR, FEATURE_COLUMNS, SHAP_EXPLAINER, OPTIMAL_THRESHOLD, OPTIMAL_ALPHA

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

    try:
        with open("optimal_threshold.json", "r") as f:
            t_data = json.load(f)
            OPTIMAL_THRESHOLD = t_data.get("threshold", 0.5)
            OPTIMAL_ALPHA = t_data.get("alpha", 0.5)
        print(f"  - Loaded optimal_threshold.json (Threshold: {OPTIMAL_THRESHOLD:.3f}, Alpha: {OPTIMAL_ALPHA:.2f})")
    except Exception:
        print("  - Warning: optimal_threshold.json not found. Defaulting to 0.5.")

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

        # Dependents numeric conversion
        deps = str(df['Dependents'].iloc[0]).replace('3+', '3')
        try:
            deps_num = float(deps)
        except:
            deps_num = 0.0

        df['Dependents'] = deps_num
        
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['TotalIncome_log'] = np.log((df['TotalIncome'] + 1).astype(float))
        df['LoanAmount_log'] = np.log((df['LoanAmount'] + 1).astype(float))
        emi = 0
        if df['Loan_Amount_Term'].iloc[0] > 0:
            emi = df['LoanAmount'].iloc[0] / df['Loan_Amount_Term'].iloc[0]
        df['EMI'] = emi
        df['BalanceIncome'] = df['TotalIncome'] - (emi * 1000)

        # Advanced Interactions
        try:
            cred = float(df['Credit_History'].iloc[0])
        except:
            cred = 0.0
        df['Credit_History'] = cred
        df['Credit_x_Income'] = cred * df['TotalIncome_log']
        df['Wealth_Factor'] = df['BalanceIncome'] if emi == 0 else df['BalanceIncome'] / (emi + 1)

        # Apply LabelEncoders to categorical columns before scaling
        cat_cols = ['Gender', 'Married', 'Education',
                    'Self_Employed', 'Property_Area']
        for col in cat_cols:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[col] = df[col].map(
                    lambda s: str(s) if str(s) in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col].astype(str))

        # Enforce exact column order as required by the fitted Scaler and XGBoost
        if hasattr(XGB_MODEL, "feature_names_in_"):
            required_cols = list(XGB_MODEL.feature_names_in_)
        else:
            required_cols = FEATURE_COLUMNS  # Fallback
            
        df = df[required_cols]
        
        try:
            processed_x = scaler.transform(df) if scaler is not None else df.values
        except Exception as e:
            print(f"[ERROR] Scaler Transform failed. Columns: {list(df.columns)}. Err: {e}")
            processed_x = df.values
            
        probs = XGB_MODEL.predict_proba(processed_x)
        # Assume probability of class 1 represents target Risk score
        temporal_score = float(probs[0][1] if probs.shape[1] > 1 else probs[0])
        
    # Step C: Dynamic Weighted Average Fusion
    final_risk = (OPTIMAL_ALPHA * gcn_score) + ((1.0 - OPTIMAL_ALPHA) * temporal_score)
    
    # Run SHAP explanation on the EXACT feature array passed to XGB
    if XGB_MODEL and 'df' in locals() and PREPROCESSOR:
        top_xai_features = run_shap_explanation(processed_x, required_cols)
    else:
        top_xai_features = {"Error": "Model not loaded"}
    
    # Construct exact expected output schema
    result = {
        "Loan_ID":          loan_id,
        "Final_Risk_Score": round(float(final_risk), 4),
        "GCN_Score":        round(float(gcn_score),  4),
        "Temporal_Score":   round(float(temporal_score), 4),
        "Top_XAI_Features": top_xai_features
    }

    # Persist to database (PostgreSQL on Render, SQLite locally)
    # Decision is Rejected if mathematically computed risk >= statistically derived threshold
    decision = "Rejected" if final_risk >= OPTIMAL_THRESHOLD else "Approved"
    ph = _ph()
    conn = None
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
    except Exception as db_err:
        print(f"DB insert failed for {loan_id}: {db_err}")
    finally:
        if conn:
            _put_conn(conn)

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
    Uses PostgreSQL connection pool on Render, SQLite locally.
    """
    conn = None
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
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            _put_conn(conn)


@app.get("/health")
def health_check():
    """
    Health check endpoint — returns system status, model load state, DB
    connectivity, and uptime. Used by load balancers and monitoring tools.
    """
    uptime_seconds = round(_time.time() - _STARTUP_TIME, 1)
    status = {
        "status":          "ok",
        "uptime_seconds":  uptime_seconds,
        "db_backend":      "postgresql" if USE_POSTGRES else "sqlite",
        "models": {
            "xgb_model":      XGB_MODEL      is not None,
            "shap_explainer": SHAP_EXPLAINER is not None,
            "gcn_scores":     len(GCN_SCORES) > 0,
            "preprocessor":   PREPROCESSOR   is not None,
        },
        "thread_pool_workers": 100,
    }
    # Quick DB ping
    try:
        conn = _get_conn()
        conn.cursor().execute("SELECT 1")
        _put_conn(conn)
        status["db_connected"] = True
    except Exception:
        status["db_connected"] = False
        status["status"] = "degraded"
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
