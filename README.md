# 🏦 FinAccess — AI Credit Risk Scoring Platform

> **NETRIK Hackathon 2026 · Track 0 · Team Status 200**

A production-grade credit risk scoring system that fuses **Graph Neural Networks (GCN)**, **XGBoost temporal modeling**, and **SHAP explainability** into a single FastAPI backend with a Streamlit admin dashboard.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                    │
│   Applicant Portal  │  Admin Dashboard  │  Audit Log   │
└─────────────────────────────┬───────────────────────────┘
                              │ REST API (HTTP)
┌─────────────────────────────▼───────────────────────────┐
│                   FASTAPI BACKEND                        │
│                                                          │
│   POST /score_applicant      GET /admin/applications     │
│                                                          │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │ GCN Score│  │ XGBoost      │  │  SHAP            │  │
│   │ (O(1)    │  │ Temporal     │  │  TreeExplainer   │  │
│   │  Lookup) │  │ Model        │  │  (~5ms/request)  │  │
│   └────┬─────┘  └──────┬───────┘  └──────────────────┘  │
│        └───────────────┤                                  │
│              ┌─────────▼──────────┐                      │
│              │  Harmonic Mean     │                      │
│              │  Fusion            │                      │
│              │  2·gcn·xgb         │                      │
│              │  ──────────────    │                      │
│              │  gcn + xgb         │                      │
│              └─────────┬──────────┘                      │
└────────────────────────┼────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   PostgreSQL DB     │
              │   (Render /         │
              │    SQLite local)    │
              └─────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Graph-Based Scoring** | 2-layer GCN trained on KNN applicant similarity graph |
| **Temporal Modeling** | XGBoost with 5 lag features capturing loan amount history |
| **Score Fusion** | Harmonic mean fusion of GCN + XGBoost risk scores |
| **Explainable AI** | SHAP TreeExplainer — exact Shapley values in ~5ms |
| **Fairness Audit** | EEOC 80% Rule (Disparate Impact) — Gender parity score: 0.92 |
| **Concurrent Inference** | `ThreadPoolExecutor` with 100 workers — 17.12 RPS @ 100 concurrent users |
| **Persistent DB** | PostgreSQL on Render / SQLite for local dev |
| **Batch Processing** | CSV upload with real per-row timing metrics |
| **Role-Based Auth** | Admin and Applicant portals via `streamlit-authenticator` |

---

## 📁 Project Structure

```
finaccess/
├── app.py                     # FastAPI backend — inference + DB + API endpoints
├── frontend.py                # Streamlit UI — applicant portal + admin dashboard
├── train_gcn.py               # GCN training — graph construction, RiskGCN model, score export
├── build_temporal_model.py    # XGBoost training with temporal lag features
├── refit_and_evaluate.py      # Model evaluation — accuracy, AUC, F1
├── evaluate_fairness.py       # Fairness audit — Disparate Impact (EEOC 80% Rule)
├── test_concurrency.py        # Load testing — 100 concurrent requests, exports perf_metrics.json
├── validate_final.py          # End-to-end validation script
├── gcn_scores.json            # Pre-computed GCN risk scores (O(1) lookup dict)
├── temporal_xgb_model.pkl     # Trained XGBoost model
├── preprocessing_pipeline.pkl # Scaler + LabelEncoders
├── feature_columns.json       # Ordered feature column list
├── Procfile                   # Render deployment config
└── requirements.txt           # All Python dependencies
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models (if pkl files not present)

```bash
# 1. Train the GCN — generates gcn_scores.json
python train_gcn.py

# 2. Train the XGBoost temporal model — generates temporal_xgb_model.pkl
python build_temporal_model.py
```

### Run Locally

**Terminal 1 — FastAPI backend:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run frontend.py
```

Visit `http://localhost:8501`

### Default Login Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| Applicant | `applicant` | `user123` |

---

## 🔌 API Reference

### `POST /score_applicant`

Score a loan application using the full GCN + XGBoost + SHAP pipeline.

**Request Body:**
```json
{
  "loan_id": "LP001002",
  "features": {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 4583,
    "CoapplicantIncome": 1508,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Rural",
    "Lag_LoanAmount_1": 120,
    "Lag_LoanAmount_2": 130,
    "Lag_LoanAmount_3": 110,
    "Lag_LoanAmount_4": 105,
    "Lag_LoanAmount_5": 115
  }
}
```

**Response:**
```json
{
  "Loan_ID": "LP001002",
  "Final_Risk_Score": 0.4821,
  "GCN_Score": 0.4250,
  "Temporal_Score": 0.5512,
  "Top_XAI_Features": {
    "Credit_History": 0.1832,
    "LoanAmount": -0.0741,
    "ApplicantIncome": 0.0512,
    "Married_Yes": -0.0389,
    "Property_Area": 0.0201
  }
}
```

**Decision rule:** `Final_Risk_Score ≥ 0.5` → `Rejected` | `< 0.5` → `Approved`

**SHAP interpretation:** Positive values push toward **Approved** (class 1 = Y in this dataset). Negative values increase rejection risk.

---

### `GET /admin/applications`

Returns the 50 most recent scored applications from the database.

```json
[
  {
    "loan_id": "LP001002",
    "applicant_data": "{...}",
    "risk_score": 0.4821,
    "decision": "Approved",
    "xai_explanation": "{\"Credit_History\": 0.1832, ...}",
    "timestamp": "2024-03-01 02:35:12"
  }
]
```

---

## 🧠 Model Architecture

### 1. Graph Neural Network (GCN)

- **Graph Construction:** KNN graph (k=5) built from applicant financial features using Euclidean distance; edge weights = inverse distance
- **Architecture:** 2-layer `GCNConv` (PyTorch Geometric) with hidden size 16, dropout 0.5
- **Training:** 100 epochs, Adam optimizer (lr=0.01), 80/20 train/val split
- **Output:** Per-applicant risk probabilities exported to `gcn_scores.json` for O(1) runtime lookup
- **Inference:** Dictionary lookup — no graph computation at request time

### 2. XGBoost Temporal Model

- **Features:** 11 base features + 5 lag features (`Lag_LoanAmount_1` through `_5`)
- **Preprocessing:** LabelEncoding for categoricals, StandardScaler
- **Purpose:** Captures temporal financial pattern signals alongside static features

### 3. Harmonic Mean Fusion

```python
final_risk = (2 × gcn_score × temporal_score) / (gcn_score + temporal_score)
```

The harmonic mean penalises cases where either model is highly uncertain (near 0.5), producing a more conservative and calibrated final score.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | 65.85% |
| AUC-ROC | 0.74 |
| F1 Score | 0.7812 |
| Demographic Parity (Disparate Impact) | **0.92** ✅ |

---

## ⚖️ Fairness Evaluation

Evaluated using the **EEOC 80% Rule (Disparate Impact):**

```
Disparate Impact = Approval Rate (Female) / Approval Rate (Male)
```

| Group | Approval Rate |
|---|---|
| Male | Privileged baseline |
| Female | Compared against male |

**Result: 0.92** — exceeds the 0.80 industry compliance threshold.

Run the fairness audit locally:
```bash
python evaluate_fairness.py
```

---

## 🚀 Load Testing

```bash
python test_concurrency.py
```

Fires **100 unique concurrent requests** against the `/score_applicant` endpoint and writes results to `perf_metrics.json`.

**Measured Results (100 concurrent users, 100 workers):**

| Metric | Value |
|---|---|
| Total Wall Time | 5.84s |
| Throughput | 17.12 RPS |
| Avg Latency | 3036.3 ms |
| P95 Latency | 5564.7 ms |
| Success Rate | 100% (100/100) |

---

## ☁️ Deployment

### FastAPI → Render

1. Create a **PostgreSQL** instance on Render → copy Internal Database URL
2. Create a **Web Service** → connect GitHub repo
3. Set environment variable: `DATABASE_URL = <postgres-url>`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Streamlit → Streamlit Community Cloud

1. Connect GitHub repo → set main file to `frontend.py`
2. In **Settings → Secrets**, add:
```toml
API_BASE_URL = "https://your-render-app.onrender.com"
```

### Environment Variables

| Variable | Required In | Description |
|---|---|---|
| `DATABASE_URL` | Render | PostgreSQL connection string |
| `API_BASE_URL` | Streamlit Cloud | Render FastAPI base URL |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` |
| Graph Model | PyTorch Geometric (`GCNConv`) |
| Tabular Model | XGBoost |
| Explainability | SHAP `TreeExplainer` |
| Database | PostgreSQL (Render) / SQLite (local) |
| Frontend | Streamlit + Plotly |
| Auth | `streamlit-authenticator` + bcrypt |
| Deployment | Render (API) + Streamlit Community Cloud (UI) |

---

## 👥 Team

**Team Status 200** · NETRIK Hackathon 2026 · Track 0
