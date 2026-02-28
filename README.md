# FinAccess Risk-Scoring Platform 🚀

Welcome to the **FinAccess Risk-Scoring Platform**, an advanced machine learning backend designed for robust, real-time credit risk evaluation. Built for our hackathon submission, this architecture intelligently blends **Graph-Based Relational Modeling (GCN)** with **Dynamic Temporal Gradient Boosting (XGBoost)** to provide highly accurate, interpretable lending decisions at scale.

## 🧠 Architectural Overview

Our risk-scoring platform operates on a dual-engine architecture to evaluate loan applicants:

1. **Transductive Graph Convolutional Network (GCN)**:
   - Evaluates the applicant's relational position within the broader lending ecosystem.
   - We construct a static $k=5$ KNN graph based on inverse Euclidean distances of applicant continuous features (e.g., Income, Loan Amount).
   - A 2-layer `GCNConv` network parses this graph structure to determine a global structural risk probability.

2. **Temporal XGBoost Engine**:
   - Evaluates the applicant's dynamic historical data and pure tabular features.
   - Processes engineered metrics such as `TotalIncome_log` and debt-to-income `EMI` ratios.

3. **Inference Blender & XAI**:
   - The FastAPI backend intercepts both predictions and calculates a final risk score using the optimized weighted ensemble: `(0.4 * GCN_Score) + (0.6 * Temporal_Score)`.
   - Every prediction is paired with a **LIME (Local Interpretable Model-Agnostic Explanations)** tabular explainer to extract the Top 5 driving factors behind the decision, ensuring full regulatory compliance and interpretability.

---

## 📂 Project Structure & Assets

*   `app.py` - The core multithreaded FastAPI server.
*   `train_gcn.py` - The PyTorch Geometric pipeline used to train the Transductive GCN graph and engineer features.
*   `gcn_scores.json` - High-speed $O(1)$ lookup dictionary containing pre-computed GCN risk scores.
*   `temporal_xgb_model.pkl` - The trained standalone XGBoost inference model.
*   `preprocessing_pipeline.pkl` - `scikit-learn` encoding and scaling pipeline.
*   `feature_columns.json` - Schema mapping defining the exact features expected by the Temporal Model.
*   `requirements.txt` - Python environment dependencies.
*   `outputs/` - Directory where finalized asynchronous inference reports are saved dynamically.

---

## ⚡ Scaling & Concurrency (Bypassing the GIL)

The platform is designed to handle mass load-testing seamlessly.
Because Python's Global Interpreter Lock (GIL) normally blocks asynchronous web loops during heavy model inference (like XGBoost or LIME matrix calculations), our core endpoint `POST /score_applicant` pushes all ML processing into a `concurrent.futures.ThreadPoolExecutor(max_workers=10)`. 

This guarantees the `uvicorn` event loop remains unblocked, allowing the server to accept and queue hundreds of concurrent requests without dropping connections.

---

## 🛠️ Installation & Execution

### 1. Install Dependencies
Make sure you have Python installed, then install the required inference packages:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Start the FastAPI server using Uvicorn. *(Note: On Windows, to prevent multiprocessing socket errors, do not use the `--workers` flag locally).*

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the Endpoint
With the server running, you can access the interactive Swagger UI directly in your browser:  
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

#### Example Payload:
```json
{
  "loan_id": "LP001002",
  "features": {
    "ApplicantIncome": 5849,
    "CoapplicantIncome": 0,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Gender_Male": 1,
    "Married_Yes": 0,
    "Property_Area_Urban": 1,
    "TotalIncome_log": 8.674,
    "LoanAmount_log": 4.859,
    "EMI": 0.355,
    "BalanceIncome": 5494
  }
}
```

The server will instantaneously return the blended scores and LIME feature explanations, and drop a comprehensive report named `{loan_id}.json` safely into the local `outputs/` directory!
