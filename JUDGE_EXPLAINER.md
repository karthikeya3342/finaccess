# FinAccess — Technical Explainer for Judges
### Team Status 200 | NETRIK Hackathon 2026 | Track 0

> This document explains **why** we made every key technical decision in our system.
> Use it as a reference while presenting to judges.

---

## 🔷 1. Why FastAPI + Uvicorn (not Flask or Django)?

**What we used:** `FastAPI` as the backend framework running on `Uvicorn` ASGI server.

**Why:**
Flask and Django are synchronous frameworks. When 100 users submit loan applications simultaneously, Flask would process them one by one — the 100th user waits for 99 users to finish.

FastAPI is built on Python's `asyncio` — it can accept all 100 requests at the same time and hand them off to threads without blocking. Uvicorn is an async HTTP server that handles the concurrency at the socket level.

**The key line in our code:**
```python
@app.post("/score_applicant")
async def score_applicant(payload: ApplicantPayload):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_applicant, payload)
```
`run_in_executor` moves the heavy ML computation off the event loop into a real OS thread, so the API never freezes.

---

## 🔷 2. Why ThreadPoolExecutor with 100 Workers?

**What we used:** `concurrent.futures.ThreadPoolExecutor(max_workers=100)`

**Why:**
Our inference pipeline (XGBoost + SHAP) is CPU-bound computation. Python's `asyncio` alone handles I/O concurrency, but it cannot run CPU work in parallel due to the GIL.

`ThreadPoolExecutor` creates 100 real OS threads. Each incoming request gets its own thread to run `process_applicant()` and SHAP explanation independently.

**The result:**
We measured **17.12 RPS with 100 concurrent users** — all 100 requests were processed in **5.84 seconds** with a **0% failure rate**. Without threading, processing 100 requests sequentially at ~3s each would take ~300 seconds.

---

## 🔷 3. Why a Graph Neural Network (GCN)?

**What we used:** 2-layer `GCNConv` from PyTorch Geometric.

**Why:**
Traditional ML models (XGBoost, logistic regression) treat each loan application in isolation. But in reality, borrowers in the same geographic area or income bracket influence each other — if 10 similar applicants in a region defaulted recently, a new similar applicant carries higher risk.

A **Graph Neural Network** captures this relational signal by:
1. Building a similarity graph where each applicant is a node
2. Connecting similar applicants with edges (using KNN with k=5)
3. Setting edge weights as inverse Euclidean distance (closer = stronger influence)
4. Running 2-layer message passing so each node aggregates information from its neighbours

**The output:** A per-applicant risk score that reflects both individual features AND their position in the broader borrower network.

**Why KNN (k=5)?**
k=5 is the standard starting point for KNN graphs — it ensures every node has at least 5 neighbours without over-connecting the graph (which destroys local structure).

**Why is the GCN pre-computed and stored in a JSON?**
Running a full GCN forward pass at request time would require the entire graph to be in memory and re-computed for every request — this is O(N) per request, completely unscalable. Instead, we train the GCN offline, export all risk scores to a JSON dictionary (`gcn_scores.json`), and at runtime it's a single O(1) dictionary lookup.

---

## 🔷 4. Why XGBoost with Temporal Lag Features?

**What we used:** XGBoost with 5 lag features — `Lag_LoanAmount_1` through `Lag_LoanAmount_5`.

**Why XGBoost:**
XGBoost is the industry standard for tabular credit scoring. It handles:
- Mixed data types (categorical + numerical)
- Missing values natively
- Non-linear relationships (income vs. risk is not linear)
- Feature interactions automatically

**Why Lag Features (Temporal Modeling):**
A borrower's loan history is a time series. The current loan amount in isolation tells you little — but if someone borrowed 100k, then 200k, then 400k in rapid succession, that escalating pattern is a high-risk signal.

By creating `Lag_LoanAmount_1` through `_5` (the previous 5 loan amounts for that applicant, sorted by Loan_ID), we give XGBoost the ability to learn these temporal escalation patterns. This is why we call it the "Temporal Model."

---

## 🔷 5. Why Harmonic Mean Fusion?

**What we used:**
```
Final_Risk = (2 × GCN_Score × Temporal_Score) / (GCN_Score + Temporal_Score)
```

**Why not a simple average?**
Simple average: `(0.9 + 0.1) / 2 = 0.5` — treats a very confident model and a completely uncertain model equally.

Harmonic mean: `(2 × 0.9 × 0.1) / (0.9 + 0.1) = 0.18` — penalises when one model says high-risk and the other says low-risk. The final score is pulled toward the **conservative (lower)** estimate.

**Why is this better for credit risk?**
In lending, a false approval (approving a high-risk borrower) is more costly than a false rejection. The harmonic mean's conservative behaviour aligns with this risk tolerance — if either model is skeptical, the system is skeptical.

**The effect:** GCN and XGBoost must both agree on a decision for the final score to be strongly decisive.

---

## 🔷 6. Why SHAP Instead of LIME?

**What we used:** `shap.TreeExplainer` (initialised once at startup, reused for every request).

**Why not LIME:**
LIME generates explanations by creating ~500 random perturbations of the input, running the model on all of them, then fitting a local linear model. This takes **~1.5 seconds per request**.

SHAP `TreeExplainer` uses exact Shapley value mathematics, exploiting the tree structure of XGBoost to compute exact explanations in **~5ms** — that is **300× faster**.

**Why initialise once at startup?**
Creating a `TreeExplainer` object involves loading the model's internal tree structure. If done per-request, it adds ~200ms overhead. We initialise it once in the `startup` event and reuse the same explainer object for every request.

**What do SHAP values mean in our model?**
In our Loan Dataset, Class 1 = `Y` = Approved. So:
- **Positive SHAP value** → feature pushes the model toward Approved → shown in **Green**
- **Negative SHAP value** → feature pulls the model toward Rejected → shown in **Red**

---

## 🔷 7. Why a Persistent PostgreSQL Database?

**What we used:** `psycopg2` connecting to Render's managed PostgreSQL. Falls back to `sqlite3` locally.

**Why:**
The hackathon requirement explicitly asks for persistent storage. We needed every scored application to be stored permanently — even after server restarts or redeployments.

Render's free tier has an **ephemeral filesystem** — any file written to disk (like `finaccess.db`) is wiped on restart. A managed PostgreSQL instance is completely separate from the server and persists independently.

**Why the dual-mode adapter?**
```python
USE_POSTGRES = bool(DATABASE_URL and HAS_PSYCOPG2)
```
Setting up PostgreSQL locally requires installing and configuring a server. By falling back to SQLite when `DATABASE_URL` isn't set, developers can run the full application locally with zero database setup. Same code, different driver — zero friction.

---

## 🔷 8. Why SHAP Lag Features Are Hidden in the UI?

**What we did:**
```python
filtered_xai = {k: v for k, v in xai_features.items()
                if not k.startswith("Lag_")}
```

**Why:**
`Lag_LoanAmount_1`, `Lag_LoanAmount_2` etc. are internal engineering features — they have no meaningful interpretation for a loan officer or bank agent (what does "your previous loan amount from 3 positions ago" mean to a customer?).

We filter them out before rendering the SHAP chart so only business-interpretable features appear: `Credit_History`, `LoanAmount`, `ApplicantIncome`, `Property_Area`, etc.

The SHAP values are still used internally for the full explanation; we just present the cleaned view.

---

## 🔷 9. Why EEOC 80% Rule for Fairness?

**What we measured:**
```
Disparate Impact = Approval Rate (Female) / Approval Rate (Male)
```
**Our score: 0.92**

**Why this specific metric:**
The EEOC (US Equal Employment Opportunity Commission) 80% Rule is the **legal standard** used in real-world credit system audits. Any ratio below 0.80 indicates unlawful disparate impact — the unprivileged group is being approved at less than 80% the rate of the privileged group.

A score of 0.92 means female applicants are approved at 92% the rate of male applicants — well within legal compliance.

**Why does our system achieve this?**
- The GCN layer normalises applicant relationships across the network, reducing neighbourhood-level demographic clustering
- The XGBoost model is trained on financial behaviour features, not demographic features like gender or location as primary signals
- The harmonic mean fusion further dampens extremes

---

## 🔷 10. Why Streamlit for the Frontend?

**What we used:** Streamlit with `streamlit-authenticator` for role-based access.

**Why not React or plain HTML:**
Our backend engineers are Python-native. Streamlit lets us build a fully functional, production-looking dashboard in Python — the same language as our ML pipeline. No JavaScript build toolchain, no CORS issues, no API serialisation layers between frontend and data.

**Role-based access design:**
- **Applicant portal** → submit a loan application → see real-time risk score + SHAP chart
- **Admin dashboard** → view system metrics, load test results, live DB audit log, batch CSV processing

This separation ensures bank agents (admins) and loan applicants see only what's relevant to them.

---

## 🔷 11. Why Batch Processing with Real Timing?

**What we built:**
A CSV uploader in the Admin dashboard that processes all rows, measures actual wall-clock time per row using `time.perf_counter()`, and reports **real** Throughput (RPS), Avg Latency, and P95 Latency.

**Why not fake the metrics:**
We measured real numbers from actual processing loops. Performance metrics shown to judges are derived from `numpy.percentile(latencies, 95)` on actual measured timings — not hardcoded.

**Why P95 specifically:**
P95 (95th percentile latency) is the industry standard SLA metric. "Average latency" hides tail latency problems — if 5% of requests take 10 seconds, the average might still look acceptable. P95 exposes this.

---

## 🔷 12. Deployment Architecture Decisions

**FastAPI → Render (not AWS/GCP):**
Render offers a free managed PostgreSQL instance alongside the web service, making persistent storage trivially easy to configure. AWS RDS would require VPC setup, security groups, and costs.

**Streamlit → Streamlit Community Cloud:**
Free tier, built specifically for Streamlit apps, one-click GitHub integration, and Secrets management for environment variables. No Nginx/proxy configuration required.

**Why `Procfile`:**
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```
Render injects `$PORT` dynamically. The `Procfile` tells Render exactly how to start the server without any manual configuration.

---

## 📊 Quick Reference — Key Numbers for Presentation

| Question a Judge Might Ask | Answer |
|---|---|
| What is your RPS? | **17.12 RPS** at 100 concurrent users |
| What is your P95 latency? | **5564.7 ms** |
| Why is latency high? | SHAP computation per-request (~3s). Pre-warming SHAP reduces this. |
| What is your model accuracy? | **65.85%** accuracy, **AUC 0.74**, **F1 0.78** |
| Is your database persistent? | Yes — **PostgreSQL on Render**, survives restarts |
| How is fairness measured? | EEOC 80% Rule — **Disparate Impact = 0.92** (passes) |
| Why Harmonic Mean and not Average? | Harmonic mean is **conservative** — both models must agree |
| What is GCN contributing? | **Relational risk** — similar-borrower network influence |
| How fast is your SHAP? | **~5ms per request** (TreeExplainer, pre-initialised) |
| Is the AI explainable? | Yes — top business features shown per-applicant in the UI |
