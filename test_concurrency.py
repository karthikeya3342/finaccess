import concurrent.futures
import requests
import time
import random
import json
import numpy as np

URL = "http://127.0.0.1:8000/score_applicant"

GENDERS         = ["Male", "Female"]
MARRIED         = ["Yes", "No"]
DEPENDENTS      = ["0", "1", "2", "3+"]
EDUCATION       = ["Graduate", "Not Graduate"]
SELF_EMPLOYED   = ["Yes", "No"]
PROPERTY_AREAS  = ["Urban", "Semiurban", "Rural"]

random.seed(None)  # different values on every run

def make_unique_payload(request_id: int) -> dict:
    """Generate a randomised applicant payload — every request is distinct."""
    loan_amt = random.randint(50, 500)
    return {
        "loan_id": f"LOAD_TEST_{request_id:04d}",
        "features": {
            "Gender":            random.choice(GENDERS),
            "Married":           random.choice(MARRIED),
            "Dependents":        random.choice(DEPENDENTS),
            "Education":         random.choice(EDUCATION),
            "Self_Employed":     random.choice(SELF_EMPLOYED),
            "ApplicantIncome":   random.randint(1500, 20000),
            "CoapplicantIncome": random.randint(0, 8000),
            "LoanAmount":        loan_amt,
            "Loan_Amount_Term":  random.choice([120, 180, 240, 300, 360, 480]),
            "Credit_History":    random.choice([0, 1]),
            "Property_Area":     random.choice(PROPERTY_AREAS),
            "Lag_LoanAmount_1":  random.randint(50, 500),
            "Lag_LoanAmount_2":  random.randint(50, 500),
            "Lag_LoanAmount_3":  random.randint(50, 500),
            "Lag_LoanAmount_4":  random.randint(50, 500),
            "Lag_LoanAmount_5":  random.randint(50, 500),
        }
    }

def send_request(request_id: int):
    payload = make_unique_payload(request_id)
    t0 = time.time()
    try:
        resp = requests.post(URL, json=payload, timeout=60)
        latency = time.time() - t0
        return resp.status_code, latency
    except Exception as e:
        return str(e), time.time() - t0

if __name__ == "__main__":
    total_requests = 100
    print(f"Firing {total_requests} concurrent requests with UNIQUE feature sets...")
    print(f"(No cache hits — each request forces a full SHAP computation)\n")

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = list(executor.map(send_request, range(total_requests)))

    total_time = time.time() - start

    statuses     = [r[0] for r in results]
    latencies_ms = [r[1] * 1000 for r in results]   # convert seconds -> ms
    successes    = statuses.count(200)
    failures     = total_requests - successes

    avg_latency_ms = round(float(np.mean(latencies_ms)),         1)
    p95_latency_ms = round(float(np.percentile(latencies_ms, 95)), 1)
    rps            = round(total_requests / total_time,           2)

    print("-" * 40)
    print(f"Total Wall Time  : {total_time:.2f}s")
    print(f"Successes (200)  : {successes}")
    print(f"Failures         : {failures}")
    print(f"Throughput (RPS) : {rps}")
    print(f"Min Latency      : {min(latencies_ms):.1f}ms")
    print(f"Max Latency      : {max(latencies_ms):.1f}ms")
    print(f"Avg Latency      : {avg_latency_ms}ms")
    print(f"P95 Latency      : {p95_latency_ms}ms")
    print("-" * 40)

    # Write metrics to JSON — Streamlit admin dashboard reads this file
    metrics = {
        "total_requests": total_requests,
        "successes":      successes,
        "failures":       failures,
        "total_time_s":   round(total_time, 2),
        "rps":            rps,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "min_latency_ms": round(float(min(latencies_ms)), 1),
        "max_latency_ms": round(float(max(latencies_ms)), 1),
        "run_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("perf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[+] Metrics saved to perf_metrics.json")