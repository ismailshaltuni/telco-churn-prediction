"""
predict_customer.py - score a single customer with the trained churn model.

Created: 16 October 2026 (v1.0)
Group project - Predicting Customer Churn for a Telecom Provider (Option A)

Usage
-----
    python src/predict_customer.py                      # scores two built-in example customers
    python src/predict_customer.py --json my_customer.json   # scores a customer described in a JSON file

The JSON file must contain the 19 raw columns of the Telco dataset (everything except
customerID and Churn). See src/example_customer.json for the exact field names.
"""
import argparse
import json
import os
import sys

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "models", "churn_model_gradient_boosting.joblib")

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# Business assumptions used for the expected-value calculation (same values as the notebook, Section 8).
RETENTION_HORIZON = 12     # months of revenue retained if the save succeeds
INTERVENTION_COST = 30.0   # $ per customer contacted
SAVE_RATE = 0.30           # probability an at-risk customer is retained after the offer

EXAMPLES = {
    "High-risk new fibre customer": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.4, "TotalCharges": 286.2,
    },
    "Loyal two-year customer": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 85.0, "TotalCharges": 5100.0,
    },
}


def prepare_customer(record: dict, features: list) -> pd.DataFrame:
    """Re-create the engineered features exactly as in the notebook (Section 4)."""
    r = dict(record)
    r["SeniorCitizen"] = {0: "No", 1: "Yes"}.get(r.get("SeniorCitizen"), r.get("SeniorCitizen"))
    total = float(r.get("TotalCharges") or 0.0)
    r["NumAddOnServices"] = sum(r[a] == "Yes" for a in ADDONS)
    r["HasSecurityOrSupport"] = int(r["OnlineSecurity"] == "Yes" or r["TechSupport"] == "Yes")
    r["AvgMonthlySpend"] = total / max(r["tenure"], 1) if r["tenure"] > 0 else r["MonthlyCharges"]
    r["ChargeRatio"] = r["MonthlyCharges"] / r["AvgMonthlySpend"]
    r["IsNewCustomer"] = int(r["tenure"] <= 12)
    r["AutoPay"] = int("automatic" in r["PaymentMethod"])
    return pd.DataFrame([r])[features]


def score(bundle: dict, label: str, record: dict) -> None:
    X = prepare_customer(record, bundle["features"])
    p = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    flagged = p >= bundle["threshold"]
    ev = p * SAVE_RATE * RETENTION_HORIZON * record["MonthlyCharges"] - INTERVENTION_COST
    print(f"\n{label}")
    print(f"  Contract={record['Contract']}, tenure={record['tenure']} months, internet={record['InternetService']}, "
          f"payment={record['PaymentMethod']}, monthly bill=${record['MonthlyCharges']:.2f}")
    print(f"  Predicted probability of churn : {p:.1%}")
    print(f"  Decision threshold             : {bundle['threshold']:.2f}")
    print(f"  Recommendation                 : {'FLAG - contact with retention offer' if flagged else 'No action'}")
    print(f"  Expected value of a $30 offer  : ${ev:,.0f}  ({'worth it' if ev > 0 else 'not worth it'})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a customer with the trained churn model.")
    ap.add_argument("--json", help="Path to a JSON file describing one customer")
    ap.add_argument("--model", default=MODEL_PATH, help="Path to the .joblib model bundle")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Model file not found: {args.model}\nRun the notebook first (it saves the model to models/).")
    bundle = joblib.load(args.model)
    print(f"Loaded {bundle['model_name']} (v{bundle['version']}, created {bundle['created']})")

    if args.json:
        with open(args.json) as f:
            record = json.load(f)
        score(bundle, f"Customer from {args.json}", record)
    else:
        for label, record in EXAMPLES.items():
            score(bundle, label, record)


if __name__ == "__main__":
    main()
