# Predicting Customer Churn for a Telecom Provider

**Data Science Portfolio Assignment — Group Project, Option A**
TETR College of Business · Instructor: Mr Ayush Jain
Created: 16 October 2026 · v1.1 (team names added) · Submitted: 16 October 2026

| Member | ID |
|---|---|
| Ismail Shaltuni | bmt_28127 |
| Ilia Macuka | [ID] |
| Thashen Naidoo | [ID] |
| Izmina Pujoor | [ID] |

## What this project does

A telecom company loses about one customer in four every year. We use the IBM Telco Customer Churn dataset (7,043 customers, 21 columns) to:

1. clean and explore the data,
2. engineer features that capture the patterns we found,
3. train and compare three classifiers (Logistic Regression, Random Forest, Gradient Boosting),
4. tune the decision threshold and check probability calibration,
5. turn predictions into a **retention decision rule** based on expected value,
6. audit the model for **fairness** across demographic groups, and
7. export the best model so it can score a new customer.

**Headline result:** all three models reach ROC-AUC ≈ 0.84. Gradient Boosting was selected for its calibration; at a tuned threshold of 0.31 it catches 74 % of churners (precision 54 %, accuracy 76 %). An expected-value targeting rule reaches 86 % of churners and produces the highest net benefit of every policy tested.

## Repository structure

```
telco-churn-prediction/
├── README.md                          <- this file (how to run)
├── DATASET.md                         <- dataset source documentation
├── requirements.txt                   <- Python dependencies
├── data/
│   └── Telco-Customer-Churn.csv       <- raw dataset (unchanged from source)
├── notebooks/
│   └── Telco_Churn_Analysis.ipynb     <- full workflow with outputs saved
├── models/
│   └── churn_model_gradient_boosting.joblib   <- trained pipeline + threshold
├── src/
│   ├── predict_customer.py            <- score a new customer from the command line
│   └── example_customer.json          <- sample input for the script
├── figures/                           <- all 15 charts (PNG) + results_summary.json
├── report/
│   └── Project_Report_Telco_Churn.pdf <- 2,500-word written report
└── demo/
    └── Demo_Video_Script.md           <- script used for the demo video
```

## How to run

### Option 1 — Google Colab (no installation)

1. Open <https://colab.research.google.com> and choose **File → Upload notebook**.
2. Upload `notebooks/Telco_Churn_Analysis.ipynb`.
3. Click **Runtime → Run all**. The notebook downloads the dataset automatically if it is not found locally, so nothing else needs uploading.
4. Runtime is about 1–2 minutes. Figures appear inline; the model is saved to `models/` in the Colab session.

### Option 2 — Local machine

```bash
git clone https://github.com/ismailshaltuni/telco-churn-prediction.git
cd telco-churn-prediction
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/Telco_Churn_Analysis.ipynb  # then Kernel -> Restart & Run All
```

### Score a new customer with the saved model

```bash
python src/predict_customer.py                              # two built-in example customers
python src/predict_customer.py --json src/example_customer.json   # your own customer
```

Example output:

```
High-risk new fibre customer
  Predicted probability of churn : 85.9%
  Decision threshold             : 0.31
  Recommendation                 : FLAG - contact with retention offer
  Expected value of a $30 offer  : $265  (worth it)
```

## Results summary (held-out test set, 1,409 customers)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.740 | 0.506 | 0.775 | 0.612 | 0.841 |
| Random Forest | 0.50 | 0.749 | 0.519 | 0.781 | 0.623 | 0.839 |
| Gradient Boosting | 0.50 | 0.795 | 0.647 | 0.500 | 0.564 | 0.838 |
| Logistic Regression | 0.58 (tuned) | 0.767 | 0.546 | 0.727 | 0.624 | 0.841 |
| Random Forest | 0.55 (tuned) | 0.764 | 0.541 | 0.730 | 0.621 | 0.839 |
| **Gradient Boosting (selected)** | **0.31 (tuned)** | **0.760** | **0.535** | **0.743** | **0.622** | **0.838** |

Thresholds were tuned on out-of-fold predictions from the training set only.

## Reproducibility

- Every random process uses `random_state = 42`.
- The notebook's saved outputs were produced with Python 3.12, pandas 3.0, scikit-learn 1.9. Any pandas ≥ 2.0 / scikit-learn ≥ 1.3 will reproduce the results to within rounding.
- `figures/results_summary.json` holds every number quoted in the report.

## Code attribution

All code was written by the group for this assignment, using the public APIs of the open-source libraries listed in `requirements.txt` (pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib). No code was copied from tutorials or other groups.

## Licence

Dataset: IBM sample data, redistributed on Kaggle under an open licence for educational use (see `DATASET.md`). Code: MIT.
