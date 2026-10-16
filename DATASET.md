# Dataset Source Documentation

Created: 16 October 2026 (v1.0)
Project: Predicting Customer Churn for a Telecom Provider (Option A)

## 1. Source

| Item | Detail |
|---|---|
| **Dataset name** | Telco Customer Churn |
| **Original publisher** | IBM (sample dataset shipped with IBM Cognos Analytics / Watson Studio) |
| **Distribution used** | Kaggle, uploaded by user *BlastChar* — <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> |
| **Identical public mirror** | IBM's GitHub repository `IBM/telco-customer-churn-on-icp4d`, file `data/Telco-Customer-Churn.csv` (the notebook downloads from here when the local copy is absent) |
| **File name as downloaded** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Kaggle) / `Telco-Customer-Churn.csv` (IBM mirror) — same content |
| **Date accessed** | 16 October 2026 |
| **Licence / terms** | Published by IBM as sample data; redistributed on Kaggle for public, educational use. Used here for educational purposes only. |
| **Size** | 7,043 rows × 21 columns; 955 KB |
| **Citation (APA 7)** | BlastChar. (2018). *Telco customer churn* [Data set]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn (Original work published by IBM) |

## 2. What the data contains

Each row is one customer of a fictional Californian telecom provider, with a snapshot of their account and whether they left ("churned") in the most recent month.

| Group | Columns |
|---|---|
| Identifier | `customerID` |
| Demographics | `gender`, `SeniorCitizen` (0/1), `Partner`, `Dependents` |
| Services | `PhoneService`, `MultipleLines`, `InternetService` (DSL / Fiber optic / No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Account & billing | `tenure` (months, 0–72), `Contract` (Month-to-month / One year / Two year), `PaperlessBilling`, `PaymentMethod` (4 types), `MonthlyCharges` ($), `TotalCharges` ($) |
| Target | `Churn` (Yes / No) — 26.5 % Yes |

## 3. Personal and sensitive data

The dataset is synthetic sample data: `customerID` is an opaque code, there are no names, addresses, phone numbers or dates of birth, and no attempt was made to re-identify anyone. It does contain demographic attributes (`gender`, `SeniorCitizen`, `Partner`, `Dependents`); these were used only to audit the model for unequal treatment (notebook Section 9), not to target anyone.

## 4. Pre-processing applied before analysis

The raw CSV in `data/` is **unchanged** from the source. All processing happens inside the notebook and is fully reproducible:

| # | Issue found | Fix applied | Rows affected |
|---|---|---|---|
| 1 | `TotalCharges` stored as text because 11 cells are blank strings | Converted to numeric; blanks set to 0 because all 11 customers have `tenure = 0` (joined this month, not billed yet) | 11 |
| 2 | `SeniorCitizen` coded 0/1 although it is a category | Mapped to "No"/"Yes" so it is treated as categorical | 7,043 |
| 3 | Possible stray whitespace in text columns | All text columns stripped | 0 changed |
| 4 | Target is text | Added `ChurnFlag` = 1 if `Churn == "Yes"` else 0 | 7,043 |
| 5 | Duplicates / impossible values | Checked — none found (0 duplicate IDs; all numeric ranges plausible) | 0 |

No rows were removed. After cleaning, the dataset still has 7,043 customers.

## 5. Features engineered for modelling

| Feature | Definition |
|---|---|
| `NumAddOnServices` | Count of the six add-on services set to "Yes" |
| `HasSecurityOrSupport` | 1 if `OnlineSecurity` or `TechSupport` is "Yes" |
| `AvgMonthlySpend` | `TotalCharges / tenure` (equals `MonthlyCharges` when tenure = 0) |
| `ChargeRatio` | `MonthlyCharges / AvgMonthlySpend` |
| `IsNewCustomer` | 1 if `tenure` ≤ 12 |
| `AutoPay` | 1 if payment method is an automatic one (bank transfer or credit card) |

`customerID` and `TotalCharges` are not used as model inputs (identifier / redundant with tenure × charges).

## 6. Train / test split

80 % / 20 % stratified split on `ChurnFlag`, `random_state = 42` → 5,634 training customers, 1,409 test customers, both with 26.5 % churn. The test set was used only for final evaluation.
