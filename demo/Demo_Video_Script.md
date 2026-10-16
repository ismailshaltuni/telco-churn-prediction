# Demo Video Script — Predicting Customer Churn for a Telecom Provider

Created: 16 October 2026 (v1.1 — names added) · Target length: **6 min 30 s** (limit 5–7 min)
Format: screen recording of the notebook (already run, so all outputs are visible) with voice-over.

**Speaker plan (4 members, ~1.5 min each).** If one person records the whole thing, just read all parts.

| Part | Time | Speaker | On screen |
|---|---|---|---|
| 1. Problem & data | 0:00 – 1:25 | Member 1 (Ismail) | Notebook top + Section 2 |
| 2. Exploration | 1:25 – 2:40 | Member 2 (Ilia) | Section 3 figures |
| 3. Features & models | 2:40 – 4:25 | Member 3 (Thashen) | Sections 4–7 |
| 4. Business, fairness, live demo, close | 4:25 – 6:30 | Member 4 (Izmina) | Sections 8–12 + terminal |

> **Before you record:** open `notebooks/Telco_Churn_Analysis.ipynb` in Colab or Jupyter and make sure every cell already shows its output (Runtime → Run all, wait ~2 min). Zoom the browser to 110–125 % so charts are readable. Keep the mouse pointer near what you are talking about.

---

## Part 1 — Problem and data (0:00 – 1:25) — Member 1

**[0:00] Screen: notebook title cell]**

> Hi, we're Ismail, Ilia, Thashen and Izmina. For our data science project we chose Option A: predicting customer churn for a telecom provider.
>
> Every month a telecom quietly loses customers. Keeping one is far cheaper than acquiring a new one, so the question we set out to answer isn't just "who will leave" — it's "who should the company actually spend money trying to keep."

**[0:25] Scroll to Section 1, the `raw.head()` output]**

> We used IBM's Telco Customer Churn dataset from Kaggle: 7,043 customers and 21 columns — demographics, the services each customer has, contract and billing details, and the target, whether they churned last month. About 26.5 % did.

**[0:45] Scroll to Section 2, the table of 11 blank TotalCharges rows]**

> The data looked clean but had one hidden problem. TotalCharges loaded as text because eleven cells were blank. Every one of them is a customer with tenure zero — they joined this month and haven't been billed yet. So instead of imputing a median or dropping them, we set those to zero, which is what they actually are. We also recoded SeniorCitizen from 0/1 to a proper category, checked every label for typos, and confirmed there were no duplicates. Nothing was removed — all 7,043 customers stay in.

---

## Part 2 — Exploration (1:25 – 2:40) — Member 2 (Ilia)

**[1:25] Screen: Figure 2 (the 2×2 grid of churn by Contract / Internet / Payment / Paperless)]**

> Three account features dominate everything else. Contract type: 43 % of month-to-month customers churn, versus 11 % on one-year and under 3 % on two-year contracts. Internet type: fibre-optic customers churn at 42 %, more than double DSL. And payment method: electronic-check payers churn at 45 %, while customers on automatic payments churn at 15 to 17 %.

**[1:55] Scroll to Figure 4 (churn by tenure band)]**

> Tenure tells the time dimension. Almost half of first-year customers leave, and that falls to under 10 % after four years. The first year is the danger zone.

**[2:10] Scroll to Figure 6 (add-on services)]**

> And here's something the company can act on: customers with Online Security or Tech Support churn at about 15 % — less than half the rate of other internet customers. Streaming add-ons don't help at all. So the protective factors are longer contracts, automatic payment and security-type add-ons — all things the company sells.

---

## Part 3 — Features and models (2:40 – 4:25) — Member 3 (Thashen)

**[2:40] Screen: Section 4 feature table]**

> From those patterns we engineered six features — each one justified by a chart you just saw: a count of add-on services, a flag for security-or-support, an average monthly spend to replace the redundant TotalCharges column, a charge ratio that catches recent price increases, a new-customer flag, and an auto-pay flag. We deliberately kept gender, senior-citizen, partner and dependents so we could audit the model for bias later.

**[3:05] Scroll to Section 5.2 cross-validation table, then the test-set table]**

> We trained three deliberately different models inside one scikit-learn pipeline: logistic regression as the interpretable baseline, a random forest for non-linear interactions, and gradient boosting. We compared them with five-fold cross-validation and then once on a held-out 20 % test set — 1,409 customers the models never saw.

**[3:25] Scroll to Figure 9 (ROC / PR curves)]**

> All three land at the same ROC-AUC of about 0.84. That's the ceiling for this dataset — the features just don't capture things like competitor offers or outages. At the default 0.5 threshold they look different, but that's an artefact of class weighting.

**[3:45] Scroll to Figure 10 (threshold tuning and calibration)]**

> So we tuned each model's threshold on out-of-fold training predictions — never on the test set — and they converge to the same F1 of 0.62. We picked gradient boosting because its probabilities are the best calibrated — lowest Brier score — which matters for the business maths in a second. At its threshold of 0.31 it catches 74 % of churners with 76 % accuracy.

**[4:05] Scroll to Figure 13 (contract within tenure bands)]**

> One more thing the brief asked: does contract type *cause* churn? Controlling for tenure, month-to-month customers still churn far more in every band — so it's not just that they're newer. But two-year contracts have termination fees, and uncertain customers self-select into month-to-month, so we can't separate lock-in from self-selection with this data. It's a great predictor and plausibly a partial cause — the honest answer is "test it with an A/B experiment."

---

## Part 4 — Business, fairness, live demo and close (4:25 – 6:30) — Member 4 (Izmina)

**[4:25] Screen: Section 8 policies table]**

> A model that flags 500 customers is useless unless you know what to do with them. So we built an expected-value rule with three stated assumptions: a saved customer is worth twelve more months of their bill, an offer costs $30, and 30 % of at-risk customers accept. Contact a customer only if probability times value beats the cost. On the test set that rule reaches 86 % of churners and gives the highest net benefit of every policy we tried — about $67,000, versus $56,000 for contacting everyone and $28,000 for a random campaign of the same size.

**[4:55] Scroll to Figure 14 (cumulative gains)]**

> With a budget of only 500 contacts, ranking by the model captures 70 % of churners; random targeting would get 35 %. We turned that into a four-tier playbook — a personal call plus a contract-migration offer for high-risk, high-value fibre customers, a cheap automated nudge for the middle tier, and nothing for low-value or low-risk customers.

**[5:15] Scroll to Figure 15 (fairness audit)]**

> On fairness: the model flags seniors and single customers more, but those groups genuinely churn more. The real issue is recall — it catches 79 % of at-risk customers *without* dependents but only 55 % *with* them. One global threshold under-serves lower-churn groups. Our guard-rails: use the score only to give benefits, never to restrict service, and monitor recall by group every month.

**[5:40] Screen: switch to Section 11 of the notebook — or a terminal — and actually run it]**

> Finally, the model working on a test case. The whole pipeline is saved as one joblib file with its threshold.

*Run `python src/predict_customer.py` in a terminal (or re-run the last code cell of Section 11). Point at the output.*

> A new fibre customer, three months in, month-to-month, paying by electronic check, no add-ons: 86 % churn probability — flag, and the expected value of a $30 offer is positive. Change that same customer to a two-year contract with auto-pay and tech support, and the probability drops to 2 % — no action.

*(Optional, if time allows: edit one field in `src/example_customer.json` — e.g. change `"Contract": "Month-to-month"` to `"Two year"` — save, and run `python src/predict_customer.py --json src/example_customer.json` to show the probability change live.)*

**[6:10] Screen: Section 12 conclusion]**

> So: churn here is a first-year, month-to-month, fibre, e-check problem, and the fixes are products the company already sells. Is the model ready to deploy? As a 90-day pilot with a control group, yes. As an automated system — not until the interventions are A/B-tested and the recall gap for family customers is closed. Thanks for watching.

**[6:30] End.**

---

## Recording checklist

- [ ] Notebook fully run, outputs visible, browser zoom 110–125 %
- [ ] Terminal open in the repo folder with `python src/predict_customer.py` already typed (press Enter on cue)
- [ ] Record at 1080p; any of: Zoom (New meeting → Share screen → Record), Loom, OBS, QuickTime (Mac: File → New Screen Recording), Windows Game Bar (Win + G) or PowerPoint's Record Slide Show
- [ ] Do one dry run with a timer — target 6:30, hard limit 7:00
- [ ] Export as .mp4; upload to YouTube (Unlisted) or Google Drive (Anyone with the link can view) and paste the link in the submission
