# Credit Card Default Prediction

Predicting which credit card clients are likely to default on their next payment, using the UCI "Default of Credit Card Clients" dataset.

## Business problem

A card issuer wants to flag likely defaulters ahead of time so it can intervene (credit line adjustments, outreach) before a missed payment. Since a missed defaulter is costlier to the business than a false alarm, **recall on the default class** is treated as the metric that matters most, not raw accuracy.

## Approach

1. **Data cleaning** — corrected undocumented category codes in `SEX`, `EDUCATION`, and `MARRIAGE` against the data dictionary.
2. **Feature engineering** — collapsed 6 months of raw bill amounts, payment amounts, and repayment-status codes (noisy on their own, single-month snapshots) into engineered features: utilization ratio, average bill/payment amounts, average repayment status, max delay, months delayed, bill change, and pay ratio.
3. **Exploratory validation** — confirmed the engineered features actually separate defaulters from non-defaulters before modeling on them.
4. **Modeling** — Logistic Regression (interpretable baseline) vs. Decision Tree (captures non-linear interactions), compared on accuracy, precision, recall, F1, and ROC-AUC.
5. **Threshold analysis** — examined the precision-recall tradeoff, since the default 0.5 classification threshold is a business choice, not a fixed rule.

## Key finding

The Decision Tree edges out Logistic Regression on ROC-AUC and recall for the default class — catching more true defaulters, which is the error type that matters most for the business. The engineered features (`MAX_DELAY`, `PAY_1`, `AVG_BILL_AMT`, `NUM_MONTHS_DELAYED`) rank among the most important, confirming the feature engineering step added real signal beyond the raw monthly snapshots.

## Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Data

Source: [UCI Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (Kaggle). Not included in this repo — download `UCI_Credit_Card.csv` and place it in the project root.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook credit_card_default_prediction.ipynb
```
# CreditCardPredictionML
