# 📉 Customer Churn Prediction + Explainability Dashboard

Predicts which telecom customers are about to cancel their subscription — and explains **why** using SHAP, so the output is something a retention team can actually act on, not just a black-box score.

**[🔗 Live Dashboard](https://customer-churn-prediction-2k4zmbpezkgtwx7czuhxxg.streamlit.app/)**

---

## Table of Contents

- [Business Problem](#business-problem)
- [Results](#results)
- [How It Works](#how-it-works)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Running It Locally](#running-it-locally)
- [What I'd Do Next](#what-id-do-next)

---

## Business Problem

Customer churn is direct, avoidable revenue loss. Catching at-risk customers early lets a business step in — a retention offer, proactive support — before they leave. A missed churner costs far more than a false alarm, so this model is tuned and evaluated on **recall** for the churn class, not raw accuracy (which is a misleading metric here: simply predicting "no churn" for everyone would already score ~85%).

**Data:** [Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) (Kaggle) — 3,333 customers, ~14.5% churn rate, pre-split into an 80% training set and a 20% holdout that stays untouched until final evaluation.

## Results

Evaluated on the holdout test set (667 customers the model never saw during training or tuning):

| Metric | Score |
|---|---|
| **Recall** (churn) | 79% |
| **Precision** (churn) | 88% |
| **ROC-AUC** | 0.92 |

Of 95 real churners in the holdout set, the model correctly flags 75 — at 88% precision, meaning a retention team acting on these alerts spends most of its outreach on customers who were genuinely at risk.

**Top churn drivers** (Random Forest feature importance): total day minutes, frequency of customer service calls, and holding an international plan.

<p align="center">
  <img src="plot_confusion_matrix.png" width="420" alt="Confusion matrix on holdout test set">
  <img src="plot_feature_importance.png" width="420" alt="Top 10 feature importances">
</p>

## How It Works

1. **Clean & de-leak the data** — dropped `State`/`Area code` (no real signal), `Account length` (~0 correlation with churn), and the four `*charge` columns (near-perfectly collinear with the matching `*minutes` columns — keeping both is redundant, borderline leakage).
2. **Baseline** — Logistic Regression through a `ColumnTransformer` (scaling + one-hot encoding).
3. **Model** — Random Forest, tuned via `GridSearchCV` (5-fold CV, scored on recall).
4. **Honest evaluation** — final numbers reported on the untouched `churn-bigml-20` holdout, not the split the model was tuned on.
5. **Explainability** — SHAP `TreeExplainer` on the tuned Random Forest, surfaced per-customer in the dashboard below.

## Dashboard

An interactive Streamlit app: enter a customer's profile (plan type, usage minutes, service calls), get a live churn risk score, and see a SHAP breakdown of exactly which factors pushed that score up or down.


## Project Structure

```
.
├── Customer_Churn_Prediction.ipynb   # EDA, cleaning, baseline, tuned RF, holdout eval
├── dashboard.py                      # Streamlit app: risk score + SHAP explanation
├── model.pkl                         # trained pipeline (preprocessor + tuned RF)
├── requirements.txt
└── README.md
```

## Running It Locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

# retrain / regenerate model.pkl (optional — it's already included)
jupyter notebook Customer_Churn_Prediction.ipynb

# launch the dashboard
streamlit run dashboard.py
```

## What I'd Do Next

- **Cost-sensitive thresholding** — weigh false negatives against the actual cost of a retention offer instead of using the default 0.5 cutoff, to pick the threshold that maximizes expected business value.
- **A/B test retention offers** against the model's risk tiers to measure real revenue impact, not just recall.
- **Richer features** — this dataset has no tenure or contract-type fields, both known strong churn signals in telecom; would add if a fuller dataset became available.

---

*Built with scikit-learn, SHAP, and Streamlit.*