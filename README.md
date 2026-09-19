# Credit Risk Analysis — Predicting Loan Default

Predicting which loans are likely to **default**, and understanding what drives that risk,
using a portfolio of **37,408 loans**. Built in Python with pandas and scikit-learn.

**Live interactive notebook (Kaggle):** https://www.kaggle.com/code/bekirnal35/credit-risk-analysis
**Author:** Bekir Onal · [LinkedIn](https://www.linkedin.com/in/bekir-onal-636b57250) · [GitHub](https://github.com/bekir35-on)

---

## Business question
For a lender, the biggest risk is a customer not repaying. Can we (1) understand **where** the
risk is highest and (2) **predict** default for new customers?

## Data
- **37,408 loans**, 11 columns (investment total, current balance, installment size, region,
  whether a penalty was charged, etc.).
- Target: `QUALITY_OF_LOAN` (Good / Bad) → converted to a binary `DEFAULT` (Bad = 1).

## Approach
1. **Cleaning** — filled numeric gaps with the median, categorical gaps with the mode; one-hot encoded categories.
2. **EDA** — overall default rate and segment breakdowns.
3. **Visualisation** — default rate by region and by penalty status (Matplotlib).
4. **Modelling** — Logistic Regression and a Decision Tree; evaluated with **ROC-AUC** (not accuracy,
   because the data is imbalanced at ~11% default).
5. **Interpretation** — feature importance and business recommendations.

## Key findings
- **Overall default rate: 11.1%** (4,154 of 37,408 loans).
- Highest-risk segments: customers with a **penalty charged** (16.1% vs 7.0%) and **rural** customers
  (12.8% vs 1.3% for urban).
- **Models:** Logistic Regression ROC-AUC **0.671**, Decision Tree ROC-AUC **0.676** — a real but
  limited signal given only 11 raw features.
- **Feature importance:** a prior penalty charge (41.7%), current balance (19.2%) and being urban (16.3%)
  explain ~77% of the model's decisions.
- **Fairness:** gender showed almost no predictive power and should not be used as a risk factor.

## Business recommendations
1. Flag applicants with a **prior penalty charge** for stricter review — the strongest single signal.
2. Use **region** and **current balance** as supporting factors in credit scoring.
3. Collect **richer data** (credit history, income, past repayment behaviour) to push predictive power beyond ~0.68.

## Tools
Python · pandas · NumPy · scikit-learn (LogisticRegression, DecisionTreeClassifier, train_test_split,
StandardScaler, roc_auc_score) · Matplotlib · Kaggle Notebooks.

## Files
- `credit-risk-analysis.ipynb` — the full notebook (cleaning → EDA → visualisation → models → conclusions).
