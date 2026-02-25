# 🔍 Explainable AI with SHAP — Diabetes Risk Prediction

A Machine Learning project that trains an XGBoost model and uses SHAP (SHapley Additive exPlanations) to explain every prediction — globally and for individual patients. This is **Project 7 of 10** in my ML learning roadmap toward computational biology research.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Dataset | Pima Indians Diabetes Dataset |
| Model | XGBoost Classifier |
| Technique | SHAP (TreeExplainer) |
| Visualisations | Summary, Bar, Waterfall, Dependence, Stability plots |
| Libraries | `shap`, `xgboost`, `scikit-learn`, `pandas`, `matplotlib` |

---

## 🧠 What Is SHAP?

SHAP assigns each feature a **contribution value** for each individual prediction, based on Shapley values from cooperative game theory. Starting from the average model prediction (baseline), SHAP shows exactly how much each feature pushed a specific patient's prediction up or down.

**Why SHAP over built-in feature importance?**
- Built-in importance: one global number per feature, no direction, no per-patient insight
- SHAP: per-patient values, shows direction (+/-), captures interactions, mathematically guaranteed consistency

---

## 📊 Visualisations Generated

| Plot | What It Shows |
|---|---|
| Summary (Beeswarm) | Every patient × feature SHAP value, coloured by feature value |
| Bar Plot | Mean absolute SHAP value per feature (global ranking) |
| SHAP vs Built-in | Side-by-side comparison of two importance methods |
| Waterfall (High-risk) | Step-by-step explanation for highest-risk patient |
| Waterfall (Low-risk) | Step-by-step explanation for lowest-risk patient |
| Dependence Plots | How top feature values relate to their SHAP values |
| Stability Analysis | Feature robustness across 20 bootstrap resamples |

---

## 🔍 Key Findings

- **Glucose** is consistently the strongest predictor — high glucose strongly pushes toward diabetic prediction
- **BMI** and **Age** are the next most important features
- SHAP reveals that **Diabetes Pedigree Function** has a non-linear effect — moderate values have little impact but extreme values significantly increase risk
- High-risk patients show multiple features simultaneously pushing toward diabetic prediction
- Robust features (≥80% bootstrap stability) are the ones to trust in clinical decision making

---

## 📂 Project Structure

```
shap-explainability/
├── diabetes.csv                    # Dataset (from Project 2)
├── shap_explainability.py          # Main script
├── plot1_shap_summary.png          # Beeswarm summary plot
├── plot2_shap_bar.png              # Global importance bar plot
├── plot3_shap_vs_builtin.png       # SHAP vs built-in comparison
├── plot4_waterfall_highrisk.png    # High-risk patient explanation
├── plot5_waterfall_lowrisk.png     # Low-risk patient explanation
├── plot6_dependence_plots.png      # Feature dependence plots
├── plot7_shap_stability.png        # Bootstrap stability analysis
└── README.md
```

---

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Shaflovescoffee19/shap-explainability.git
cd shap-explainability
```

**2. Install dependencies**
```bash
pip3 install shap xgboost scikit-learn pandas matplotlib seaborn
```

**3. Add the dataset**
Copy `diabetes.csv` from Project 2 into this folder.

**4. Run the script**
```bash
python3 shap_explainability.py
```

---

## 🔬 Connection to Research Proposal

This project directly implements the interpretability framework of **Aim 3** of a computational biology research proposal on CRC risk prediction in the Emirati population:

> *"Individual predictions will be decomposed into feature contributions using SHAP values, revealing how specific variants, microbial taxa, and clinical factors influence risk relative to the population baseline."*

> *"Explanation stability will be assessed across 100 bootstrap resamples, with features appearing in the top-10 importance for at least 80% of models designated as robust."*

> *"Interpretable predictions showing SHAP concordance > 0.7 across methods"*

The same SHAP pipeline applied here to diabetes features will be applied to Emirati CRC patients — replacing glucose/BMI/age with genomic variants, microbial taxa abundances, and polygenic risk scores.

---

## 📚 What I Learned

- What **Shapley values** are and why they provide theoretically grounded feature attribution
- Why **SHAP is superior** to built-in feature importance — direction, per-patient, interactions
- How to build and interpret all 4 key SHAP plots: Summary, Bar, Waterfall, Dependence
- How to identify **robust features** using bootstrap stability analysis
- How to explain an individual patient's prediction in clinical terms
- Why **interpretability is required** for regulatory approval of clinical AI (FDA, EU AI Act)
- The difference between **global explanations** (across all patients) and **local explanations** (one patient)

---

## 🗺️ Part of My ML Learning Roadmap

| # | Project | Status |
|---|---|---|
| 1 | Heart Disease EDA | ✅ Complete |
| 2 | Diabetes Data Cleaning | ✅ Complete |
| 3 | Cancer Risk Classification | ✅ Complete |
| 4 | Survival Analysis | ✅ Complete |
| 5 | Customer Segmentation | ✅ Complete |
| 6 | Gene Expression Clustering | ✅ Complete |
| 7 | Explainable AI with SHAP | ✅ Complete |
| 8 | Counterfactual Explanations | 🔜 Next |
| 9 | Multi-Modal Data Fusion | ⏳ Upcoming |
| 10 | Transfer Learning | ⏳ Upcoming |

---

## 🙋 Author

**Shaflovescoffee19** — building ML skills from scratch toward computational biology research.
