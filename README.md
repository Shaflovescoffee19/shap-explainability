# 🔍 Explainable AI with SHAP -> Understanding Model Predictions

A highly accurate model that no one can explain is a model no one will trust or use. Explainability is not a nice-to-have; it is a requirement for deploying ML in any high-stakes domain. This project goes beyond building a good model to asking a harder question: *why* does it make each prediction, and which features are genuinely driving outcomes versus acting as noise?

---

## 📌 Project Snapshot

| | |
|---|---|
| **Dataset** | Pima Indians Diabetes Dataset |
| **Model** | XGBoost Classifier |
| **Explainability Method** | SHAP (SHapley Additive exPlanations) |
| **Key Output** | Per-patient feature contributions + bootstrap stability analysis |
| **Libraries** | `shap` · `xgboost` · `scikit-learn` · `pandas` · `matplotlib` |

---

## 🧠 What Is SHAP?

SHAP assigns each feature a contribution value for each individual prediction, grounded in cooperative game theory (Shapley values). Starting from the average model prediction across all patients, SHAP decomposes any individual prediction into a sum of feature contributions, showing exactly how much each feature pushed the prediction up or down from that baseline.

This is fundamentally different from standard feature importance, which gives one global number per feature with no direction and no per-patient breakdown.

| Property | Feature Importance | SHAP |
|----------|--------------------|------|
| Scope | Global only | Global and local (per patient) |
| Direction | No | Yes — positive or negative |
| Per-patient breakdown | No | Yes |
| Mathematical guarantee | Not always consistent | Guaranteed by Shapley axioms |

---

## 📐 Analysis Performed

### Global Explanation -> Summary Plot
Every patient represented as a dot for every feature. X-axis position = SHAP value (impact on prediction). Colour = feature value (red = high, blue = low). Features sorted by mean absolute SHAP. The single most informative explainability plot, shows direction, magnitude, and patient-level variation simultaneously.

### Global Explanation -> Bar Plot
Mean absolute SHAP value per feature. A quick global importance ranking without directional information.

### SHAP vs Built-in Feature Importance
Side-by-side comparison of SHAP importance rankings against XGBoost's built-in gain-based importance. Spearman rank correlation quantifies agreement between the two methods.

### Local Explanation -> Waterfall Plots
For the highest-risk and lowest-risk patients, step-by-step decomposition of how the prediction was built from the baseline. Each bar is one feature's contribution. This is the format used to explain individual predictions to a clinical decision-maker.

### Dependence Plots
For the top 3 features, a scatter of feature value vs SHAP value across all patients. Reveals non-linear relationships and thresholds: at what glucose level does risk begin to accelerate? Is the relationship monotonic or does it plateau?

### Bootstrap Stability Analysis
SHAP importance rankings are computed across 20 bootstrap resamples of the test set. Features appearing in the top 5 in ≥80% of resamples are designated as robust, their importance is not an artefact of the specific data sample but a genuine signal.

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `plot1_shap_summary.png` | Beeswarm summary plot (every patient, every feature) |
| `plot2_shap_bar.png` | Global importance bar chart |
| `plot3_shap_vs_builtin.png` | SHAP vs XGBoost built-in importance comparison |
| `plot4_waterfall_highrisk.png` | Step-by-step explanation for highest-risk patient |
| `plot5_waterfall_lowrisk.png` | Step-by-step explanation for lowest-risk patient |
| `plot6_dependence_plots.png` | Feature value vs SHAP value for top 3 features |
| `plot7_shap_stability.png` | Bootstrap stability of feature importance rankings |

---

## 🔍 Key Findings

**Glucose** is consistently the most important feature by SHAP, high glucose strongly and reliably pushes predictions toward the diabetic class. This holds across all 20 bootstrap resamples (100% stability).

**BMI** is the second most important. The dependence plot reveals a non-linear relationship, moderate BMI values have near-zero SHAP values, but beyond a threshold the contribution increases sharply. A linear model would miss this threshold effect.

**Diabetes Pedigree Function** shows highly variable SHAP values, it matters a lot for some patients and almost nothing for others. This kind of patient-level variability is only visible through per-patient SHAP analysis, not global importance scores.

Comparing SHAP and built-in importance, the top-3 features are consistent but the ordering of lower-ranked features diverges, a reminder that different importance methods ask subtly different questions and should be used together rather than in isolation.

---

## 📂 Repository Structure

```
shap-explainability/
├── diabetes.csv
├── shap_explainability.py
├── plot1_shap_summary.png
├── plot2_shap_bar.png
├── plot3_shap_vs_builtin.png
├── plot4_waterfall_highrisk.png
├── plot5_waterfall_lowrisk.png
├── plot6_dependence_plots.png
├── plot7_shap_stability.png
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Shaflovescoffee19/shap-explainability.git
cd shap-explainability
pip3 install shap xgboost scikit-learn pandas matplotlib seaborn
python3 shap_explainability.py
```

---

## 📚 Skills Developed

- Shapley values -> the game-theoretic foundation and why they provide theoretically sound attribution
- Global vs local explanations -> population-level patterns vs individual patient breakdown
- Building and interpreting all four key SHAP plot types: summary, bar, waterfall, dependence
- Bootstrap stability analysis -> distinguishing robust signal from sample-dependent artefacts
- Comparing SHAP with built-in feature importance and understanding when they agree or diverge
- Threshold effects in dependence plots -> identifying the feature values where risk accelerates

---

## 🗺️ Learning Roadmap

_**Project 7 of 10**_ -> a structured series building from data exploration through to advanced ML techniques.

| # | Project | Focus |
|---|---------|-------|
| 1 | Heart Disease EDA | Exploratory analysis, visualisation |
| 2 | Diabetes Data Cleaning | Missing data, outliers, feature engineering |
| 3 | Cancer Risk Classification | Supervised learning, model comparison |
| 4 | Survival Analysis | Time-to-event modelling, Cox regression |
| 5 | Customer Segmentation | Clustering, unsupervised learning |
| 6 | Gene Expression Clustering | High-dimensional data, heatmaps |
| 7 | **Explainable AI with SHAP** ← | Model interpretability |
| 8 | Counterfactual Explanations | Actionable predictions |
| 9 | Multi-Modal Data Fusion | Stacking, ensemble methods |
| 10 | Transfer Learning | Neural networks, domain adaptation |
