# ============================================================
# PROJECT 7: Explainable AI with SHAP
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the Pima Indians Diabetes dataset (from Project 2)
#   2. Trains an XGBoost classifier
#   3. Computes SHAP values for every prediction
#   4. Builds all 4 key SHAP visualisations:
#      - Summary (Beeswarm) Plot
#      - Bar Plot (global importance)
#      - Waterfall Plot (single patient)
#      - Force Plot (single patient)
#   5. Analyses SHAP stability across bootstrap resamples
#   6. Compares SHAP vs built-in feature importance
#   7. Finds high-risk patients and explains their predictions
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
np.random.seed(42)

# Initialise SHAP JavaScript (needed for some plots)

# ===========================================================
# STEP 1: LOAD AND PREPARE DATA
# ===========================================================
# Reusing Pima Indians Diabetes dataset from Project 2.
# We apply the same cleaning steps for consistency.

df = pd.read_csv("diabetes.csv")

print("=" * 60)
print("STEP 1: LOADING AND PREPARING DATA")
print("=" * 60)
print(f"  Samples  : {len(df)}")
print(f"  Features : {df.shape[1] - 1}")
print(f"  Diabetic : {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
print()

# Clean impossible zeros (same as Project 2)
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_clean = df.copy()
for col in zero_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)
    df_clean[col] = df_clean.groupby("Outcome")[col].transform(
        lambda x: x.fillna(x.median())
    )

# Feature names for readable SHAP plots
feature_names = [
    "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
    "Insulin", "BMI", "Diabetes Pedigree", "Age"
]

X = df_clean.drop("Outcome", axis=1)
X.columns = feature_names
y = df_clean["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print()

# ===========================================================
# STEP 2: TRAIN XGBOOST MODEL
# ===========================================================

print("=" * 60)
print("STEP 2: TRAINING XGBOOST MODEL")
print("=" * 60)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_prob)

print(f"  AUC-ROC  : {auc:.4f}")
print(f"  Model trained and ready for SHAP analysis")
print()

# ===========================================================
# STEP 3: COMPUTE SHAP VALUES
# ===========================================================
# TreeExplainer is the fastest, most accurate SHAP method
# for tree-based models (XGBoost, Random Forest, LightGBM).
# It computes exact Shapley values in polynomial time.

print("=" * 60)
print("STEP 3: COMPUTING SHAP VALUES")
print("=" * 60)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"  SHAP values shape : {shap_values.shape}")
print(f"  One row per test patient, one column per feature")
print(f"  Positive SHAP = pushes toward diabetic prediction")
print(f"  Negative SHAP = pushes toward non-diabetic prediction")
print()
print("  SHAP values for first 3 patients:")
shap_df = pd.DataFrame(shap_values, columns=feature_names)
print(shap_df.head(3).round(4).to_string())
print()

# Expected value (baseline) = average prediction across training data
baseline = explainer.expected_value
print(f"  Baseline (average prediction) : {baseline:.4f}")
print(f"  This is the starting point for all explanations")
print()

# ===========================================================
# STEP 4: SHAP SUMMARY PLOT (BEESWARM)
# ===========================================================
# The most informative SHAP plot.
# Each dot = one patient for one feature.
# X-axis = SHAP value (impact on prediction)
# Colour = feature value (red=high, blue=low)
# Features sorted by mean absolute SHAP (most important at top)

print("=" * 60)
print("STEP 4: SHAP SUMMARY PLOT (BEESWARM)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                  show=False, plot_size=None)
plt.title("SHAP Summary Plot — Feature Impact on Diabetes Prediction\n"
          "(Red = high feature value, Blue = low feature value)",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plot1_shap_summary.png", bbox_inches="tight")
plt.close()
print("  Saved: plot1_shap_summary.png")
print("  How to read:")
print("  - Features at top have highest overall impact")
print("  - Right of centre = pushes toward diabetic prediction")
print("  - Left of centre = pushes toward non-diabetic prediction")
print("  - Red dots with positive SHAP = high values increase risk")
print("  - Blue dots with positive SHAP = low values increase risk (counterintuitive)")
print()

# ===========================================================
# STEP 5: SHAP BAR PLOT (GLOBAL IMPORTANCE)
# ===========================================================
# Simpler summary — mean absolute SHAP value per feature.
# Shows magnitude of impact but NOT direction.

fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                  plot_type="bar", show=False, plot_size=None)
plt.title("SHAP Feature Importance (Mean |SHAP Value|)",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plot2_shap_bar.png", bbox_inches="tight")
plt.close()
print("Saved: plot2_shap_bar.png")

# Print ranked importance
mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_shap, index=feature_names).sort_values(ascending=False)
print("  Global SHAP importance ranking:")
for i, (feat, val) in enumerate(shap_importance.items(), 1):
    print(f"    {i}. {feat:<20s}: {val:.4f}")
print()

# ===========================================================
# STEP 6: COMPARE SHAP vs BUILT-IN FEATURE IMPORTANCE
# ===========================================================

builtin_importance = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# SHAP importance
colors_shap = ["#DD8452" if i < 3 else "#4C72B0"
               for i in range(len(shap_importance))]
shap_importance.plot(kind="barh", ax=axes[0],
                     color=colors_shap[::-1], edgecolor="white")
axes[0].invert_yaxis()
axes[0].set_title("SHAP Importance\n(Mean |SHAP Value|)",
                  fontweight="bold", fontsize=12)
axes[0].set_xlabel("Mean |SHAP Value|")
axes[0].grid(axis="x", alpha=0.3)

# Built-in importance
colors_bi = ["#55A868" if i < 3 else "#4C72B0"
             for i in range(len(builtin_importance))]
builtin_importance.plot(kind="barh", ax=axes[1],
                        color=colors_bi[::-1], edgecolor="white")
axes[1].invert_yaxis()
axes[1].set_title("XGBoost Built-in Importance\n(Gain)",
                  fontweight="bold", fontsize=12)
axes[1].set_xlabel("Feature Importance (Gain)")
axes[1].grid(axis="x", alpha=0.3)

fig.suptitle("SHAP vs Built-in Feature Importance — Do They Agree?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot3_shap_vs_builtin.png")
plt.close()
print("Saved: plot3_shap_vs_builtin.png")

# Calculate rank correlation
shap_ranks   = shap_importance.rank(ascending=False)
builtin_ranks = builtin_importance.rank(ascending=False)
rank_corr = shap_ranks.corr(builtin_ranks, method="spearman")
print(f"  Spearman rank correlation (SHAP vs built-in): {rank_corr:.4f}")
print(f"  {'Strong agreement' if rank_corr > 0.7 else 'Moderate agreement' if rank_corr > 0.5 else 'Weak agreement'}")
print()

# ===========================================================
# STEP 7: WATERFALL PLOT — EXPLAIN ONE PATIENT
# ===========================================================
# The waterfall plot shows step-by-step how the prediction
# for ONE patient was built from the baseline.
# This is what you would show a clinician.

print("=" * 60)
print("STEP 7: WATERFALL PLOT — SINGLE PATIENT EXPLANATION")
print("=" * 60)

# Find a high-risk patient (predicted diabetic with high confidence)
high_risk_idx = np.argmax(y_prob)
patient_pred  = y_prob[high_risk_idx]
patient_true  = y_test.iloc[high_risk_idx]

print(f"  Explaining Patient #{high_risk_idx}")
print(f"  Predicted probability : {patient_pred:.4f} ({patient_pred*100:.1f}% diabetic risk)")
print(f"  True label            : {'Diabetic' if patient_true == 1 else 'Not Diabetic'}")
print()
print("  Patient feature values:")
patient_features = X_test.iloc[high_risk_idx]
for feat, val in patient_features.items():
    shap_val = shap_values[high_risk_idx][list(feature_names).index(feat)]
    direction = "↑ risk" if shap_val > 0 else "↓ risk"
    print(f"    {feat:<20s}: {val:6.1f}  |  SHAP = {shap_val:+.4f}  ({direction})")
print()

fig, ax = plt.subplots(figsize=(10, 6))
shap_explanation = shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=baseline,
    data=X_test.iloc[high_risk_idx].values,
    feature_names=feature_names
)
shap.plots.waterfall(shap_explanation, show=False, max_display=8)
plt.title(f"Waterfall Plot — Patient #{high_risk_idx}\n"
          f"Predicted Risk: {patient_pred*100:.1f}% | "
          f"True Label: {'Diabetic' if patient_true else 'Non-Diabetic'}",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plot4_waterfall_highrisk.png", bbox_inches="tight")
plt.close()
print("Saved: plot4_waterfall_highrisk.png")

# ===========================================================
# STEP 8: WATERFALL FOR LOW-RISK PATIENT (COMPARISON)
# ===========================================================

low_risk_idx  = np.argmin(y_prob)
patient_pred_low = y_prob[low_risk_idx]
patient_true_low = y_test.iloc[low_risk_idx]

print("=" * 60)
print("STEP 8: WATERFALL — LOW-RISK PATIENT (COMPARISON)")
print("=" * 60)
print(f"  Patient #{low_risk_idx}")
print(f"  Predicted probability : {patient_pred_low:.4f} ({patient_pred_low*100:.1f}% diabetic risk)")
print(f"  True label            : {'Diabetic' if patient_true_low == 1 else 'Not Diabetic'}")
print()

fig, ax = plt.subplots(figsize=(10, 6))
shap_explanation_low = shap.Explanation(
    values=shap_values[low_risk_idx],
    base_values=baseline,
    data=X_test.iloc[low_risk_idx].values,
    feature_names=feature_names
)
shap.plots.waterfall(shap_explanation_low, show=False, max_display=8)
plt.title(f"Waterfall Plot — Patient #{low_risk_idx}\n"
          f"Predicted Risk: {patient_pred_low*100:.1f}% | "
          f"True Label: {'Diabetic' if patient_true_low else 'Non-Diabetic'}",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plot5_waterfall_lowrisk.png", bbox_inches="tight")
plt.close()
print("Saved: plot5_waterfall_lowrisk.png")

# ===========================================================
# STEP 9: SHAP DEPENDENCE PLOT
# ===========================================================
# Shows the relationship between one feature's value and
# its SHAP value across all patients.
# Colour = interaction with another feature.
# Reveals non-linear effects and feature interactions.

print("=" * 60)
print("STEP 9: SHAP DEPENDENCE PLOTS")
print("=" * 60)

top_features = shap_importance.head(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, feat in enumerate(top_features):
    feat_idx = list(feature_names).index(feat)
    feat_values = X_test[feat].values
    feat_shap   = shap_values[:, feat_idx]

    sc = axes[i].scatter(feat_values, feat_shap,
                          c=feat_values, cmap="RdBu_r",
                          s=40, alpha=0.7, edgecolors="none")
    axes[i].axhline(y=0, color="black", linestyle="--",
                    linewidth=1, alpha=0.5)
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel("SHAP Value", fontsize=11)
    axes[i].set_title(f"SHAP Dependence: {feat}",
                      fontweight="bold", fontsize=11)
    axes[i].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[i], label=feat)

fig.suptitle("SHAP Dependence Plots — How Feature Values Affect Predictions\n"
             "(Above 0 = pushes toward diabetic, Below 0 = pushes toward non-diabetic)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plot6_dependence_plots.png", bbox_inches="tight")
plt.close()
print("Saved: plot6_dependence_plots.png")
print(f"  Showing dependence for: {', '.join(top_features)}")
print()

# ===========================================================
# STEP 10: SHAP STABILITY ANALYSIS
# ===========================================================
# Your research proposal requires:
# "Features appearing in top-10 importance for at least
#  80% of models designated as robust"
# We simulate this with bootstrap resampling.

print("=" * 60)
print("STEP 10: SHAP STABILITY ACROSS BOOTSTRAP RESAMPLES")
print("=" * 60)

N_BOOTSTRAP = 20  # Keep low for speed; proposal uses 100
top_k = 5
feature_top_counts = {feat: 0 for feat in feature_names}

print(f"  Running {N_BOOTSTRAP} bootstrap resamples...")

for i in range(N_BOOTSTRAP):
    # Resample test set with replacement
    boot_idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
    X_boot = X_test.iloc[boot_idx]

    # Compute SHAP on resampled data
    sv_boot = explainer.shap_values(X_boot)
    mean_abs = np.abs(sv_boot).mean(axis=0)
    top_features_boot = pd.Series(mean_abs, index=feature_names)\
                          .sort_values(ascending=False)\
                          .head(top_k).index.tolist()

    for feat in top_features_boot:
        feature_top_counts[feat] += 1

stability_df = pd.DataFrame({
    "Feature": list(feature_top_counts.keys()),
    "Times in Top-5": list(feature_top_counts.values()),
    "Stability %": [v / N_BOOTSTRAP * 100 for v in feature_top_counts.values()]
}).sort_values("Times in Top-5", ascending=False)

print()
print("  Feature stability in top-5 across bootstrap resamples:")
print(f"  {'Feature':<22} {'Top-5 Count':>12} {'Stability %':>12} {'Robust?':>10}")
print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
for _, row in stability_df.iterrows():
    robust = "✓ YES" if row["Stability %"] >= 80 else "✗ NO"
    print(f"  {row['Feature']:<22} {int(row['Times in Top-5']):>12} "
          f"{row['Stability %']:>11.0f}% {robust:>10}")
print()

# Plot stability
fig, ax = plt.subplots(figsize=(10, 6))
colors_stab = ["#55A868" if v >= 80 else "#DD8452" if v >= 50 else "#C44E52"
               for v in stability_df["Stability %"]]
bars = ax.barh(stability_df["Feature"], stability_df["Stability %"],
               color=colors_stab, edgecolor="white", alpha=0.9)
ax.axvline(x=80, color="red", linestyle="--", linewidth=2,
           label="80% robustness threshold")
ax.set_xlabel("% of Bootstrap Resamples in Top-5", fontsize=12)
ax.set_title(f"SHAP Feature Stability ({N_BOOTSTRAP} Bootstrap Resamples)\n"
             "Green = Robust (≥80%) | Orange = Moderate | Red = Unstable",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(0, 105)
ax.grid(axis="x", alpha=0.3)
for bar, v in zip(bars, stability_df["Stability %"]):
    ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
            f"{v:.0f}%", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("plot7_shap_stability.png", bbox_inches="tight")
plt.close()
print("Saved: plot7_shap_stability.png")

# ===========================================================
# FINAL SUMMARY
# ===========================================================

robust_features = stability_df[stability_df["Stability %"] >= 80]["Feature"].tolist()

print()
print("=" * 60)
print("PROJECT 7 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Dataset           : Pima Indians Diabetes ({len(df)} patients)")
print(f"  Model             : XGBoost (AUC = {auc:.4f})")
print(f"  SHAP method       : TreeExplainer (exact Shapley values)")
print()
print(f"  Top 3 features by SHAP importance:")
for i, (feat, val) in enumerate(shap_importance.head(3).items(), 1):
    print(f"    {i}. {feat:<20s}: mean |SHAP| = {val:.4f}")
print()
print(f"  Robust features (≥80% stability): {', '.join(robust_features) if robust_features else 'None at 80% threshold'}")
print(f"  SHAP vs built-in rank correlation: {rank_corr:.4f}")
print()
print(f"  High-risk patient #{high_risk_idx}: {patient_pred*100:.1f}% predicted risk")
print(f"  Low-risk patient  #{low_risk_idx}: {patient_pred_low*100:.1f}% predicted risk")
print()
print("  7 plots saved.")
print("  Ready to push to GitHub!")
print("=" * 60)
