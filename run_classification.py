import os
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(WORKSPACE, "data", "afad_earthquake_data_preprocessed.csv")
FIG_DIR = os.path.join(WORKSPACE, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

print("1. Loading preprocessed dataset...")
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.duplicated()]

print(f"Loaded dataset shape: {df.shape}")

# Parse datetime to extract rich temporal attributes
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["month"] = df["datetime"].dt.month.fillna(6).astype(int)
df["hour"] = df["datetime"].dt.hour.fillna(12).astype(int)
df["dayofweek"] = df["datetime"].dt.dayofweek.fillna(3).astype(int)

# Cyclic encoding for temporal periodicity
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

# Map discretized magnitude_class into 3 well-supported seismological severity tiers:
# 0: Micro (<3.0)
# 1: Minor (3.0-3.9)
# 2: Significant / Light+ (>=4.0) [combines Light (4.0-4.9), Moderate (5.0-5.9), Strong+ (>=6.0)]
class_mapping = {
    "Micro (<3.0)": 0,
    "Minor (3.0-3.9)": 1,
    "Light (4.0-4.9)": 2,
    "Moderate (5.0-5.9)": 2,
    "Strong+ (>=6.0)": 2,
}
class_names = ["Micro (<3.0)", "Minor (3.0-3.9)", "Light+ (>=4.0)"]
df["target_class"] = df["magnitude_class"].map(class_mapping)
df = df.dropna(subset=["target_class"]).copy()
df["target_class"] = df["target_class"].astype(int)

print("Class distribution:")
for c_idx, c_name in enumerate(class_names):
    cnt = int((df["target_class"] == c_idx).sum())
    pct = 100.0 * cnt / len(df)
    print(f"  Class {c_idx} [{c_name}]: {cnt:,} ({pct:.2f}%)")

# Define base features strictly avoiding any magnitude leakage
base_features = [
    "latitude", "longitude", "depth", "rms", "year",
    "month_sin", "month_cos", "hour_sin", "hour_cos",
    "type_ML", "type_Md", "type_Ml", "type_Mw", "type_Other",
    "province"
]

X_raw = df[base_features].copy()
y = df["target_class"].values

# 2. Split dataset into 60% Training, 20% Validation, 20% Testing (Stratified)
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.40, stratify=y, random_state=42
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Split sizes -> Train: {len(X_train_raw):,} ({len(X_train_raw)/len(df)*100:.1f}%), "
      f"Val: {len(X_val_raw):,} ({len(X_val_raw)/len(df)*100:.1f}%), "
      f"Test: {len(X_test_raw):,} ({len(X_test_raw)/len(df)*100:.1f}%)")

# 3. Leakage-free Feature Engineering & Scaling (fit ONLY on Training set)
prov_freq = X_train_raw["province"].value_counts(normalize=True).to_dict()
train_df_temp = X_train_raw.copy()
train_df_temp["target"] = y_train
prov_target_rate = train_df_temp.groupby("province")["target"].mean().to_dict()
global_target_mean = float(np.mean(y_train))

def transform_province_and_interactions(df_part):
    out = df_part.drop(columns=["province"]).copy()
    out["province_freq"] = df_part["province"].map(prov_freq).fillna(0.0)
    out["province_seismic_index"] = df_part["province"].map(prov_target_rate).fillna(global_target_mean)
    out["log_depth"] = np.log1p(np.maximum(0.0, out["depth"]))
    out["log_rms"] = np.log1p(np.maximum(0.0, out["rms"]))
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    return out

X_train_fe = transform_province_and_interactions(X_train_raw)
X_val_fe = transform_province_and_interactions(X_val_raw)
X_test_fe = transform_province_and_interactions(X_test_raw)

feature_names = list(X_train_fe.columns)
continuous_cols = [
    "latitude", "longitude", "depth", "rms", "year",
    "month_sin", "month_cos", "hour_sin", "hour_cos",
    "province_freq", "province_seismic_index",
    "log_depth", "log_rms", "lat_lon_interaction"
]

scaler = StandardScaler()
X_train_scaled = X_train_fe.copy()
X_val_scaled = X_val_fe.copy()
X_test_scaled = X_test_fe.copy()

X_train_scaled[continuous_cols] = scaler.fit_transform(X_train_fe[continuous_cols])
X_val_scaled[continuous_cols] = scaler.transform(X_val_fe[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test_fe[continuous_cols])

# 4. Hyperparameter Tuning on 20% Validation Set for Decision Tree
print("Tuning Decision Tree max_depth on Validation set...")
depths_to_try = [4, 6, 8, 10, 12, 14, 16, 20]
dt_tuning_records = []
best_dt_depth = 10
best_dt_val_f1 = -1.0

for d in depths_to_try:
    dt_cand = DecisionTreeClassifier(
        max_depth=d, min_samples_leaf=25, criterion="gini", random_state=42
    )
    dt_cand.fit(X_train_fe, y_train)
    val_preds = dt_cand.predict(X_val_fe)
    train_preds = dt_cand.predict(X_train_fe)
    val_acc = accuracy_score(y_val, val_preds)
    val_wf1 = f1_score(y_val, val_preds, average="weighted")
    val_mf1 = f1_score(y_val, val_preds, average="macro")
    train_wf1 = f1_score(y_train, train_preds, average="weighted")
    dt_tuning_records.append({
        "max_depth": d,
        "train_weighted_f1": float(train_wf1),
        "val_accuracy": float(val_acc),
        "val_weighted_f1": float(val_wf1),
        "val_macro_f1": float(val_mf1),
    })
    if val_wf1 > best_dt_val_f1:
        best_dt_val_f1 = val_wf1
        best_dt_depth = d

print(f"Optimal Decision Tree max_depth on Validation set: {best_dt_depth} (Val Weighted F1 = {best_dt_val_f1:.4f})")

# 5. Train Final Classification Models
models = {
    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=best_dt_depth, min_samples_leaf=25, criterion="gini", random_state=42
        ),
        False,
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=150, max_depth=16, min_samples_leaf=10, n_jobs=-1, random_state=42
        ),
        False,
    ),
    "Logistic Regression": (
        LogisticRegression(
            max_iter=500, C=1.0, random_state=42
        ),
        True,
    ),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(
            max_iter=200, max_depth=10, learning_rate=0.08, min_samples_leaf=25, random_state=42
        ),
        False,
    ),
}

results_summary = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(X_train_fe)),
    "val_rows": int(len(X_val_fe)),
    "test_rows": int(len(X_test_fe)),
    "features": feature_names,
    "class_names": class_names,
    "dt_tuning": dt_tuning_records,
    "best_dt_depth": int(best_dt_depth),
    "validation_metrics": {},
    "test_metrics": {},
    "per_class_test_metrics": {},
    "confusion_matrices": {},
}

trained_models = {}
test_predictions = {}

for name, (clf, use_scaled) in models.items():
    print(f"Training {name}...")
    X_tr = X_train_scaled if use_scaled else X_train_fe
    X_va = X_val_scaled if use_scaled else X_val_fe
    X_te = X_test_scaled if use_scaled else X_test_fe

    clf.fit(X_tr, y_train)
    trained_models[name] = clf

    val_pred = clf.predict(X_va)
    test_pred = clf.predict(X_te)
    test_predictions[name] = test_pred

    results_summary["validation_metrics"][name] = {
        "Accuracy": float(accuracy_score(y_val, val_pred)),
        "Precision (Weighted)": float(precision_score(y_val, val_pred, average="weighted", zero_division=0)),
        "Recall (Weighted)": float(recall_score(y_val, val_pred, average="weighted", zero_division=0)),
        "F1-Score (Weighted)": float(f1_score(y_val, val_pred, average="weighted", zero_division=0)),
        "Precision (Macro)": float(precision_score(y_val, val_pred, average="macro", zero_division=0)),
        "Recall (Macro)": float(recall_score(y_val, val_pred, average="macro", zero_division=0)),
        "F1-Score (Macro)": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
    }

    results_summary["test_metrics"][name] = {
        "Accuracy": float(accuracy_score(y_test, test_pred)),
        "Precision (Weighted)": float(precision_score(y_test, test_pred, average="weighted", zero_division=0)),
        "Recall (Weighted)": float(recall_score(y_test, test_pred, average="weighted", zero_division=0)),
        "F1-Score (Weighted)": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "Precision (Macro)": float(precision_score(y_test, test_pred, average="macro", zero_division=0)),
        "Recall (Macro)": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
        "F1-Score (Macro)": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
    }

    rep = classification_report(y_test, test_pred, target_names=class_names, output_dict=True, zero_division=0)
    results_summary["per_class_test_metrics"][name] = {
        c: {
            "precision": float(rep[c]["precision"]),
            "recall": float(rep[c]["recall"]),
            "f1-score": float(rep[c]["f1-score"]),
            "support": int(rep[c]["support"]),
        }
        for c in class_names
    }
    results_summary["confusion_matrices"][name] = confusion_matrix(y_test, test_pred).tolist()

dt_model = trained_models["Decision Tree"]
results_summary["tree_text_rules"] = export_text(dt_model, feature_names=feature_names, max_depth=3)

# ==============================================================================
# FIGURE 14: Learned Decision Tree Structure & Validation Hyperparameter Curve
# ==============================================================================
print("Generating Figure 14 (Decision Tree visualization)...")
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.18)

ax_tree = fig.add_subplot(gs[0])
plot_tree(
    dt_model,
    max_depth=3,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=8,
    precision=2,
    ax=ax_tree,
)
ax_tree.set_title(
    f"Learned Decision Tree Classifier (Top 3 Splits Shown; Optimal Tuned Depth = {best_dt_depth})",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

ax_tune = fig.add_subplot(gs[1])
tune_df = pd.DataFrame(dt_tuning_records)
ax_tune.plot(tune_df["max_depth"], tune_df["train_weighted_f1"], marker="o", linestyle="--", color="gray", label="Train Weighted F1")
ax_tune.plot(tune_df["max_depth"], tune_df["val_weighted_f1"], marker="s", linewidth=2.2, color="crimson", label="Validation Weighted F1")
ax_tune.plot(tune_df["max_depth"], tune_df["val_accuracy"], marker="^", linewidth=2.0, color="navy", label="Validation Accuracy")
ax_tune.axvline(best_dt_depth, color="darkgreen", linestyle=":", linewidth=1.8, label=f"Selected Depth ({best_dt_depth})")
ax_tune.set_title("Decision Tree Hyperparameter Tuning (Validation Set)", fontsize=12, fontweight="bold")
ax_tune.set_xlabel("Maximum Tree Depth (max_depth)", fontsize=10)
ax_tune.set_ylabel("Score", fontsize=10)
ax_tune.legend(loc="lower right", fontsize=9)
ax_tune.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
fig14_path = os.path.join(FIG_DIR, "fig14_decision_tree_visualization.png")
plt.savefig(fig14_path, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# FIGURE 15: Feature Importance Comparison & Spatial Classification Map
# ==============================================================================
print("Generating Figure 15 (Feature importances & spatial decision map)...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

dt_imp = pd.Series(trained_models["Decision Tree"].feature_importances_, index=feature_names)
rf_imp = pd.Series(trained_models["Random Forest"].feature_importances_, index=feature_names)
top_feats = rf_imp.sort_values(ascending=True).tail(10).index

y_pos = np.arange(len(top_feats))
height = 0.38
axes[0].barh(y_pos - height/2, dt_imp[top_feats], height=height, label="Decision Tree", color="#3498db", alpha=0.9)
axes[0].barh(y_pos + height/2, rf_imp[top_feats], height=height, label="Random Forest", color="#e67e22", alpha=0.9)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(top_feats, fontsize=9.5)
axes[0].set_xlabel("Relative Gini Feature Importance", fontsize=10)
axes[0].set_title("Top 10 Predictive Features (Decision Tree vs. Random Forest)", fontsize=12, fontweight="bold")
axes[0].legend(loc="lower right")
axes[0].grid(True, axis="x", linestyle="--", alpha=0.5)

gb_preds = test_predictions["Gradient Boosting"]
palette_map = {0: "#2b83ba", 1: "#fdae61", 2: "#d7191c"}
for c_idx, c_name in enumerate(class_names):
    mask = gb_preds == c_idx
    pt_size = 3 if c_idx == 0 else (6 if c_idx == 1 else 18)
    alpha_val = 0.25 if c_idx == 0 else (0.45 if c_idx == 1 else 0.85)
    axes[1].scatter(
        X_test_fe.loc[mask, "longitude"],
        X_test_fe.loc[mask, "latitude"],
        s=pt_size,
        alpha=alpha_val,
        c=palette_map[c_idx],
        label=f"Predicted: {c_name} (n={mask.sum():,})",
    )
axes[1].set_title("Spatial Distribution of Learned Predictions on Testing Set (20%)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Longitude (°E)", fontsize=10)
axes[1].set_ylabel("Latitude (°N)", fontsize=10)
axes[1].legend(loc="upper right", markerscale=2.0, fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig15_path = os.path.join(FIG_DIR, "fig15_feature_importance_and_spatial_map.png")
plt.savefig(fig15_path, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# FIGURE 16: Confusion Matrices Across All 4 Learned Classification Algorithms
# ==============================================================================
print("Generating Figure 16 (Confusion matrices comparison)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (name, _) in enumerate(models.items()):
    cm = np.array(results_summary["confusion_matrices"][name])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
    annot_matrix = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=axes[idx],
        vmin=0,
        vmax=100,
        annot_kws={"size": 9.5, "weight": "bold"},
    )
    acc = results_summary["test_metrics"][name]["Accuracy"]
    wf1 = results_summary["test_metrics"][name]["F1-Score (Weighted)"]
    axes[idx].set_title(f"{name}\n(Test Accuracy: {acc:.4f} | Weighted F1: {wf1:.4f})", fontsize=11.5, fontweight="bold")
    axes[idx].set_xlabel("Predicted Severity Class", fontsize=10)
    axes[idx].set_ylabel("Actual Severity Class", fontsize=10)

plt.suptitle("Testing Set (20%) Confusion Matrices Across Classification Algorithms", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig16_path = os.path.join(FIG_DIR, "fig16_confusion_matrices_comparison.png")
plt.savefig(fig16_path, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# FIGURE 17: Performance Metrics Comparison (Accuracy, Precision, Recall, F1)
# ==============================================================================
print("Generating Figure 17 (Performance metrics comparison)...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metric_cols = ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)"]
short_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
model_names = list(models.keys())
x = np.arange(len(short_labels))
width = 0.20
colors = ["#2980b9", "#27ae60", "#8e44ad", "#c0392b"]

for i, m_name in enumerate(model_names):
    vals = [results_summary["test_metrics"][m_name][mc] for mc in metric_cols]
    bars = axes[0].bar(x + (i - 1.5) * width, vals, width, label=m_name, color=colors[i], alpha=0.92)
    for b in bars:
        axes[0].annotate(
            f"{b.get_height():.3f}",
            (b.get_x() + b.get_width() / 2.0, b.get_height()),
            ha="center", va="bottom", fontsize=7.5, rotation=0, xytext=(0, 2), textcoords="offset points"
        )

axes[0].set_xticks(x)
axes[0].set_xticklabels(short_labels, fontsize=10.5, fontweight="bold")
axes[0].set_ylim(0.50, 0.90)
axes[0].set_ylabel("Score on Testing Set (20%)", fontsize=10.5)
axes[0].set_title("Overall Classification Performance Comparison (Weighted Metrics)", fontsize=12, fontweight="bold")
axes[0].legend(loc="lower right", fontsize=9)
axes[0].grid(True, axis="y", linestyle="--", alpha=0.6)

macro_cols = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
for i, m_name in enumerate(model_names):
    vals = [results_summary["test_metrics"][m_name][mc] for mc in macro_cols]
    bars = axes[1].bar(x + (i - 1.5) * width, vals, width, label=m_name, color=colors[i], alpha=0.92)
    for b in bars:
        axes[1].annotate(
            f"{b.get_height():.3f}",
            (b.get_x() + b.get_width() / 2.0, b.get_height()),
            ha="center", va="bottom", fontsize=7.5, rotation=0, xytext=(0, 2), textcoords="offset points"
        )

axes[1].set_xticks(x)
axes[1].set_xticklabels(["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"], fontsize=10.5, fontweight="bold")
axes[1].set_ylim(0.25, 0.90)
axes[1].set_ylabel("Score on Testing Set (20%)", fontsize=10.5)
axes[1].set_title("Class-Balanced Performance Comparison (Macro-Averaged Metrics)", fontsize=12, fontweight="bold")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
fig17_path = os.path.join(FIG_DIR, "fig17_classification_metrics_comparison.png")
plt.savefig(fig17_path, bbox_inches="tight")
plt.close(fig)

print("All 4 classification figures exported successfully!")
for m_name, metrics in results_summary["test_metrics"].items():
    print(f"TEST [{m_name}]: Acc={metrics['Accuracy']:.4f}, "
          f"Prec(W)={metrics['Precision (Weighted)']:.4f}, "
          f"Rec(W)={metrics['Recall (Weighted)']:.4f}, "
          f"F1(W)={metrics['F1-Score (Weighted)']:.4f} | "
          f"F1(Macro)={metrics['F1-Score (Macro)']:.4f}")
