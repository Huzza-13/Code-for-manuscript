import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

input_dir = r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"
output_dir = os.environ.get("PFAS_FIG_OUTPUT_DIR", input_dir)
os.makedirs(output_dir, exist_ok=True)

shap_file = os.path.join(input_dir, "Final_XGBoost_raw_SHAP_values.csv")
X_file = os.path.join(input_dir, "Final_XGBoost_transformed_X.csv")
importance_file = os.path.join(
    input_dir, "Overall_aggregated_SHAP_importance.csv"
)


# ============================================================
# 2. Read model outputs
# ============================================================

for file_path in [shap_file, X_file, importance_file]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required input file not found:\n{file_path}")

raw_shap = pd.read_csv(shap_file)
X_transformed = pd.read_csv(X_file)
importance_df = pd.read_csv(importance_file)

if list(raw_shap.columns) != list(X_transformed.columns):
    raise ValueError(
        "The columns of Final_XGBoost_raw_SHAP_values.csv and "
        "Final_XGBoost_transformed_X.csv do not match."
    )

print("SHAP shape:", raw_shap.shape)
print("Transformed X shape:", X_transformed.shape)


# ============================================================
# 3. Final continuous molecular descriptors only
#
# Carbon Chain Length and PFAS class are deliberately excluded:
# they are used only for post-hoc chemical interpretation, not
# as model inputs or as SHAP features in this figure.
# ============================================================

molecular_features = [
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi",
]

display_labels = {
    "Chi3v": "Chi3v",
    "MinPartialCharge": "Min partial charge",
    "TPSA": "TPSA",
    "ALogP": "ALogP",
    "GATS3c": "GATS3c",
    "SpMin8_Bhi": "SpMin8_Bhi",
}

required_importance_columns = {"Feature", "Mean_abs_SHAP"}
if not required_importance_columns.issubset(importance_df.columns):
    raise ValueError(
        "Overall_aggregated_SHAP_importance.csv must contain Feature and "
        "Mean_abs_SHAP columns."
    )

missing_model_features = [
    feature for feature in molecular_features
    if feature not in raw_shap.columns or feature not in X_transformed.columns
]
if missing_model_features:
    raise ValueError(
        "Missing molecular model features: "
        f"{missing_model_features}"
    )

molecular_importance = (
    importance_df[importance_df["Feature"].isin(molecular_features)]
    .sort_values("Mean_abs_SHAP", ascending=False)
    .copy()
)

ordered_features = molecular_importance["Feature"].tolist()

if set(ordered_features) != set(molecular_features):
    missing = set(molecular_features) - set(ordered_features)
    raise ValueError(
        "The overall SHAP importance table does not contain all six "
        f"molecular descriptors. Missing: {sorted(missing)}"
    )

print("\nMolecular feature order (highest mean |SHAP| first):")
for feature in ordered_features:
    print(feature)


# ============================================================
# 4. Export source data for this panel
# ============================================================

long_tables = []
for feature in ordered_features:
    long_tables.append(
        pd.DataFrame({
            "Feature": feature,
            "Feature_value": X_transformed[feature].to_numpy(),
            "SHAP_value": raw_shap[feature].to_numpy(),
        })
    )

long_df = pd.concat(long_tables, ignore_index=True)
long_df.to_csv(
    os.path.join(output_dir, "Fig3d_Molecular_SHAP_plot_data.csv"),
    index=False,
)


# ============================================================
# 5. Publication style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "font.weight": "bold",
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

cmap = mpl.colormaps["coolwarm"]
rng = np.random.default_rng(2026)


# ============================================================
# 6. Draw molecular SHAP beeswarm
# ============================================================

fig, ax = plt.subplots(figsize=(8.4, 5.9))
n_features = len(ordered_features)

for row_index, feature in enumerate(ordered_features):
    # First feature is most important and is placed at the top.
    y_base = n_features - 1 - row_index

    shap_values = raw_shap[feature].to_numpy()
    feature_values = X_transformed[feature].to_numpy()

    low, high = np.nanpercentile(feature_values, [2, 98])
    if np.isclose(high, low):
        normalized_values = np.full_like(feature_values, 0.5, dtype=float)
    else:
        normalized_values = np.clip(
            (feature_values - low) / (high - low), 0, 1
        )

    jitter = rng.normal(loc=0, scale=0.10, size=len(shap_values))

    ax.scatter(
        shap_values,
        y_base + jitter,
        s=28,
        c=normalized_values,
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )

ax.axvline(
    x=0,
    color="#6B6B6B",
    linestyle="--",
    linewidth=1.5,
    zorder=1,
)


# ============================================================
# 7. Axes and colour scale
# ============================================================

all_shap_values = long_df["SHAP_value"].to_numpy()
xmin = np.nanmin(all_shap_values)
xmax = np.nanmax(all_shap_values)
x_padding = max((xmax - xmin) * 0.08, 0.03)
ax.set_xlim(xmin - x_padding, xmax + x_padding)

ax.set_yticks(np.arange(n_features))
ax.set_yticklabels(
    [display_labels[feature] for feature in ordered_features[::-1]],
    fontsize=17,
    fontweight="bold",
)
ax.set_xlabel("SHAP value", fontsize=20, fontweight="bold")
ax.set_ylabel("")
ax.set_ylim(-0.55, n_features - 0.45)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=17, width=1.5, length=7, pad=7)
ax.tick_params(axis="x", labelsize=15, width=1.5, length=7, pad=7)

for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")

ax.text(
    -0.18,
    1.02,
    "(d)",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=24,
    fontweight="bold",
)

sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
sm.set_array([])

cax = fig.add_axes([0.79, 0.38, 0.018, 0.25])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Low", "High"])
cbar.set_label(
    "Feature value",
    fontsize=16,
    fontweight="bold",
    labelpad=10,
)
cbar.ax.tick_params(labelsize=14, width=1.3, length=5)

for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight("bold")

fig.subplots_adjust(left=0.30, right=0.74, top=0.95, bottom=0.16)


# ============================================================
# 8. Export
# ============================================================

output_stem = os.path.join(output_dir, "Fig3d_Molecular_SHAP_optimized")

fig.savefig(output_stem + ".svg", bbox_inches="tight", facecolor="white")
fig.savefig(output_stem + ".pdf", bbox_inches="tight", facecolor="white")
fig.savefig(
    output_stem + ".tiff",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pil_kwargs={"compression": "tiff_lzw"},
)
fig.savefig(
    output_stem + ".png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print("\nSaved files:")
for suffix in [".svg", ".pdf", ".tiff", ".png"]:
    print(output_stem + suffix)

plt.show()
plt.close(fig)
