import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# 1. Paths
# ============================================================

output_dir = r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"

original_file = (
    r"D:\python\pythonProject1"
    r"\Machine-learning original dataset.xlsx"
)

shap_file = os.path.join(
    output_dir,
    "Final_XGBoost_raw_SHAP_values.csv"
)

X_file = os.path.join(
    output_dir,
    "Final_XGBoost_transformed_X.csv"
)

importance_file = os.path.join(
    output_dir,
    "Overall_aggregated_SHAP_importance.csv"
)


# ============================================================
# 2. Read data
# ============================================================

raw_shap = pd.read_csv(shap_file)
X_transformed = pd.read_csv(X_file)
original_data = pd.read_excel(original_file)
importance_df = pd.read_csv(importance_file)

print("SHAP shape:", raw_shap.shape)
print("Transformed X shape:", X_transformed.shape)
print("Original data shape:", original_data.shape)


# ============================================================
# 3. Define features for Fig. 3c
# ============================================================

env_plant_exposure_features = [
    "Tissue group",
    "SOM",
    "Exposure concentration",
    "Soil pH",
    "Growth temperature",
    "Growth duration",
    "Morphotype"
]


# ============================================================
# 4. Determine plotting order from Fig. 3b importance ranking
# ============================================================

subset_importance = (
    importance_df[
        importance_df["Feature"].isin(env_plant_exposure_features)
    ]
    .sort_values("Mean_abs_SHAP", ascending=False)
    .reset_index(drop=True)
)

ordered_display_features = subset_importance["Feature"].tolist()

print("\nFeature order for Fig. 3c:")
for f in ordered_display_features:
    print(f)


# ============================================================
# 5. Mapping display names to model/original names
# ============================================================

display_to_model = {
    "Tissue group": "Plant tissue group classify",
    "SOM": "SOM",
    "Exposure concentration": "Exposure Concentration",
    "Soil pH": "Soil pH",
    "Growth temperature": "Growth Temperature",
    "Growth duration": "Growth Duration",
    "Morphotype": "Morphotypes"
}


# ============================================================
# 6. Prepare aggregated SHAP for categorical predictors
# ============================================================

tissue_dummy_cols = [
    "Plant tissue group classify_0",
    "Plant tissue group classify_1",
    "Plant tissue group classify_2",
    "Plant tissue group classify_3"
]

morph_dummy_cols = [
    "Morphotypes_0",
    "Morphotypes_1"
]

for col in tissue_dummy_cols + morph_dummy_cols:
    if col not in raw_shap.columns:
        raise ValueError(f"Missing SHAP column: {col}")

tissue_group_shap = raw_shap[tissue_dummy_cols].sum(axis=1)
morphotype_shap = raw_shap[morph_dummy_cols].sum(axis=1)


# ============================================================
# 7. Read categorical codes from original dataset
# ============================================================

tissue_group_code = (
    original_data["Plant tissue group classify"]
    .astype(int)
    .reset_index(drop=True)
)

morphotype_code = (
    original_data["Morphotypes"]
    .astype(int)
    .reset_index(drop=True)
)


# ============================================================
# 8. IMPORTANT:
# Check these mappings against your own dataset coding
# If your code meanings differ, only edit the labels below.
# ============================================================

# ---- Tissue group labels: please check and revise if needed ----
tissue_group_labels = {
    0: "Root",
    1: "Leaf",
    2: "Stem",
    3: "Fruit/seed"
}

tissue_group_colors = {
    0: "#1B9E77",
    1: "#66A61E",
    2: "#7570B3",
    3: "#E7298A"
}

# ---- Morphotype labels: please check and revise if needed ----
morphotype_labels = {
    0: "Monocot",
    1: "Dicot"
}

morphotype_colors = {
    0: "#4C78A8",
    1: "#F28E2B"
}


# ============================================================
# 9. Save plotting data table
# ============================================================

long_rows = []

for display_feature in ordered_display_features:

    model_feature = display_to_model[display_feature]

    if model_feature == "Plant tissue group classify":

        for i in range(len(raw_shap)):
            code = tissue_group_code.iloc[i]

            long_rows.append({
                "Feature": display_feature,
                "Feature_value": code,
                "Feature_value_label": tissue_group_labels.get(code, f"Group {code}"),
                "SHAP_value": tissue_group_shap.iloc[i]
            })

    elif model_feature == "Morphotypes":

        for i in range(len(raw_shap)):
            code = morphotype_code.iloc[i]

            long_rows.append({
                "Feature": display_feature,
                "Feature_value": code,
                "Feature_value_label": morphotype_labels.get(code, f"Type {code}"),
                "SHAP_value": morphotype_shap.iloc[i]
            })

    else:
        feature_values = X_transformed[model_feature].values
        shap_values = raw_shap[model_feature].values

        for i in range(len(raw_shap)):
            long_rows.append({
                "Feature": display_feature,
                "Feature_value": feature_values[i],
                "Feature_value_label": "",
                "SHAP_value": shap_values[i]
            })

long_df = pd.DataFrame(long_rows)

long_df.to_csv(
    os.path.join(output_dir, "Fig3c_EnvPlantExposure_SHAP_plot_data.csv"),
    index=False
)


# ============================================================
# 10. Figure style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "font.weight": "bold",

    "axes.labelsize": 18,       # x/y axis title
    "axes.labelweight": "bold",

    "xtick.labelsize": 14,      # x-axis tick labels
    "ytick.labelsize": 15,      # y-axis tick labels

    "axes.linewidth": 1.4,

    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,

    "xtick.major.size": 7,
    "ytick.major.size": 7,

    "xtick.direction": "out",
    "ytick.direction": "out"
})



# ============================================================
# 11. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(8.6, 5.9))


# ============================================================
# 12. SHAP = 0 reference line
# ============================================================

ax.axvline(
    x=0,
    color="0.25",
    linestyle="--",
    linewidth=1.5,
    alpha=0.85,
    zorder=1
)


# ============================================================
# 13. Random jitter
# ============================================================

rng = np.random.default_rng(2026)


# ============================================================
# 14. Blue-red colormap for continuous variables
# ============================================================

cmap = mpl.colormaps["coolwarm"]


# ============================================================
# 15. Plot rows
# ============================================================

n_features = len(ordered_display_features)

for row_index, display_feature in enumerate(ordered_display_features):

    y_base = n_features - 1 - row_index
    model_feature = display_to_model[display_feature]

    # --------------------------------------------------------
    # A. Tissue group (categorical)
    # --------------------------------------------------------
    if model_feature == "Plant tissue group classify":

        category_offsets = {
            0: -0.24,
            1: -0.08,
            2:  0.08,
            3:  0.24
        }

        for code in sorted(tissue_group_labels.keys()):

            mask = (tissue_group_code.values == code)
            x_vals = tissue_group_shap.values[mask]

            if len(x_vals) == 0:
                continue

            jitter = rng.normal(
                loc=0,
                scale=0.018,
                size=len(x_vals)
            )

            y_vals = y_base + category_offsets[code] + jitter

            ax.scatter(
                x_vals,
                y_vals,
                s=34,
                color=tissue_group_colors[code],
                alpha=0.75,
                edgecolors="none",
                rasterized=True,
                zorder=3
            )

    # --------------------------------------------------------
    # B. Morphotype (categorical)
    # --------------------------------------------------------
    elif model_feature == "Morphotypes":

        category_offsets = {
            0: -0.10,
            1:  0.10
        }

        for code in sorted(morphotype_labels.keys()):

            mask = (morphotype_code.values == code)
            x_vals = morphotype_shap.values[mask]

            if len(x_vals) == 0:
                continue

            jitter = rng.normal(
                loc=0,
                scale=0.018,
                size=len(x_vals)
            )

            y_vals = y_base + category_offsets[code] + jitter

            ax.scatter(
                x_vals,
                y_vals,
                s=34,
                color=morphotype_colors[code],
                alpha=0.75,
                edgecolors="none",
                rasterized=True,
                zorder=3
            )

    # --------------------------------------------------------
    # C. Continuous variables
    # --------------------------------------------------------
    else:

        shap_vals = raw_shap[model_feature].values
        feature_vals = X_transformed[model_feature].values

        low = np.nanpercentile(feature_vals, 2)
        high = np.nanpercentile(feature_vals, 98)

        if high == low:
            high = low + 1e-9

        normalized_values = (feature_vals - low) / (high - low)
        normalized_values = np.clip(normalized_values, 0, 1)

        jitter = rng.normal(
            loc=0,
            scale=0.105,
            size=len(shap_vals)
        )

        y_vals = y_base + jitter

        ax.scatter(
            shap_vals,
            y_vals,
            s=28,
            c=normalized_values,
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=0.55,
            edgecolors="none",
            rasterized=True,
            zorder=3
        )


# ============================================================
# ============================================================
# 16. Y-axis labels
# ============================================================

ordered_labels_for_axis = ordered_display_features[::-1]

ax.set_yticks(
    np.arange(n_features)
)

ax.set_yticklabels(
    ordered_labels_for_axis,
    fontsize=17,
    fontweight="bold"
)


# ============================================================
# 17. X-axis
# ============================================================

all_shap_values = (
    long_df["SHAP_value"]
    .values
)

xmin = np.nanmin(
    all_shap_values
)

xmax = np.nanmax(
    all_shap_values
)

padding = (
    xmax - xmin
) * 0.08


ax.set_xlim(
    xmin - padding,
    xmax + padding
)


ax.set_xlabel(
    "SHAP value",
    fontsize=20,
    fontweight="bold"
)

ax.set_ylabel("")


# ============================================================
# 18. No background grid
# ============================================================

ax.grid(False)


# ============================================================
# 19. Frame
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# 20. Tick style
# ============================================================

ax.tick_params(
    axis="x",
    labelsize=15,
    width=1.5,
    length=7,
    pad=7
)

ax.tick_params(
    axis="y",
    labelsize=17,
    width=1.5,
    length=7,
    pad=7
)


for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")


# ============================================================
# 21. Panel label
# ============================================================

ax.text(
    -0.17,
    1.03,
    "(c)",
    transform=ax.transAxes,
    fontsize=24,
    fontweight="bold",
    ha="left",
    va="bottom"
)


# ============================================================
# 22. Colorbar
# Place at UPPER-RIGHT
# ============================================================

sm = ScalarMappable(
    norm=Normalize(
        vmin=0,
        vmax=1
    ),
    cmap=cmap
)

sm.set_array([])


# ------------------------------------------------------------
# [left, bottom, width, height]
# figure coordinates
# ------------------------------------------------------------

cax = fig.add_axes([
    0.79,   # left
    0.38,   # bottom
    0.018,  # width
    0.25    # height
])

cbar = fig.colorbar(
    sm,
    cax=cax
)


cbar.set_ticks(
    [0, 1]
)

cbar.set_ticklabels(
    ["Low", "High"]
)


cbar.set_label(
    "Feature value",
    fontsize=16,
    fontweight="bold",
    labelpad=10
)


cbar.ax.tick_params(
    labelsize=14,
    width=1.3,
    length=5
)


for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight("bold")


# ============================================================
# 23. Categorical legends
# ============================================================


# ------------------------------------------------------------
# A. Tissue group legend
# Place at RIGHT-MIDDLE
# ------------------------------------------------------------

tissue_handles = []

for code, label in tissue_group_labels.items():

    tissue_handles.append(

        Line2D(
            [0], [0],

            marker="o",

            color="none",

            markerfacecolor=
                tissue_group_colors[code],

            markeredgecolor=
                "none",

            markersize=10,

            label=label
        )
    )


legend1 = ax.legend(
    handles=tissue_handles,
    title="Tissue group",
    loc="upper left",
    bbox_to_anchor=(1.18, 0.98),   # 右上
    frameon=False,
    fontsize=14,
    title_fontsize=15,
    labelspacing=0.60,
    handletextpad=0.7,
    borderpad=0.2
)

legend1.get_title().set_fontweight("bold")

for text in legend1.get_texts():
    text.set_fontweight("bold")

ax.add_artist(legend1)




# ------------------------------------------------------------
# B. Morphotype legend
# Place at LOWER-RIGHT
# ------------------------------------------------------------

morph_handles = []

for code, label in morphotype_labels.items():

    morph_handles.append(

        Line2D(
            [0], [0],

            marker="o",

            color="none",

            markerfacecolor=
                morphotype_colors[code],

            markeredgecolor=
                "none",

            markersize=10,

            label=label
        )
    )


legend2 = ax.legend(

    handles=
        morph_handles,

    title=
        "Morphotype",

    loc=
        "lower left",

    bbox_to_anchor=
        (1.18, 0.02),

    frameon=False,

    fontsize=14,

    title_fontsize=15,

    labelspacing=0.60,

    handletextpad=0.7,

    borderpad=0.2
)


legend2.get_title().set_fontweight(
    "bold"
)


for text in legend2.get_texts():

    text.set_fontweight(
        "bold"
    )


# ============================================================
# 24. Layout
#
# Reserve enough space on the right for:
# colorbar + tissue legend + morphotype legend
# ============================================================

plt.subplots_adjust(
    left=0.30,
    right=0.72,
    top=0.95,
    bottom=0.16
)

# ============================================================
# 25. Save
# ============================================================

tiff_file = os.path.join(output_dir, "Fig3c_EnvPlantExposure_SHAP.tiff")
png_file = os.path.join(output_dir, "Fig3c_EnvPlantExposure_SHAP.png")
pdf_file = os.path.join(output_dir, "Fig3c_EnvPlantExposure_SHAP.pdf")

plt.savefig(tiff_file, dpi=600, bbox_inches="tight")
plt.savefig(png_file, dpi=300, bbox_inches="tight")
plt.savefig(pdf_file, bbox_inches="tight")

plt.show()

print("\nFigure saved to:")
print(tiff_file)
print(png_file)
print(pdf_file)