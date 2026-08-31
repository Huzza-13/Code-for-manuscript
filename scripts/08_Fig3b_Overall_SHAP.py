import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# 1. Paths
# ============================================================

output_dir = r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"

os.makedirs(output_dir, exist_ok=True)

pipeline_file = os.path.join(
    output_dir,
    "Final_XGBoost_pipeline.joblib"
)

X_file = os.path.join(
    output_dir,
    "Final_XGBoost_transformed_X.csv"
)

feature_file = os.path.join(
    output_dir,
    "Final_XGBoost_transformed_feature_names.csv"
)


# ============================================================
# 2. Load final model and transformed data
# ============================================================

pipeline = joblib.load(
    pipeline_file
)

final_model = pipeline.named_steps[
    "model"
]

X_transformed = pd.read_csv(
    X_file
)

feature_names = (
    pd.read_csv(feature_file)
    ["Model_feature"]
    .tolist()
)


# Make absolutely sure column names match
X_transformed.columns = feature_names


print(
    "Transformed data shape:",
    X_transformed.shape
)

print(
    "Number of model features:",
    len(feature_names)
)


# ============================================================
# 3. Calculate SHAP values
#
# TreeExplainer is appropriate for XGBoost tree models.
# ============================================================

print(
    "\nCalculating SHAP values..."
)

explainer = shap.TreeExplainer(
    final_model
)

shap_values = explainer(
    X_transformed
)


print(
    "SHAP matrix shape:",
    shap_values.values.shape
)


# ============================================================
# 4. Save raw 17-feature SHAP values
#
# Keep these because later environmental and molecular
# beeswarm plots will use the same SHAP calculations.
# ============================================================

raw_shap_df = pd.DataFrame(
    shap_values.values,
    columns=feature_names
)

raw_shap_df.to_csv(
    os.path.join(
        output_dir,
        "Final_XGBoost_raw_SHAP_values.csv"
    ),
    index=False
)


# ============================================================
# 5. Define mapping from model-input columns
#    back to ORIGINAL predictors
#
# IMPORTANT:
# Categories are aggregated so that Tissue group,
# Morphotype and Functional group each appear once.
# ============================================================

feature_groups = {

    "Tissue group": [
        "Plant tissue group classify_0",
        "Plant tissue group classify_1",
        "Plant tissue group classify_2",
        "Plant tissue group classify_3"
    ],

    "Morphotype": [
        "Morphotypes_0",
        "Morphotypes_1"
    ],

    "Soil pH": [
        "Soil pH"
    ],

    "SOM": [
        "SOM"
    ],

    "Exposure concentration": [
        "Exposure Concentration"
    ],

    "Growth duration": [
        "Growth Duration"
    ],

    "Growth temperature": [
        "Growth Temperature"
    ],

    "Chi3v": [
        "Chi3v"
    ],

    "MinPartialCharge": [
        "MinPartialCharge"
    ],

    "TPSA": [
        "TPSA"
    ],

    "ALogP": [
        "ALogP"
    ],

    "GATS3c": [
        "GATS3c"
    ],

    "SpMin8_Bhi": [
        "SpMin8_Bhi"
    ]
}


# ============================================================
# 6. Quality check:
#    every expected transformed feature must exist
# ============================================================

for original_feature, model_features in feature_groups.items():

    for f in model_features:

        if f not in raw_shap_df.columns:

            raise ValueError(
                f"Missing transformed feature: {f}"
            )


print(
    "\nFeature aggregation check passed."
)


# ============================================================
# 7. Calculate aggregated mean |SHAP|
#
# For categorical variables:
# sum mean absolute SHAP across dummy variables.
#
# For continuous variables:
# this reduces to mean absolute SHAP of that variable.
# ============================================================

importance_results = []


for original_feature, model_features in feature_groups.items():

    individual_importance = (
        raw_shap_df[
            model_features
        ]
        .abs()
        .mean(axis=0)
    )

    aggregated_importance = (
        individual_importance.sum()
    )

    importance_results.append({

        "Feature":
            original_feature,

        "Mean_abs_SHAP":
            aggregated_importance

    })


importance_df = pd.DataFrame(
    importance_results
)


importance_df = (
    importance_df
    .sort_values(
        "Mean_abs_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 8. Assign feature domains - 4 categories
# ============================================================

plant_features = [
    "Tissue group",
    "Morphotype"
]

environment_features = [
    "Soil pH",
    "SOM"
]

exposure_features = [
    "Exposure concentration",
    "Growth duration",
    "Growth temperature"
]

molecular_features = [
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi"
]


def assign_domain(feature):

    if feature in plant_features:
        return "Plant"

    elif feature in environment_features:
        return "Environment"

    elif feature in exposure_features:
        return "Exposure"

    elif feature in molecular_features:
        return "Molecular"

    else:
        return "Other"
# VERY IMPORTANT:
# Add Domain column to importance_df
# ============================================================

importance_df["Domain"] = (
    importance_df["Feature"]
    .apply(assign_domain)
)


print("\nDomain assignment:")
print(
    importance_df[
        ["Feature", "Domain"]
    ].to_string(index=False)
)
# ============================================================
# 9. Calculate percentage contribution
#
# IMPORTANT:
# This is a relative summary of mean |SHAP| in THIS MODEL.
# It is NOT a causal percentage.
# ============================================================

total_importance = (
    importance_df[
        "Mean_abs_SHAP"
    ]
    .sum()
)


importance_df[
    "Relative_importance_percent"
] = (

    importance_df[
        "Mean_abs_SHAP"
    ]

    / total_importance

    * 100
)


# ============================================================
# 10. Print table
# ============================================================

print("\n")
print("=" * 90)
print("OVERALL AGGREGATED SHAP IMPORTANCE")
print("=" * 90)

print(
    importance_df.to_string(
        index=False,
        formatters={

            "Mean_abs_SHAP":
                "{:.4f}".format,

            "Relative_importance_percent":
                "{:.2f}".format
        }
    )
)

print("=" * 90)


# ============================================================
# 11. Domain-level summary
# ============================================================

domain_summary = (

    importance_df
    .groupby(
        "Domain",
        as_index=False
    )
    .agg(

        Mean_abs_SHAP=(
            "Mean_abs_SHAP",
            "sum"
        ),

        Relative_importance_percent=(
            "Relative_importance_percent",
            "sum"
        )
    )

    .sort_values(
        "Mean_abs_SHAP",
        ascending=False
    )
)


print("\n")
print("=" * 70)
print("DOMAIN-LEVEL SHAP SUMMARY")
print("=" * 70)

print(
    domain_summary.to_string(
        index=False,
        formatters={

            "Mean_abs_SHAP":
                "{:.4f}".format,

            "Relative_importance_percent":
                "{:.2f}".format
        }
    )
)

print("=" * 70)


# ============================================================
# 12. Save tables
# ============================================================

importance_df.to_csv(

    os.path.join(
        output_dir,
        "Overall_aggregated_SHAP_importance.csv"
    ),

    index=False
)


domain_summary.to_csv(

    os.path.join(
        output_dir,
        "SHAP_domain_summary.csv"
    ),

    index=False
)


# ============================================================
# ============================================================
# ============================================================
# 13. Plot settings
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
# 14. Prepare plotting order
# Highest importance at TOP
# ============================================================

plot_df = (
    importance_df
    .sort_values("Mean_abs_SHAP", ascending=True)
    .reset_index(drop=True)
    .copy()
)

plot_df["Domain"] = plot_df["Feature"].apply(assign_domain)


# ============================================================
# 15. Clean feature labels (optional but recommended)
# ============================================================

rename_map = {
    "Tissue group": "Tissue group",
    "Morphotype": "Morphotype",
    "Soil pH": "Soil pH",
    "SOM": "SOM",
    "Exposure concentration": "Exposure concentration",
    "Growth duration": "Growth duration",
    "Growth temperature": "Growth temperature",
    "Chi3v": "Chi3v",
    "MinPartialCharge": "Min partial charge",
    "TPSA": "TPSA",
    "ALogP": "ALogP",
    "GATS3c": "GATS3c",
    "SpMin8_Bhi": "SpMin8_Bhi"
}

plot_df["Feature_label"] = plot_df["Feature"].map(rename_map)


# ============================================================
# 16. Domain colors - 4 categories
# Keep them muted and publication-friendly
# ============================================================

domain_colors = {
    "Plant": "#2E8B57",        # muted green
    "Environment": "#2C7FB8",  # muted blue
    "Exposure": "#7A5195",     # muted purple
    "Molecular": "#D95F0E"     # muted orange
}

plot_df["Color"] = plot_df["Domain"].map(domain_colors)


# ============================================================
# 17. Create figure
# Wider layout to avoid crowding
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 5.6))

y_position = np.arange(len(plot_df))


# ============================================================
# 18. Draw lollipop lines
# ============================================================

for i, row in plot_df.iterrows():
    ax.hlines(
        y=i,
        xmin=0,
        xmax=row["Mean_abs_SHAP"],
        color=row["Color"],
        linewidth=1.9,
        alpha=0.30,
        zorder=1
    )


# ============================================================
# 19. Draw points
# ============================================================

ax.scatter(
    plot_df["Mean_abs_SHAP"],
    y_position,
    s=165,
    c=plot_df["Color"],
    edgecolors="white",
    linewidths=1.0,
    zorder=3
)


# ============================================================
# 20. Axes labels and ticks
# ============================================================

ax.set_yticks(y_position)

ax.set_yticklabels(
    plot_df["Feature_label"],
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel(
    "Mean absolute SHAP value",
    fontsize=18,
    fontweight="bold"
)

ax.set_ylabel("")

xmax = plot_df["Mean_abs_SHAP"].max() * 1.10
ax.set_xlim(0, xmax)

ax.tick_params(
    axis="x",
    which="major",
    labelsize=15,
    width=1.5,
    length=7,
    pad=7
)

ax.tick_params(
    axis="y",
    which="major",
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
# 21. Grid and frame
# ============================================================

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.8,
    alpha=0.22
)

ax.grid(
    axis="y",
    visible=False
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# 22. Panel label
# ============================================================

ax.text(
    -0.16,
    1.03,
    "(b)",
    transform=ax.transAxes,
    fontsize=22,
    fontweight="bold",
    ha="left",
    va="bottom"
)


# ============================================================
# 23. Legend - OUTSIDE plotting area
# This avoids overlap with data points
# ============================================================

from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=domain_colors["Plant"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=10.5,
        label="Plant"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=domain_colors["Environment"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=10.5,
        label="Environment"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=domain_colors["Exposure"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=10.5,
        label="Exposure"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=domain_colors["Molecular"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=10.5,
        label="Molecular"
    )
]

legend = ax.legend(
    handles=legend_elements,
    loc="lower left",
    bbox_to_anchor=(1.01, 0.02),   # 图外右下
    frameon=False,
    fontsize=12,
    handletextpad=0.8,
    borderpad=0.2,
    labelspacing=0.8
)

for text in legend.get_texts():
    text.set_fontweight("bold")


# ============================================================
# 24. Layout
# Manually reserve space at right for legend
# ============================================================

plt.subplots_adjust(
    left=0.34,
    right=0.80,
    top=0.95,
    bottom=0.14
)


# ============================================================
# 25. Save
# ============================================================

tiff_file = os.path.join(
    output_dir,
    "Fig3b_Overall_SHAP_importance_4domains.tiff"
)

png_file = os.path.join(
    output_dir,
    "Fig3b_Overall_SHAP_importance_4domains.png"
)

pdf_file = os.path.join(
    output_dir,
    "Fig3b_Overall_SHAP_importance_4domains.pdf"
)

plt.savefig(
    tiff_file,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.show()

print("\nFigure saved to:")
print(tiff_file)
print(png_file)
print(pdf_file)