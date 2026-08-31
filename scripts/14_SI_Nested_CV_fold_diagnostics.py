import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = (
    r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"
    r"\Nested_XGBoost_OOF_predictions.csv"
)

output_dir = r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"

figure_base = os.path.join(
    output_dir,
    "Fig_S_Nested_CV_fold_diagnostics_final"
)


# ============================================================
# 2. FONT SETTINGS
# ============================================================

# Windows Arial fonts
arial_regular_path = r"C:\Windows\Fonts\arial.ttf"
arial_bold_path = r"C:\Windows\Fonts\arialbd.ttf"

if os.path.exists(arial_regular_path):
    font_regular = FontProperties(
        fname=arial_regular_path
    )
else:
    font_regular = FontProperties(
        family="Arial"
    )

if os.path.exists(arial_bold_path):
    font_bold = FontProperties(
        fname=arial_bold_path
    )
else:
    font_bold = FontProperties(
        family="Arial",
        weight="bold"
    )


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,

    "axes.linewidth": 1.8,

    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,

    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,

    # Preserve editable/vector text in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# ============================================================
# 3. READ DATA
# ============================================================

df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_cols = [
    "Observed",
    "Predicted",
    "Study_Compound_ID"
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing required columns: {missing_cols}\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# 5. CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=[
        "Observed",
        "Predicted",
        "Study_Compound_ID"
    ]
).copy()

df["Observed"] = pd.to_numeric(
    df["Observed"],
    errors="coerce"
)

df["Predicted"] = pd.to_numeric(
    df["Predicted"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Observed",
        "Predicted"
    ]
).reset_index(drop=True)


print("\nNumber of observations:", len(df))

print(
    "Number of unique Study×PFAS groups:",
    df["Study_Compound_ID"].nunique()
)


# ============================================================
# 6. RECONSTRUCT OUTER GROUPKFOLD
#
# IMPORTANT:
# This reconstruction assumes the original nested CV used:
#
# GroupKFold(n_splits=5)
#
# with Study_Compound_ID as the grouping variable.
# ============================================================

groups = (
    df["Study_Compound_ID"]
    .astype(str)
    .to_numpy()
)

X_dummy = np.zeros(
    (len(df), 1)
)

y_dummy = (
    df["Observed"]
    .to_numpy(dtype=float)
)

outer_cv = GroupKFold(
    n_splits=5
)

outer_fold = np.zeros(
    len(df),
    dtype=int
)

for fold_id, (_, test_idx) in enumerate(
    outer_cv.split(
        X_dummy,
        y_dummy,
        groups=groups
    ),
    start=1
):

    outer_fold[test_idx] = fold_id


df["Outer_Fold"] = outer_fold


# ============================================================
# 7. SCIENTIFIC CONSISTENCY CHECKS
# ============================================================

# Check that each Study×PFAS group occurs in one test fold only
group_fold_counts = (
    df.groupby("Study_Compound_ID")
    ["Outer_Fold"]
    .nunique()
)

if group_fold_counts.max() != 1:

    raise RuntimeError(
        "Group leakage detected: at least one "
        "Study×PFAS group appears in multiple folds."
    )


print("\nGroup leakage check: PASSED")

print(
    "Every Study×PFAS combination belongs "
    "to exactly one outer fold."
)


unique_folds = sorted(
    df["Outer_Fold"].unique()
)

if unique_folds != [1, 2, 3, 4, 5]:

    raise RuntimeError(
        f"Unexpected outer-fold IDs: {unique_folds}"
    )


print(
    "Five-fold reconstruction check: PASSED"
)


# ============================================================
# 8. OVERALL OOF PERFORMANCE
# ============================================================

y_all = (
    df["Observed"]
    .to_numpy(dtype=float)
)

pred_all = (
    df["Predicted"]
    .to_numpy(dtype=float)
)


overall_r2 = r2_score(
    y_all,
    pred_all
)

overall_rmse = np.sqrt(
    mean_squared_error(
        y_all,
        pred_all
    )
)


print("\n======================================")
print("Overall OOF performance")
print("======================================")

print(
    f"R2   = {overall_r2:.6f}"
)

print(
    f"RMSE = {overall_rmse:.6f}"
)

print(
    f"n    = {len(df)}"
)


# ============================================================
# 9. CHECK AGAINST MANUSCRIPT VALUES
# ============================================================

EXPECTED_R2 = 0.624
EXPECTED_RMSE = 0.869

if abs(overall_r2 - EXPECTED_R2) > 0.01:

    warnings.warn(
        f"Calculated R2 ({overall_r2:.3f}) differs "
        f"from manuscript R2 ({EXPECTED_R2:.3f})."
    )

if abs(overall_rmse - EXPECTED_RMSE) > 0.01:

    warnings.warn(
        f"Calculated RMSE ({overall_rmse:.3f}) differs "
        f"from manuscript RMSE ({EXPECTED_RMSE:.3f})."
    )


# ============================================================
# 10. FOLD-LEVEL PERFORMANCE
# ============================================================

fold_results = []

for fold_id in range(1, 6):

    sub = df[
        df["Outer_Fold"] == fold_id
    ].copy()

    y_true = (
        sub["Observed"]
        .to_numpy(dtype=float)
    )

    y_pred = (
        sub["Predicted"]
        .to_numpy(dtype=float)
    )

    fold_r2 = r2_score(
        y_true,
        y_pred
    )

    fold_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    fold_results.append({

        "Outer_Fold": fold_id,

        "n": len(sub),

        "Number_of_Study_PFAS_groups":
            sub["Study_Compound_ID"].nunique(),

        "R2": fold_r2,

        "RMSE": fold_rmse
    })


fold_results = pd.DataFrame(
    fold_results
)


print("\n======================================")
print("Fold-level diagnostics")
print("======================================")

print(
    fold_results.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE NUMERICAL RESULTS
# ============================================================

oof_output = os.path.join(
    output_dir,
    "Nested_XGBoost_OOF_predictions_with_folds.csv"
)

df.to_csv(
    oof_output,
    index=False
)


diagnostics_output = os.path.join(
    output_dir,
    "Nested_CV_fold_diagnostics_values.csv"
)

fold_results.to_csv(
    diagnostics_output,
    index=False
)


# ============================================================
# 12. PREPARE FIGURE DATA
# ============================================================

x = np.arange(5)

r2_values = (
    fold_results["R2"]
    .to_numpy(dtype=float)
)

rmse_values = (
    fold_results["RMSE"]
    .to_numpy(dtype=float)
)


fold_labels = [
    f"Fold {int(row.Outer_Fold)}\n(n={int(row.n)})"
    for _, row in fold_results.iterrows()
]


# ============================================================
# 13. COLORS
# ============================================================

r2_color = "#5F86AD"
rmse_color = "#C68155"

stem_color = "#B8B8B8"
reference_color = "#595959"


# ============================================================
# 14. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.6, 5.9)
)


# ============================================================
# 15. DRAW FUNCTION
# ============================================================

def draw_fold_panel(
    ax,
    values,
    overall_value,
    ylabel,
    panel_label,
    point_color,
    ylim,
    yticks
):

    values = np.asarray(
        values,
        dtype=float
    )

    # --------------------------------------------------------
    # Axis range
    # --------------------------------------------------------

    ax.set_ylim(
        ylim
    )

    ax.set_yticks(
        yticks
    )


    # --------------------------------------------------------
    # Overall OOF reference line
    # --------------------------------------------------------

    ax.axhline(
        overall_value,
        color=reference_color,
        linewidth=2.0,
        linestyle=(0, (5, 4)),
        zorder=1
    )


    # --------------------------------------------------------
    # Deviation stems
    #
    # No lines connecting fold 1 → 5 because outer folds
    # have no ordered temporal/continuous interpretation.
    # --------------------------------------------------------

    for xi, value in zip(
        x,
        values
    ):

        ax.plot(
            [xi, xi],
            [overall_value, value],
            color=stem_color,
            linewidth=2.5,
            solid_capstyle="round",
            zorder=1
        )


    # --------------------------------------------------------
    # Fold-level points
    # --------------------------------------------------------

    ax.scatter(
        x,
        values,
        s=235,
        facecolor=point_color,
        edgecolor="black",
        linewidth=1.4,
        zorder=4
    )


    # --------------------------------------------------------
    # Fold-level numerical labels
    #
    # Above OOF -> label above point
    # Below OOF -> label below point
    # --------------------------------------------------------

    for xi, value in zip(
            x,
            values
    ):

        if value >= overall_value:
            offset_y = 15
            vertical_alignment = "bottom"
        else:
            offset_y = -17
            vertical_alignment = "top"

        # Fold 1 label moves slightly to the right
        if xi == 0:
            offset_x = 12
        else:
            offset_x = 0

        ax.annotate(
            f"{value:.3f}",
            xy=(xi, value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",

            ha="center",
            va=vertical_alignment,

            fontsize=15.5,
            fontproperties=font_bold,

            annotation_clip=False,
            zorder=6
        )


    # --------------------------------------------------------
    # X-axis labels
    # --------------------------------------------------------

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        fold_labels
    )


    for label in ax.get_xticklabels():

        label.set_fontproperties(
            font_bold
        )

        label.set_fontsize(
            15.5
        )


    # --------------------------------------------------------
    # Y-axis label
    # --------------------------------------------------------

    ax.set_ylabel(
        ylabel,
        fontproperties=font_bold,
        fontsize=22
    )


    # --------------------------------------------------------
    # Y tick labels
    # --------------------------------------------------------

    for label in ax.get_yticklabels():

        label.set_fontproperties(
            font_bold
        )

        label.set_fontsize(
            16
        )


    # --------------------------------------------------------
    # Axis style
    # --------------------------------------------------------

    ax.grid(
        False
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_linewidth(
        1.8
    )

    ax.spines["bottom"].set_linewidth(
        1.8
    )


    ax.tick_params(
        axis="both",
        direction="out",
        width=1.6,
        length=5.5
    )


    # --------------------------------------------------------
    # Panel label
    #
    # Positioned at the upper-left corner
    # --------------------------------------------------------

    ax.text(
        -0.18,
        1.035,
        panel_label,

        transform=ax.transAxes,

        fontproperties=font_bold,
        fontsize=27,

        ha="left",
        va="top",

        clip_on=False
    )


    # --------------------------------------------------------
    # Overall OOF legend
    #
    # Use legend instead of placing text directly on the
    # dashed reference line to avoid overlap.
    # --------------------------------------------------------

    proxy_line = Line2D(
        [0],
        [0],

        color=reference_color,

        linewidth=2.0,

        linestyle=(0, (5, 4))
    )


    legend = ax.legend(
        [proxy_line],
        [
            f"Overall OOF = {overall_value:.3f}"
        ],

        loc="upper right",

        bbox_to_anchor=(
            0.99,
            0.98
        ),

        frameon=False,

        handlelength=2.7,
        handletextpad=0.7,

        borderaxespad=0.0,

        prop=font_bold
    )


    for text in legend.get_texts():

        text.set_fontsize(
            14.5
        )


# ============================================================
# 16. PANEL (a): R2
# ============================================================

draw_fold_panel(

    axes[0],

    values=r2_values,

    overall_value=overall_r2,

    ylabel=r"$R^2$",

    panel_label="(a)",

    point_color=r2_color,

    ylim=(0.42, 0.83),

    yticks=[
        0.4,
        0.5,
        0.6,
        0.7,
        0.8
    ]
)


# ============================================================
# 17. PANEL (b): RMSE
# ============================================================

draw_fold_panel(

    axes[1],

    values=rmse_values,

    overall_value=overall_rmse,

    ylabel="RMSE",

    panel_label="(b)",

    point_color=rmse_color,

    ylim=(0.53, 1.23),

    yticks=[
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2
    ]
)


# ============================================================
# 18. FINAL LAYOUT
# ============================================================

plt.subplots_adjust(

    left=0.10,

    right=0.985,

    bottom=0.205,

    top=0.93,

    wspace=0.30
)


# ============================================================
# 19. SAVE FIGURE
# ============================================================

plt.savefig(
    figure_base + ".png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    figure_base + ".tiff",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)



plt.show()