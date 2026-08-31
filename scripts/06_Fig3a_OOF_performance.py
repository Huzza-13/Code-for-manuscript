import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.lines import Line2D


# ============================================================
# 1. File paths
# ============================================================

output_dir = r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"

input_file = os.path.join(
    output_dir,
    "Nested_XGBoost_OOF_predictions.csv"
)


# ============================================================
# 2. Read OOF predictions
# ============================================================

data = pd.read_csv(input_file)

y_obs = data["Observed"].values
y_pred = data["Predicted"].values


# ============================================================
# 3. Performance metrics
# ============================================================

r2 = r2_score(y_obs, y_pred)

rmse = np.sqrt(
    mean_squared_error(
        y_obs,
        y_pred
    )
)

n = len(y_obs)


print(f"OOF R²   = {r2:.3f}")
print(f"OOF RMSE = {rmse:.3f}")
print(f"n        = {n}")


# ============================================================
# 4. Plot range
# ============================================================

global_min = min(
    y_obs.min(),
    y_pred.min()
)

global_max = max(
    y_obs.max(),
    y_pred.max()
)

padding = (
    global_max - global_min
) * 0.055

plot_min = global_min - padding
plot_max = global_max + padding

x_line = np.linspace(
    plot_min,
    plot_max,
    300
)


# ============================================================
# 5. Calibration regression
#
# Predicted = a + b × Observed
# ============================================================

reg = LinearRegression()

reg.fit(
    y_obs.reshape(-1, 1),
    y_pred
)

calibration_fit = reg.predict(
    x_line.reshape(-1, 1)
)


# ============================================================
# 6. 95% confidence interval
# ============================================================

fitted_values = reg.predict(
    y_obs.reshape(-1, 1)
)

residuals = (
    y_pred
    - fitted_values
)

dof = n - 2

residual_se = np.sqrt(
    np.sum(residuals ** 2)
    / dof
)

t_value = stats.t.ppf(
    0.975,
    dof
)

x_mean = np.mean(y_obs)

ss_x = np.sum(
    (y_obs - x_mean) ** 2
)

confidence_interval = (
    t_value
    * residual_se
    * np.sqrt(
        1 / n
        + (
            (x_line - x_mean) ** 2
            / ss_x
        )
    )
)


# ============================================================
# 7. Global figure style

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
# 8. Create figure
#
# Wider than previous version
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.8, 5.4)
)


# ============================================================
# 9. OOF scatter
# ============================================================

ax.scatter(

    y_obs,
    y_pred,

    s=46,

    alpha=0.42,

    edgecolors="white",

    linewidths=0.35,

    rasterized=True,

    zorder=3
)


# ============================================================
# 10. 95% confidence band
#
# Do NOT give this a legend label:
# explanation can remain in the caption.
# ============================================================

ax.fill_between(

    x_line,

    calibration_fit - confidence_interval,

    calibration_fit + confidence_interval,

    alpha=0.13,

    linewidth=0,

    zorder=1
)


# ============================================================
# 11. 1:1 reference line
# ============================================================

ax.plot(

    [plot_min, plot_max],
    [plot_min, plot_max],

    linestyle="--",

    linewidth=1.8,

    zorder=2
)


# ============================================================
# 12. Calibration fit
# ============================================================

ax.plot(

    x_line,
    calibration_fit,

    linewidth=2.3,

    zorder=4
)


# ============================================================
# 13. Metrics annotation
#
# No box / no frame
# ============================================================

metrics_text = (
    f"$R^2$ = {r2:.3f}\n"
    f"RMSE = {rmse:.3f}\n"
    f"$n$ = {n}"
)

ax.text(

    0.055,
    0.945,

    metrics_text,

    transform=ax.transAxes,

    ha="left",
    va="top",

    fontsize=14,

    fontweight="bold",

    linespacing=1.20
)


# ============================================================
# 14. Axes
# ============================================================

ax.set_xlim(
    plot_min,
    plot_max
)

ax.set_ylim(
    plot_min,
    plot_max
)

ax.set_aspect(
    "equal",
    adjustable="box"
)


ax.set_xlabel(
    r"Observed log$_{10}$ PFAS concentration",
    fontsize=18,
    fontweight="bold"
)

ax.set_ylabel(
    r"Predicted log$_{10}$ PFAS concentration",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# 15. No grid
# ============================================================

ax.grid(False)


# ============================================================
# 16. Clean frame
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# 17. Bold tick labels
# ============================================================

for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")


ax.tick_params(
    axis="both",
    which="major",
    pad=7
)


# ============================================================
# 18. Panel label
# ============================================================

ax.text(

    -0.18,
    1.035,

    "(a)",

    transform=ax.transAxes,

    fontsize=24,

    fontweight="bold",

    ha="left",
    va="bottom"
)


# ============================================================
# 19. Clean custom legend
#
# Only line types are shown.
# No OOF point legend.
# No 95% CI square.
# No legend frame.
# ============================================================

legend_handles = [

    Line2D(
        [0],
        [0],
        linestyle="--",
        linewidth=1.8,
        label="1:1 line"
    ),

    Line2D(
        [0],
        [0],
        linestyle="-",
        linewidth=2.3,
        label="Calibration fit"
    )
]


legend = ax.legend(

    handles=legend_handles,

    loc="lower right",

    frameon=False,

    fontsize=12,

    handlelength=2.6,

    handletextpad=0.8,

    borderpad=0.1,

    labelspacing=0.6
)


for text in legend.get_texts():
    text.set_fontweight("bold")


# ============================================================
# 20. Layout
# ============================================================

plt.subplots_adjust(

    left=0.19,

    right=0.97,

    bottom=0.16,

    top=0.95
)


# ============================================================
# 21. Save
# ============================================================

tiff_file = os.path.join(
    output_dir,
    "Fig3a_Nested_OOF_performance_optimized.tiff"
)

pdf_file = os.path.join(
    output_dir,
    "Fig3a_Nested_OOF_performance_optimized.pdf"
)

png_file = os.path.join(
    output_dir,
    "Fig3a_Nested_OOF_performance_optimized.png"
)


plt.savefig(
    tiff_file,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("\nFigure saved to:")
print(tiff_file)
print(pdf_file)
print(png_file)