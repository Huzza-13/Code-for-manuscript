import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

MODEL_OUTPUT_DIR = Path(
    r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"
)
ORIGINAL_FILE = Path(
    r"D:\python\pythonProject1\Machine-learning original dataset.xlsx"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PFAS_SHAP_CHAIN_FIG_OUTPUT_DIR",
        str(MODEL_OUTPUT_DIR),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAP_FILE = MODEL_OUTPUT_DIR / "Final_XGBoost_raw_SHAP_values.csv"
METADATA_FILE = MODEL_OUTPUT_DIR / "Final_XGBoost_SHAP_metadata.csv"
OUTPUT_STEM = OUTPUT_DIR / "Fig_Molecular_SHAP_vs_chain_and_class"
SHOW_FIGURE = os.environ.get("PFAS_SHAP_CHAIN_FIG_SHOW", "1").strip() != "0"


# ============================================================
# 2. Scientific data contract
# ============================================================

PFAS_COLUMN = "PFASs Name"
CHAIN_COLUMN = "Carbon Chain Length"
CLASS_COLUMN = "PFAS class"

MOLECULAR_FEATURES = [
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi",
]

DISPLAY_LABELS = {
    "ALogP": "ALogP",
    "Chi3v": "Chi3v",
    "GATS3c": "GATS3c",
    "MinPartialCharge": "Min partial charge",
    "SpMin8_Bhi": "SpMin8_Bhi",
    "TPSA": "TPSA",
}

PFAS_CLASS_CODE_MAP = {
    1: "PFCA",
    2: "PFSA",
    3: "PAP",
    4: "Ether-PFAS",
    5: "FOSA/FOSAA",
}

CLASS_ORDER = ["PFCA", "PFSA", "PAP", "Ether-PFAS", "FOSA/FOSAA"]

CLASS_COLORS = {
    "PFCA": "#3D6F8E",
    "PFSA": "#5F8F82",
    "PAP": "#C08B45",
    "Ether-PFAS": "#8174A2",
    "FOSA/FOSAA": "#AC6875",
}

CLASS_MARKERS = {
    "PFCA": "o",
    "PFSA": "s",
    "PAP": "^",
    "Ether-PFAS": "D",
    "FOSA/FOSAA": "P",
}

HOMOLOGUE_LINE_STYLES = {"PFCA": "-", "PFSA": "--"}

# Labels are limited to a few chemically distinctive compounds.
LABEL_COMPOUNDS = ["GenX", "8:2 diPAP", "N-EtFOSAA"]


# ============================================================
# 3. Helpers
# ============================================================

def canonicalize_pfas_class(series):
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        if not np.allclose(numeric.to_numpy(), numeric.round().to_numpy()):
            raise ValueError("PFAS-class codes must be integers.")
        labels = numeric.round().astype(int).map(PFAS_CLASS_CODE_MAP)
        if labels.isna().any():
            unknown = sorted(numeric[labels.isna()].unique())
            raise ValueError(f"Undefined PFAS-class codes: {unknown}")
        return labels

    if numeric.isna().all():
        accepted = {
            "pfca": "PFCA",
            "pfsa": "PFSA",
            "pap": "PAP",
            "ether-pfas": "Ether-PFAS",
            "ether pfas": "Ether-PFAS",
            "fosa/fosaa": "FOSA/FOSAA",
            "fosa-fosaa": "FOSA/FOSAA",
        }
        labels = series.astype(str).str.strip().str.casefold().map(accepted)
        if labels.isna().any():
            unknown = sorted(series[labels.isna()].astype(str).unique())
            raise ValueError(f"Unrecognized PFAS-class labels: {unknown}")
        return labels

    raise ValueError("PFAS class mixes numeric codes and text labels.")


def q25(values):
    return values.quantile(0.25)


def q75(values):
    return values.quantile(0.75)


def add_panel_label(ax, label):
    ax.text(
        -0.17,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


# ============================================================
# 4. Read and validate the final SHAP outputs
# ============================================================

for required_file in [SHAP_FILE, METADATA_FILE, ORIGINAL_FILE]:
    if not required_file.exists():
        raise FileNotFoundError(f"Required file not found:\n{required_file}")

raw_shap = pd.read_csv(SHAP_FILE)
metadata = pd.read_csv(METADATA_FILE)
original_data = pd.read_excel(ORIGINAL_FILE)

raw_shap.columns = raw_shap.columns.astype(str).str.strip()
metadata.columns = metadata.columns.astype(str).str.strip()
original_data.columns = original_data.columns.astype(str).str.strip()

if len(raw_shap) != len(metadata):
    raise ValueError(
        "SHAP and metadata row counts differ: "
        f"{len(raw_shap)} versus {len(metadata)}."
    )

missing_shap = [
    feature for feature in MOLECULAR_FEATURES
    if feature not in raw_shap.columns
]
if missing_shap:
    raise ValueError(f"Missing molecular SHAP columns: {missing_shap}")

required_chemistry = [PFAS_COLUMN, CHAIN_COLUMN, CLASS_COLUMN]
missing_chemistry = [
    column for column in required_chemistry
    if column not in original_data.columns
]
if missing_chemistry:
    raise ValueError(f"Missing chemistry columns: {missing_chemistry}")

if PFAS_COLUMN not in metadata.columns:
    raise ValueError(f"Metadata is missing: {PFAS_COLUMN}")

metadata[PFAS_COLUMN] = metadata[PFAS_COLUMN].astype(str).str.strip()
original_data[PFAS_COLUMN] = original_data[PFAS_COLUMN].astype(str).str.strip()


# ============================================================
# 5. One chain-length and class record per compound
# ============================================================

chemistry_source = original_data[required_chemistry].copy()
within_compound_nunique = (
    chemistry_source
    .groupby(PFAS_COLUMN)[[CHAIN_COLUMN, CLASS_COLUMN]]
    .nunique(dropna=False)
)
inconsistent = within_compound_nunique.index[
    (within_compound_nunique > 1).any(axis=1)
].tolist()
if inconsistent:
    raise ValueError(
        "Chain length or PFAS class varies within these compounds: "
        f"{inconsistent}"
    )

chemistry = (
    chemistry_source
    .drop_duplicates(subset=PFAS_COLUMN, keep="first")
    .copy()
)
chemistry[CHAIN_COLUMN] = pd.to_numeric(
    chemistry[CHAIN_COLUMN],
    errors="raise",
)
chemistry["PFAS class display"] = canonicalize_pfas_class(
    chemistry[CLASS_COLUMN]
)


# ============================================================
# 6. Observation-level and compound-level SHAP data
# ============================================================

observation_data = metadata[[PFAS_COLUMN]].copy()
for feature in MOLECULAR_FEATURES:
    observation_data[f"SHAP_{feature}"] = raw_shap[feature].to_numpy()

observation_data = observation_data.merge(
    chemistry[[PFAS_COLUMN, CHAIN_COLUMN, "PFAS class display"]],
    on=PFAS_COLUMN,
    how="left",
    validate="many_to_one",
)

if observation_data[[CHAIN_COLUMN, "PFAS class display"]].isna().any().any():
    unmatched = sorted(
        observation_data.loc[
            observation_data[[CHAIN_COLUMN, "PFAS class display"]]
            .isna()
            .any(axis=1),
            PFAS_COLUMN,
        ].unique()
    )
    raise ValueError(f"Chemical metadata could not be matched for: {unmatched}")

aggregation = {
    "n_observations": (f"SHAP_{MOLECULAR_FEATURES[0]}", "size")
}
for feature in MOLECULAR_FEATURES:
    aggregation[f"median_SHAP_{feature}"] = (f"SHAP_{feature}", "median")
    aggregation[f"Q1_SHAP_{feature}"] = (f"SHAP_{feature}", q25)
    aggregation[f"Q3_SHAP_{feature}"] = (f"SHAP_{feature}", q75)

compound_summary = (
    observation_data
    .groupby(
        [PFAS_COLUMN, CHAIN_COLUMN, "PFAS class display"],
        as_index=False,
    )
    .agg(**aggregation)
)

mean_abs_shap = raw_shap[MOLECULAR_FEATURES].abs().mean()
feature_order = mean_abs_shap.sort_values(ascending=False).index.tolist()

class_rank = {name: index for index, name in enumerate(CLASS_ORDER)}
compound_summary["_class_rank"] = (
    compound_summary["PFAS class display"].map(class_rank)
)
compound_summary = (
    compound_summary
    .sort_values(["_class_rank", CHAIN_COLUMN, PFAS_COLUMN])
    .drop(columns="_class_rank")
    .reset_index(drop=True)
)

observation_output = OUTPUT_DIR / "Molecular_SHAP_by_chain_observations.csv"
summary_output = OUTPUT_DIR / "Molecular_SHAP_by_chain_compound_summary.csv"
observation_data.to_csv(observation_output, index=False, encoding="utf-8-sig")
compound_summary.to_csv(summary_output, index=False, encoding="utf-8-sig")


# ============================================================
# 7. Figure style
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "font.weight": "bold",
    "axes.labelsize": 9,
    "axes.labelweight": "bold",
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.35), sharex=True)
axes = axes.ravel()


# ============================================================
# 8. Six descriptor-specific SHAP panels
# ============================================================

for panel_index, (ax, feature) in enumerate(zip(axes, feature_order)):
    median_column = f"median_SHAP_{feature}"
    q1_column = f"Q1_SHAP_{feature}"
    q3_column = f"Q3_SHAP_{feature}"

    ax.axhline(
        0,
        color="#8A8A8A",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )

    # Connect only the PFCA and PFSA homologous series.
    for pfas_class in ["PFCA", "PFSA"]:
        series = compound_summary.loc[
            compound_summary["PFAS class display"] == pfas_class
        ].sort_values(CHAIN_COLUMN)
        if len(series) >= 2:
            ax.plot(
                series[CHAIN_COLUMN],
                series[median_column],
                linestyle=HOMOLOGUE_LINE_STYLES[pfas_class],
                color=CLASS_COLORS[pfas_class],
                linewidth=1.35,
                alpha=0.82,
                zorder=1,
            )

    for pfas_class in CLASS_ORDER:
        class_data = compound_summary.loc[
            compound_summary["PFAS class display"] == pfas_class
        ]
        if class_data.empty:
            continue

        medians = class_data[median_column].to_numpy()
        lower = medians - class_data[q1_column].to_numpy()
        upper = class_data[q3_column].to_numpy() - medians

        ax.errorbar(
            class_data[CHAIN_COLUMN],
            medians,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=CLASS_COLORS[pfas_class],
            elinewidth=0.85,
            capsize=2.3,
            capthick=0.8,
            alpha=0.60,
            zorder=2,
        )
        ax.scatter(
            class_data[CHAIN_COLUMN],
            medians,
            s=48,
            marker=CLASS_MARKERS[pfas_class],
            facecolor=CLASS_COLORS[pfas_class],
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
        )

    for compound in LABEL_COMPOUNDS:
        selected = compound_summary.loc[
            compound_summary[PFAS_COLUMN] == compound
        ]
        if selected.empty:
            continue
        row = selected.iloc[0]
        vertical_offset = 5 if row[median_column] >= 0 else -10
        ax.annotate(
            compound,
            xy=(row[CHAIN_COLUMN], row[median_column]),
            xytext=(5, vertical_offset),
            textcoords="offset points",
            fontsize=6.1,
            fontweight="bold",
            color=CLASS_COLORS[row["PFAS class display"]],
            ha="left",
            va="bottom" if vertical_offset >= 0 else "top",
            zorder=4,
        )

    ax.set_title(
        f"{DISPLAY_LABELS[feature]}\n"
        f"mean |SHAP| = {mean_abs_shap[feature]:.3f}",
        pad=6,
    )
    ax.set_ylabel("SHAP value")
    ax.set_xlim(3.4, 14.6)
    ax.set_xticks([4, 6, 8, 10, 12, 14])
    ax.margins(y=0.18)
    ax.grid(False)
    ax.tick_params(axis="both", which="major", pad=3)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    add_panel_label(ax, f"({chr(97 + panel_index)})")

for ax in axes[3:]:
    ax.set_xlabel("Carbon-chain length")


# ============================================================
# 9. Shared PFAS-class legend
# ============================================================

class_counts = (
    compound_summary.groupby("PFAS class display")[PFAS_COLUMN]
    .nunique()
    .to_dict()
)

legend_handles = []
for pfas_class in CLASS_ORDER:
    line_style = HOMOLOGUE_LINE_STYLES.get(pfas_class, "none")
    legend_handles.append(
        Line2D(
            [0],
            [0],
            linestyle=line_style,
            linewidth=1.25 if line_style != "none" else 0,
            color=CLASS_COLORS[pfas_class],
            marker=CLASS_MARKERS[pfas_class],
            markersize=6.5,
            markerfacecolor=CLASS_COLORS[pfas_class],
            markeredgecolor="white",
            label=f"{pfas_class} (n = {class_counts.get(pfas_class, 0)})",
        )
    )

legend = fig.legend(
    handles=legend_handles,
    title="PFAS class",
    loc="lower center",
    bbox_to_anchor=(0.50, 0.004),
    ncol=5,
    fontsize=7,
    title_fontsize=8,
    handletextpad=0.45,
    columnspacing=0.85,
    borderpad=0.2,
)
legend.get_title().set_fontweight("bold")
for legend_text in legend.get_texts():
    legend_text.set_fontweight("bold")

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.145,
    top=0.955,
    wspace=0.36,
    hspace=0.43,
)


# ============================================================
# 10. Export
# ============================================================

fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(
    OUTPUT_STEM.with_suffix(".tiff"),
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pil_kwargs={"compression": "tiff_lzw"},
)
fig.savefig(
    OUTPUT_STEM.with_suffix(".png"),
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print("\nPanel order and mean absolute SHAP:")
print(mean_abs_shap.loc[feature_order].to_string())
print(f"\nObservations summarized: {len(observation_data)}")
print(f"PFAS compounds plotted: {compound_summary[PFAS_COLUMN].nunique()}")
print("\nSaved source-data files:")
print(observation_output)
print(summary_output)
print("\nSaved figure files:")
for suffix in [".svg", ".pdf", ".tiff", ".png"]:
    print(OUTPUT_STEM.with_suffix(suffix))

if SHOW_FIGURE:
    plt.show()

plt.close(fig)
