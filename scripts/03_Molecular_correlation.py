import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# 1. Paths
# ============================================================

INPUT_FILE = Path(
    r"D:\python\pythonProject1\Machine-learning original dataset.xlsx"
)

OUTPUT_DIR = Path(os.environ.get(
    "PFAS_CORRELATION_OUTPUT_DIR",
    r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass\03_correlation",
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STEM = OUTPUT_DIR / "PFAS_descriptor_and_PFAS_class_figure"
SHOW_FIGURE = True


# ============================================================
# 2. Data contract
# ============================================================

PFAS_COLUMN = "PFASs Name"
PFAS_CLASS_COLUMN = "PFAS class"
CORRELATION_THRESHOLD = 0.90

# These are candidate molecular descriptors and the structural chain-length
# variable used for post-hoc chemical interpretation. This script does not
# train, validate, or explain the XGBoost model.
MOLECULAR_DESCRIPTORS = [
    "Carbon Chain Length",
    "Chi3v",
    "MinPartialCharge",
    "MolWt",
    "NumRotatableBonds",
    "PEOE_VSA4",
    "TPSA",
    "NumF",
    "ALogP",
    "ATS3s",
    "GATS3c",
    "SpMin8_Bhi",
]

# Dataset codes are nominal PFAS-class identifiers, not numeric values.
PFAS_CLASS_CODE_MAP = {
    1: "PFCA",
    2: "PFSA",
    3: "PAP",
    4: "Ether-PFAS",
    5: "FOSA/FOSAA",
}

# Independent compound-to-class mapping used to audit the original dataset.
EXPECTED_COMPOUND_CLASS = {
    "PFBA": "PFCA",
    "PFPeA": "PFCA",
    "PFHxA": "PFCA",
    "PFHpA": "PFCA",
    "PFOA": "PFCA",
    "PFNA": "PFCA",
    "PFDA": "PFCA",
    "PFUnDA": "PFCA",
    "PFTeDA": "PFCA",
    "PFBS": "PFSA",
    "PFHxS": "PFSA",
    "PFOS": "PFSA",
    "8:2 diPAP": "PAP",
    "GenX": "Ether-PFAS",
    "FOSA": "FOSA/FOSAA",
    "N-EtFOSA": "FOSA/FOSAA",
    "N-EtFOSAA": "FOSA/FOSAA",
}

CLASS_ORDER = ["PFCA", "PFSA", "PAP", "Ether-PFAS", "FOSA/FOSAA"]

CLASS_COLORS = {
    "PFCA": "#3D6F8E",
    "PFSA": "#5F8F82",
    "PAP": "#C08B45",
    "Ether-PFAS": "#8174A2",
    "FOSA/FOSAA": "#AC6875",
}


# ============================================================
# 3. Typography and helpers
# ============================================================

try:
    ARIAL_PATH = fm.findfont("Arial", fallback_to_default=False)
    print("Arial font:", ARIAL_PATH)
except ValueError as exc:
    raise RuntimeError("Arial was not found. Install Arial before plotting.") from exc

sns.set_theme(style="white")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 8,
    "font.weight": "bold",
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def normalize_name(value):
    return (
        str(value)
        .strip()
        .replace("–", "-")
        .replace("—", "-")
        .casefold()
    )


def split_into_lines(values, items_per_line=3):
    return [
        "  ·  ".join(values[start:start + items_per_line])
        for start in range(0, len(values), items_per_line)
    ]


def text_color_for_cell(cmap, norm, value):
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.53 else "#252525"


# ============================================================
# 4. Read and validate the raw chemical data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

data = pd.read_excel(INPUT_FILE)
data.columns = data.columns.astype(str).str.strip()

required_columns = [PFAS_COLUMN, PFAS_CLASS_COLUMN] + MOLECULAR_DESCRIPTORS
missing_columns = [
    column for column in required_columns if column not in data.columns
]
if missing_columns:
    raise KeyError("Required columns are missing:\n" + "\n".join(missing_columns))

print("Original rows:", len(data))
print("Unique PFAS compounds:", data[PFAS_COLUMN].nunique())

# Each descriptor and PFAS class must be constant within a compound.
columns_to_check = MOLECULAR_DESCRIPTORS + [PFAS_CLASS_COLUMN]
within_compound_nunique = (
    data.groupby(PFAS_COLUMN)[columns_to_check].nunique(dropna=False)
)

integrity_problems = []
for column in columns_to_check:
    compounds = within_compound_nunique.index[
        within_compound_nunique[column] > 1
    ].tolist()
    integrity_problems.extend((compound, column) for compound in compounds)

if integrity_problems:
    details = "\n".join(
        f"  - {compound}: {column}"
        for compound, column in integrity_problems
    )
    raise ValueError("Values vary within the same PFAS compound:\n" + details)

# Pearson correlations are calculated from one row per PFAS compound.
compound_data = (
    data[[PFAS_COLUMN, PFAS_CLASS_COLUMN] + MOLECULAR_DESCRIPTORS]
    .drop_duplicates(subset=PFAS_COLUMN, keep="first")
    .copy()
)

compound_data[PFAS_COLUMN] = compound_data[PFAS_COLUMN].astype(str).str.strip()
compound_data["_normalized_name"] = compound_data[PFAS_COLUMN].map(normalize_name)

for descriptor in MOLECULAR_DESCRIPTORS:
    compound_data[descriptor] = pd.to_numeric(
        compound_data[descriptor], errors="raise"
    )

if compound_data[MOLECULAR_DESCRIPTORS].isna().any().any():
    missing_counts = compound_data[MOLECULAR_DESCRIPTORS].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    raise ValueError("Missing descriptor values:\n" + missing_counts.to_string())


# ============================================================
# 5. PFAS-class audit
# ============================================================

expected_by_normalized_name = {
    normalize_name(compound): pfas_class
    for compound, pfas_class in EXPECTED_COMPOUND_CLASS.items()
}

observed_names = set(compound_data["_normalized_name"])
expected_names = set(expected_by_normalized_name)
unexpected_compounds = sorted(observed_names - expected_names)
missing_expected_compounds = sorted(expected_names - observed_names)

if unexpected_compounds or missing_expected_compounds:
    raise ValueError(
        "The compound list does not match the expected 17-PFAS set.\n"
        f"Unexpected names: {unexpected_compounds}\n"
        f"Missing expected names: {missing_expected_compounds}"
    )

compound_data["PFAS class display"] = (
    compound_data["_normalized_name"].map(expected_by_normalized_name)
)

raw_class = compound_data[PFAS_CLASS_COLUMN]
numeric_class = pd.to_numeric(raw_class, errors="coerce")

if numeric_class.notna().all():
    if not np.allclose(numeric_class.to_numpy(), numeric_class.round().to_numpy()):
        raise ValueError("PFAS-class codes must be integers.")

    class_codes = numeric_class.round().astype(int)
    unknown_codes = sorted(set(class_codes.unique()) - set(PFAS_CLASS_CODE_MAP))
    if unknown_codes:
        raise ValueError(f"Undefined PFAS-class codes: {unknown_codes}")

    class_from_code = class_codes.map(PFAS_CLASS_CODE_MAP)
    mismatched = compound_data.loc[
        class_from_code.to_numpy()
        != compound_data["PFAS class display"].to_numpy(),
        [PFAS_COLUMN, PFAS_CLASS_COLUMN, "PFAS class display"],
    ]
    if not mismatched.empty:
        raise ValueError(
            "PFAS-class codes conflict with compound chemistry:\n"
            + mismatched.to_string(index=False)
        )

elif numeric_class.isna().all():
    accepted_labels = {
        "pfca": "PFCA",
        "pfsa": "PFSA",
        "pap": "PAP",
        "ether-pfas": "Ether-PFAS",
        "ether pfas": "Ether-PFAS",
        "fosa/fosaa": "FOSA/FOSAA",
        "fosa-fosaa": "FOSA/FOSAA",
    }
    class_from_text = (
        raw_class.astype(str).str.strip().str.casefold().map(accepted_labels)
    )
    if class_from_text.isna().any():
        unknown = sorted(raw_class[class_from_text.isna()].astype(str).unique())
        raise ValueError(f"Unrecognized PFAS-class labels: {unknown}")
    if not np.array_equal(
        class_from_text.to_numpy(),
        compound_data["PFAS class display"].to_numpy(),
    ):
        raise ValueError("PFAS-class labels conflict with compound chemistry.")

else:
    raise ValueError("PFAS class mixes numeric codes and text labels.")

print("\nVerified PFAS-class assignments:")
print(compound_data[[PFAS_COLUMN, "PFAS class display"]].to_string(index=False))


# ============================================================
# 6. Descriptor correlations and PFAS-class ordering
# ============================================================

correlation_matrix = compound_data[MOLECULAR_DESCRIPTORS].corr(method="pearson")

strong_pairs = []
for row_index in range(1, len(MOLECULAR_DESCRIPTORS)):
    for column_index in range(row_index):
        value = correlation_matrix.iloc[row_index, column_index]
        if abs(value) >= CORRELATION_THRESHOLD:
            strong_pairs.append({
                "Descriptor 1": MOLECULAR_DESCRIPTORS[column_index],
                "Descriptor 2": MOLECULAR_DESCRIPTORS[row_index],
                "Pearson r": value,
                "Absolute r": abs(value),
            })

strong_pairs_table = pd.DataFrame(
    strong_pairs,
    columns=["Descriptor 1", "Descriptor 2", "Pearson r", "Absolute r"],
)
if not strong_pairs_table.empty:
    strong_pairs_table = strong_pairs_table.sort_values("Absolute r", ascending=False)

class_rank = {pfas_class: index for index, pfas_class in enumerate(CLASS_ORDER)}
compound_data["_class_rank"] = compound_data["PFAS class display"].map(class_rank)
compound_data = compound_data.sort_values(
    ["_class_rank", "Carbon Chain Length", PFAS_COLUMN]
).reset_index(drop=True)

class_compounds = {
    pfas_class: compound_data.loc[
        compound_data["PFAS class display"] == pfas_class, PFAS_COLUMN
    ].tolist()
    for pfas_class in CLASS_ORDER
}


# ============================================================
# 7. Figure layout
# ============================================================

fig = plt.figure(figsize=(8.60, 6.10), facecolor="white")

outer_grid = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[1.78, 0.92],
    left=0.185,
    right=0.985,
    bottom=0.185,
    top=0.875,
    wspace=0.26,
)

heatmap_grid = outer_grid[0].subgridspec(
    nrows=1,
    ncols=2,
    width_ratios=[1.0, 0.038],
    wspace=0.055,
)

ax_heatmap = fig.add_subplot(heatmap_grid[0, 0])
ax_colorbar = fig.add_subplot(heatmap_grid[0, 1])
ax_classes = fig.add_subplot(outer_grid[1])


# ============================================================
# 8. Panel a: descriptor-correlation heatmap
# ============================================================

row_labels = MOLECULAR_DESCRIPTORS[1:]
column_labels = MOLECULAR_DESCRIPTORS[:-1]
plot_matrix = correlation_matrix.loc[row_labels, column_labels]
upper_mask = np.triu(np.ones(plot_matrix.shape, dtype=bool), k=1)

correlation_cmap = LinearSegmentedColormap.from_list(
    "correlation_muted", ["#315F8C", "#F7F7F5", "#B64C52"], N=256
)
correlation_cmap.set_bad("white")
correlation_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

sns.heatmap(
    plot_matrix,
    mask=upper_mask,
    cmap=correlation_cmap,
    norm=correlation_norm,
    square=True,
    linewidths=0.42,
    linecolor="white",
    annot=False,
    cbar=True,
    cbar_ax=ax_colorbar,
    cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]},
    ax=ax_heatmap,
)

for row_index in range(plot_matrix.shape[0]):
    for column_index in range(row_index + 1):
        value = plot_matrix.iloc[row_index, column_index]
        is_strong = abs(value) >= CORRELATION_THRESHOLD
        label = f"{value:.2f}" + ("*" if is_strong else "")
        ax_heatmap.text(
            column_index + 0.5,
            row_index + 0.5,
            label,
            ha="center",
            va="center",
            fontsize=6.45,
            fontfamily="Arial",
            fontweight="bold",
            color=text_color_for_cell(correlation_cmap, correlation_norm, value),
        )

ax_heatmap.set_xticklabels(
    column_labels, rotation=50, ha="right", rotation_mode="anchor", fontsize=7.2
)
ax_heatmap.set_yticklabels(row_labels, rotation=0, fontsize=7.2)
ax_heatmap.tick_params(axis="both", length=0, pad=2)
ax_heatmap.set_xlabel("")
ax_heatmap.set_ylabel("")
for spine in ax_heatmap.spines.values():
    spine.set_visible(False)

for tick in ax_heatmap.get_xticklabels() + ax_heatmap.get_yticklabels():
    tick.set_fontweight("bold")

ax_colorbar.set_ylabel(
    "Pearson r", rotation=90, fontsize=8.2, fontweight="bold", labelpad=6
)
ax_colorbar.tick_params(axis="y", labelsize=7.0, length=3.0, width=0.7)
for tick in ax_colorbar.get_yticklabels():
    tick.set_fontweight("bold")
for spine in ax_colorbar.spines.values():
    spine.set_linewidth(0.6)
    spine.set_edgecolor("#5A5A5A")

ax_heatmap.text(
    -0.39, 1.095, "(a)", transform=ax_heatmap.transAxes,
    ha="left", va="bottom", fontsize=12, fontweight="bold"
)
ax_heatmap.text(
    0.0, 1.095, "Descriptor correlations", transform=ax_heatmap.transAxes,
    ha="left", va="bottom", fontsize=10, fontweight="bold"
)
ax_heatmap.text(
    0.0,
    1.035,
    f"Pearson r across {len(compound_data)} unique PFAS compounds; "
    f"* |r| ≥ {CORRELATION_THRESHOLD:.2f}",
    transform=ax_heatmap.transAxes,
    ha="left",
    va="bottom",
    fontsize=7.2,
    color="#555555",
)


# ============================================================
# 9. Panel b: direct compound-to-PFAS-class assignment
# ============================================================

ax_classes.set_axis_off()

ax_classes.text(
    -0.08, 1.095, "(b)", transform=ax_classes.transAxes,
    ha="left", va="bottom", fontsize=12, fontweight="bold"
)
ax_classes.text(
    0.055, 1.095, "PFAS classes", transform=ax_classes.transAxes,
    ha="left", va="bottom", fontsize=10, fontweight="bold"
)
ax_classes.text(
    0.055, 1.035, "Direct compound-to-class assignment",
    transform=ax_classes.transAxes, ha="left", va="bottom",
    fontsize=7.2, color="#555555"
)

wrapped_lines = {
    pfas_class: split_into_lines(class_compounds[pfas_class], items_per_line=3)
    for pfas_class in CLASS_ORDER
}
card_weights = {
    pfas_class: 1.0 + 0.50 * (len(wrapped_lines[pfas_class]) - 1)
    for pfas_class in CLASS_ORDER
}

card_top = 0.955
card_bottom = 0.070
card_gap = 0.022
available_height = card_top - card_bottom - card_gap * (len(CLASS_ORDER) - 1)
height_scale = available_height / sum(card_weights.values())
current_top = card_top

for pfas_class in CLASS_ORDER:
    compounds = class_compounds[pfas_class]
    color = CLASS_COLORS[pfas_class]
    card_height = card_weights[pfas_class] * height_scale
    card_y = current_top - card_height

    ax_classes.add_patch(FancyBboxPatch(
        (0.0, card_y), 1.0, card_height,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        transform=ax_classes.transAxes,
        facecolor=to_rgba(color, 0.085),
        edgecolor=to_rgba(color, 0.32),
        linewidth=0.65,
        clip_on=False,
    ))
    ax_classes.add_patch(Rectangle(
        (0.0, card_y), 0.018, card_height,
        transform=ax_classes.transAxes,
        facecolor=color,
        edgecolor="none",
        clip_on=False,
    ))

    header_y = current_top - 0.17 * card_height
    body_y = card_y + 0.12 * card_height
    ax_classes.text(
        0.055, header_y, pfas_class, transform=ax_classes.transAxes,
        ha="left", va="top", fontsize=8.1, fontweight="bold", color=color
    )
    ax_classes.text(
        0.95, header_y, f"n = {len(compounds)}", transform=ax_classes.transAxes,
        ha="right", va="top", fontsize=6.8, fontweight="bold", color="#666666"
    )
    ax_classes.text(
        0.055, body_y, "\n".join(wrapped_lines[pfas_class]),
        transform=ax_classes.transAxes, ha="left", va="bottom",
        fontsize=7.0, fontweight="bold", linespacing=1.38, color="#252525"
    )
    current_top = card_y - card_gap

ax_classes.text(
    0.0,
    0.012,
    "Classes are nominal; vertical order does not imply magnitude.",
    transform=ax_classes.transAxes,
    ha="left",
    va="bottom",
    fontsize=6.5,
    fontweight="bold",
    color="#666666",
)


# ============================================================
# 10. Export source data and figure files
# ============================================================

correlation_matrix.to_csv(
    OUTPUT_DIR / "PFAS_unique_compound_Pearson_correlation.csv",
    encoding="utf-8-sig",
)
compound_data[[PFAS_COLUMN, "PFAS class display", "Carbon Chain Length"]].to_csv(
    OUTPUT_DIR / "PFAS_verified_PFAS_classes.csv",
    index=False,
    encoding="utf-8-sig",
)
strong_pairs_table.to_csv(
    OUTPUT_DIR / "PFAS_descriptor_pairs_abs_r_ge_0.90.csv",
    index=False,
    encoding="utf-8-sig",
)

export_options = {
    "bbox_inches": "tight",
    "pad_inches": 0.035,
    "facecolor": "white",
}
fig.savefig(OUTPUT_STEM.with_suffix(".svg"), format="svg", **export_options)
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), format="pdf", **export_options)
fig.savefig(
    OUTPUT_STEM.with_suffix(".tiff"),
    format="tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
    **export_options,
)
fig.savefig(
    OUTPUT_STEM.with_suffix(".png"), format="png", dpi=300, **export_options
)

print("\nSaved figure files:")
for extension in [".svg", ".pdf", ".tiff", ".png"]:
    print(OUTPUT_STEM.with_suffix(extension))

if SHOW_FIGURE:
    plt.show()

plt.close(fig)
