import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error
)


# ============================================================
# 1. Paths
# ============================================================

input_file = (
    r"D:\python\pythonProject1"
    r"\Machine-learning original dataset.xlsx"
)

output_dir = (
    r"D:\python\pythonProject1"
    r"\机器学习新 202608\no_chain_PFASclass"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# 2. Read dataset
# ============================================================

data = pd.read_excel(
    input_file
)

# Clean possible spaces in column names
data.columns = (
    data.columns
    .astype(str)
    .str.strip()
)


print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    "Dataset shape:",
    data.shape
)

print(
    "Studies:",
    data["Reference"].nunique()
)

print(
    "PFAS compounds:",
    data["PFASs Name"].nunique()
)


# ============================================================
# 3. Study × PFAS grouping
# ============================================================

data["Study_Compound_ID"] = (
    data["Reference"].astype(str)
    + "_"
    + data["PFASs Name"].astype(str)
)

groups = (
    data["Study_Compound_ID"]
)

print(
    "Study × PFAS groups:",
    groups.nunique()
)


# ============================================================
# 4. Target
# ============================================================

target = (
    "Log PFASs concentration"
)

y = (
    data[target]
    .copy()
)


# ============================================================
# 5. Final predictor set used for algorithm screening
#
# This screening uses the same 13 original predictors as the
# final XGBoost model. Candidate algorithms use fixed settings
# here; final XGBoost performance is reported separately from
# nested hyperparameter optimization.

features = [

    # Plant
    "Plant tissue group classify",
    "Morphotypes",

    # Environmental / exposure
    "Soil pH",
    "SOM",
    "Exposure Concentration",
    "Growth Duration",
    "Growth Temperature",

    # Final molecular descriptors
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi"
]


# ============================================================
# 6. Quality check
# ============================================================

required_columns = (
    features
    + [
        target,
        "Reference",
        "PFASs Name",
        "Study_Compound_ID"
    ]
)

missing_columns = [
    col
    for col in required_columns
    if col not in data.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(
            missing_columns
        )
    )


missing_values = (
    data[features + [target]]
    .isna()
    .sum()
)

missing_values = (
    missing_values[
        missing_values > 0
    ]
)

if len(missing_values) > 0:

    print(
        "\nWARNING: Missing values detected:"
    )

    print(
        missing_values
    )

    raise ValueError(
        "Please resolve missing values "
        "before model comparison."
    )


X = (
    data[features]
    .copy()
)


# ============================================================
# 7. Categorical variables
# ============================================================

categorical_features = [
    "Plant tissue group classify",
    "Morphotypes"
]

numerical_features = [
    col
    for col in features
    if col not in categorical_features
]


# ============================================================
# 8. Preprocessor generator
#
# A new preprocessor is created for every model.
# ============================================================

def make_preprocessor():

    return ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),

                categorical_features
            ),

            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ],

        verbose_feature_names_out=False
    )


# ============================================================
# 9. Candidate models
#
# IMPORTANT:
# These are the INITIAL screening settings.
# They are not the final nested-optimized models.
# ============================================================

models = {

    "Random Forest":

        RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=2026,
            n_jobs=-1
        ),


    "Gradient Boosting":

        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=2026
        ),


    "LightGBM":

        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=2026,
            verbosity=-1,
            n_jobs=-1
        ),


    "XGBoost":

        XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=2026,
            n_jobs=-1
        )
}


model_order = [
    "Random Forest",
    "Gradient Boosting",
    "LightGBM",
    "XGBoost"
]


# ============================================================
# 10. Create EXACTLY the same five grouped folds
#     for all four models
# ============================================================

cv = GroupKFold(
    n_splits=5
)

dummy_X = np.zeros(
    (len(data), 1)
)

splits = list(

    cv.split(
        dummy_X,
        y,
        groups=groups
    )
)


# ============================================================
# 11. Leakage check
# ============================================================

for fold, (
    train_idx,
    test_idx
) in enumerate(
    splits,
    start=1
):

    train_groups = set(
        groups.iloc[
            train_idx
        ]
    )

    test_groups = set(
        groups.iloc[
            test_idx
        ]
    )

    overlap = (
        train_groups
        .intersection(
            test_groups
        )
    )

    if len(overlap) > 0:

        raise ValueError(
            f"Group leakage detected "
            f"in Fold {fold}"
        )


print(
    "\nGroup leakage check passed."
)


# ============================================================
# 12. Storage
# ============================================================

fold_results = []

overall_results = []

oof_predictions = pd.DataFrame({

    "Observed":
        y,

    "Reference":
        data["Reference"],

    "PFASs Name":
        data["PFASs Name"],

    "Study_Compound_ID":
        groups
})


# ============================================================
# 13. Fit four candidate models
# ============================================================

for model_name in model_order:

    model = models[
        model_name
    ]

    print("\n")
    print("=" * 72)
    print(
        "MODEL:",
        model_name
    )
    print("=" * 72)


    pipeline = Pipeline(

        steps=[

            (
                "preprocess",
                make_preprocessor()
            ),

            (
                "model",
                model
            )
        ]
    )


    # OOF predictions
    y_oof = np.full(
        len(data),
        np.nan
    )


    # --------------------------------------------------------
    # Fold-by-fold evaluation
    # --------------------------------------------------------

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        splits,
        start=1
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]


        pipeline.fit(
            X_train,
            y_train
        )


        y_pred = (
            pipeline.predict(
                X_test
            )
        )


        y_oof[
            test_idx
        ] = y_pred


        fold_r2 = (
            r2_score(
                y_test,
                y_pred
            )
        )


        fold_rmse = np.sqrt(

            mean_squared_error(
                y_test,
                y_pred
            )
        )


        fold_results.append({

            "Model":
                model_name,

            "Fold":
                fold,

            "N_test":
                len(test_idx),

            "N_test_groups":
                groups.iloc[
                    test_idx
                ].nunique(),

            "R2":
                fold_r2,

            "RMSE":
                fold_rmse
        })


        print(

            f"Fold {fold}: "

            f"R² = "
            f"{fold_r2:.3f}, "

            f"RMSE = "
            f"{fold_rmse:.3f}"
        )


    # --------------------------------------------------------
    # Overall OOF performance
    # --------------------------------------------------------

    if np.isnan(
        y_oof
    ).any():

        raise ValueError(
            f"Missing OOF predictions "
            f"for {model_name}"
        )


    overall_r2 = (
        r2_score(
            y,
            y_oof
        )
    )


    overall_rmse = np.sqrt(

        mean_squared_error(
            y,
            y_oof
        )
    )


    overall_results.append({

        "Model":
            model_name,

        "Overall_OOF_R2":
            overall_r2,

        "Overall_OOF_RMSE":
            overall_rmse
    })


    oof_predictions[
        model_name
    ] = y_oof


    print(
        "-" * 72
    )

    print(
        f"Overall OOF R²   = "
        f"{overall_r2:.3f}"
    )

    print(
        f"Overall OOF RMSE = "
        f"{overall_rmse:.3f}"
    )


# ============================================================
# 14. Tables
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

overall_df = pd.DataFrame(
    overall_results
)


summary_df = (

    fold_df
    .groupby(
        "Model",
        as_index=False
    )
    .agg(

        Mean_fold_R2=(
            "R2",
            "mean"
        ),

        SD_fold_R2=(
            "R2",
            "std"
        ),

        Mean_fold_RMSE=(
            "RMSE",
            "mean"
        ),

        SD_fold_RMSE=(
            "RMSE",
            "std"
        )
    )
)


summary_df = (
    overall_df
    .merge(
        summary_df,
        on="Model",
        how="left"
    )
)


# Preserve desired model order
summary_df["Model"] = pd.Categorical(
    summary_df["Model"],
    categories=model_order,
    ordered=True
)

summary_df = (
    summary_df
    .sort_values("Model")
    .reset_index(drop=True)
)


print("\n")
print("=" * 105)
print(
    "INITIAL MODEL SCREENING SUMMARY"
)
print("=" * 105)

print(
    summary_df.to_string(
        index=False
    )
)

print("=" * 105)


# ============================================================
# 15. Save numerical results
# ============================================================

fold_df.to_csv(

    os.path.join(
        output_dir,
        "FigS_Model_screening_fold_results.csv"
    ),

    index=False
)


summary_df.to_csv(

    os.path.join(
        output_dir,
        "FigS_Model_screening_summary.csv"
    ),

    index=False
)


oof_predictions.to_csv(

    os.path.join(
        output_dir,
        "FigS_Model_screening_OOF_predictions.csv"
    ),

    index=False
)


# ============================================================
# 16. Figure style
#
# Arial, bold, large fonts, NO GRID
# ============================================================

mpl.rcParams.update({

    "font.family":
        "Arial",

    "font.weight":
        "bold",

    "font.size":
        12,

    "axes.labelsize":
        15,

    "axes.labelweight":
        "bold",

    "xtick.labelsize":
        12,

    "ytick.labelsize":
        12,

    "axes.linewidth":
        1.25,

    "xtick.major.width":
        1.25,

    "ytick.major.width":
        1.25,

    "xtick.major.size":
        6,

    "ytick.major.size":
        6,

    "xtick.direction":
        "out",

    "ytick.direction":
        "out"
})


# ============================================================
# 17. Model colors
#
# Use Matplotlib's default color cycle rather than
# hard-coding a decorative palette.
# ============================================================

default_colors = (
    mpl.rcParams[
        "axes.prop_cycle"
    ]
    .by_key()["color"]
)

model_colors = {

    model:
        default_colors[i]

    for i, model
    in enumerate(
        model_order
    )
}


# ============================================================
# 18. Create 2-panel figure
# ============================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(
        12.0,
        5.5
    )
)


# Base x positions
x_base = np.arange(
    len(model_order)
)


# Small deterministic offsets:
# each offset corresponds to the SAME fold across models
fold_offsets = np.linspace(
    -0.14,
    0.14,
    5
)


# ============================================================
# 19. Helper function
# ============================================================

def draw_panel(
    ax,
    metric,
    ylabel,
    panel_label
):

    # --------------------------------------------------------
    # A. Paired fold lines
    #
    # Each gray line represents the SAME grouped-CV fold
    # across all four algorithms.
    # --------------------------------------------------------

    for fold in range(
        1,
        6
    ):

        fold_values = []

        fold_x = []

        offset = (
            fold_offsets[
                fold - 1
            ]
        )


        for model_idx, model_name in enumerate(
            model_order
        ):

            value = (
                fold_df[
                    (
                        fold_df["Model"]
                        == model_name
                    )
                    &
                    (
                        fold_df["Fold"]
                        == fold
                    )
                ][metric]
                .iloc[0]
            )


            fold_values.append(
                value
            )

            fold_x.append(
                x_base[
                    model_idx
                ]
                + offset
            )


        ax.plot(

            fold_x,
            fold_values,

            color="0.82",

            linewidth=1.15,

            alpha=0.80,

            zorder=1
        )


    # --------------------------------------------------------
    # B. Individual fold points
    # --------------------------------------------------------

    for model_idx, model_name in enumerate(
        model_order
    ):

        model_fold = (

            fold_df[
                fold_df["Model"]
                == model_name
            ]
            .sort_values(
                "Fold"
            )
        )


        x_fold = (
            x_base[
                model_idx
            ]
            + fold_offsets
        )


        ax.scatter(

            x_fold,

            model_fold[
                metric
            ],

            s=72,

            color=
                model_colors[
                    model_name
                ],

            alpha=0.78,

            edgecolors="white",

            linewidths=0.65,

            zorder=3
        )


        # ----------------------------------------------------
        # C. Mean ± SD
        # ----------------------------------------------------

        mean_value = (
            model_fold[
                metric
            ].mean()
        )

        sd_value = (
            model_fold[
                metric
            ].std()
        )


        ax.errorbar(

            x_base[
                model_idx
            ],

            mean_value,

            yerr=
                sd_value,

            fmt="D",

            markersize=9.5,

            color=
                model_colors[
                    model_name
                ],

            markeredgecolor=
                "0.15",

            markeredgewidth=1.0,

            elinewidth=2.0,

            capsize=5,

            capthick=1.8,

            zorder=5
        )


    # --------------------------------------------------------
    # D. Axes
    # --------------------------------------------------------

    ax.set_xticks(
        x_base
    )

    ax.set_xticklabels(
        [
            "Random\nForest",
            "Gradient\nBoosting",
            "LightGBM",
            "XGBoost"
        ],
        fontweight="bold"
    )


    ax.set_ylabel(
        ylabel,
        fontsize=16,
        fontweight="bold"
    )


    # NO GRID
    ax.grid(False)


    # Clean frame
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    # Bold tick labels
    for tick in (
        ax.get_xticklabels()
    ):

        tick.set_fontweight(
            "bold"
        )


    for tick in (
        ax.get_yticklabels()
    ):

        tick.set_fontweight(
            "bold"
        )


    ax.tick_params(
        axis="both",
        which="major",
        pad=7
    )


    # --------------------------------------------------------
    # E. Panel label
    # --------------------------------------------------------

    ax.text(

        -0.13,
        1.04,

        panel_label,

        transform=
            ax.transAxes,

        fontsize=20,

        fontweight=
            "bold",

        ha="left",

        va="bottom"
    )


    # --------------------------------------------------------
    # F. Automatic y range with padding
    # --------------------------------------------------------

    all_values = (
        fold_df[
            metric
        ].values
    )

    y_min = (
        np.nanmin(
            all_values
        )
    )

    y_max = (
        np.nanmax(
            all_values
        )
    )

    y_range = (
        y_max
        - y_min
    )


    if y_range == 0:

        y_range = 1


    ax.set_ylim(

        y_min
        - 0.20
        * y_range,

        y_max
        + 0.30
        * y_range
    )


# ============================================================
# 20. Panel (a): R²
# ============================================================

draw_panel(

    axes[0],

    metric="R2",

    ylabel=r"Grouped-CV $R^2$",

    panel_label="(a)"
)


# ============================================================
# 21. Panel (b): RMSE
# ============================================================

draw_panel(

    axes[1],

    metric="RMSE",

    ylabel="Grouped-CV RMSE",

    panel_label="(b)"
)


# ============================================================
# 22. Layout
# ============================================================

plt.subplots_adjust(

    left=0.09,

    right=0.98,

    bottom=0.20,

    top=0.92,

    wspace=0.34
)


# ============================================================
# 23. Save figure
# ============================================================

tiff_file = os.path.join(
    output_dir,
    "FigS_Model_screening_4_algorithms.tiff"
)

png_file = os.path.join(
    output_dir,
    "FigS_Model_screening_4_algorithms.png"
)

pdf_file = os.path.join(
    output_dir,
    "FigS_Model_screening_4_algorithms.pdf"
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


print("\n")
print("=" * 70)
print("FIGURE SAVED")
print("=" * 70)

print(tiff_file)
print(png_file)
print(pdf_file)