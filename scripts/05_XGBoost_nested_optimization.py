import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

from xgboost import XGBRegressor


# ============================================================
# 1. Paths
# ============================================================

input_file = (
    r"D:\python\pythonProject1"
    r"\Machine-learning original dataset.xlsx"
)

OUTPUT_DIR = Path(
    r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"
)

output_dir = str(OUTPUT_DIR)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# 2. Read data
# ============================================================

data = pd.read_excel(
    input_file
)

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
# 3. Study × Compound grouping
# ============================================================

data["Study_Compound_ID"] = (
    data["Reference"].astype(str)
    + "_"
    + data["PFASs Name"].astype(str)
)

groups = data[
    "Study_Compound_ID"
]

print(
    "Study × Compound groups:",
    groups.nunique()
)


# ============================================================
# 4. Target
# ============================================================

target = (
    "Log PFASs concentration"
)

y = data[target]


# ============================================================
# 5. Final candidate predictor set
#
# 15-variable Reduced Full model
# ============================================================

features = [

    # Plant / biological
    "Plant tissue group classify",
    "Morphotypes",

    # Environmental / experimental
    "Soil pH",
    "SOM",
    "Exposure Concentration",
    "Growth Duration",
    "Growth Temperature",

    # Selected molecular descriptors
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi"
]


X = data[
    features
].copy()


# ============================================================
# 6. Categorical and numerical variables
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
# 7. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(

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
    ]
)


# ============================================================
# 8. Base XGBoost model
#
# tree_method="hist" improves computational efficiency.
# ============================================================

xgb = XGBRegressor(

    objective="reg:squarederror",

    tree_method="hist",

    random_state=2026,

    n_jobs=-1
)


# ============================================================
# 9. Pipeline
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocess",
            preprocessor
        ),

        (
            "model",
            xgb
        )
    ]
)


# ============================================================
# 10. Hyperparameter search space
#
# These parameters control:
#
# n_estimators      -> number of boosting trees
# learning_rate     -> contribution of each tree
# max_depth         -> tree complexity
# min_child_weight  -> minimum child-node requirement
# subsample         -> row sampling
# colsample_bytree  -> feature sampling
# reg_alpha         -> L1 regularization
# reg_lambda        -> L2 regularization
# ============================================================

param_distributions = {

    "model__n_estimators": [
        200,
        300,
        400,
        500,
        700,
        900
    ],

    "model__learning_rate": [
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.10
    ],

    "model__max_depth": [
        2,
        3,
        4,
        5,
        6
    ],

    "model__min_child_weight": [
        1,
        2,
        3,
        5,
        8,
        10
    ],

    "model__subsample": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__reg_alpha": [
        0,
        0.01,
        0.05,
        0.1,
        0.5,
        1.0
    ],

    "model__reg_lambda": [
        0.1,
        0.5,
        1,
        2,
        5,
        10
    ]
}


# ============================================================
# 11. OUTER grouped CV
#
# The outer folds are used ONLY for final performance
# evaluation.
# ============================================================

outer_cv = GroupKFold(
    n_splits=5
)


outer_splits = list(

    outer_cv.split(
        X,
        y,
        groups=groups
    )
)


# ============================================================
# 12. Storage
# ============================================================

oof_predictions = np.full(
    len(data),
    np.nan
)

outer_results = []

best_params_list = []


# ============================================================
# 13. Nested cross-validation
# ============================================================

for outer_fold, (
    train_idx,
    test_idx
) in enumerate(
    outer_splits,
    start=1
):

    print("\n")
    print("=" * 80)
    print(
        f"OUTER FOLD {outer_fold} / 5"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Outer training / testing data
    # --------------------------------------------------------

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


    groups_train = groups.iloc[
        train_idx
    ]

    groups_test = groups.iloc[
        test_idx
    ]


    # --------------------------------------------------------
    # Check group leakage
    # --------------------------------------------------------

    overlap = set(
        groups_train
    ).intersection(
        set(groups_test)
    )

    if len(overlap) > 0:

        raise ValueError(
            f"Group leakage detected "
            f"in outer fold {outer_fold}"
        )


    print(
        "Training observations:",
        len(train_idx)
    )

    print(
        "Test observations:",
        len(test_idx)
    )

    print(
        "Training groups:",
        groups_train.nunique()
    )

    print(
        "Test groups:",
        groups_test.nunique()
    )


    # ========================================================
    # 14. INNER grouped CV
    #
    # Used only for hyperparameter tuning.
    # ========================================================

    inner_cv = GroupKFold(
        n_splits=4
    )


    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=
            param_distributions,

        # Number of random parameter combinations
        n_iter=50,

        # Optimize RMSE
        scoring=
            "neg_root_mean_squared_error",

        cv=inner_cv,

        random_state=(
            2026
            + outer_fold
        ),

        n_jobs=1,

        verbose=1,

        refit=True,

        return_train_score=False
    )


    # --------------------------------------------------------
    # Hyperparameter search
    #
    # IMPORTANT:
    # groups are supplied to the inner GroupKFold.
    # --------------------------------------------------------

    search.fit(

        X_train,

        y_train,

        groups=groups_train
    )


    # --------------------------------------------------------
    # Best parameters for this outer fold
    # --------------------------------------------------------

    best_params = (
        search.best_params_
    )


    print(
        "\nBest inner-CV RMSE:",
        f"{-search.best_score_:.4f}"
    )

    print(
        "\nBest parameters:"
    )

    for key, value in (
        best_params.items()
    ):

        print(
            f"{key}: {value}"
        )


    # --------------------------------------------------------
    # Predict completely held-out outer test fold
    # --------------------------------------------------------

    best_model = (
        search.best_estimator_
    )


    y_pred = (
        best_model.predict(
            X_test
        )
    )


    oof_predictions[
        test_idx
    ] = y_pred


    # --------------------------------------------------------
    # Outer-fold performance
    # --------------------------------------------------------

    fold_r2 = r2_score(
        y_test,
        y_pred
    )


    fold_rmse = np.sqrt(

        mean_squared_error(
            y_test,
            y_pred
        )
    )


    print(
        "\nOUTER TEST PERFORMANCE"
    )

    print(
        f"R²   = {fold_r2:.3f}"
    )

    print(
        f"RMSE = {fold_rmse:.3f}"
    )


    # --------------------------------------------------------
    # Store fold results
    # --------------------------------------------------------

    outer_results.append({

        "Outer_Fold":
            outer_fold,

        "N_train":
            len(train_idx),

        "N_test":
            len(test_idx),

        "N_train_groups":
            groups_train.nunique(),

        "N_test_groups":
            groups_test.nunique(),

        "Inner_best_RMSE":
            -search.best_score_,

        "Outer_R2":
            fold_r2,

        "Outer_RMSE":
            fold_rmse
    })


    # --------------------------------------------------------
    # Store best hyperparameters
    # --------------------------------------------------------

    params_clean = {

        key.replace(
            "model__",
            ""
        ):
        value

        for key, value
        in best_params.items()
    }


    params_clean[
        "Outer_Fold"
    ] = outer_fold


    best_params_list.append(
        params_clean
    )


# ============================================================
# 15. Final nested OOF performance
# ============================================================

if np.isnan(
    oof_predictions
).any():

    raise ValueError(
        "Some observations did not "
        "receive an OOF prediction."
    )


overall_r2 = r2_score(
    y,
    oof_predictions
)


overall_rmse = np.sqrt(

    mean_squared_error(
        y,
        oof_predictions
    )
)


print("\n")
print("=" * 80)
print(
    "FINAL NESTED GROUPED-CV PERFORMANCE"
)
print("=" * 80)

print(
    f"Overall OOF R²   = "
    f"{overall_r2:.3f}"
)

print(
    f"Overall OOF RMSE = "
    f"{overall_rmse:.3f}"
)

print("=" * 80)


# ============================================================
# 16. Save fold-level results
# ============================================================

outer_results_df = (
    pd.DataFrame(
        outer_results
    )
)


outer_results_df.to_csv(

    os.path.join(
        output_dir,
        "Nested_XGBoost_outer_fold_results.csv"
    ),

    index=False
)


# ============================================================
# 17. Save best parameters from each outer fold
# ============================================================

best_params_df = (
    pd.DataFrame(
        best_params_list
    )
)


best_params_df.to_csv(

    os.path.join(
        output_dir,
        "Nested_XGBoost_best_parameters.csv"
    ),

    index=False
)


# ============================================================
# 18. Save OOF predictions
# ============================================================

oof_df = pd.DataFrame({

    "Observed":
        y,

    "Predicted":
        oof_predictions,

    "Residual":
        y - oof_predictions,

    "Reference":
        data["Reference"],

    "PFASs Name":
        data["PFASs Name"],

    "Study_Compound_ID":
        groups
})


oof_df.to_csv(

    os.path.join(
        output_dir,
        "Nested_XGBoost_OOF_predictions.csv"
    ),

    index=False
)


# ============================================================
# 19. Save overall summary
# ============================================================

summary_df = pd.DataFrame({

    "Metric": [
        "Overall OOF R2",
        "Overall OOF RMSE",
        "Mean outer-fold R2",
        "SD outer-fold R2",
        "Mean outer-fold RMSE",
        "SD outer-fold RMSE"
    ],

    "Value": [

        overall_r2,

        overall_rmse,

        outer_results_df[
            "Outer_R2"
        ].mean(),

        outer_results_df[
            "Outer_R2"
        ].std(),

        outer_results_df[
            "Outer_RMSE"
        ].mean(),

        outer_results_df[
            "Outer_RMSE"
        ].std()
    ]
})


summary_df.to_csv(

    os.path.join(
        output_dir,
        "Nested_XGBoost_performance_summary.csv"
    ),

    index=False
)


print(
    "\nResults saved to:"
)

print(
    output_dir
)
