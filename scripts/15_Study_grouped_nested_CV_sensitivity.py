import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor


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

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Read data
# ============================================================

data = pd.read_excel(input_file)
data = data.reset_index(drop=True)

print("Dataset shape:", data.shape)
print("Number of studies:", data["Reference"].nunique())
print("Number of PFAS compounds:", data["PFASs Name"].nunique())


# ============================================================
# 3. Target and predictors
# ============================================================

target = "Log PFASs concentration"

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

required_columns = [
    "Reference",
    "PFASs Name",
    target
] + features

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if data[required_columns].isnull().any().any():
    missing_counts = (
        data[required_columns]
        .isnull()
        .sum()
    )

    missing_counts = missing_counts[
        missing_counts > 0
    ]

    raise ValueError(
        "Missing values were detected in the analysis dataset:\n"
        f"{missing_counts}"
    )


X = data[features].copy()

y = pd.to_numeric(
    data[target],
    errors="raise"
)


# ============================================================
# 4. Study-level grouping
#
# Key difference:
# The grouping variable is Reference only.
# All observations from the same study remain in one fold.
# ============================================================

study_groups = (
    data["Reference"]
    .astype(str)
    .str.strip()
)

number_of_studies = study_groups.nunique()

print("Study groups used for validation:", number_of_studies)

if number_of_studies < 5:
    raise ValueError(
        "At least five independent studies are required "
        "for five-fold study-grouped cross-validation."
    )


# ============================================================
# 5. Categorical and numerical variables
# ============================================================

categorical_features = [
    "Plant tissue group classify",
    "Morphotypes"
]

numerical_features = [
    column
    for column in features
    if column not in categorical_features
]


# ============================================================
# 6. Preprocessing
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
# 7. XGBoost model
# ============================================================

xgb = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=2026,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", xgb)
    ]
)


# ============================================================
# 8. Same hyperparameter search space as the primary analysis
# ============================================================

param_distributions = {
    "model__n_estimators": [
        200, 300, 400, 500, 700, 900
    ],
    "model__learning_rate": [
        0.01, 0.02, 0.03, 0.05, 0.08, 0.10
    ],
    "model__max_depth": [
        2, 3, 4, 5, 6
    ],
    "model__min_child_weight": [
        1, 2, 3, 5, 8, 10
    ],
    "model__subsample": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],
    "model__colsample_bytree": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],
    "model__reg_alpha": [
        0, 0.01, 0.05, 0.10, 0.50, 1.0
    ],
    "model__reg_lambda": [
        0.1, 0.5, 1.0, 2.0, 5.0, 10.0
    ]
}


# ============================================================
# 9. Outer study-grouped cross-validation
# ============================================================

outer_cv = GroupKFold(n_splits=5)

outer_splits = list(
    outer_cv.split(
        X,
        y,
        groups=study_groups
    )
)

oof_predictions = np.full(
    len(data),
    np.nan
)

outer_fold_ids = np.zeros(
    len(data),
    dtype=int
)

outer_results = []
best_params_list = []


# ============================================================
# 10. Nested study-grouped cross-validation
# ============================================================

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_splits,
    start=1
):

    print("\n" + "=" * 80)
    print(f"STUDY-GROUPED OUTER FOLD {outer_fold} / 5")
    print("=" * 80)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = study_groups.iloc[train_idx]
    groups_test = study_groups.iloc[test_idx]

    # --------------------------------------------------------
    # Check study-level leakage
    # --------------------------------------------------------

    overlapping_studies = set(
        groups_train
    ).intersection(
        set(groups_test)
    )

    if overlapping_studies:
        raise RuntimeError(
            f"Study leakage detected in outer fold "
            f"{outer_fold}: {overlapping_studies}"
        )

    print("Training observations:", len(train_idx))
    print("Test observations:", len(test_idx))
    print("Training studies:", groups_train.nunique())
    print("Test studies:", groups_test.nunique())

    if groups_train.nunique() < 4:
        raise ValueError(
            f"Outer fold {outer_fold} contains fewer than "
            "four training studies, so four-fold inner "
            "GroupKFold cannot be performed."
        )

    # --------------------------------------------------------
    # Inner study-grouped CV for hyperparameter optimization
    # --------------------------------------------------------

    inner_cv = GroupKFold(n_splits=4)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        random_state=2026 + outer_fold,
        n_jobs=1,
        verbose=1,
        refit=True,
        return_train_score=False
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)

    oof_predictions[test_idx] = y_pred
    outer_fold_ids[test_idx] = outer_fold

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

    print("\nOUTER TEST PERFORMANCE")
    print(f"R²   = {fold_r2:.3f}")
    print(f"RMSE = {fold_rmse:.3f}")

    outer_results.append({
        "Outer_Fold": outer_fold,
        "N_train": len(train_idx),
        "N_test": len(test_idx),
        "N_train_studies": groups_train.nunique(),
        "N_test_studies": groups_test.nunique(),
        "Inner_best_RMSE": -search.best_score_,
        "Outer_R2": fold_r2,
        "Outer_RMSE": fold_rmse
    })

    cleaned_parameters = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }

    cleaned_parameters["Outer_Fold"] = outer_fold

    best_params_list.append(
        cleaned_parameters
    )


# ============================================================
# 11. Final checks
# ============================================================

if np.isnan(oof_predictions).any():
    raise RuntimeError(
        "Some observations did not receive an "
        "out-of-fold prediction."
    )

if (outer_fold_ids == 0).any():
    raise RuntimeError(
        "Some observations were not assigned "
        "to an outer fold."
    )

fold_assignment_check = pd.DataFrame({
    "Study_ID": study_groups,
    "Outer_Fold": outer_fold_ids
})

study_fold_counts = (
    fold_assignment_check
    .groupby("Study_ID")["Outer_Fold"]
    .nunique()
)

if study_fold_counts.max() != 1:
    raise RuntimeError(
        "Study leakage detected: at least one study "
        "appears in more than one outer test fold."
    )

print("\nStudy leakage check: PASSED")
print(
    "Every study was assigned to exactly one "
    "outer test fold."
)


# ============================================================
# 12. Overall pooled OOF performance
# ============================================================

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

outer_results_df = pd.DataFrame(
    outer_results
)

best_params_df = pd.DataFrame(
    best_params_list
)


print("\n" + "=" * 80)
print("FINAL STUDY-GROUPED NESTED-CV PERFORMANCE")
print("=" * 80)
print(f"Observations       = {len(data)}")
print(f"Independent studies = {number_of_studies}")
print(f"Overall OOF R²     = {overall_r2:.6f}")
print(f"Overall OOF RMSE   = {overall_rmse:.6f}")
print("=" * 80)


# ============================================================
# 13. Save fold-level results
# ============================================================

outer_results_file = os.path.join(
    output_dir,
    "Study_grouped_Nested_XGBoost_outer_fold_results.csv"
)

outer_results_df.to_csv(
    outer_results_file,
    index=False
)


# ============================================================
# 14. Save best parameters
# ============================================================

best_parameters_file = os.path.join(
    output_dir,
    "Study_grouped_Nested_XGBoost_best_parameters.csv"
)

best_params_df.to_csv(
    best_parameters_file,
    index=False
)


# ============================================================
# 15. Save OOF predictions and fold assignments
# ============================================================

oof_results = pd.DataFrame({
    "Observed": y,
    "Predicted": oof_predictions,
    "Residual": y - oof_predictions,
    "Outer_Fold": outer_fold_ids,
    "Reference": data["Reference"],
    "PFASs Name": data["PFASs Name"],
    "Study_ID": study_groups
})

oof_predictions_file = os.path.join(
    output_dir,
    "Study_grouped_Nested_XGBoost_OOF_predictions.csv"
)

oof_results.to_csv(
    oof_predictions_file,
    index=False
)


# ============================================================
# 16. Save performance summary
# ============================================================

summary_df = pd.DataFrame({
    "Metric": [
        "Number of observations",
        "Number of studies",
        "Overall OOF R2",
        "Overall OOF RMSE",
        "Mean outer-fold R2",
        "SD outer-fold R2",
        "Mean outer-fold RMSE",
        "SD outer-fold RMSE"
    ],
    "Value": [
        len(data),
        number_of_studies,
        overall_r2,
        overall_rmse,
        outer_results_df["Outer_R2"].mean(),
        outer_results_df["Outer_R2"].std(),
        outer_results_df["Outer_RMSE"].mean(),
        outer_results_df["Outer_RMSE"].std()
    ]
})

summary_file = os.path.join(
    output_dir,
    "Study_grouped_Nested_XGBoost_performance_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 17. Compare with the existing Study-PFAS grouped result
# ============================================================

primary_summary_file = os.path.join(
    output_dir,
    "Nested_XGBoost_performance_summary.csv"
)

if os.path.exists(primary_summary_file):

    primary_summary = pd.read_csv(
        primary_summary_file
    )

    primary_metrics = (
        primary_summary
        .set_index("Metric")["Value"]
    )

    primary_r2 = float(
        primary_metrics["Overall OOF R2"]
    )

    primary_rmse = float(
        primary_metrics["Overall OOF RMSE"]
    )

    comparison_df = pd.DataFrame({
        "Validation_scheme": [
            "Study-PFAS grouped nested CV",
            "Study-grouped nested CV"
        ],
        "Overall_OOF_R2": [
            primary_r2,
            overall_r2
        ],
        "Overall_OOF_RMSE": [
            primary_rmse,
            overall_rmse
        ]
    })

    comparison_file = os.path.join(
        output_dir,
        "Study_grouped_vs_Study_PFAS_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    print("\nVALIDATION-SCHEME COMPARISON")
    print(comparison_df.to_string(index=False))

else:

    print(
        "\nExisting Study-PFAS performance summary "
        "was not found, so the comparison table "
        "was not generated."
    )


# ============================================================
# 18. Output locations
# ============================================================

print("\nResults saved:")
print(outer_results_file)
print(best_parameters_file)
print(oof_predictions_file)
print(summary_file)