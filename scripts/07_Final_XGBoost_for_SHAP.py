import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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

print("Dataset shape:", data.shape)
print("Studies:", data["Reference"].nunique())
print("PFAS compounds:", data["PFASs Name"].nunique())


# ============================================================
# 3. Study × Compound groups
# ============================================================

data["Study_Compound_ID"] = (
    data["Reference"].astype(str)
    + "_"
    + data["PFASs Name"].astype(str)
)

groups = data["Study_Compound_ID"]

print(
    "Study × Compound groups:",
    groups.nunique()
)


# ============================================================
# 4. Target variable
# ============================================================

target = "Log PFASs concentration"

y = data[target]


# ============================================================
# 5. Final Reduced Full predictor set
#
# 15 original predictors
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

    # Selected detailed molecular descriptors
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi"
]


X = data[features].copy()


# ============================================================
# 6. Categorical / numerical predictors
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
#
# The original Excel coding is retained.
# Categorical labels are converted to one-hot representation.
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
    ],

    verbose_feature_names_out=False
)


# ============================================================
# 8. Base XGBoost model
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
# Same search space as nested-CV analysis
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
# 11. Full-data grouped CV
#
# IMPORTANT:
# This CV is for final hyperparameter selection.
# It is NOT the final unbiased performance estimate.
#
# Final reported performance remains:
# Nested grouped-CV OOF R² = 0.643
# Nested grouped-CV OOF RMSE = 0.848
# ============================================================

group_cv = GroupKFold(
    n_splits=5
)


# ============================================================
# 12. Randomized hyperparameter search
#
# 100 candidates × 5 folds = approximately 500 fits
# ============================================================

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=
        param_distributions,

    n_iter=100,

    scoring=
        "neg_root_mean_squared_error",

    cv=group_cv,

    random_state=2026,

    n_jobs=1,

    verbose=1,

    refit=True,

    return_train_score=False
)


print("\n" + "=" * 80)
print("FINAL FULL-DATA GROUPED HYPERPARAMETER SEARCH")
print("=" * 80)

search.fit(

    X,
    y,

    groups=groups
)


# ============================================================
# 13. Best tuning result
# ============================================================

best_rmse = (
    -search.best_score_
)

best_params = (
    search.best_params_
)


print("\n")
print("=" * 80)
print("BEST FULL-DATA GROUPED-CV PARAMETERS")
print("=" * 80)

print(
    f"Best tuning CV RMSE = "
    f"{best_rmse:.4f}"
)

print(
    "\nBest parameters:"
)

for key, value in best_params.items():

    print(
        f"{key}: {value}"
    )

print("=" * 80)


# ============================================================
# 14. Best refitted pipeline
#
# RandomizedSearchCV(refit=True) automatically refits the
# selected parameter combination using ALL 825 observations.
# ============================================================

final_pipeline = (
    search.best_estimator_
)


# ============================================================
# 15. Extract preprocessing and XGBoost objects
# ============================================================

final_preprocessor = (
    final_pipeline.named_steps[
        "preprocess"
    ]
)

final_model = (
    final_pipeline.named_steps[
        "model"
    ]
)


# ============================================================
# 16. Transform the complete dataset
#
# This matrix will be used for SHAP in the next step.
# ============================================================

X_transformed = (
    final_preprocessor.transform(
        X
    )
)


# ============================================================
# 17. Get transformed feature names
#
# One-hot encoding means that the 15 original predictors
# become more than 15 model-input columns.
# ============================================================

transformed_feature_names = (
    final_preprocessor
    .get_feature_names_out()
)


print(
    "\nOriginal predictors:",
    len(features)
)

print(
    "Model-input features after one-hot encoding:",
    len(transformed_feature_names)
)


print(
    "\nTransformed feature names:"
)

for name in transformed_feature_names:

    print(name)


# ============================================================
# 18. Save transformed feature names
# ============================================================

feature_names_df = pd.DataFrame({

    "Model_feature":
        transformed_feature_names
})


feature_names_df.to_csv(

    os.path.join(
        output_dir,
        "Final_XGBoost_transformed_feature_names.csv"
    ),

    index=False
)


# ============================================================
# 19. Save best parameters
# ============================================================

clean_params = {

    key.replace(
        "model__",
        ""
    ):
    value

    for key, value
    in best_params.items()
}


best_params_df = pd.DataFrame(

    [
        {
            **clean_params,

            "Best_grouped_CV_RMSE":
                best_rmse
        }
    ]
)


best_params_df.to_csv(

    os.path.join(
        output_dir,
        "Final_XGBoost_best_parameters.csv"
    ),

    index=False
)


# Also save as JSON
with open(

    os.path.join(
        output_dir,
        "Final_XGBoost_best_parameters.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        {
            **clean_params,
            "Best_grouped_CV_RMSE":
                float(best_rmse)
        },
        f,
        indent=4
    )


# ============================================================
# 20. Save all RandomizedSearchCV candidate results
# ============================================================

cv_results = pd.DataFrame(
    search.cv_results_
)


# Convert negative RMSE to positive RMSE
cv_results[
    "Mean_CV_RMSE"
] = (
    -cv_results[
        "mean_test_score"
    ]
)


cv_results[
    "SD_CV_RMSE"
] = (
    cv_results[
        "std_test_score"
    ]
)


keep_columns = [

    "rank_test_score",

    "Mean_CV_RMSE",
    "SD_CV_RMSE",

    "mean_fit_time",
    "std_fit_time",

    "param_model__n_estimators",
    "param_model__learning_rate",
    "param_model__max_depth",
    "param_model__min_child_weight",
    "param_model__subsample",
    "param_model__colsample_bytree",
    "param_model__reg_alpha",
    "param_model__reg_lambda"
]


cv_results[
    keep_columns
].sort_values(

    by="rank_test_score"

).to_csv(

    os.path.join(
        output_dir,
        "Final_XGBoost_hyperparameter_search_results.csv"
    ),

    index=False
)


# ============================================================
# 21. Save final fitted pipeline
#
# This file contains:
# preprocessing + one-hot encoder + final XGBoost model
# ============================================================

pipeline_file = os.path.join(

    output_dir,

    "Final_XGBoost_pipeline.joblib"
)


joblib.dump(

    final_pipeline,

    pipeline_file
)


# ============================================================
# 22. Save transformed X matrix
#
# This makes the next SHAP script easier and reproducible.
# ============================================================

X_transformed_df = pd.DataFrame(

    X_transformed,

    columns=
        transformed_feature_names
)


X_transformed_df.to_csv(

    os.path.join(
        output_dir,
        "Final_XGBoost_transformed_X.csv"
    ),

    index=False
)


# ============================================================
# 23. Save metadata for later SHAP aggregation
# ============================================================

metadata = data[
    [
        "Reference",
        "PFASs Name",
        "Study_Compound_ID",
        target
    ]
].copy()


metadata.to_csv(

    os.path.join(
        output_dir,
        "Final_XGBoost_SHAP_metadata.csv"
    ),

    index=False
)


# ============================================================
# 24. Finished
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 80)

print(
    "Final performance used in manuscript:"
)

print(
    "Nested grouped-CV OOF R²   = 0.643"
)

print(
    "Nested grouped-CV OOF RMSE = 0.848"
)

print(
    "\nThe full-data model generated here "
    "is for SHAP interpretation."
)

print(
    "\nSaved to:"
)

print(
    output_dir
)

print(
    "\nPipeline file:"
)

print(
    pipeline_file
)