import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# 1. Paths
# ============================================================

input_file = r"D:\python\pythonProject1\Machine-learning original dataset.xlsx"
output_dir = r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Helper functions
# ============================================================

def calculate_rmse(y_true, y_pred):
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_r2_if_defined(y_true, y_pred):
    """Return R2 only when the held-out target has non-zero variation."""
    y_true_array = np.asarray(y_true, dtype=float)

    if len(y_true_array) < 2 or np.isclose(np.var(y_true_array), 0.0):
        return np.nan

    return float(r2_score(y_true_array, y_pred))


# ============================================================
# 3. Read and validate data
# ============================================================

data = pd.read_excel(input_file).reset_index(drop=True)

target = "Log PFASs concentration"

features = [
    # Plant / biological variables
    "Plant tissue group classify",
    "Morphotypes",

    # Environmental / experimental variables
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
    "SpMin8_Bhi",
]

required_columns = ["Reference", "PFASs Name", target] + features

missing_columns = [
    column for column in required_columns if column not in data.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if data[required_columns].isnull().any().any():
    missing_counts = data[required_columns].isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    raise ValueError(
        "Missing values were detected in the analysis dataset:\n"
        f"{missing_counts}"
    )

X = data[features].copy()
y = pd.to_numeric(data[target], errors="raise")

# Reference is the independent grouping unit for external validation.
study_groups = data["Reference"].astype(str).str.strip()

number_of_studies = int(study_groups.nunique())
number_of_compounds = int(data["PFASs Name"].nunique())

print("Dataset shape:", data.shape)
print("Number of studies:", number_of_studies)
print("Number of PFAS compounds:", number_of_compounds)

if number_of_studies < 5:
    raise ValueError(
        "At least five independent studies are required because the inner "
        "cross-validation uses four study-grouped folds."
    )


# ============================================================
# 4. Preprocessing
# ============================================================

categorical_features = [
    "Plant tissue group classify",
    "Morphotypes"
]

numerical_features = [
    column for column in features if column not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
        ("numerical", "passthrough", numerical_features),
    ]
)


# ============================================================
# 5. XGBoost pipeline and hyperparameter search space
# ============================================================

xgb = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=2026,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", xgb),
    ]
)

param_distributions = {
    "model__n_estimators": [200, 300, 400, 500, 700, 900],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__min_child_weight": [1, 2, 3, 5, 8, 10],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__reg_alpha": [0, 0.01, 0.05, 0.10, 0.50, 1.0],
    "model__reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
}


# ============================================================
# 6. Outer leave-one-study-out splits
# ============================================================

outer_cv = LeaveOneGroupOut()

outer_splits = list(
    outer_cv.split(
        X,
        y,
        groups=study_groups,
    )
)

if len(outer_splits) != number_of_studies:
    raise RuntimeError(
        "The number of LOSO outer folds does not equal the number of studies."
    )

oof_predictions = np.full(len(data), np.nan, dtype=float)
dummy_oof_predictions = np.full(len(data), np.nan, dtype=float)
outer_fold_ids = np.zeros(len(data), dtype=int)
held_out_study_ids = np.full(len(data), None, dtype=object)

outer_results = []
best_params_list = []


# ============================================================
# 7. Nested LOSO cross-validation
# ============================================================

for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
    print("\n" + "=" * 80)
    print(
        f"LEAVE-ONE-STUDY-OUT OUTER FOLD "
        f"{outer_fold} / {len(outer_splits)}"
    )
    print("=" * 80)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = study_groups.iloc[train_idx]
    groups_test = study_groups.iloc[test_idx]

    test_studies = groups_test.unique()

    if len(test_studies) != 1:
        raise RuntimeError(
            f"Outer fold {outer_fold} contains {len(test_studies)} test "
            "studies instead of exactly one."
        )

    test_study = str(test_studies[0])

    overlapping_studies = set(groups_train).intersection(set(groups_test))

    if overlapping_studies:
        raise RuntimeError(
            f"Study leakage detected in outer fold {outer_fold}: "
            f"{overlapping_studies}"
        )

    number_of_training_studies = int(groups_train.nunique())

    if number_of_training_studies < 4:
        raise ValueError(
            f"Outer fold {outer_fold} contains fewer than four training "
            "studies, so four-fold inner GroupKFold cannot be performed."
        )

    print("Held-out study:", test_study)
    print("Training observations:", len(train_idx))
    print("Test observations:", len(test_idx))
    print("Training studies:", number_of_training_studies)

    # Inner grouped CV prevents study leakage during hyperparameter tuning.
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
        return_train_score=False,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Training-mean baseline. This baseline uses no information from the
    # held-out study and therefore follows the same outer validation split.
    dummy_value = float(y_train.mean())
    dummy_pred = np.full(len(test_idx), dummy_value, dtype=float)

    oof_predictions[test_idx] = y_pred
    dummy_oof_predictions[test_idx] = dummy_pred
    outer_fold_ids[test_idx] = outer_fold
    held_out_study_ids[test_idx] = test_study

    model_r2 = calculate_r2_if_defined(y_test, y_pred)
    model_rmse = calculate_rmse(y_test, y_pred)
    model_mae = float(mean_absolute_error(y_test, y_pred))
    model_bias = float(np.mean(y_pred - y_test.to_numpy()))

    dummy_r2 = calculate_r2_if_defined(y_test, dummy_pred)
    dummy_rmse = calculate_rmse(y_test, dummy_pred)
    dummy_mae = float(mean_absolute_error(y_test, dummy_pred))

    print("\nOUTER TEST PERFORMANCE")
    print(f"Model R2   = {model_r2:.6f}")
    print(f"Model RMSE = {model_rmse:.6f}")
    print(f"Model MAE  = {model_mae:.6f}")
    print(f"Dummy RMSE = {dummy_rmse:.6f}")
    print(f"Dummy MAE  = {dummy_mae:.6f}")

    outer_results.append(
        {
            "Outer_Fold": outer_fold,
            "Held_out_study": test_study,
            "N_train": len(train_idx),
            "N_test": len(test_idx),
            "N_train_studies": number_of_training_studies,
            "Inner_best_RMSE": -float(search.best_score_),
            "Model_R2": model_r2,
            "Model_RMSE": model_rmse,
            "Model_MAE": model_mae,
            "Model_mean_bias": model_bias,
            "Dummy_R2": dummy_r2,
            "Dummy_RMSE": dummy_rmse,
            "Dummy_MAE": dummy_mae,
        }
    )

    cleaned_parameters = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }
    cleaned_parameters["Outer_Fold"] = outer_fold
    cleaned_parameters["Held_out_study"] = test_study
    best_params_list.append(cleaned_parameters)


# ============================================================
# 8. Leakage and completeness checks
# ============================================================

if np.isnan(oof_predictions).any():
    raise RuntimeError("Some observations did not receive a model OOF prediction.")

if np.isnan(dummy_oof_predictions).any():
    raise RuntimeError("Some observations did not receive a dummy OOF prediction.")

if (outer_fold_ids == 0).any():
    raise RuntimeError("Some observations were not assigned to an outer fold.")

fold_assignment_check = pd.DataFrame(
    {
        "Study_ID": study_groups,
        "Outer_Fold": outer_fold_ids,
        "Held_out_study": held_out_study_ids,
    }
)

study_fold_counts = (
    fold_assignment_check.groupby("Study_ID")["Outer_Fold"].nunique()
)

if study_fold_counts.max() != 1 or study_fold_counts.min() != 1:
    raise RuntimeError(
        "Study leakage or incomplete assignment detected: each study must "
        "appear in exactly one LOSO test fold."
    )

if set(held_out_study_ids) != set(study_groups.unique()):
    raise RuntimeError("Not every study was used exactly once as a held-out study.")

print("\nStudy leakage check: PASSED")
print("Every study was used exactly once as the complete outer test set.")


# ============================================================
# 9. Pooled observation-weighted OOF performance
# ============================================================

model_overall_r2 = float(r2_score(y, oof_predictions))
model_overall_rmse = calculate_rmse(y, oof_predictions)
model_overall_mae = float(mean_absolute_error(y, oof_predictions))

dummy_overall_r2 = float(r2_score(y, dummy_oof_predictions))
dummy_overall_rmse = calculate_rmse(y, dummy_oof_predictions)
dummy_overall_mae = float(mean_absolute_error(y, dummy_oof_predictions))

print("\n" + "=" * 80)
print("FINAL NESTED LEAVE-ONE-STUDY-OUT PERFORMANCE")
print("=" * 80)
print(f"Observations            = {len(data)}")
print(f"Independent studies     = {number_of_studies}")
print(f"Model pooled OOF R2     = {model_overall_r2:.6f}")
print(f"Model pooled OOF RMSE   = {model_overall_rmse:.6f}")
print(f"Model pooled OOF MAE    = {model_overall_mae:.6f}")
print(f"Dummy pooled OOF R2     = {dummy_overall_r2:.6f}")
print(f"Dummy pooled OOF RMSE   = {dummy_overall_rmse:.6f}")
print(f"Dummy pooled OOF MAE    = {dummy_overall_mae:.6f}")
print("=" * 80)


# ============================================================
# 10. Observation-level OOF output
# ============================================================

oof_results = pd.DataFrame(
    {
        "Observed": y,
        "Predicted": oof_predictions,
        "Dummy_predicted": dummy_oof_predictions,
        "Residual": y.to_numpy() - oof_predictions,
        "Absolute_error": np.abs(y.to_numpy() - oof_predictions),
        "Outer_Fold": outer_fold_ids,
        "Reference": data["Reference"],
        "PFASs_Name": data["PFASs Name"],
        "Study_ID": study_groups,
        "Held_out_study": held_out_study_ids,
    }
)


# ============================================================
# 11. Study-balanced performance summary
# ============================================================

study_level_rows = []

for study_id, subset in oof_results.groupby("Study_ID", sort=False):
    observed = subset["Observed"].to_numpy()
    predicted = subset["Predicted"].to_numpy()
    dummy_predicted = subset["Dummy_predicted"].to_numpy()

    study_level_rows.append(
        {
            "Study_ID": study_id,
            "N": len(subset),
            "Observed_mean": float(np.mean(observed)),
            "Observed_SD": float(np.std(observed, ddof=1))
            if len(observed) > 1
            else np.nan,
            "Predicted_mean": float(np.mean(predicted)),
            "Model_R2": calculate_r2_if_defined(observed, predicted),
            "Model_RMSE": calculate_rmse(observed, predicted),
            "Model_MAE": float(mean_absolute_error(observed, predicted)),
            "Model_mean_bias": float(np.mean(predicted - observed)),
            "Dummy_R2": calculate_r2_if_defined(observed, dummy_predicted),
            "Dummy_RMSE": calculate_rmse(observed, dummy_predicted),
            "Dummy_MAE": float(
                mean_absolute_error(observed, dummy_predicted)
            ),
        }
    )

study_level_df = pd.DataFrame(study_level_rows)

study_balanced_summary = {
    "Median_study_RMSE": float(study_level_df["Model_RMSE"].median()),
    "Q1_study_RMSE": float(study_level_df["Model_RMSE"].quantile(0.25)),
    "Q3_study_RMSE": float(study_level_df["Model_RMSE"].quantile(0.75)),
    "Median_study_MAE": float(study_level_df["Model_MAE"].median()),
    "Q1_study_MAE": float(study_level_df["Model_MAE"].quantile(0.25)),
    "Q3_study_MAE": float(study_level_df["Model_MAE"].quantile(0.75)),
    "Median_dummy_study_RMSE": float(
        study_level_df["Dummy_RMSE"].median()
    ),
    "Median_dummy_study_MAE": float(
        study_level_df["Dummy_MAE"].median()
    ),
}

print("\nSTUDY-BALANCED ERROR SUMMARY")
print(
    "Median study RMSE "
    f"= {study_balanced_summary['Median_study_RMSE']:.6f} "
    f"(Q1-Q3: {study_balanced_summary['Q1_study_RMSE']:.6f}-"
    f"{study_balanced_summary['Q3_study_RMSE']:.6f})"
)
print(
    "Median study MAE  "
    f"= {study_balanced_summary['Median_study_MAE']:.6f} "
    f"(Q1-Q3: {study_balanced_summary['Q1_study_MAE']:.6f}-"
    f"{study_balanced_summary['Q3_study_MAE']:.6f})"
)


# ============================================================
# 12. Save all LOSO outputs
# ============================================================

outer_results_df = pd.DataFrame(outer_results)
best_params_df = pd.DataFrame(best_params_list)

outer_results_file = os.path.join(
    output_dir,
    "LOSO_Nested_XGBoost_outer_study_results.csv",
)
best_parameters_file = os.path.join(
    output_dir,
    "LOSO_Nested_XGBoost_best_parameters.csv",
)
oof_predictions_file = os.path.join(
    output_dir,
    "LOSO_Nested_XGBoost_OOF_predictions.csv",
)
study_level_file = os.path.join(
    output_dir,
    "LOSO_Nested_XGBoost_study_level_metrics.csv",
)

outer_results_df.to_csv(outer_results_file, index=False)
best_params_df.to_csv(best_parameters_file, index=False)
oof_results.to_csv(oof_predictions_file, index=False)
study_level_df.to_csv(study_level_file, index=False)

summary_rows = [
    ("Number of observations", len(data)),
    ("Number of studies", number_of_studies),
    ("Model pooled OOF R2", model_overall_r2),
    ("Model pooled OOF RMSE", model_overall_rmse),
    ("Model pooled OOF MAE", model_overall_mae),
    ("Dummy pooled OOF R2", dummy_overall_r2),
    ("Dummy pooled OOF RMSE", dummy_overall_rmse),
    ("Dummy pooled OOF MAE", dummy_overall_mae),
]

summary_rows.extend(study_balanced_summary.items())

summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

summary_file = os.path.join(
    output_dir,
    "LOSO_Nested_XGBoost_performance_summary.csv",
)
summary_df.to_csv(summary_file, index=False)


# ============================================================
# 13. Compare LOSO with the existing Study-PFAS result
# ============================================================

primary_summary_file = os.path.join(
    output_dir,
    "Nested_XGBoost_performance_summary.csv",
)

comparison_file = None

if os.path.exists(primary_summary_file):
    primary_summary = pd.read_csv(primary_summary_file)
    primary_metrics = primary_summary.set_index("Metric")["Value"]

    primary_r2 = float(primary_metrics["Overall OOF R2"])
    primary_rmse = float(primary_metrics["Overall OOF RMSE"])

    comparison_df = pd.DataFrame(
        {
            "Validation_scheme": [
                "Study-PFAS grouped nested CV",
                "Nested leave-one-study-out CV",
            ],
            "Overall_OOF_R2": [primary_r2, model_overall_r2],
            "Overall_OOF_RMSE": [primary_rmse, model_overall_rmse],
        }
    )

    comparison_file = os.path.join(
        output_dir,
        "LOSO_vs_Study_PFAS_comparison.csv",
    )
    comparison_df.to_csv(comparison_file, index=False)

    print("\nVALIDATION-SCHEME COMPARISON")
    print(comparison_df.to_string(index=False))
else:
    print(
        "\nExisting Study-PFAS performance summary was not found. "
        "The LOSO results were saved without a comparison table."
    )


# ============================================================
# 14. Output locations
# ============================================================

print("\nResults saved:")
print(outer_results_file)
print(best_parameters_file)
print(oof_predictions_file)
print(study_level_file)
print(summary_file)

if comparison_file is not None:
    print(comparison_file)

