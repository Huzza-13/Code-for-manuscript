import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor


# ============================================================
# 1. File paths
# ============================================================

input_file = r"D:\python\pythonProject1\Machine-learning original dataset.xlsx"

output_dir = r"D:\python\pythonProject1\机器学习新 202608"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Read data
# ============================================================

data = pd.read_excel(input_file)

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
# 4. Target
# ============================================================

y = data["Log PFASs concentration"]


# ============================================================
# 5. Feature blocks
# ============================================================

# Environmental / biological / experimental
E_features = [
    "Plant tissue group classify",
    "Morphotypes",
    "Soil pH",
    "SOM",
    "Exposure Concentration",
    "Growth Duration",
    "Growth Temperature"
]


# Basic chemical descriptors
C_features = [
    "Carbon Chain Length",
    "Functional group"
]


# Reduced detailed molecular descriptors
M_reduced = [
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi"
]


# ============================================================
# 6. Four models
# ============================================================

feature_sets = {

    "E":
        E_features,

    "E+C":
        E_features + C_features,

    "Reduced E+M":
        E_features + M_reduced,

    "Reduced Full":
        E_features + C_features + M_reduced
}


# ============================================================
# 7. Categorical features
# ============================================================

all_categorical = [
    "Plant tissue group classify",
    "Morphotypes",
    "Functional group"
]


# ============================================================
# 8. XGBoost
#
# Keep the SAME parameters as before.
# We are comparing feature sets, not tuning yet.
# ============================================================

def make_model():

    return XGBRegressor(
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


# ============================================================
# 9. Create EXACTLY THE SAME five folds
# ============================================================

cv = GroupKFold(n_splits=5)

# X here is only used to generate row indices.
# The split is determined by groups.
dummy_X = np.zeros((len(data), 1))

splits = list(
    cv.split(
        dummy_X,
        y,
        groups=groups
    )
)


# Check group leakage
for fold, (train_idx, test_idx) in enumerate(splits, start=1):

    train_groups = set(
        groups.iloc[train_idx]
    )

    test_groups = set(
        groups.iloc[test_idx]
    )

    overlap = (
        train_groups.intersection(test_groups)
    )

    if len(overlap) > 0:
        raise ValueError(
            f"Group leakage detected in Fold {fold}"
        )

print("\nGroup leakage check passed.")


# ============================================================
# 10. Storage
# ============================================================

overall_results = []
fold_results = []

oof_df = pd.DataFrame({

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
# 11. Run each model using identical folds
# ============================================================

for model_name, features in feature_sets.items():

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("Number of original predictors:", len(features))
    print("=" * 70)

    X = data[features].copy()


    categorical_features = [
        col for col in all_categorical
        if col in features
    ]

    numerical_features = [
        col for col in features
        if col not in categorical_features
    ]


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


    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", make_model())
        ]
    )


    # --------------------------------------------------------
    # Empty OOF prediction vector
    # --------------------------------------------------------

    y_oof = np.full(
        len(data),
        np.nan
    )


    # --------------------------------------------------------
    # Fit and evaluate each fold separately
    # --------------------------------------------------------

    for fold, (train_idx, test_idx) in enumerate(
        splits,
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]


        pipeline.fit(
            X_train,
            y_train
        )


        y_pred = pipeline.predict(
            X_test
        )


        y_oof[test_idx] = y_pred


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


        fold_results.append({

            "Model":
                model_name,

            "Fold":
                fold,

            "N_test":
                len(test_idx),

            "N_groups_test":
                groups.iloc[test_idx].nunique(),

            "R2":
                fold_r2,

            "RMSE":
                fold_rmse
        })


        print(
            f"Fold {fold}: "
            f"R² = {fold_r2:.3f}, "
            f"RMSE = {fold_rmse:.3f}, "
            f"n = {len(test_idx)}"
        )


    # --------------------------------------------------------
    # Overall OOF performance
    # --------------------------------------------------------

    overall_r2 = r2_score(
        y,
        y_oof
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y,
            y_oof
        )
    )


    print("-" * 70)

    print(
        f"Overall OOF R²   = {overall_r2:.3f}"
    )

    print(
        f"Overall OOF RMSE = {overall_rmse:.3f}"
    )


    overall_results.append({

        "Model":
            model_name,

        "Number_of_predictors":
            len(features),

        "Overall_OOF_R2":
            overall_r2,

        "Overall_OOF_RMSE":
            overall_rmse
    })


    oof_df[model_name] = y_oof


# ============================================================
# 12. Convert results to tables
# ============================================================

overall_df = pd.DataFrame(
    overall_results
)

fold_df = pd.DataFrame(
    fold_results
)


# ============================================================
# 13. Fold-level summary
# ============================================================

fold_summary = (
    fold_df
    .groupby("Model")
    .agg(
        Mean_Fold_R2=("R2", "mean"),
        SD_Fold_R2=("R2", "std"),
        Mean_Fold_RMSE=("RMSE", "mean"),
        SD_Fold_RMSE=("RMSE", "std")
    )
    .reset_index()
)


# ============================================================
# 14. Merge summaries
# ============================================================

final_summary = overall_df.merge(
    fold_summary,
    on="Model",
    how="left"
)


print("\n")
print("=" * 100)
print("FINAL REDUCED MODEL COMPARISON")
print("=" * 100)

print(
    final_summary.to_string(
        index=False
    )
)

print("=" * 100)


# ============================================================
# 15. Save outputs
# ============================================================

final_summary.to_csv(
    os.path.join(
        output_dir,
        "Reduced_model_comparison_summary.csv"
    ),
    index=False
)


fold_df.to_csv(
    os.path.join(
        output_dir,
        "Reduced_model_fold_results.csv"
    ),
    index=False
)


oof_df.to_csv(
    os.path.join(
        output_dir,
        "Reduced_model_OOF_predictions.csv"
    ),
    index=False
)


print(
    "\nAll results saved to:"
)

print(
    output_dir
)