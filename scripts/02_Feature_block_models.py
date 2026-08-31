import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor


# ============================================================
# 1. Read dataset
# ============================================================

os.chdir(r"D:\python\pythonProject1")

data = pd.read_excel("Machine-learning original dataset.xlsx")

print("Dataset shape:", data.shape)
print("Studies:", data["Reference"].nunique())
print("PFAS compounds:", data["PFASs Name"].nunique())


# ============================================================
# 2. Construct Study × Compound groups
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
print("\nDataset consistency check")
print("-------------------------")
print("Observations:", len(data))
print("Studies:", data["Reference"].nunique())
print("PFAS compounds:", data["PFASs Name"].nunique())
print("Study × PFAS groups:", groups.nunique())

assert len(data) == 825
assert groups.nunique() == 110

# ============================================================
# 3. Target
# ============================================================

target = "Log PFASs concentration"

y = data[target]


# ============================================================
# 4. Define the three feature blocks
# ============================================================

# ---------------- Layer 1 ----------------
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





# ---------------- Layer 3 ----------------
# Detailed molecular descriptors
M_features = [
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi"
]

feature_sets = {
    "E": E_features,
    "E+M": E_features + M_features
}

print("\nFinal feature-block definitions:")
for model_name, features in feature_sets.items():
    print(
        f"{model_name:6s}: "
        f"{len(features):2d} predictors"
    )

assert len(feature_sets["E"]) == 7
assert len(feature_sets["E+M"]) == 13
# ============================================================
# 5. Categorical variables
# ============================================================

all_categorical = [
    "Plant tissue group classify",
    "Morphotypes",
    "Functional group"
]


# ============================================================
# 6. Fixed grouped CV
#
# VERY IMPORTANT:
# All three models use exactly the same folds.
# ============================================================

cv = GroupKFold(
    n_splits=5
)


# ============================================================
# 7. XGBoost model
#
# For now we deliberately use the same hyperparameters
# for all three models.
#
# This stage evaluates the information added by feature blocks,
# not final hyperparameter optimization.
# ============================================================

def make_xgb_model():

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
# 8. Store results
# ============================================================

results = []

predictions_df = pd.DataFrame({
    "Observed": y,
    "Reference": data["Reference"],
    "PFASs Name": data["PFASs Name"],
    "Study_Compound_ID": groups
})


# ============================================================
# 9. Fit the three feature-block models
# ============================================================

for model_name, features in feature_sets.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("Number of original features:", len(features))
    print("=" * 60)

    X = data[features].copy()


    # --------------------------------------------------------
    # Identify categorical variables actually present
    # in this specific feature set
    # --------------------------------------------------------

    categorical_features = [
        col for col in all_categorical
        if col in features
    ]

    numerical_features = [
        col for col in features
        if col not in categorical_features
    ]


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", make_xgb_model())
        ]
    )


    # --------------------------------------------------------
    # Out-of-fold prediction
    # --------------------------------------------------------

    y_pred = cross_val_predict(

        estimator=pipeline,

        X=X,

        y=y,

        groups=groups,

        cv=cv,

        n_jobs=1

    )


    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    r2 = r2_score(
        y,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            y_pred
        )
    )


    print(
        f"Grouped-CV R²   = {r2:.3f}"
    )

    print(
        f"Grouped-CV RMSE = {rmse:.3f}"
    )


    results.append({

        "Model": model_name,

        "Number_of_features":
            len(features),

        "R2":
            r2,

        "RMSE":
            rmse

    })


    predictions_df[
        model_name
    ] = y_pred


# ============================================================
# 10. Results table
# ============================================================

results_df = pd.DataFrame(
    results
)




print("\n")
print("=" * 70)
print("FEATURE-BLOCK MODEL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("=" * 70)


# ============================================================
# 11. Save
# ============================================================

results_df.to_csv(
    r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"
    r"\Feature_block_model_results_final15.csv",
    index=False
)

predictions_df.to_csv(
    r"D:\python\pythonProject1\机器学习新 202608\no_chain_PFASclass"
    r"\Feature_block_OOF_predictions_final15.csv",
    index=False
)

print(
    "\nResults saved successfully."
)