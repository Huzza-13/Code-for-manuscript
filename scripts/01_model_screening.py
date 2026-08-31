import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_squared_error

import sys

print("当前运行 Python：")
print(sys.executable)

import pandas as pd
# ============================================================
# 1. 读取数据
# ============================================================

os.chdir(r"D:\python\pythonProject1")

data = pd.read_excel("Machine-learning original dataset.xlsx")

print("Dataset shape:", data.shape)
print("Number of studies:", data["Reference"].nunique())
print("Number of PFAS compounds:", data["PFASs Name"].nunique())


# ============================================================
# 2. 构建 Study × Compound 分组
# ============================================================

data["Study_Compound_ID"] = (
    data["Reference"].astype(str)
    + "_"
    + data["PFASs Name"].astype(str)
)

print(
    "Number of Study × Compound groups:",
    data["Study_Compound_ID"].nunique()
)


# ============================================================
# 3. 定义目标变量
# ============================================================

target = "Log PFASs concentration"

y = data[target]


# ============================================================
# 4. 定义全部预测变量
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

    # Detailed molecular descriptors
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
    "SpMin8_Bhi"
]

X = data[features].copy()


# ============================================================
# 5. 区分类别变量和连续变量
# ============================================================

categorical_features = [
    "Plant tissue group classify",
    "Morphotypes"
]

numerical_features = [
    col for col in features
    if col not in categorical_features
]


# ============================================================
# 6. One-hot encoding
#
# 重要：
# 这三个变量是 nominal categories，
# 不能让模型把 1,2,3,4,5 当成连续大小关系
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
# 7. 定义四种候选模型
#
# 这里暂时不做最终超参数优化。
# 目的只是：
# 在新的 grouped CV 框架下重新筛选算法。
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
            random_state=2026,
            n_jobs=-1
        )
}


# ============================================================
# 8. Study × Compound grouped 5-fold CV
# ============================================================

groups = data["Study_Compound_ID"]

cv = GroupKFold(n_splits=5)


# ============================================================
# 9. 逐个模型进行 OOF prediction
#
# 每个 observation 的 prediction 都来自：
# 该 observation 所属 group 没有参与训练的模型
# ============================================================

results = []

oof_predictions = pd.DataFrame({
    "Observed": y,
    "Reference": data["Reference"],
    "PFASs Name": data["PFASs Name"],
    "Study_Compound_ID": groups
})


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )

    y_pred = cross_val_predict(
        estimator=pipeline,
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        n_jobs=1
    )

    r2 = r2_score(y, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y, y_pred)
    )

    print(f"Grouped-CV R²   = {r2:.3f}")
    print(f"Grouped-CV RMSE = {rmse:.3f}")

    results.append({
        "Model": model_name,
        "R2": r2,
        "RMSE": rmse
    })

    oof_predictions[model_name] = y_pred


# ============================================================
# 10. 输出模型性能表
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n")
print("=" * 60)
print("FINAL MODEL SCREENING RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))
print("=" * 60)


# ============================================================
# 11. 保存结果
# ============================================================

results_df.to_csv(
    "Model_screening_GroupCV_results.csv",
    index=False
)

oof_predictions.to_csv(
    "Model_screening_OOF_predictions.csv",
    index=False
)

print("\nResults saved successfully.")