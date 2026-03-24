# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:22:50 2026

@author: Z01788
"""
# %% 1. packages
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


# %% 2. 模型欄位準備
ID_COL = "被保人身分證字號"
POLICY_ID_COL = "保單申請案號"
DATE_COL = "投保日"
TARGET_COL = "保單是否躉繳投資型"

DATA_END_DATE = pd.Timestamp("2026-03-01")
HORIZON_DAYS = 365
LABEL_CUTOFF_DATE = DATA_END_DATE - pd.Timedelta(days=HORIZON_DAYS)  # 2025-03-01


# %% 3. 建立正樣本 snapshot
"""
正樣本客戶 = 曾買過躉繳投資型
snapshot_date = 首次購買日 - 365 天
只保留 snapshot 前已經有保單的客戶
"""

def build_positive_snapshot_dates(policy_df: pd.DataFrame,
                                  horizon_days: int = 365) -> pd.DataFrame:
    df = policy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0)

    first_buy_df = (
        df[df[TARGET_COL] == 1]
        .sort_values([ID_COL, DATE_COL, POLICY_ID_COL])
        .groupby(ID_COL, as_index=False)
        .first()[[ID_COL, DATE_COL, POLICY_ID_COL]]
        .rename(columns={
            DATE_COL: "first_buy_date",
            POLICY_ID_COL: "first_buy_policy_id"
        })
    )

    if first_buy_df.empty:
        return pd.DataFrame(columns=[ID_COL, "snapshot_date", "label", "event_date"])

    first_buy_df["snapshot_date"] = first_buy_df["first_buy_date"] - pd.Timedelta(days=horizon_days)
    first_buy_df["label"] = 1
    first_buy_df["event_date"] = first_buy_df["first_buy_date"]

    first_policy_df = (
        df.groupby(ID_COL, as_index=False)[DATE_COL]
        .min()
        .rename(columns={DATE_COL: "first_policy_date"})
    )

    pos_df = first_buy_df.merge(first_policy_df, on=ID_COL, how="left")
    pos_df = pos_df[pos_df["first_policy_date"] <= pos_df["snapshot_date"]].copy()

    return pos_df[[ID_COL, "snapshot_date", "label", "event_date"]]

# %% 4. 建立負樣本 snapshot
"""
從未買過躉繳投資型的人
snapshot_date = 最後一張可完整觀察一年結果的保單日
label = 0
"""

def build_negative_snapshot_dates(policy_df: pd.DataFrame,
                                  label_cutoff_date: pd.Timestamp) -> pd.DataFrame:
    df = policy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0)

    ever_buy_ids = set(df.loc[df[TARGET_COL] == 1, ID_COL].dropna().unique())

    never_buy_df = df[~df[ID_COL].isin(ever_buy_ids)].copy()
    eligible_df = never_buy_df[never_buy_df[DATE_COL] <= label_cutoff_date].copy()

    neg_df = (
        eligible_df
        .sort_values([ID_COL, DATE_COL, POLICY_ID_COL])
        .groupby(ID_COL, as_index=False)
        .last()[[ID_COL, DATE_COL]]
        .rename(columns={DATE_COL: "snapshot_date"})
    )

    neg_df["label"] = 0
    neg_df["event_date"] = pd.NaT

    return neg_df[[ID_COL, "snapshot_date", "label", "event_date"]]

# %% 5. 合併成 snapshot master

def build_snapshot_master(policy_df: pd.DataFrame,
                          data_end_date: pd.Timestamp,
                          horizon_days: int = 365) -> pd.DataFrame:
    label_cutoff_date = data_end_date - pd.Timedelta(days=horizon_days)

    pos_df = build_positive_snapshot_dates(policy_df, horizon_days=horizon_days)
    neg_df = build_negative_snapshot_dates(policy_df, label_cutoff_date=label_cutoff_date)

    snapshot_master_df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)
    snapshot_master_df = snapshot_master_df[
        snapshot_master_df["snapshot_date"] <= label_cutoff_date
    ].copy()

    return snapshot_master_df

# %% 6.【關鍵】建 snapshot features
def build_snapshot_features(policy_df: pd.DataFrame,
                            snapshot_master_df: pd.DataFrame) -> pd.DataFrame:
    df = policy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    snap = snapshot_master_df.copy()
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"], errors="coerce")

    merged = df.merge(
        snap[[ID_COL, "snapshot_date", "label", "event_date"]],
        on=ID_COL,
        how="inner"
    )

    hist_df = merged[merged[DATE_COL] <= merged["snapshot_date"]].copy()

    if hist_df.empty:
        return pd.DataFrame()

    # ===== 動態時間窗 =====
    hist_df["近1年保單"] = (
        hist_df[DATE_COL] >= (hist_df["snapshot_date"] - pd.Timedelta(days=365))
    ).astype("Int64")

    hist_df["近2年保單"] = (
        hist_df[DATE_COL] >= (hist_df["snapshot_date"] - pd.Timedelta(days=730))
    ).astype("Int64")

    hist_df["近3年保單"] = (
        hist_df[DATE_COL] >= (hist_df["snapshot_date"] - pd.Timedelta(days=1095))
    ).astype("Int64")

    # 若沒有拆分欄位就補
    for c in [
        "保單件數_壽險", "保單件數_產險",
        "保單總保費_壽險", "保單總保費_產險",
        "保單總繳款FYC_壽險", "保單總繳款FYC_產險"
    ]:
        if c not in hist_df.columns:
            hist_df[c] = 0

    hist_df["近1年保單_壽險"] = hist_df["近1年保單"] * hist_df["保單件數_壽險"]
    hist_df["近1年保單_產險"] = hist_df["近1年保單"] * hist_df["保單件數_產險"]

    # ===== 間隔 =====
    hist_df = hist_df.sort_values([ID_COL, "snapshot_date", DATE_COL, POLICY_ID_COL]).reset_index(drop=True)
    hist_df["前一張投保日"] = hist_df.groupby([ID_COL, "snapshot_date"])[DATE_COL].shift(1)
    hist_df["保單間隔天數"] = (hist_df[DATE_COL] - hist_df["前一張投保日"]).dt.days

    # ===== route 摘要 =====
    if "主約商品險種主類別" in hist_df.columns:
        hist_df["前一張主約商品險種主類別"] = hist_df.groupby([ID_COL, "snapshot_date"])["主約商品險種主類別"].shift(1)
        hist_df["主類別是否切換"] = (
            hist_df["主約商品險種主類別"] != hist_df["前一張主約商品險種主類別"]
        ).astype("Int64")
        hist_df.loc[hist_df["前一張主約商品險種主類別"].isna(), "主類別是否切換"] = 0
    else:
        hist_df["主類別是否切換"] = 0
    
    if "主約是否投資型" not in hist_df.columns:
        if "主約商品險種主類別" in hist_df.columns:
            hist_df["主約是否投資型"] = (
                hist_df["主約商品險種主類別"] == "投資型"
            ).astype("Int64")
        else:
            hist_df["主約是否投資型"] = 0

    group_keys = [ID_COL, "snapshot_date", "label", "event_date"]

    agg_dict = {
        POLICY_ID_COL: "count",
        DATE_COL: ["min", "max"],

        "保單件數_壽險": "sum",
        "保單件數_產險": "sum",

        "保單總保費": ["sum", "mean", "median", "max"],
        "保單總保費_壽險": ["sum", "mean", "median", "max"],
        "保單總保費_產險": ["sum", "mean", "median", "max"],

        "保單總繳款FYC": ["sum", "mean", "max"],
        "保單總繳款FYC_壽險": ["sum", "mean", "max"],
        "保單總繳款FYC_產險": ["sum", "mean", "max"],

        "近1年保單": "sum",
        "近2年保單": "sum",
        "近3年保單": "sum",
        "近1年保單_壽險": "sum",
        "近1年保單_產險": "sum",

        "保單間隔天數": ["mean", "median", "min", "max"],
        "主類別是否切換": "sum",
        "主約是否投資型": "sum",
    }

    optional_nunique_cols = [
        "主約商品險種主類別",
        "主約商品險種次類別",
        "主約商品名稱",
        "主約繳別"
    ]
    for c in optional_nunique_cols:
        if c in hist_df.columns:
            agg_dict[c] = pd.Series.nunique

    feat_df = hist_df.groupby(group_keys, dropna=False).agg(agg_dict)

    def flatten_columns(columns):
        out = []
        for col in columns:
            if isinstance(col, tuple):
                parts = [str(x) for x in col if x not in ("", None)]
                out.append("_".join(parts))
            else:
                out.append(str(col))
        return out

    feat_df.columns = flatten_columns(feat_df.columns)
    feat_df = feat_df.reset_index()

    rename_map = {
        f"{POLICY_ID_COL}_count": "保單數",
        f"{DATE_COL}_min": "首次投保日",
        f"{DATE_COL}_max": "最近投保日",

        "保單件數_壽險_sum": "壽險保單數",
        "保單件數_產險_sum": "產險保單數",

        "保單總保費_sum": "累計保單總保費",
        "保單總保費_mean": "平均每張保單保費",
        "保單總保費_median": "保單保費中位數",
        "保單總保費_max": "最大單張保單保費",

        "保單總保費_壽險_sum": "累計壽險保單總保費",
        "保單總保費_產險_sum": "累計產險保單總保費",

        "保單總繳款FYC_sum": "累計保單總繳款FYC",
        "保單總繳款FYC_壽險_sum": "累計壽險保單總繳款FYC",
        "保單總繳款FYC_產險_sum": "累計產險保單總繳款FYC",

        "近1年保單_sum": "近1年保單數",
        "近2年保單_sum": "近2年保單數",
        "近3年保單_sum": "近3年保單數",
        "近1年保單_壽險_sum": "近1年壽險保單數",
        "近1年保單_產險_sum": "近1年產險保單數",

        "保單間隔天數_mean": "平均保單間隔天數",
        "保單間隔天數_median": "保單間隔天數中位數",
        "保單間隔天數_min": "最短保單間隔天數",
        "保單間隔天數_max": "最長保單間隔天數",

        "主類別是否切換_sum": "主約商品主類別切換次數",
        "主約是否投資型_sum": "主約投資型保單數",
        "主約商品險種主類別_nunique": "主約商品險種主類別數",
        "主約商品險種次類別_nunique": "主約商品險種次類別數",
        "主約商品名稱_nunique": "主約商品名稱數",
        "主約繳別_nunique": "主約繳別類型數",
    }

    feat_df = feat_df.rename(columns=rename_map)

    feat_df["投保年資天數"] = (feat_df["snapshot_date"] - feat_df["首次投保日"]).dt.days
    feat_df["距離最近投保天數"] = (feat_df["snapshot_date"] - feat_df["最近投保日"]).dt.days

    if {"保單數", "壽險保單數"}.issubset(feat_df.columns):
        feat_df["壽險保單占比"] = np.where(
            feat_df["保單數"] > 0,
            feat_df["壽險保單數"] / feat_df["保單數"],
            np.nan
        )

    if {"累計保單總保費", "累計壽險保單總保費"}.issubset(feat_df.columns):
        feat_df["壽險保費占比"] = np.where(
            feat_df["累計保單總保費"] > 0,
            feat_df["累計壽險保單總保費"] / feat_df["累計保單總保費"],
            np.nan
        )

    if {"累計保單總繳款FYC", "累計壽險保單總繳款FYC"}.issubset(feat_df.columns):
        feat_df["壽險繳款FYC占比"] = np.where(
            feat_df["累計保單總繳款FYC"] > 0,
            feat_df["累計壽險保單總繳款FYC"] / feat_df["累計保單總繳款FYC"],
            np.nan
        )

    if {"保單數", "主約投資型保單數"}.issubset(feat_df.columns):
        feat_df["主約投資型保單比例"] = np.where(
            feat_df["保單數"] > 0,
            feat_df["主約投資型保單數"] / feat_df["保單數"],
            np.nan
        )

    return feat_df

# %% 7. 切 train / valid / test 
def split_train_valid_test(snapshot_feature_df: pd.DataFrame):
    df = snapshot_feature_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")

    train_df = df[
        (df["snapshot_date"] >= pd.Timestamp("2022-03-01")) &
        (df["snapshot_date"] <= pd.Timestamp("2024-03-31"))
    ].copy()

    valid_df = df[
        (df["snapshot_date"] >= pd.Timestamp("2024-04-01")) &
        (df["snapshot_date"] <= pd.Timestamp("2024-08-31"))
    ].copy()

    test_df = df[
        (df["snapshot_date"] >= pd.Timestamp("2024-09-01")) &
        (df["snapshot_date"] <= pd.Timestamp("2025-03-01"))
    ].copy()

    return train_df, valid_df, test_df





# %% 8. train 內做負樣本下採樣
def balance_train(train_df: pd.DataFrame,
                  label_col: str = "label",
                  neg_to_pos_ratio: float = 3.0,
                  random_state: int = 42) -> pd.DataFrame:
    df = train_df.copy()

    pos_df = df[df[label_col] == 1].copy()
    neg_df = df[df[label_col] == 0].copy()

    if len(pos_df) == 0 or len(neg_df) == 0:
        return df

    target_neg_n = min(int(len(pos_df) * neg_to_pos_ratio), len(neg_df))
    neg_sampled = neg_df.sample(n=target_neg_n, random_state=random_state)

    out = pd.concat([pos_df, neg_sampled], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out

# %% 9. 模型欄位

def get_model_feature_cols(df: pd.DataFrame):
    feature_cols = [
        # rule 類
        "保單數",
        "壽險保單數",
        "產險保單數",
        "壽險保單占比",
        "累計保單總保費",
        "累計壽險保單總保費",
        "累計保單總繳款FYC",
        "累計壽險保單總繳款FYC",
        "壽險保費占比",
        "壽險繳款FYC占比",

        # timing
        "投保年資天數",
        "距離最近投保天數",

        # frequency
        "近1年保單數",
        "近2年保單數",
        "近3年保單數",
        "近1年壽險保單數",
        "近1年產險保單數",
        "平均保單間隔天數",
        "保單間隔天數中位數",
        "最短保單間隔天數",
        "最長保單間隔天數",

        # route / 結構
        "主約商品主類別切換次數",
        # "主約投資型保單數",
        # "主約投資型保單比例",
        "主約商品險種主類別數",
        "主約商品險種次類別數",
        # "主約商品名稱數",
        "主約繳別類型數",

        # 時間背景
        "snapshot_year",
        "snapshot_month",
    ]

    out = [c for c in feature_cols if c in df.columns]
    return out

# %% 10. 加時間背景特徵
def add_snapshot_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    out["snapshot_year"] = out["snapshot_date"].dt.year
    out["snapshot_month"] = out["snapshot_date"].dt.month
    return out


# %% 11. 前處理器
def build_preprocessor(X_train: pd.DataFrame):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


# %% 12. 評估函式
def evaluate_binary_model(y_true, y_prob, name="model"):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob
    }).sort_values("y_prob", ascending=False)

    # 1. 基本指標
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    # 2. Top K Precision（最重要🔥）
    def top_k_precision(df, top_pct):
        k = max(int(len(df) * top_pct), 1)
        top_df = df.head(k)
        return top_df["y_true"].mean()

    p10 = top_k_precision(df, 0.10)
    p20 = top_k_precision(df, 0.20)

    # 3. Lift（業務很愛看🔥）
    base_rate = df["y_true"].mean()

    lift_10 = p10 / base_rate if base_rate > 0 else np.nan
    lift_20 = p20 / base_rate if base_rate > 0 else np.nan

    # 4. 結果整理
    result = pd.DataFrame([{
        "model": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "base_rate": base_rate,
        "top10_precision": p10,
        "top20_precision": p20,
        "lift_10": lift_10,
        "lift_20": lift_20
    }])

    return result

# %% 13. 建立模型資料集主函式
def build_model_dataset(policy_df: pd.DataFrame):
    snapshot_master_df = build_snapshot_master(
        policy_df=policy_df,
        data_end_date=DATA_END_DATE,
        horizon_days=HORIZON_DAYS
    )

    snapshot_feature_df = build_snapshot_features(
        policy_df=policy_df,
        snapshot_master_df=snapshot_master_df
    )

    snapshot_feature_df = add_snapshot_calendar_features(snapshot_feature_df)

    train_df, valid_df, test_df = split_train_valid_test(snapshot_feature_df)
    train_df_balanced = balance_train(train_df, label_col="label", neg_to_pos_ratio=3.0)

    feature_cols = get_model_feature_cols(snapshot_feature_df)

    X_train = train_df_balanced[feature_cols].copy()
    y_train = train_df_balanced["label"].copy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df["label"].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df["label"].copy()

    return {
        "snapshot_master_df": snapshot_master_df,
        "snapshot_feature_df": snapshot_feature_df,
        "train_df": train_df,
        "train_df_balanced": train_df_balanced,
        "valid_df": valid_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test
    }

# %% 14. Logistic Regression Baseline
def train_logistic_baseline(X_train, y_train, X_valid, y_valid, X_test, y_test):
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    clf.fit(X_train, y_train)

    valid_prob = clf.predict_proba(X_valid)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    valid_metrics = evaluate_binary_model(y_valid, valid_prob, name="logistic_valid")
    test_metrics = evaluate_binary_model(y_test, test_prob, name="logistic_test")

    return {
        "model": clf,
        "valid_prob": valid_prob,
        "test_prob": test_prob,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics
    }

# %% 15. XGBoost 主模型
def train_xgboost_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    # 先 fit transform train，再 transform valid/test
    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)
    X_test_t = preprocessor.transform(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_t, y_train)

    valid_prob = model.predict_proba(X_valid_t)[:, 1]
    test_prob = model.predict_proba(X_test_t)[:, 1]

    valid_metrics = evaluate_binary_model(y_valid, valid_prob, name="xgb_valid")
    test_metrics = evaluate_binary_model(y_test, test_prob, name="xgb_test")

    return {
        "preprocessor": preprocessor,
        "model": model,
        "valid_prob": valid_prob,
        "test_prob": test_prob,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics
    }

# %% 16. 跑完整 Model Score pipeline
model_data = build_model_dataset(policy_df)

snapshot_master_df = model_data["snapshot_master_df"]
snapshot_feature_df = model_data["snapshot_feature_df"]
train_df = model_data["train_df"]
train_df_balanced = model_data["train_df_balanced"]
valid_df = model_data["valid_df"]
test_df = model_data["test_df"]
feature_cols = model_data["feature_cols"]

X_train = model_data["X_train"]
y_train = model_data["y_train"]
X_valid = model_data["X_valid"]
y_valid = model_data["y_valid"]
X_test = model_data["X_test"]
y_test = model_data["y_test"]

# %% 17. 看結果

print("Train 原始:", train_df.shape, train_df["label"].value_counts(dropna=False).to_dict())
print("Train 平衡後:", train_df_balanced.shape, train_df_balanced["label"].value_counts(dropna=False).to_dict())
print("Valid:", valid_df.shape, valid_df["label"].value_counts(dropna=False).to_dict())
print("Test:", test_df.shape, test_df["label"].value_counts(dropna=False).to_dict())

print(feature_cols)

# %% 18. 訓練 Baseline
logit_result = train_logistic_baseline(
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test
)

pd.concat([
    logit_result["valid_metrics"],
    logit_result["test_metrics"]
], axis=0)

# %% 19. 訓練 XGBoost
xgb_result = train_xgboost_model(
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test
)

pd.concat([
    xgb_result["valid_metrics"],
    xgb_result["test_metrics"]
], axis=0)


# %% 20. Logistic 打分
candidate_model_df["model_prob_logistic"] = logit_result["model"].predict_proba(candidate_model_X)[:, 1]

# %% 21. XGBoost 打分
candidate_X_t = xgb_result["preprocessor"].transform(candidate_model_X)
candidate_model_df["model_prob_xgb"] = xgb_result["model"].predict_proba(candidate_X_t)[:, 1]


