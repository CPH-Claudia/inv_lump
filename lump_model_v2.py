# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:52:51 2026

@author: Z01788
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# =========================================================
# 基本欄位
# =========================================================
ID_COL = "被保人身分證字號"
POLICY_ID_COL = "保單申請案號"
DATE_COL = "投保日"
TARGET_COL = "保單是否躉繳投資型"

MOBILE_COL = "行動投保受理號"
INCOME_FAMILY_COL = "(要)保人-家庭年收入(萬)"
INCOME_OTHER_COL = "(要)保人-其他年收入(萬)"
INCOME_WORK_COL = "(要)保人-工作年收入(萬)"

DATA_END_DATE = pd.Timestamp("2026-03-01")
HORIZON_DAYS = 365


# =========================================================
# 1. Snapshot 標記
# =========================================================
def build_positive_snapshot_dates(policy_df: pd.DataFrame,
                                  horizon_days: int = 365) -> pd.DataFrame:
    df = policy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0)
    df[ID_COL] = df[ID_COL].astype("string")

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
        return pd.DataFrame(columns=[
            ID_COL, "snapshot_date", "label", "event_date", "first_buy_policy_id"
        ])

    merged = df.merge(
        first_buy_df[[ID_COL, "first_buy_date", "first_buy_policy_id"]],
        on=ID_COL,
        how="inner"
    )

    candidate_pos = merged[
        (merged[DATE_COL] < merged["first_buy_date"]) &
        (merged["first_buy_date"] <= merged[DATE_COL] + pd.Timedelta(days=horizon_days))
    ].copy()

    if candidate_pos.empty:
        return pd.DataFrame(columns=[
            ID_COL, "snapshot_date", "label", "event_date", "first_buy_policy_id"
        ])

    pos_df = (
        candidate_pos
        .sort_values([ID_COL, DATE_COL, POLICY_ID_COL])
        .groupby(ID_COL, as_index=False)
        .last()[[ID_COL, DATE_COL, "first_buy_date", "first_buy_policy_id"]]
        .rename(columns={
            DATE_COL: "snapshot_date",
            "first_buy_date": "event_date"
        })
    )

    pos_df["label"] = 1
    return pos_df[[ID_COL, "snapshot_date", "label", "event_date", "first_buy_policy_id"]]


def build_negative_snapshot_dates(policy_df: pd.DataFrame,
                                  label_cutoff_date: pd.Timestamp) -> pd.DataFrame:
    df = policy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0)
    df[ID_COL] = df[ID_COL].astype("string")

    ever_buy_ids = set(df.loc[df[TARGET_COL] == 1, ID_COL].dropna().astype("string").unique())

    never_buy_df = df[~df[ID_COL].isin(ever_buy_ids)].copy()
    eligible_df = never_buy_df[never_buy_df[DATE_COL] <= label_cutoff_date].copy()

    if eligible_df.empty:
        return pd.DataFrame(columns=[ID_COL, "snapshot_date", "label", "event_date", "snapshot_policy_id"])

    neg_df = (
        eligible_df
        .sort_values([ID_COL, DATE_COL, POLICY_ID_COL])
        .groupby(ID_COL, as_index=False)
        .last()[[ID_COL, DATE_COL, POLICY_ID_COL]]
        .rename(columns={
            DATE_COL: "snapshot_date",
            POLICY_ID_COL: "snapshot_policy_id"
        })
    )

    neg_df["label"] = 0
    neg_df["event_date"] = pd.NaT
    return neg_df[[ID_COL, "snapshot_date", "label", "event_date", "snapshot_policy_id"]]


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

    snapshot_master_df[ID_COL] = snapshot_master_df[ID_COL].astype("string")
    return snapshot_master_df


# =========================================================
# 2. Snapshot 特徵
# =========================================================
def build_snapshot_features(policy_df: pd.DataFrame,
                            snapshot_master_df: pd.DataFrame) -> pd.DataFrame:
    df = policy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[ID_COL] = df[ID_COL].astype("string")

    snap = snapshot_master_df.copy()
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"], errors="coerce")
    snap[ID_COL] = snap[ID_COL].astype("string")

    merged = df.merge(
        snap[[ID_COL, "snapshot_date", "label", "event_date"]],
        on=ID_COL,
        how="inner"
    )

    hist_df = merged[merged[DATE_COL] <= merged["snapshot_date"]].copy()

    if hist_df.empty:
        return pd.DataFrame()

    hist_df["近1年保單"] = (
        hist_df[DATE_COL] >= (hist_df["snapshot_date"] - pd.Timedelta(days=365))
    ).astype("Int64")

    hist_df["近2年保單"] = (
        hist_df[DATE_COL] >= (hist_df["snapshot_date"] - pd.Timedelta(days=730))
    ).astype("Int64")

    hist_df["近3年保單"] = (
        hist_df[DATE_COL] >= (hist_df["snapshot_date"] - pd.Timedelta(days=1095))
    ).astype("Int64")

    for c in [
        "保單件數_壽險", "保單件數_產險",
        "保單總保費_壽險", "保單總保費_產險",
        "保單總繳款FYC_壽險", "保單總繳款FYC_產險"
    ]:
        if c not in hist_df.columns:
            hist_df[c] = 0

    hist_df["近1年保單_壽險"] = hist_df["近1年保單"] * hist_df["保單件數_壽險"]
    hist_df["近1年保單_產險"] = hist_df["近1年保單"] * hist_df["保單件數_產險"]

    hist_df = hist_df.sort_values([ID_COL, "snapshot_date", DATE_COL, POLICY_ID_COL]).reset_index(drop=True)
    hist_df["前一張投保日"] = hist_df.groupby([ID_COL, "snapshot_date"])[DATE_COL].shift(1)
    hist_df["保單間隔天數"] = (hist_df[DATE_COL] - hist_df["前一張投保日"]).dt.days

    if "主約商品險種主類別" in hist_df.columns:
        hist_df["前一張主約商品險種主類別"] = hist_df.groupby([ID_COL, "snapshot_date"])["主約商品險種主類別"].shift(1)
        hist_df["主類別是否切換"] = (
            hist_df["主約商品險種主類別"] != hist_df["前一張主約商品險種主類別"]
        ).astype("Int64")
        hist_df.loc[hist_df["前一張主約商品險種主類別"].isna(), "主類別是否切換"] = 0
    else:
        hist_df["主類別是否切換"] = 0

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

    return feat_df


def add_snapshot_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    out["snapshot_year"] = out["snapshot_date"].dt.year
    out["snapshot_month"] = out["snapshot_date"].dt.month
    return out


# =========================================================
# 3. 切資料 / 模型
# =========================================================
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


def top_k_precision(y_true, y_prob, top_pct=0.10):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values("y_prob", ascending=False)
    k = max(int(len(df) * top_pct), 1)
    top_df = df.head(k)
    return top_df["y_true"].mean()


def evaluate_binary_model(y_true, y_prob, name="model"):
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    p10 = top_k_precision(y_true, y_prob, top_pct=0.10)
    p20 = top_k_precision(y_true, y_prob, top_pct=0.20)
    base_rate = np.mean(y_true)
    lift_10 = p10 / base_rate if base_rate > 0 else np.nan
    lift_20 = p20 / base_rate if base_rate > 0 else np.nan

    out = pd.DataFrame([{
        "model": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "base_rate": base_rate,
        "top10_precision": p10,
        "top20_precision": p20,
        "lift_10": lift_10,
        "lift_20": lift_20
    }])
    return out


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


def train_xgboost_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

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

    importance = pd.DataFrame({
        "feature": list(X_train.columns),
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    valid_metrics = evaluate_binary_model(y_valid, valid_prob, name="xgb_valid")
    test_metrics = evaluate_binary_model(y_test, test_prob, name="xgb_test")

    return {
        "preprocessor": preprocessor,
        "model": model,
        "valid_prob": valid_prob,
        "test_prob": test_prob,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "importance_df": importance
    }


# =========================================================
# 4. 主模型（A 路徑）
# =========================================================
def get_model_feature_cols_main(df: pd.DataFrame):
    feature_cols = [
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

        "投保年資天數",
        "距離最近投保天數",

        "近1年保單數",
        "近2年保單數",
        "近3年保單數",
        "近1年壽險保單數",
        "近1年產險保單數",
        "平均保單間隔天數",
        "保單間隔天數中位數",
        "最短保單間隔天數",
        "最長保單間隔天數",

        "主約商品主類別切換次數",
        "主約商品險種主類別數",
        "主約商品險種次類別數",
        "主約繳別類型數",

        "snapshot_year",
        "snapshot_month",
    ]
    return [c for c in feature_cols if c in df.columns]


def build_main_model_dataset(policy_df: pd.DataFrame):
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

    feature_cols = get_model_feature_cols_main(snapshot_feature_df)

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


def run_main_model_pipeline(policy_df: pd.DataFrame):
    main_data = build_main_model_dataset(policy_df)

    logit_result = train_logistic_baseline(
        main_data["X_train"], main_data["y_train"],
        main_data["X_valid"], main_data["y_valid"],
        main_data["X_test"], main_data["y_test"]
    )

    xgb_result = train_xgboost_model(
        main_data["X_train"], main_data["y_train"],
        main_data["X_valid"], main_data["y_valid"],
        main_data["X_test"], main_data["y_test"]
    )

    return {
        "data": main_data,
        "feature_cols": main_data["feature_cols"],   # ✅ 一起補上
        "logit_result": logit_result,
        "xgb_result": xgb_result, 
    }


# =========================================================
# 5. 收入模型（B 路徑）
#    注意：只收「年收入有值」的 snapshot
# =========================================================
def get_mobile_customer_ids(raw_df: pd.DataFrame) -> set:
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    required_cols = [ID_COL, MOBILE_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"raw_df 缺少必要欄位: {missing_cols}")

    mobile_ids = set(
        df.loc[df[MOBILE_COL].notna(), ID_COL]
        .dropna()
        .astype("string")
        .unique()
    )
    return mobile_ids


def subset_policy_by_mobile_customers(policy_df: pd.DataFrame, mobile_customer_ids: set) -> pd.DataFrame:
    df = policy_df.copy()
    df.columns = df.columns.str.strip()
    df[ID_COL] = df[ID_COL].astype("string")
    out = df[df[ID_COL].isin(mobile_customer_ids)].copy()
    return out


def prepare_income_detail(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    required_cols = [ID_COL, DATE_COL, MOBILE_COL, INCOME_FAMILY_COL, INCOME_OTHER_COL, INCOME_WORK_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"raw_df 缺少必要欄位: {missing_cols}")

    df[ID_COL] = df[ID_COL].astype("string")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    for c in [INCOME_FAMILY_COL, INCOME_OTHER_COL, INCOME_WORK_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = np.nan

    # 只用有行動投保受理號的紀錄當收入來源
    df = df[df[MOBILE_COL].notna()].copy()

    df["年收入_合成"] = np.where(
        df[INCOME_FAMILY_COL].notna(),
        df[INCOME_FAMILY_COL],
        df[[INCOME_WORK_COL, INCOME_OTHER_COL]].fillna(0).sum(axis=1)
    )

    both_missing = df[INCOME_FAMILY_COL].isna() & df[INCOME_WORK_COL].isna() & df[INCOME_OTHER_COL].isna()
    df.loc[both_missing, "年收入_合成"] = np.nan

    df["是否有收入資料"] = df["年收入_合成"].notna().astype(int)
    df["log_年收入_合成"] = np.log1p(df["年收入_合成"])

    keep_cols = [
        ID_COL, DATE_COL,
        INCOME_FAMILY_COL, INCOME_OTHER_COL, INCOME_WORK_COL,
        "年收入_合成", "log_年收入_合成", "是否有收入資料"
    ]

    income_df = df[keep_cols].copy()

    income_df = (
        income_df.sort_values([ID_COL, DATE_COL])
        .groupby([ID_COL, DATE_COL], as_index=False)
        .last()
    )

    return income_df


def build_income_snapshot_features(snapshot_master_df: pd.DataFrame,
                                   income_detail_df: pd.DataFrame) -> pd.DataFrame:
    snap = snapshot_master_df[[ID_COL, "snapshot_date"]].copy()
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"], errors="coerce")
    snap[ID_COL] = snap[ID_COL].astype("string")

    income = income_detail_df.copy()
    income[DATE_COL] = pd.to_datetime(income[DATE_COL], errors="coerce")
    income[ID_COL] = income[ID_COL].astype("string")

    snap = snap[snap[ID_COL].notna() & snap["snapshot_date"].notna()].copy()
    income = income[income[ID_COL].notna() & income[DATE_COL].notna()].copy()

    common_ids = set(snap[ID_COL].unique()) & set(income[ID_COL].unique())
    snap = snap[snap[ID_COL].isin(common_ids)].copy()
    income = income[income[ID_COL].isin(common_ids)].copy()

    merged_list = []

    for cust_id, snap_g in snap.groupby(ID_COL, dropna=False):
        income_g = income[income[ID_COL] == cust_id].copy()

        if income_g.empty:
            tmp = snap_g.copy()
            tmp[INCOME_FAMILY_COL] = np.nan
            tmp[INCOME_OTHER_COL] = np.nan
            tmp[INCOME_WORK_COL] = np.nan
            tmp["年收入_合成"] = np.nan
            tmp["log_年收入_合成"] = np.nan
            tmp["是否有收入資料"] = 0
            merged_list.append(tmp)
            continue

        snap_g = snap_g.sort_values("snapshot_date").reset_index(drop=True)
        income_g = income_g.sort_values(DATE_COL).reset_index(drop=True)

        merged_g = pd.merge_asof(
            left=snap_g,
            right=income_g,
            left_on="snapshot_date",
            right_on=DATE_COL,
            direction="backward",
            allow_exact_matches=True
        )

        if f"{ID_COL}_x" in merged_g.columns:
            merged_g = merged_g.rename(columns={f"{ID_COL}_x": ID_COL})
        if f"{ID_COL}_y" in merged_g.columns:
            merged_g = merged_g.drop(columns=[f"{ID_COL}_y"], errors="ignore")

        merged_list.append(merged_g)

    if merged_list:
        merged = pd.concat(merged_list, axis=0, ignore_index=True)
    else:
        merged = snap.copy()
        merged[INCOME_FAMILY_COL] = np.nan
        merged[INCOME_OTHER_COL] = np.nan
        merged[INCOME_WORK_COL] = np.nan
        merged["年收入_合成"] = np.nan
        merged["log_年收入_合成"] = np.nan
        merged["是否有收入資料"] = 0

    merged = merged.drop(columns=[DATE_COL], errors="ignore")

    rename_map = {
        INCOME_FAMILY_COL: "snapshot_家庭年收入_萬",
        INCOME_OTHER_COL: "snapshot_其他年收入_萬",
        INCOME_WORK_COL: "snapshot_工作年收入_萬",
        "年收入_合成": "snapshot_年收入_合成_萬",
        "log_年收入_合成": "snapshot_log_年收入_合成",
        "是否有收入資料": "snapshot_是否有收入資料"
    }

    merged = merged.rename(columns=rename_map)

    for c in [
        "snapshot_家庭年收入_萬",
        "snapshot_其他年收入_萬",
        "snapshot_工作年收入_萬",
        "snapshot_年收入_合成_萬",
        "snapshot_log_年收入_合成",
        "snapshot_是否有收入資料"
    ]:
        if c not in merged.columns:
            merged[c] = np.nan if c != "snapshot_是否有收入資料" else 0

    return merged


def get_model_feature_cols_income(df: pd.DataFrame):
    feature_cols = get_model_feature_cols_main(df)

    income_cols = [
        "snapshot_年收入_合成_萬",
        "snapshot_log_年收入_合成",
    ]

    feature_cols += [c for c in income_cols if c in df.columns]
    return feature_cols


def build_income_model_dataset_B(raw_df: pd.DataFrame,
                                 policy_df: pd.DataFrame,
                                 data_end_date: str = "2026-03-01",
                                 horizon_days: int = 365,
                                 balance_train_set: bool = True,
                                 neg_to_pos_ratio: float = 3.0):
    raw_df = raw_df.copy()
    policy_df = policy_df.copy()

    raw_df[ID_COL] = raw_df[ID_COL].astype("string")
    policy_df[ID_COL] = policy_df[ID_COL].astype("string")

    mobile_customer_ids = get_mobile_customer_ids(raw_df)
    policy_mobile_df = subset_policy_by_mobile_customers(policy_df, mobile_customer_ids)

    if policy_mobile_df.empty:
        raise ValueError("行動投保客戶子集為空，無法建模。")

    snapshot_master_df = build_snapshot_master(
        policy_df=policy_mobile_df,
        data_end_date=pd.Timestamp(data_end_date),
        horizon_days=horizon_days
    )

    snapshot_feature_df = build_snapshot_features(
        policy_df=policy_mobile_df,
        snapshot_master_df=snapshot_master_df
    )

    income_detail_df = prepare_income_detail(raw_df)
    income_snapshot_df = build_income_snapshot_features(
        snapshot_master_df=snapshot_master_df,
        income_detail_df=income_detail_df
    )

    snapshot_feature_df = snapshot_feature_df.merge(
        income_snapshot_df,
        on=[ID_COL, "snapshot_date"],
        how="left"
    )

    snapshot_feature_df = add_snapshot_calendar_features(snapshot_feature_df)

    # ====== 核心：只保留年收入有值的 snapshot ======
    snapshot_feature_df = snapshot_feature_df[
        snapshot_feature_df["snapshot_年收入_合成_萬"].notna()
    ].copy()

    if snapshot_feature_df.empty:
        raise ValueError("收入版模型資料為空：沒有任何 snapshot 有年收入資料。")

    train_df, valid_df, test_df = split_train_valid_test(snapshot_feature_df)

    if balance_train_set:
        train_df_balanced = balance_train(
            train_df,
            label_col="label",
            neg_to_pos_ratio=neg_to_pos_ratio
        )
    else:
        train_df_balanced = train_df.copy()

    feature_cols = get_model_feature_cols_income(snapshot_feature_df)

    X_train = train_df_balanced[feature_cols].copy()
    y_train = train_df_balanced["label"].copy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df["label"].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df["label"].copy()

    return {
        "mobile_customer_ids": mobile_customer_ids,
        "policy_mobile_df": policy_mobile_df,
        "snapshot_master_df": snapshot_master_df,
        "snapshot_feature_df": snapshot_feature_df,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "train_df_balanced": train_df_balanced,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test
    }


def run_income_model_pipeline_B(raw_df: pd.DataFrame, policy_df: pd.DataFrame):
    income_data = build_income_model_dataset_B(raw_df, policy_df)

    logit_result = train_logistic_baseline(
        income_data["X_train"], income_data["y_train"],
        income_data["X_valid"], income_data["y_valid"],
        income_data["X_test"], income_data["y_test"]
    )

    xgb_result = train_xgboost_model(
        income_data["X_train"], income_data["y_train"],
        income_data["X_valid"], income_data["y_valid"],
        income_data["X_test"], income_data["y_test"]
    )

    return {
        "data": income_data,
        "feature_cols": income_data["feature_cols"],   # ✅ 正確加法
        "logit_result": logit_result,
        "xgb_result": xgb_result
    }


# =========================================================
# 6. Candidate 打分：做法 B
#    A 與 B 不重疊
#    B 只有年收入有值者才納入
# =========================================================
def calc_product_main_type_switch(df):
    df = df.copy()

    required_cols = [ID_COL, DATE_COL, POLICY_ID_COL, "主約商品險種主類別"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[ID_COL] = df[ID_COL].astype("string")

    df = df.sort_values([ID_COL, DATE_COL, POLICY_ID_COL]).copy()

    df["prev_main_type"] = df.groupby(ID_COL)["主約商品險種主類別"].shift(1)

    is_switch = (
        (df["主約商品險種主類別"] != df["prev_main_type"]) &
        df["prev_main_type"].notna() &
        df["主約商品險種主類別"].notna()
    )

    df["is_switch"] = is_switch.fillna(False).astype(int)

    switch_df = (
        df.groupby(ID_COL, dropna=False)["is_switch"]
        .sum()
        .reset_index()
        .rename(columns={"is_switch": "主約商品主類別切換次數"})
    )

    return switch_df


def add_missing_main_model_features(candidate_df: pd.DataFrame,
                                    policy_df: pd.DataFrame) -> pd.DataFrame:
    out = candidate_df.copy()
    out[ID_COL] = out[ID_COL].astype("string")

    if "主約商品主類別切換次數" not in out.columns:
        switch_df = calc_product_main_type_switch(policy_df)
        switch_df[ID_COL] = switch_df[ID_COL].astype("string")

        out = out.merge(
            switch_df,
            on=ID_COL,
            how="left"
        )
        out["主約商品主類別切換次數"] = out["主約商品主類別切換次數"].fillna(0)

    return out


def add_snapshot_proxy_for_candidate(candidate_df: pd.DataFrame) -> pd.DataFrame:
    out = candidate_df.copy()

    if "snapshot_date" not in out.columns:
        if "最近投保日" in out.columns:
            out["snapshot_date"] = pd.to_datetime(out["最近投保日"], errors="coerce")
        else:
            out["snapshot_date"] = pd.Timestamp("2026-03-01")

    out = add_snapshot_calendar_features(out)
    return out


def build_candidate_income_features_B(raw_df: pd.DataFrame,
                                      candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    對 candidate 補收入特徵，且只保留年收入有值的客戶作為 B 路徑候選
    """
    out = candidate_df.copy()
    out[ID_COL] = out[ID_COL].astype("string")

    if "最近投保日" not in out.columns:
        out["最近投保日"] = pd.Timestamp("2026-03-01")
    else:
        out["最近投保日"] = pd.to_datetime(out["最近投保日"], errors="coerce")

    income_detail_df = prepare_income_detail(raw_df)
    income_detail_df[ID_COL] = income_detail_df[ID_COL].astype("string")

    candidate_snap = out[[ID_COL, "最近投保日"]].rename(columns={"最近投保日": "snapshot_date"}).copy()
    candidate_snap["snapshot_date"] = pd.to_datetime(candidate_snap["snapshot_date"], errors="coerce")

    income_snap = build_income_snapshot_features(
        snapshot_master_df=candidate_snap,
        income_detail_df=income_detail_df
    )

    # income_snap 只有 ID + income features
    keep_cols = [
        ID_COL,
        "snapshot_家庭年收入_萬",
        "snapshot_其他年收入_萬",
        "snapshot_工作年收入_萬",
        "snapshot_年收入_合成_萬",
        "snapshot_log_年收入_合成",
        "snapshot_是否有收入資料"
    ]
    keep_cols = [c for c in keep_cols if c in income_snap.columns]

    income_snap = income_snap[keep_cols].copy()

    out = out.merge(
        income_snap,
        on=ID_COL,
        how="left"
    )

    if "snapshot_是否有收入資料" not in out.columns:
        out["snapshot_是否有收入資料"] = 0

    out["snapshot_是否有收入資料"] = out["snapshot_是否有收入資料"].fillna(0)

    return out


def score_candidates_with_model_B(candidate_rule_scored_df: pd.DataFrame,
                                  policy_df: pd.DataFrame,
                                  raw_df: pd.DataFrame,
                                  main_pack: dict,
                                  income_pack_B: dict,
                                  rule_weight: float = 0.40,
                                  model_weight: float = 0.60):
    """
    做法 B：
    - A 路徑：沒有年收入資料 → 主模型
    - B 路徑：只有年收入有值 → 收入模型
    - A/B 完全不重疊
    - 兩群都各自和 rule score 融合
    """

    candidate_df = candidate_rule_scored_df.copy()
    candidate_df[ID_COL] = candidate_df[ID_COL].astype("string")

    # 補主模型共用特徵
    candidate_df = add_missing_main_model_features(candidate_df, policy_df)
    candidate_df = add_snapshot_proxy_for_candidate(candidate_df)

    # 補收入特徵
    candidate_df = build_candidate_income_features_B(raw_df, candidate_df)

    # ====== 定義 B 路徑：只有年收入有值者 ======
    candidate_df["is_income_B"] = candidate_df["snapshot_年收入_合成_萬"].notna().astype(int)

    # A / B 切分
    candidate_A_df = candidate_df[candidate_df["is_income_B"] == 0].copy()
    candidate_B_df = candidate_df[candidate_df["is_income_B"] == 1].copy()

    # ====== 檢查 A / B 完全不重疊 ======
    ids_A = set(candidate_A_df[ID_COL].dropna().astype("string"))
    ids_B = set(candidate_B_df[ID_COL].dropna().astype("string"))
    overlap_ids = ids_A & ids_B
    assert len(overlap_ids) == 0, f"A / B 路徑有重疊客戶：{len(overlap_ids)}"

    # ====== Rule normalize：全 candidate 共用同一尺標 ======
    rule_min = candidate_df["rule_score"].min()
    rule_max = candidate_df["rule_score"].max()

    if rule_max > rule_min:
        candidate_df["rule_score_norm"] = (
            (candidate_df["rule_score"] - rule_min) / (rule_max - rule_min)
        )
    else:
        candidate_df["rule_score_norm"] = 0.5

    candidate_A_df = candidate_A_df.merge(
        candidate_df[[ID_COL, "rule_score_norm"]],
        on=ID_COL,
        how="left"
    )
    candidate_B_df = candidate_B_df.merge(
        candidate_df[[ID_COL, "rule_score_norm"]],
        on=ID_COL,
        how="left"
    )

    # =====================================================
    # A 路徑：主模型
    # =====================================================
    if len(candidate_A_df) > 0:
        main_feature_cols = main_pack["data"]["feature_cols"]
        missing_main_cols = set(main_feature_cols) - set(candidate_A_df.columns)
        if missing_main_cols:
            raise ValueError(f"A 路徑 candidate 缺少主模型欄位: {missing_main_cols}")

        X_A = candidate_A_df[main_feature_cols].copy()

        main_logit = main_pack["logit_result"]["model"]
        main_xgb_pre = main_pack["xgb_result"]["preprocessor"]
        main_xgb = main_pack["xgb_result"]["model"]

        candidate_A_df["model_prob_logistic"] = main_logit.predict_proba(X_A)[:, 1]

        X_A_t = main_xgb_pre.transform(X_A)
        candidate_A_df["model_prob_xgb"] = main_xgb.predict_proba(X_A_t)[:, 1]

        candidate_A_df["final_score"] = (
            rule_weight * candidate_A_df["rule_score_norm"] +
            model_weight * candidate_A_df["model_prob_xgb"]
        )
        candidate_A_df["模型路徑"] = "A_main_only"
        candidate_A_df["income_model_prob_xgb"] = np.nan
    else:
        candidate_A_df["model_prob_logistic"] = []
        candidate_A_df["model_prob_xgb"] = []
        candidate_A_df["final_score"] = []
        candidate_A_df["模型路徑"] = []
        candidate_A_df["income_model_prob_xgb"] = []

    # =====================================================
    # B 路徑：收入模型
    # =====================================================
    if len(candidate_B_df) > 0:
        income_feature_cols = income_pack_B["data"]["feature_cols"]
        missing_income_cols = set(income_feature_cols) - set(candidate_B_df.columns)
        if missing_income_cols:
            raise ValueError(f"B 路徑 candidate 缺少收入模型欄位: {missing_income_cols}")

        X_B = candidate_B_df[income_feature_cols].copy()

        income_logit = income_pack_B["logit_result"]["model"]
        income_xgb_pre = income_pack_B["xgb_result"]["preprocessor"]
        income_xgb = income_pack_B["xgb_result"]["model"]

        candidate_B_df["model_prob_logistic"] = income_logit.predict_proba(X_B)[:, 1]

        X_B_t = income_xgb_pre.transform(X_B)
        candidate_B_df["income_model_prob_xgb"] = income_xgb.predict_proba(X_B_t)[:, 1]

        # B 路徑 final score：rule + income model
        candidate_B_df["final_score"] = (
            rule_weight * candidate_B_df["rule_score_norm"] +
            model_weight * candidate_B_df["income_model_prob_xgb"]
        )

        candidate_B_df["model_prob_xgb"] = np.nan
        candidate_B_df["模型路徑"] = "B_income_only"
    else:
        candidate_B_df["model_prob_logistic"] = []
        candidate_B_df["income_model_prob_xgb"] = []
        candidate_B_df["final_score"] = []
        candidate_B_df["model_prob_xgb"] = []
        candidate_B_df["模型路徑"] = []

    # ====== 合併，確保與做法 A 不重疊 ======
    scored_df = pd.concat([candidate_A_df, candidate_B_df], axis=0, ignore_index=True)

    # 與做法 A 相比，做法 B 是分流；這裡客戶只能出現一次
    dup_cnt = scored_df[ID_COL].duplicated().sum()
    assert dup_cnt == 0, f"做法 B 最終輸出有重複客戶：{dup_cnt}"

    # ====== 排序 / 分群 / 主因 ======
    scored_df = scored_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    scored_df["final_rank"] = (
        scored_df["final_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    scored_df["final_rank_pct"] = scored_df["final_rank"] / len(scored_df)

    def assign_segment(p):
        if p <= 0.10:
            return "A"
        elif p <= 0.30:
            return "B"
        else:
            return "C"

    scored_df["名單等級"] = scored_df["final_rank_pct"].apply(assign_segment)

    reason_cols = [
        "壽險參與度_score",
        "保費能力_score",
        "關係深度_score",
        "FYC結構_score",
        "近期行為_score"
    ]
    available_reason_cols = [c for c in reason_cols if c in scored_df.columns]

    if available_reason_cols:
        scored_df["推薦主因"] = (
            scored_df[available_reason_cols]
            .idxmax(axis=1)
            .str.replace("_score", "", regex=False)
        )
    else:
        scored_df["推薦主因"] = pd.NA

    output_cols = [
        ID_COL,
        "目前營業單位",
        "目前經紀人1業代",
        "保單數",
        "壽險保單數",
        "產險保單數",
        "累計保單總保費",
        "累計壽險保單總保費",
        "近1年保單數",
        "距離最近投保天數",
        "snapshot_年收入_合成_萬",
        "rule_score",
        "rule_score_norm",
        "model_prob_logistic",
        "model_prob_xgb",          # A 路徑會有
        "income_model_prob_xgb",   # B 路徑會有
        "final_score",
        "final_rank",
        "final_rank_pct",
        "名單等級",
        "推薦主因",
        "模型路徑",
        "is_income_B"
    ]
    output_cols = [c for c in output_cols if c in scored_df.columns]

    return scored_df[output_cols].copy()


# =========================================================
# 7. 摘要 / 報表
# =========================================================
def summarize_split(df: pd.DataFrame, split_name: str):
    total_n = len(df)
    pos_n = int(df["label"].sum())
    neg_n = int(total_n - pos_n)
    pos_rate = pos_n / total_n if total_n > 0 else np.nan

    unique_customers = df[ID_COL].nunique() if ID_COL in df.columns else np.nan

    return {
        "split": split_name,
        "rows": total_n,
        "customers": unique_customers,
        "pos_n": pos_n,
        "neg_n": neg_n,
        "pos_rate": pos_rate
    }


def summarize_model_splits(train_df, valid_df, test_df):
    return pd.DataFrame([
        summarize_split(train_df, "train"),
        summarize_split(valid_df, "valid"),
        summarize_split(test_df, "test")
    ])


# =========================================================
# 8. 做法 B：模型結果摘要表
# =========================================================

def build_model_metrics_summary_B(main_pack: dict,
                                  income_pack_B: dict) -> pd.DataFrame:
    rows = []

    # 主模型
    for df_, model_family, split_name in [
        (main_pack["logit_result"]["valid_metrics"], "main_logistic", "valid"),
        (main_pack["logit_result"]["test_metrics"],  "main_logistic", "test"),
        (main_pack["xgb_result"]["valid_metrics"],   "main_xgb", "valid"),
        (main_pack["xgb_result"]["test_metrics"],    "main_xgb", "test"),
    ]:
        row = df_.iloc[0].to_dict()
        row["model_family"] = model_family
        row["split"] = split_name
        rows.append(row)

    # 收入模型（B）
    for df_, model_family, split_name in [
        (income_pack_B["logit_result"]["valid_metrics"], "incomeB_logistic", "valid"),
        (income_pack_B["logit_result"]["test_metrics"],  "incomeB_logistic", "test"),
        (income_pack_B["xgb_result"]["valid_metrics"],   "incomeB_xgb", "valid"),
        (income_pack_B["xgb_result"]["test_metrics"],    "incomeB_xgb", "test"),
    ]:
        row = df_.iloc[0].to_dict()
        row["model_family"] = model_family
        row["split"] = split_name
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    front_cols = [
        "model_family", "split", "model",
        "roc_auc", "pr_auc", "base_rate",
        "top10_precision", "top20_precision",
        "lift_10", "lift_20"
    ]
    front_cols = [c for c in front_cols if c in summary_df.columns]
    other_cols = [c for c in summary_df.columns if c not in front_cols]

    summary_df = summary_df[front_cols + other_cols]
    return summary_df


def build_split_summary_report_B(main_pack: dict,
                                 income_pack_B: dict) -> pd.DataFrame:
    rows = []

    # 主模型
    main_data = main_pack["data"]
    for split_name in ["train_df", "valid_df", "test_df"]:
        df_ = main_data[split_name]
        rows.append({
            "model_family": "main",
            "split": split_name.replace("_df", ""),
            "rows": len(df_),
            "customers": df_[ID_COL].nunique(),
            "pos_n": int(df_["label"].sum()),
            "neg_n": int(len(df_) - df_["label"].sum()),
            "pos_rate": df_["label"].mean()
        })

    # 收入模型 B
    income_data = income_pack_B["data"]
    for split_name in ["train_df", "valid_df", "test_df"]:
        df_ = income_data[split_name]
        rows.append({
            "model_family": "income_B",
            "split": split_name.replace("_df", ""),
            "rows": len(df_),
            "customers": df_[ID_COL].nunique(),
            "pos_n": int(df_["label"].sum()),
            "neg_n": int(len(df_) - df_["label"].sum()),
            "pos_rate": df_["label"].mean()
        })

    return pd.DataFrame(rows)


def build_feature_importance_report_B(main_pack: dict,
                                      income_pack_B: dict,
                                      top_n: int = 20) -> pd.DataFrame:
    rows = []

    main_imp = main_pack["xgb_result"]["importance_df"].copy().head(top_n)
    main_imp["model_family"] = "main_xgb"
    rows.append(main_imp)

    income_imp = income_pack_B["xgb_result"]["importance_df"].copy().head(top_n)
    income_imp["model_family"] = "incomeB_xgb"
    rows.append(income_imp)

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out


def build_candidate_scoring_summary_B(candidate_output_B_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "metric": "total_candidates",
        "value": len(candidate_output_B_df)
    })

    if "名單等級" in candidate_output_B_df.columns:
        for seg in ["A", "B", "C"]:
            rows.append({
                "metric": f"segment_{seg}_count",
                "value": int((candidate_output_B_df["名單等級"] == seg).sum())
            })

    if "模型路徑" in candidate_output_B_df.columns:
        route_counts = candidate_output_B_df["模型路徑"].value_counts(dropna=False).to_dict()
        for k, v in route_counts.items():
            rows.append({
                "metric": f"route_{k}_count",
                "value": v
            })

    if "is_income_B" in candidate_output_B_df.columns:
        rows.append({
            "metric": "income_B_candidate_count",
            "value": int(candidate_output_B_df["is_income_B"].sum())
        })
        rows.append({
            "metric": "income_B_candidate_rate",
            "value": candidate_output_B_df["is_income_B"].mean()
        })

    if "final_score" in candidate_output_B_df.columns:
        rows.extend([
            {"metric": "final_score_mean", "value": candidate_output_B_df["final_score"].mean()},
            {"metric": "final_score_p50", "value": candidate_output_B_df["final_score"].quantile(0.50)},
            {"metric": "final_score_p90", "value": candidate_output_B_df["final_score"].quantile(0.90)},
            {"metric": "final_score_p95", "value": candidate_output_B_df["final_score"].quantile(0.95)},
            {"metric": "final_score_p99", "value": candidate_output_B_df["final_score"].quantile(0.99)},
        ])

    if "model_prob_xgb" in candidate_output_B_df.columns:
        rows.append({
            "metric": "main_model_scored_count",
            "value": int(candidate_output_B_df["model_prob_xgb"].notna().sum())
        })

    if "income_model_prob_xgb" in candidate_output_B_df.columns:
        rows.append({
            "metric": "income_model_scored_count",
            "value": int(candidate_output_B_df["income_model_prob_xgb"].notna().sum())
        })

    return pd.DataFrame(rows)


def export_modelB_report(main_pack: dict,
                         income_pack_B: dict,
                         candidate_output_B_df: pd.DataFrame,
                         output_path: str = "modelB_report.xlsx"):
    """
    輸出做法 B 的完整報表
    """
    model_metrics_df = build_model_metrics_summary_B(main_pack, income_pack_B)
    split_summary_df = build_split_summary_report_B(main_pack, income_pack_B)
    feature_importance_df = build_feature_importance_report_B(main_pack, income_pack_B, top_n=20)
    candidate_summary_df = build_candidate_scoring_summary_B(candidate_output_B_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        model_metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        split_summary_df.to_excel(writer, sheet_name="split_summary", index=False)
        feature_importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
        candidate_summary_df.to_excel(writer, sheet_name="candidate_summary", index=False)
        candidate_output_B_df.to_excel(writer, sheet_name="candidate_output", index=False)

    print(f"做法 B 報表已輸出：{output_path}")
    


# 執行
main_pack = run_main_model_pipeline(policy_df)

income_pack_B = run_income_model_pipeline_B(df_raw, policy_df)


candidate_output_B_df = score_candidates_with_model_B(
    candidate_rule_scored_df=candidate_rule_scored_df,
    policy_df=policy_df,
    raw_df=df_raw,
    main_pack=main_pack,
    income_pack_B=income_pack_B,
    rule_weight=0.40,
    model_weight=0.60
)

candidate_output_B_df.to_excel("D:/投資型/lump/candidate_output.xlsx", index=False)

candidate_output_B_df["模型路徑"].value_counts(dropna=False)
candidate_output_B_df["名單等級"].value_counts(dropna=False)
candidate_output_B_df.head(20)


export_modelB_report(
    main_pack=main_pack,
    income_pack_B=income_pack_B,
    candidate_output_B_df=candidate_output_B_df,
    output_path="D:/投資型/lump/modelB_report.xlsx"
)


# 評估 main model / income model
compare_df = pd.concat([
    main_pack["xgb_result"]["test_metrics"].assign(model="main"),
    income_pack_B["xgb_result"]["test_metrics"].assign(model="income_B")
])

# 
main_feature_cols = main_pack["feature_cols"]
income_feature_cols = income_pack_B["feature_cols"]

# 取 income test set
income_test_df = income_pack_B["data"]["test_df"].copy()

# 同一批人，用主模型打分
# X_income_test_main = income_test_df[main_pack["feature_cols"]]
X_income_test_main = income_test_df[main_feature_cols]

income_test_df["main_prob"] = main_pack["xgb_result"]["model"].predict_proba(
    main_pack["xgb_result"]["preprocessor"].transform(X_income_test_main)
)[:, 1]

# 收入模型分數
# X_income_test_income = income_test_df[income_pack_B["feature_cols"]]
X_income_test_income = income_test_df[income_feature_cols]

income_test_df["income_prob"] = income_pack_B["xgb_result"]["model"].predict_proba(
    income_pack_B["xgb_result"]["preprocessor"].transform(X_income_test_income)
)[:, 1]


def eval_binary(y_true, y_prob):
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }

main_eval = eval_binary(income_test_df["label"], income_test_df["main_prob"])
income_eval = eval_binary(income_test_df["label"], income_test_df["income_prob"])

print("Main model on income users:", main_eval)
print("Income model:", income_eval)




# 權重
def find_best_weight(eval_df, rule_col="rule_score", model_col="model_prob_xgb"):
    results = []

    for w in np.arange(0, 1.01, 0.05):
        eval_df["final"] = w * eval_df[rule_col] + (1 - w) * eval_df[model_col]

        # top10 precision
        df_sorted = eval_df.sort_values("final", ascending=False)
        top10 = df_sorted.head(int(len(df_sorted)*0.1))

        precision = top10["label"].mean()

        results.append({
            "rule_weight": w,
            "model_weight": 1-w,
            "top10_precision": precision
        })

    return pd.DataFrame(results)

# 主模型 test
main_test_df = main_pack["data"]["test_df"].copy()

# 收入模型 test
income_test_df = income_pack_B["data"]["test_df"].copy()

# 標記來源
main_test_df["is_income_B"] = False
income_test_df["is_income_B"] = True

# 合併（不重疊）
unified_test_df = pd.concat([main_test_df, income_test_df], axis=0, ignore_index=True)

main_feature_cols = main_pack["feature_cols"]

X_main = unified_test_df[main_feature_cols]

unified_test_df["main_prob"] = main_pack["xgb_result"]["model"].predict_proba(
    main_pack["xgb_result"]["preprocessor"].transform(X_main)
)[:, 1]

income_feature_cols = income_pack_B["feature_cols"]

mask_income = unified_test_df["is_income_B"] == True

X_income = unified_test_df.loc[mask_income, income_feature_cols]

unified_test_df.loc[mask_income, "income_prob"] = income_pack_B["xgb_result"]["model"].predict_proba(
    income_pack_B["xgb_result"]["preprocessor"].transform(X_income)
)[:, 1]

unified_test_df["model_prob"] = unified_test_df["main_prob"]

# 有收入 → 用 income model
unified_test_df.loc[mask_income, "model_prob"] = unified_test_df.loc[mask_income, "income_prob"]

# 假設你有 score_tables
unified_test_df = apply_rule_score(unified_test_df, score_tables)

test_df = unified_test_df

weight_df = find_best_weight(
    test_df,
    rule_col="rule_score_norm",
    model_col="model_prob"
)

print(weight_df.sort_values("top10_precision", ascending=False).head())






#
# 假設 income_test_df 已經有：
# label, main_prob, income_prob, rule_score

rule_min = income_test_df["rule_score"].min()
rule_max = income_test_df["rule_score"].max()

if rule_max > rule_min:
    income_test_df["rule_score_norm"] = (
        (income_test_df["rule_score"] - rule_min) / (rule_max - rule_min)
    )
else:
    income_test_df["rule_score_norm"] = 0.5

# 做法 B 的 final score（有收入者用 income model）
RULE_WEIGHT = 0.40
MODEL_WEIGHT = 0.60

income_test_df["final_score"] = (
    RULE_WEIGHT * income_test_df["rule_score_norm"] +
    MODEL_WEIGHT * income_test_df["income_prob"]
)


#

def build_cumulative_curve(df, score_col, label_col="label", n_bins=100):
    """
    依 score 排序後，計算 cumulative conversion rate curve
    """
    tmp = df[[score_col, label_col]].dropna().copy()
    tmp = tmp.sort_values(score_col, ascending=False).reset_index(drop=True)

    # 切成百分位
    tmp["bucket"] = pd.qcut(
        tmp.index,
        q=min(n_bins, len(tmp)),
        labels=False,
        duplicates="drop"
    )

    perf = (
        tmp.groupby("bucket")
        .agg(
            cnt=(label_col, "count"),
            pos=(label_col, "sum")
        )
        .reset_index()
        .sort_values("bucket")
    )

    perf["cum_cnt"] = perf["cnt"].cumsum()
    perf["cum_pos"] = perf["pos"].cumsum()
    perf["cum_conversion_rate"] = perf["cum_pos"] / perf["cum_cnt"]
    perf["population_pct"] = perf["cum_cnt"] / perf["cnt"].sum()

    return perf


curve_main = build_cumulative_curve(income_test_df, "main_prob")
curve_income = build_cumulative_curve(income_test_df, "income_prob")
curve_final = build_cumulative_curve(income_test_df, "final_score")

plt.figure(figsize=(8, 5))
plt.plot(curve_main["population_pct"], curve_main["cum_conversion_rate"], label="Main model")
plt.plot(curve_income["population_pct"], curve_income["cum_conversion_rate"], label="Income model")
plt.plot(curve_final["population_pct"], curve_final["cum_conversion_rate"], label="Final score")

plt.xlabel("Population %")
plt.ylabel("Cumulative Conversion Rate")
plt.title("Main vs Income vs Final Score")
plt.grid(True)
plt.legend()
plt.show()


def top_pct_conversion(df, score_col, label_col="label", p_list=[0.1, 0.2, 0.3]):
    tmp = df[[score_col, label_col]].dropna().copy()
    tmp = tmp.sort_values(score_col, ascending=False).reset_index(drop=True)

    results = []
    for p in p_list:
        top_n = max(int(len(tmp) * p), 1)
        conv = tmp.head(top_n)[label_col].mean()
        results.append({
            "score_type": score_col,
            "top_pct": p,
            "conversion_rate": conv
        })
    return pd.DataFrame(results)

top_compare_df = pd.concat([
    top_pct_conversion(income_test_df, "main_prob"),
    top_pct_conversion(income_test_df, "income_prob"),
    top_pct_conversion(income_test_df, "final_score"),
], axis=0, ignore_index=True)

print(top_compare_df)


