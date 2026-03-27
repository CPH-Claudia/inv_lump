# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:29:16 2026

@author: Z01788
"""

import pandas as pd
import numpy as np

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
LABEL_CUTOFF_DATE = DATA_END_DATE - pd.Timedelta(days=HORIZON_DAYS)


# 抓曾有 "行動投保受理號" 的保戶保單資料
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
        .astype(str)
        .unique()
    )
    return mobile_ids

def subset_policy_by_mobile_customers(policy_df: pd.DataFrame, mobile_customer_ids: set) -> pd.DataFrame:
    df = policy_df.copy()
    df.columns = df.columns.str.strip()

    out = df[df[ID_COL].astype(str).isin(mobile_customer_ids)].copy()
    return out

# 準備 income 明細資料
def prepare_income_detail(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    required_cols = [ID_COL, DATE_COL, MOBILE_COL, INCOME_FAMILY_COL, INCOME_OTHER_COL, INCOME_WORK_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"raw_df 缺少必要欄位: {missing_cols}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    for c in [INCOME_FAMILY_COL, INCOME_OTHER_COL, INCOME_WORK_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 只保留有行動投保受理號的紀錄來取收入
    df = df[df[MOBILE_COL].notna()].copy()

    # 負值視為缺值
    for c in [INCOME_FAMILY_COL, INCOME_OTHER_COL, INCOME_WORK_COL]:
        df.loc[df[c] < 0, c] = np.nan

    # 合成年收入：家庭優先；否則用工作+其他
    df["年收入_合成"] = np.where(
        df[INCOME_FAMILY_COL].notna(),
        df[INCOME_FAMILY_COL],
        df[[INCOME_WORK_COL, INCOME_OTHER_COL]].fillna(0).sum(axis=1)
    )

    # 如果家庭缺值且工作+其他都缺，改回 NaN
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

    # 若同一天多筆，保留最後一筆有資料的
    income_df = (
        income_df.sort_values([ID_COL, DATE_COL])
        .groupby([ID_COL, DATE_COL], as_index=False)
        .last()
    )

    return income_df

# income 做成 as-of snapshot 特徵
def build_income_snapshot_features(snapshot_master_df: pd.DataFrame,
                                   income_detail_df: pd.DataFrame) -> pd.DataFrame:
    snap = snapshot_master_df[[ID_COL, "snapshot_date"]].copy()
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"], errors="coerce")

    income = income_detail_df.copy()
    income[DATE_COL] = pd.to_datetime(income[DATE_COL], errors="coerce")

    # 型別統一
    snap[ID_COL] = snap[ID_COL].astype("string")
    income[ID_COL] = income[ID_COL].astype("string")

    # 去掉關鍵鍵值缺失
    snap = snap[snap[ID_COL].notna() & snap["snapshot_date"].notna()].copy()
    income = income[income[ID_COL].notna() & income[DATE_COL].notna()].copy()

    # 只保留雙方交集客戶，減少不必要運算
    common_ids = set(snap[ID_COL].unique()) & set(income[ID_COL].unique())
    snap = snap[snap[ID_COL].isin(common_ids)].copy()
    income = income[income[ID_COL].isin(common_ids)].copy()

    merged_list = []

    for cust_id, snap_g in snap.groupby(ID_COL, dropna=False):
        income_g = income[income[ID_COL] == cust_id].copy()

        if income_g.empty:
            # 沒有任何收入資料，直接補空欄
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

        # merge_asof 後會有兩個 ID 欄位，保留左邊的
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

    # 若 merge 後沒有收入資料欄位，補上
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


# 建立「行動投保客戶 subset」的 snapshot dataset
def build_mobile_income_model_dataset(raw_df: pd.DataFrame,
                                      policy_df: pd.DataFrame,
                                      data_end_date: str = "2026-03-01",
                                      horizon_days: int = 365,
                                      balance_train_set: bool = True,
                                      neg_to_pos_ratio: float = 3.0):
    # 1) 行動投保客戶
    mobile_customer_ids = get_mobile_customer_ids(raw_df)

    # 2) 保留這群客戶的所有保單歷史
    policy_mobile_df = subset_policy_by_mobile_customers(policy_df, mobile_customer_ids)

    if policy_mobile_df.empty:
        raise ValueError("行動投保客戶子集為空，無法建模。")

    # 3) 建 snapshot master
    snapshot_master_df = build_snapshot_master(
        policy_df=policy_mobile_df,
        data_end_date=pd.Timestamp(data_end_date),
        horizon_days=horizon_days
    )

    # 4) 建一般 snapshot features
    snapshot_feature_df = build_snapshot_features(
        policy_df=policy_mobile_df,
        snapshot_master_df=snapshot_master_df
    )

    # 5) 建 income snapshot features
    income_detail_df = prepare_income_detail(raw_df)
    income_snapshot_df = build_income_snapshot_features(
        snapshot_master_df=snapshot_master_df,
        income_detail_df=income_detail_df
    )

    # 6) merge income features
    snapshot_feature_df = snapshot_feature_df.merge(
        income_snapshot_df,
        on=[ID_COL, "snapshot_date"],
        how="left"
    )

    # 7) 補 snapshot calendar features
    snapshot_feature_df = add_snapshot_calendar_features(snapshot_feature_df)

    # 8) 時間切分
    train_df, valid_df, test_df = split_train_valid_test(snapshot_feature_df)

    # 9) 平衡 train
    train_df_balanced = balance_train(
        train_df,
        label_col="label",
        neg_to_pos_ratio=neg_to_pos_ratio
    ) if balance_train_set else train_df.copy()

    return {
        "mobile_customer_ids": mobile_customer_ids,
        "policy_mobile_df": policy_mobile_df,
        "snapshot_master_df": snapshot_master_df,
        "snapshot_feature_df": snapshot_feature_df,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "train_df_balanced": train_df_balanced
    }

# 
def get_model_feature_cols_base(df: pd.DataFrame):
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


def get_model_feature_cols_with_income(df: pd.DataFrame):
    feature_cols = get_model_feature_cols_base(df)

    income_cols = [
        "snapshot_是否有收入資料",
        "snapshot_年收入_合成_萬",
        "snapshot_log_年收入_合成",
    ]

    feature_cols += [c for c in income_cols if c in df.columns]
    return feature_cols

def summarize_split(df: pd.DataFrame, split_name: str):
    total_n = len(df)
    pos_n = int(df["label"].sum())
    neg_n = int(total_n - pos_n)
    pos_rate = pos_n / total_n if total_n > 0 else np.nan

    unique_customers = df[ID_COL].nunique() if ID_COL in df.columns else np.nan

    summary = {
        "split": split_name,
        "rows": total_n,
        "customers": unique_customers,
        "pos_n": pos_n,
        "neg_n": neg_n,
        "pos_rate": pos_rate
    }
    return summary

def summarize_model_splits(train_df, valid_df, test_df):
    summary_df = pd.DataFrame([
        summarize_split(train_df, "train"),
        summarize_split(valid_df, "valid"),
        summarize_split(test_df, "test")
    ])
    return summary_df

def prepare_xy_by_feature_cols(train_df, valid_df, test_df, feature_cols, label_col="label"):
    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].copy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[label_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[label_col].copy()

    return X_train, y_train, X_valid, y_valid, X_test, y_test

policy_df[ID_COL] = policy_df[ID_COL].astype("string")
df_raw[ID_COL] = df_raw[ID_COL].astype("string")

# === 建子集資料 ===
mobile_model_data = build_mobile_income_model_dataset(
    raw_df=df_raw,
    policy_df=policy_df,
    data_end_date="2026-03-01",
    horizon_days=365,
    balance_train_set=True,
    neg_to_pos_ratio=3.0
)

train_df = mobile_model_data["train_df"]
valid_df = mobile_model_data["valid_df"]
test_df = mobile_model_data["test_df"]
train_df_balanced = mobile_model_data["train_df_balanced"]
snapshot_feature_df_mobile = mobile_model_data["snapshot_feature_df"]

# === 看資料量與比例 ===
split_summary_df = summarize_model_splits(train_df, valid_df, test_df)
print(split_summary_df.to_string(index=False))


train_pos = int(train_df["label"].sum())
valid_pos = int(valid_df["label"].sum())
test_pos = int(test_df["label"].sum())

enough_data = (train_pos >= 200) and (valid_pos >= 50) and (test_pos >= 50)

print("是否建議進行收入模型實驗：", enough_data)

# 基礎版
base_feature_cols = get_model_feature_cols_base(snapshot_feature_df_mobile)

X_train_base, y_train_base, X_valid_base, y_valid_base, X_test_base, y_test_base = prepare_xy_by_feature_cols(
    train_df_balanced, valid_df, test_df, base_feature_cols, label_col="label"
)

logit_base = train_logistic_baseline(
    X_train_base, y_train_base,
    X_valid_base, y_valid_base,
    X_test_base, y_test_base
)

xgb_base = train_xgboost_model(
    X_train_base, y_train_base,
    X_valid_base, y_valid_base,
    X_test_base, y_test_base
)

base_metrics = pd.concat([
    logit_base["valid_metrics"],
    logit_base["test_metrics"],
    xgb_base["valid_metrics"],
    xgb_base["test_metrics"]
], axis=0)

print("\n=== 基礎版模型 ===")
print(base_metrics.to_string(index=False))

# 加收入版
income_feature_cols = get_model_feature_cols_with_income(snapshot_feature_df_mobile)

X_train_inc, y_train_inc, X_valid_inc, y_valid_inc, X_test_inc, y_test_inc = prepare_xy_by_feature_cols(
    train_df_balanced, valid_df, test_df, income_feature_cols, label_col="label"
)

logit_income = train_logistic_baseline(
    X_train_inc, y_train_inc,
    X_valid_inc, y_valid_inc,
    X_test_inc, y_test_inc
)

xgb_income = train_xgboost_model(
    X_train_inc, y_train_inc,
    X_valid_inc, y_valid_inc,
    X_test_inc, y_test_inc
)

income_metrics = pd.concat([
    logit_income["valid_metrics"],
    logit_income["test_metrics"],
    xgb_income["valid_metrics"],
    xgb_income["test_metrics"]
], axis=0)

print("\n=== 加收入版模型 ===")
print(income_metrics.to_string(index=False))

# 收入欄位覆蓋率
income_coverage = {
    "train_income_known_rate": train_df["snapshot_是否有收入資料"].mean() if "snapshot_是否有收入資料" in train_df.columns else np.nan,
    "valid_income_known_rate": valid_df["snapshot_是否有收入資料"].mean() if "snapshot_是否有收入資料" in valid_df.columns else np.nan,
    "test_income_known_rate": test_df["snapshot_是否有收入資料"].mean() if "snapshot_是否有收入資料" in test_df.columns else np.nan,
}

print("\n=== 收入資料覆蓋率 ===")
for k, v in income_coverage.items():
    print(k, v)
    





